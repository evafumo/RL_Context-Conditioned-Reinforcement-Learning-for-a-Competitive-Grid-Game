"""Ambienti paralleli CPU per DQN in modalita' MDP o POMDP.

Una chiamata step() = UNA attivazione per ambiente, inclusi Opponent e
DUMMY. I worker Windows-spawn non importano PyTorch, non creano Q-network e
non inizializzano CUDA: l'inferenza batched resta al collector nel
processo principale.

Modalita' POMDP: ogni worker utilizza POMDPGame e conserva un belief bayesiano
separato per ciascun ambiente. Le ultime 6 feature riportano b_t, NON una
one-hot della policy vera. opponent_ids ed episode_seeds sono oscurati negli
snapshot pubblici (sentinella -1); i seed reali rimangono nei worker.
Il belief viene aggiornato a ogni decisione osservabile dell'Opponent usando
lo stato PRE-azione; viene resettato soltanto a inizio episodio.
Per attivare il percorso POMDP: ParallelGridGames(..., observation_mode="pomdp").
I moduli environment.pomdp, environment.belief e le funzioni POMDP di
environment.observations devono essere presenti nel progetto.

Distinzione FONDAMENTALE per DQN:
  * step.next_observation: stati correnti per lo step SEGUENTE; dopo done
    appartengono gia' al NUOVO episodio (auto-reset).
  * step.transition_next: successori del replay. Per done=True contiene
    SOLTANTO segnaposto a zero, NON lo stato appena resettato. Il collector
    DEVE passare next_observation=None e next_action_mask=None al ReplayBuffer
    quando done=True; per gli altri usa step.transition_next.

L'API accetta seed_plan=SeedPlan(...) (quando utils/seeds.py e' disponibile,
come nel PPO) oppure base_seed=<int> con un piano locale deterministico.
I seed degli episodi dipendono dall'ID ambiente LOGICO e dal suo contatore,
non dal numero di worker. Nessuna policy RL gira nei processi figli.

Su Windows invocare l'avvio dentro if __name__ == '__main__':.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from numbers import Integral
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any, Protocol, Sequence

import numpy as np

from environment.action_mask import DUMMY_ACTION_ID, NUM_ACTIONS
from environment.context import ContextDistribution, DEFAULT_CONTEXT_DISTRIBUTION
from environment.game import HORIZON
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE


class SeedPlanLike(Protocol):
    """Interfaccia minima del SeedPlan gia' usato dal progetto PPO."""

    def environment_seed(self, environment_id: int) -> int: ...
    def episode_seed(self, episode_id: int) -> int: ...


@dataclass(frozen=True, slots=True)
class _LocalSeedPlan:
    """Fallback deterministico: evita di obbligare a copiare subito seeds.py."""

    base_seed: int

    def _derive(self, namespace: bytes, index: int) -> int:
        if type(index) is not int or index < 0 or index >= 2**64:
            raise ValueError("ID del seed fuori dall'intervallo uint64")
        payload = self.base_seed.to_bytes(8, "little") + index.to_bytes(8, "little")
        return int.from_bytes(blake2b(payload, digest_size=8, person=namespace).digest(), "little")

    def environment_seed(self, environment_id: int) -> int:
        return self._derive(b"dqn-env", environment_id)

    def episode_seed(self, episode_id: int) -> int:
        return self._derive(b"dqn-ep", episode_id)


@dataclass(frozen=True, slots=True)
class ParallelObservation:
    """Snapshot N stati NON terminali, ordinati per ID ambiente logico.

    Gli ndarray sono copie CPU indipendenti dalla memoria dei worker.
    I tre array usati come input PyTorch restano scrivibili per consentire
    torch.from_numpy() senza warning o copie; NON modificarli durante lo step.
    """

    spatial: np.ndarray           # (N,4,9,9) float32
    features: np.ndarray          # (N,13) float32
    action_masks: np.ndarray      # (N,7) bool
    is_agent_decision: np.ndarray # (N,) bool
    map_ids: np.ndarray           # (N,) int64
    opponent_ids: np.ndarray      # (N,) int64; -1 in modalita' POMDP (K nascosto)
    activations: np.ndarray       # (N,) int64, 1..100
    episode_seeds: tuple[int, ...]  # -1 in modalita' POMDP (nessun seed pubblico)

    @property
    def num_envs(self) -> int:
        return len(self.episode_seeds)


@dataclass(frozen=True, slots=True)
class ParallelTransitionNext:
    """Successori da usare nel replay solo sulle righe con done=False.

    Per done=True: tutti i campi della riga sono zero/False e NON sono una
    Observation valida dell'MDP/POMDP. Usare None nel replay per tali righe.
    """

    spatial: np.ndarray       # (N,4,9,9) float32
    features: np.ndarray      # (N,13) float32
    action_masks: np.ndarray  # (N,7) bool


@dataclass(frozen=True, slots=True)
class ParallelStep:
    """Risultati PRE-step, successori del replay e snapshot per il prossimo step.

    terminal_utilities: -1 se non terminale, 0 pareggio/sconfitta, 1 vittoria.
    terminal_episode_metrics: opzionale; presente soltanto con diagnostica
    attiva, e contenente dizionari per gli episodi terminati in questo step.
    """

    rewards: np.ndarray                  # (N,) float32
    dones: np.ndarray                    # (N,) bool
    terminal_utilities: np.ndarray       # (N,) int8
    transition_next: ParallelTransitionNext
    next_observation: ParallelObservation
    # Solo se collect_episode_metrics=True. None negli step non terminali.
    # I riepiloghi sono quelli dell'episodio TERMINATO, mai del successivo.
    terminal_episode_metrics: tuple[dict[str, Any] | None, ...] | None = None


def _local_snapshot(games: Sequence[object]) -> dict[str, object]:
    """Produce batch CPU MDP oppure POMDP, senza pubblicare K nel POMDP.

    Per compatibilita' con i test originali accetta GridGame in modalita' MDP.
    POMDPGame usa esclusivamente observe(), senza accedere al vero contesto.
    Un batch non puo' mischiare le due modalita'.
    """
    from environment.game import GameState, GridGame
    from environment.observations import stack_observations, encode_observation
    from environment.action_mask import (
        legal_action_mask, legal_actions_to_mask, stack_action_masks,
    )

    if not games:
        raise ValueError("Snapshot senza ambienti")
    if all(isinstance(game, GridGame) for game in games):
        states = [game.state for game in games]
        if not all(isinstance(state, GameState) for state in states):
            raise RuntimeError("Non pubblicare uno stato terminale come osservazione")
        contexts = [game.context for game in games]
        batch = stack_observations([
            encode_observation(state, context)
            for state, context in zip(states, contexts)
        ])
        masks = [legal_action_mask(game) for game in games]
        map_ids = [context.map_id for context in contexts]
        opponent_ids = [context.opponent_id for context in contexts]
        episode_seeds = tuple(game.episode_seed for game in games)
    else:
        # L'import e' lazy: la versione MDP originale non richiede i moduli
        # POMDP se observation_mode='mdp'.
        from environment.pomdp import POMDPGame
        from environment.observations import encode_pomdp_observation

        if not all(isinstance(game, POMDPGame) for game in games):
            raise TypeError("Non mescolare GridGame e POMDPGame nello stesso worker")
        visible = [game.observe() for game in games]
        states = [observation.state for observation in visible]
        batch = stack_observations([
            encode_pomdp_observation(observation) for observation in visible
        ])
        # action_mask.legal_action_mask accetta SOLO GridGame: la vista
        # POMDP usa la medesima fonte di verita' game.legal_actions().
        masks = [legal_actions_to_mask(game.legal_actions()) for game in games]
        map_ids = [observation.map_id for observation in visible]
        # Anche la diagnostica dell'osservazione e' non informativa su K.
        opponent_ids = [-1] * len(games)
        # Il seed del simulatore consentirebbe di ricostruire il contesto
        # tramite il PRNG: non deve far parte dell'osservazione POMDP.
        episode_seeds = (-1,) * len(games)

    return {
        "spatial": batch.spatial,
        "features": batch.features,
        "action_masks": stack_action_masks(masks),
        "is_agent_decision": np.asarray(
            [not mask[DUMMY_ACTION_ID] for mask in masks], dtype=np.bool_,
        ),
        "map_ids": np.asarray(map_ids, dtype=np.int64),
        "opponent_ids": np.asarray(opponent_ids, dtype=np.int64),
        "activations": np.asarray([state.activation for state in states], dtype=np.int64),
        "episode_seeds": episode_seeds,
    }


def _worker_main(
    connection: Connection,
    worker_id: int,
    environment_ids: tuple[int, ...],
    num_envs: int,
    seed_plan: SeedPlanLike,
    shaping_lambda: float,
    context_distribution: ContextDistribution,
    initial_episode_indices: tuple[int, ...],
    collect_episode_metrics: bool,
    observation_mode: str,
    belief_prior: tuple[float, ...] | None,
) -> None:
    """Target top-level pickle-friendly per Windows spawn: SOLTANTO CPU."""
    try:
        from environment.game import GridGame, HORIZON, GameState, TERMINAL_STATE
        if collect_episode_metrics:
            from agent.behavior_metrics import EpisodeEventCounters

        if observation_mode == "pomdp":
            from environment.pomdp import POMDPGame
            game_class = POMDPGame
        elif observation_mode == "mdp":
            game_class = GridGame
        else:
            raise ValueError("Modalita' osservazione sconosciuta nel worker")

        games = [
            game_class(
                seed=seed_plan.environment_seed(env_id),
                shaping_lambda=shaping_lambda,
                context_distribution=context_distribution,
                **({"belief_prior": belief_prior} if observation_mode == "pomdp" else {}),
            )
            for env_id in environment_ids
        ]
        episode_indices = list(initial_episode_indices)
        for local_id, game in enumerate(games):
            episode_id = environment_ids[local_id] + episode_indices[local_id] * num_envs
            game.reset(seed=seed_plan.episode_seed(episode_id))

        counters = ([EpisodeEventCounters() for _ in games]
                    if collect_episode_metrics else None)
        connection.send(("ok", _local_snapshot(games)))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(f"Comando IPC sconosciuto: {command!r}")
            actions: tuple[int, ...] = argument
            if len(actions) != len(games):
                raise ValueError("Numero di azioni non valido nel worker")

            rewards = np.empty(len(games), dtype=np.float32)
            dones = np.empty(len(games), dtype=np.bool_)
            utilities = np.full(len(games), -1, dtype=np.int8)
            terminal_metrics: list[dict[str, Any] | None] | None = (
                [None] * len(games) if collect_episode_metrics else None
            )
            for local_id, (game, action) in enumerate(zip(games, actions)):
                previous = game.state
                result = game.step(action)
                # Non saltare step né saltare turni avversari/forzati.
                if (result.state != previous
                        or result.action.value != action
                        or result.terminated != (previous.activation == HORIZON)):
                    raise RuntimeError("Violata la semantica di una attivazione MDP")
                if result.terminated != (result.next_state is TERMINAL_STATE):
                    raise RuntimeError("Violata la semantica dello stato terminale")
                if not result.terminated and not isinstance(result.next_state, GameState):
                    raise RuntimeError("Stato successivo non terminale non valido")
                if counters is not None:
                    counters[local_id].record(result)
                rewards[local_id] = result.reward
                dones[local_id] = result.terminated
                if result.terminated:
                    if result.outcome is None:
                        raise RuntimeError("Terminale senza outcome")
                    utilities[local_id] = result.outcome.utility
                    if counters is not None:
                        assert terminal_metrics is not None
                        terminal_metrics[local_id] = counters[local_id].finish(result.outcome)
                        counters[local_id] = EpisodeEventCounters()
                    episode_indices[local_id] += 1
                    episode_id = environment_ids[local_id] + episode_indices[local_id] * num_envs
                    game.reset(seed=seed_plan.episode_seed(episode_id))

            # Snapshot di tutti gli ambienti dopo lo step / eventuale reset.
            # Il main rende separati e non confondibili i successori del replay.
            response = {
                "rewards": rewards,
                "dones": dones,
                "terminal_utilities": utilities,
                "next_observation": _local_snapshot(games),
            }
            if terminal_metrics is not None:
                response['terminal_episode_metrics'] = tuple(terminal_metrics)
            connection.send(("ok", response))
    except EOFError:
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class ParallelGridGames:
    """N giochi CPU MDP/POMDP su W worker 'spawn'; CUDA nel main.

    Si possono passare:
      seed_plan=SeedPlan(seed)  (compatibile con l'API PPO), oppure
      base_seed=seed           (autonomo, senza utils/seeds.py).

    I due sistemi danno stream differenti ma ciascuno e' riproducibile per
    qualunque numero di worker, a parita' di num_envs e azioni.

    collect_episode_metrics=False di default: training e benchmark non
    ricevono dati diagnostici aggiuntivi. True per la valutazione C18-C20.

    observation_mode='mdp' mantiene la retrocompatibilita' originale;
    observation_mode='pomdp' usa POMDPGame + BayesianOpponentBelief.
    belief_prior e' indipendente dal context_distribution del simulatore:
    anche se il valutatore fissa un avversario, il belief parte dal prior
    esplicito oppure dal prior uniforme delle sei policy.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        num_workers: int,
        seed_plan: SeedPlanLike | None = None,
        base_seed: int | None = None,
        shaping_lambda: float = 0.1,
        context_distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
        timeout_s: float = 90.0,
        episode_indices: Sequence[int] | None = None,
        collect_episode_metrics: bool = False,
        observation_mode: str = "mdp",
        belief_prior: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if observation_mode not in ("mdp", "pomdp") or type(observation_mode) is not str:
            raise ValueError("observation_mode deve essere 'mdp' o 'pomdp'")
        if observation_mode == "mdp" and belief_prior is not None:
            raise ValueError("belief_prior e' disponibile solo in observation_mode='pomdp'")
        if observation_mode == "pomdp":
            from environment.belief import initial_belief
            validated_prior = initial_belief(belief_prior)
            worker_prior: tuple[float, ...] | None = tuple(
                float(probability) for probability in validated_prior
            )
        else:
            worker_prior = None

        if type(collect_episode_metrics) is not bool:
            raise TypeError('collect_episode_metrics deve essere bool')
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs deve essere un int positivo")
        if type(num_workers) is not int or not 1 <= num_workers <= num_envs:
            raise ValueError("num_workers deve essere in [1, num_envs]")
        if (seed_plan is None) == (base_seed is None):
            raise ValueError("Fornire esattamente uno fra seed_plan e base_seed")
        if base_seed is not None:
            if type(base_seed) is not int or not 0 <= base_seed < 2**64:
                raise ValueError("base_seed deve essere un intero uint64, non bool")
            selected_plan: SeedPlanLike = _LocalSeedPlan(base_seed)
        else:
            if not all(callable(getattr(seed_plan, attr, None))
                       for attr in ("environment_seed", "episode_seed")):
                raise TypeError("seed_plan deve esporre environment_seed ed episode_seed")
            selected_plan = seed_plan  # type: ignore[assignment]
        if not isinstance(context_distribution, ContextDistribution):
            raise TypeError("context_distribution deve essere ContextDistribution")
        if (isinstance(shaping_lambda, bool) or not isinstance(shaping_lambda, (int, float))
                or not isfinite(shaping_lambda) or shaping_lambda < 0):
            raise ValueError("shaping_lambda deve essere finito e >= 0")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s deve essere finito e > 0")
        if episode_indices is None:
            initial_indices = (0,) * num_envs
        else:
            if (isinstance(episode_indices, (str, bytes))
                    or not isinstance(episode_indices, Sequence)):
                raise TypeError("episode_indices deve essere una sequenza")
            initial_indices = tuple(episode_indices)
            if len(initial_indices) != num_envs:
                raise ValueError("episode_indices deve avere lunghezza num_envs")
            if any(type(i) is not int or i < 0 for i in initial_indices):
                raise ValueError("episode_indices richiede interi >= 0 (non bool)")

        # Fail-fast sui limiti di seed, prima di avviare nuovi processi.
        for env_id, index in enumerate(initial_indices):
            selected_plan.environment_seed(env_id)
            selected_plan.episode_seed(env_id + index * num_envs)

        self.collect_episode_metrics = collect_episode_metrics
        self.observation_mode = observation_mode
        self.belief_prior = worker_prior
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.seed_plan = selected_plan
        self.shaping_lambda = float(shaping_lambda)
        self.context_distribution = context_distribution
        self.timeout_s = float(timeout_s)
        self._episode_indices = list(initial_indices)
        self._groups = tuple(
            tuple(range(worker_id, num_envs, num_workers))
            for worker_id in range(num_workers)
        )
        self._workers: list[mp.Process] = []
        self._connections: list[Connection] = []
        self._observation: ParallelObservation | None = None
        self._started = False
        self._closed = False

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """PID dei worker spawn per diagnostica e benchmark."""
        return tuple(int(p.pid) for p in self._workers if p.pid is not None)

    @property
    def observation(self) -> ParallelObservation:
        if not self._started or self._closed or self._observation is None:
            raise RuntimeError("Avviare ParallelGridGames con start() o with")
        return self._observation

    @property
    def episode_indices(self) -> tuple[int, ...]:
        _ = self.observation
        return tuple(self._episode_indices)

    def checkpoint_episode_indices(self) -> tuple[int, ...]:
        """Ripristinabile soltanto se TUTTI gli ambienti sono all'attivazione 1.

        Non serializza stati intermedi: salvare i contatori a meta' partita
        e ripartire da un nuovo episodio produrrebbe una traiettoria diversa.
        """
        if not np.all(self.observation.activations == 1):
            raise RuntimeError("Checkpoint ambientale solo al confine episodio")
        return tuple(self._episode_indices)

    def __enter__(self) -> ParallelGridGames:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _receive_all(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for worker_id, connection in enumerate(self._connections):
            if not connection.poll(self.timeout_s):
                raise RuntimeError(f"Timeout del worker {worker_id}")
            try:
                status, payload = connection.recv()
            except (OSError, EOFError) as exc:
                raise RuntimeError(f"Pipe interrotta dal worker {worker_id}") from exc
            if status != "ok":
                raise RuntimeError(f"Errore nel worker {worker_id}:\n{payload}")
            if not isinstance(payload, dict):
                raise RuntimeError("Risposta IPC malformata")
            payloads.append(payload)
        return payloads

    def _combine_snapshots(
        self, payloads: Sequence[dict[str, object]]
    ) -> ParallelObservation:
        if len(payloads) != self.num_workers:
            raise RuntimeError("Numero di snapshot IPC errato")
        n = self.num_envs
        arrays: dict[str, np.ndarray] = {
            "spatial": np.empty((n, *SPATIAL_SHAPE), dtype=np.float32),
            "features": np.empty((n, FEATURE_SIZE), dtype=np.float32),
            "action_masks": np.empty((n, NUM_ACTIONS), dtype=np.bool_),
            "is_agent_decision": np.empty(n, dtype=np.bool_),
            "map_ids": np.empty(n, dtype=np.int64),
            "opponent_ids": np.empty(n, dtype=np.int64),
            "activations": np.empty(n, dtype=np.int64),
        }
        seeds: list[int] = [0] * n
        for indices, payload in zip(self._groups, payloads):
            ix = np.asarray(indices, dtype=np.intp)
            for name, target in arrays.items():
                value = payload[name]
                if (not isinstance(value, np.ndarray)
                        or value.shape != (len(ix), *target.shape[1:])
                        or value.dtype != target.dtype):
                    raise RuntimeError(f"Snapshot IPC non valido: {name}")
                target[ix] = value
            values = payload["episode_seeds"]
            if not isinstance(values, tuple) or len(values) != len(indices):
                raise RuntimeError("Snapshot IPC: episode_seeds non valido")
            for env_id, value in zip(indices, values):
                if type(value) is not int:
                    raise RuntimeError("Episode seed non intero")
                seeds[env_id] = value

        masks = arrays["action_masks"]
        if (not masks.any(axis=1).all()
                or not np.array_equal(arrays["is_agent_decision"], ~masks[:, DUMMY_ACTION_ID])
                ):
            raise RuntimeError("Maschere / turni forzati incoerenti")
        # Non duplicare le regole: applicare il validatore della Fase 2.
        from environment.action_mask import _validate_action_mask
        for mask in masks:
            _validate_action_mask(mask)
        if not np.isfinite(arrays["spatial"]).all() or not np.isfinite(arrays["features"]).all():
            raise RuntimeError("Osservazioni IPC non finite")
        if np.any((arrays["activations"] < 1) | (arrays["activations"] > HORIZON)):
            raise RuntimeError("Attivazioni IPC fuori orizzonte")
        if self.observation_mode == "pomdp":
            if not np.all(arrays["opponent_ids"] == -1):
                raise RuntimeError("Identita' K esposta nelle osservazioni POMDP")
            if any(seed != -1 for seed in seeds):
                raise RuntimeError("Episode seed esposto nelle osservazioni POMDP")
            # Le sei feature devono essere un posterior normalizzato.
            from environment.observations import BELIEF_FEATURE_OFFSET, BELIEF_SIZE
            belief = arrays["features"][:, BELIEF_FEATURE_OFFSET:]
            if (belief.shape != (n, BELIEF_SIZE)
                    or np.any(belief < 0.0)
                    or not np.allclose(belief.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)):
                raise RuntimeError("Belief POMDP malformato nell'IPC")
        # I batch osservazione sono scrivibili: to_torch_batch/to_torch_action_masks
        # usano torch.from_numpy() senza copia (PyTorch richiede array writable).
        # L'applicazione deve trattare gli snapshot come dati in sola lettura.
        return ParallelObservation(**arrays, episode_seeds=tuple(seeds))

    def start(self) -> ParallelGridGames:
        """Avvia una sola volta; su Windows serve il main guard."""
        if self._closed or self._started:
            raise RuntimeError("ParallelGridGames non riavviabile")
        ctx = mp.get_context("spawn")
        try:
            for worker_id, indices in enumerate(self._groups):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_worker_main,
                    args=(
                        child_conn, worker_id, indices, self.num_envs,
                        self.seed_plan, self.shaping_lambda,
                        self.context_distribution,
                        tuple(self._episode_indices[i] for i in indices),
                        self.collect_episode_metrics,
                        self.observation_mode, self.belief_prior,
                    ),
                    name=f"DQNGridGameWorker-{worker_id}",
                )
                try:
                    proc.start()
                except Exception:
                    parent_conn.close()
                    raise
                finally:
                    child_conn.close()
                self._connections.append(parent_conn)
                self._workers.append(proc)
            self._observation = self._combine_snapshots(self._receive_all())
            self._started = True
            return self
        except Exception:
            self.close()
            raise

    def step(self, actions: Sequence[int] | np.ndarray) -> ParallelStep:
        """Esegue una attivazione per ambiente con prevalidazione delle azioni.

        Errori di input (shape/tipo/masking) lasciano gli ambienti immutati.
        Errori IPC chiudono invece l'intero gruppo: nessuno stato parziale.
        """
        obs = self.observation
        if isinstance(actions, np.ndarray):
            if actions.shape != (self.num_envs,):
                raise ValueError("actions deve avere shape (num_envs,)")
            chosen = actions.tolist()
        else:
            if isinstance(actions, (str, bytes)):
                raise TypeError("actions deve essere una sequenza di ID interi")
            chosen = list(actions)
        if (len(chosen) != self.num_envs
                or any(not isinstance(action, Integral) or isinstance(action, (bool, np.bool_))
                       or not 0 <= int(action) < NUM_ACTIONS
                       for action in chosen)):
            raise ValueError("actions: richiesti N ID interi tra 0 e 6 (non bool)")
        chosen = tuple(int(action) for action in chosen)
        if any(not obs.action_masks[i, action] for i, action in enumerate(chosen)):
            raise ValueError("Azione illegale rispetto alla maschera dell'ambiente")

        try:
            for connection, indices in zip(self._connections, self._groups):
                connection.send(("step", tuple(chosen[i] for i in indices)))
            payloads = self._receive_all()
            rewards = np.empty(self.num_envs, dtype=np.float32)
            dones = np.empty(self.num_envs, dtype=np.bool_)
            utilities = np.empty(self.num_envs, dtype=np.int8)
            for indices, payload in zip(self._groups, payloads):
                ix = np.asarray(indices, dtype=np.intp)
                for name, target in (
                    ("rewards", rewards), ("dones", dones),
                    ("terminal_utilities", utilities),
                ):
                    value = payload[name]
                    if (not isinstance(value, np.ndarray)
                            or value.dtype != target.dtype
                            or value.shape != (len(ix),)):
                        raise RuntimeError(f"Payload IPC non valido: {name}")
                    target[ix] = value
            terminal_metrics: tuple[dict[str, Any] | None, ...] | None = None
            if self.collect_episode_metrics:
                summaries: list[dict[str, Any] | None] = [None] * self.num_envs
                for indices, payload in zip(self._groups, payloads):
                    received = payload.get('terminal_episode_metrics')
                    if not isinstance(received, tuple) or len(received) != len(indices):
                        raise RuntimeError('Payload metriche terminali malformato')
                    for env_id, item in zip(indices, received):
                        if (item is not None and not isinstance(item, dict)):
                            raise RuntimeError('Metriche terminali non serializzabili')
                        if (item is None) != (not bool(dones[env_id])):
                            raise RuntimeError('Metriche episodio non allineate al done')
                        if item is not None and item.get('activations') != HORIZON:
                            raise RuntimeError('Metriche episodio con orizzonte errato')
                        summaries[env_id] = item
                terminal_metrics = tuple(summaries)
            next_obs = self._combine_snapshots([
                payload["next_observation"] for payload in payloads
            ])
            if not np.isfinite(rewards).all():
                raise RuntimeError("Reward non finito")
            if not np.array_equal(dones, obs.activations == HORIZON):
                raise RuntimeError("done deve coincidere con attivazione 100")
            if (np.any(utilities[~dones] != -1)
                    or np.any(utilities[dones] < 0)
                    or np.any(utilities[dones] > 1)):
                raise RuntimeError("Utilita' terminale non valida")
            expected_activation = np.where(dones, 1, obs.activations + 1)
            if not np.array_equal(next_obs.activations, expected_activation):
                raise RuntimeError("La sequenza di attivazioni non e' corretta")
            for i in np.flatnonzero(~dones):
                if (obs.episode_seeds[i] != next_obs.episode_seeds[i]
                        or obs.map_ids[i] != next_obs.map_ids[i]
                        or obs.opponent_ids[i] != next_obs.opponent_ids[i]):
                    raise RuntimeError("Contesto / seed modificato durante l'episodio")

            # Il successore terminale NON e' l'osservazione dopo auto-reset.
            spatial = next_obs.spatial.copy()
            features = next_obs.features.copy()
            masks = next_obs.action_masks.copy()
            spatial[dones] = 0.0
            features[dones] = 0.0
            masks[dones] = False
            transition_next = ParallelTransitionNext(
                spatial=_read_only(spatial),
                features=_read_only(features),
                action_masks=_read_only(masks),
            )
            # Pubblicazione atomica dopo tutte le risposte valide.
            self._observation = next_obs
            for env_id, done in enumerate(dones):
                if done:
                    self._episode_indices[env_id] += 1
            return ParallelStep(
                rewards=_read_only(rewards),
                dones=_read_only(dones),
                terminal_utilities=_read_only(utilities),
                transition_next=transition_next,
                next_observation=next_obs,
                terminal_episode_metrics=terminal_metrics,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Arresto idempotente; chiude worker anche in caso di eccezione IPC."""
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError, BrokenPipeError):
                pass
        for connection in self._connections:
            connection.close()
        for proc in self._workers:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
        self._observation = None


__all__ = [
    "ParallelObservation",
    "ParallelTransitionNext",
    "ParallelStep",
    "ParallelGridGames",
    "SeedPlanLike",
]
