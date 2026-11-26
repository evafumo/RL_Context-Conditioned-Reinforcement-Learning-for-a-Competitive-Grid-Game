"""Valutazione FOW/Belief/QMDP compatibile con il protocollo MDP originale.

Uso (dalla radice FOW,::

    python evaluate.py --checkpoint runs/Training_tot/checkpoint_1000000.pt \
        --device cuda:0 --num-envs 8 --num-workers 4 \
        --episodes-per-context 100 --eval-seed 20261015 \
        --mode greedy --observation belief --fog-radius 4 \
        --output-dir evaluations/Training_tot/checkpoint_1000000

Valutazione primaria: greedy mascherata (epsilon=0), su TUTTE le 24 coppie
(mappa, avversario), con esattamente E episodi di 100 attivazioni per coppia.
La modalita' opzionale epsilon_greedy NON e' la valutazione primaria.
Selezione osservazione: full (MDP), mask (FOW), belief (FOW + filtro),
qmdp (media pesata Q della rete full su supporto della belief).

I 24 contesti vengono fissati mediante le ContextDistribution ufficiali;
non si modificano GridGame, la selezione dei turni, il reward shaping o le
maschere legali. I worker sono solo CPU, la Q-network e' nel coordinatore;
nessun replay, nessun gradiente e nessun aggiornamento dei pesi.

Si legge solo lo state_dict della online_network da un checkpoint completo,
 con torch.load(weights_only=True, map_location='cpu'). Non si
ripristinano optimizer, replay o RNG di training.

Output atomici: summary.json (P(win), IC Wilson, media return/decisioni)
e episodes.jsonl (seed, esito, return per episodio). I semi evaluation sono
indipendenti da quelli del training e non dipendono dal numero di worker CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import ceil, isfinite, sqrt
import json
import os
from pathlib import Path
from statistics import NormalDist
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import torch

from analysis.performance_metrics import snapshot_metrics
from agent.action_selection import epsilon_greedy_actions, select_actions
from agent.network import QNetwork
from agent.parallel_env import ParallelGridGames
from environment.context import (
    ContextDistribution, MAP_IDS, OPPONENT_IDS, get_all_contexts,
)
from environment.game import HORIZON
from environment.observations import ObservationBatch, SpatialChannel, to_torch_batch
from environment.action_mask import DUMMY_ACTION_ID, NUM_ACTIONS, to_torch_action_masks
from environment.rules import Action, GRID_SIZE
from environment.fog_of_war import validate_fog_radius
from utils.seeds import SeedPlan, derive_seed, make_torch_generator


_ALL_PAIRS = tuple((ctx.map_id, ctx.opponent_id) for ctx in get_all_contexts())
_PRIMITIVE_ACTIONS = tuple(action.name for action in Action if action is not Action.DUMMY)


def _ratio(numerator: int, denominator: int) -> float | None:
    """None significa che il denominatore non e' mai stato osservato."""
    return numerator / denominator if denominator else None


def _episode_action_stats(counts: np.ndarray, opportunities: np.ndarray,
                          decisions: int, forced: int) -> dict[str, Any]:
    """Solo le decisioni reali Agent entrano nella distribuzione delle azioni."""
    if (int(counts.sum()) != decisions + forced or
            int(counts[DUMMY_ACTION_ID]) != forced or
            int(counts[:DUMMY_ACTION_ID].sum()) != decisions or
            int(opportunities[DUMMY_ACTION_ID]) != forced or
            np.any(counts > opportunities)):
        raise RuntimeError('Contatori azioni non coerenti con decisioni/maschere')
    names = _PRIMITIVE_ACTIONS
    c = {name: int(counts[i]) for i, name in enumerate(names)}
    o = {name: int(opportunities[i]) for i, name in enumerate(names)}
    return {
        'counts': c,
        'legal_opportunities': o,
        'frequencies': {name: _ratio(c[name], decisions) for name in names},
        'selection_given_legal': {name: _ratio(c[name], o[name]) for name in names},
    }


def _aggregate_behavior(episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregazioni esatte dai conteggi grezzi, non medie di percentuali.

    Un batch sintetico usato nei test precedenti puo' non avere 'behavior'.
    In produzione tutte le partite devono disporne.
    """
    present = ['behavior' in ep for ep in episodes]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError('Episodi con e senza metriche comportamentali nello stesso report')
    names = _PRIMITIVE_ACTIONS
    n = len(episodes)
    action_counts = {name: sum(ep['behavior']['actions']['counts'][name]
                               for ep in episodes) for name in names}
    opportunities = {name: sum(ep['behavior']['actions']['legal_opportunities'][name]
                               for ep in episodes) for name in names}
    decisions = sum(ep['agent_decisions'] for ep in episodes)
    if sum(action_counts.values()) != decisions:
        raise RuntimeError('Somma azioni aggregata non corrisponde alle decisioni Agent')
    objectives = {key: sum(ep['behavior']['objective_captures'][key] for ep in episodes)
                  for key in ('agent', 'opponent')}
    combat_keys = tuple(episodes[0]['behavior']['combat'])
    combat = {key: sum(ep['behavior']['combat'][key] for ep in episodes)
              for key in combat_keys}
    if (combat['shot_kills_agent'] + combat['melee_kills_agent'] != combat['kills_agent']
            or combat['shot_kills_opponent'] + combat['melee_kills_opponent'] != combat['kills_opponent']
            or combat['melee_initiated_agent'] + combat['melee_initiated_opponent'] != combat['melee_encounters']):
        raise RuntimeError('Somme combattimenti incoerenti')
    return {
        'episodes': n,
        'actions': {
            'counts': action_counts,
            'legal_opportunities': opportunities,
            'frequencies': {name: _ratio(action_counts[name], decisions) for name in names},
            'selection_given_legal': {name: _ratio(action_counts[name], opportunities[name])
                                      for name in names},
            'agent_decisions': decisions,
        },
        'objective_captures': {
            'agent_total': objectives['agent'], 'opponent_total': objectives['opponent'],
            'agent_mean_per_episode': objectives['agent'] / n,
            'opponent_mean_per_episode': objectives['opponent'] / n,
            'agent_minus_opponent_mean_per_episode': (objectives['agent'] - objectives['opponent']) / n,
        },
        'combat': {
            **{key + '_total': value for key, value in combat.items()},
            'shoot_kill_rate_agent': _ratio(combat['shot_kills_agent'], combat['shots_agent']),
            'shoot_kill_rate_opponent': _ratio(combat['shot_kills_opponent'], combat['shots_opponent']),
            'kills_agent_mean_per_episode': combat['kills_agent'] / n,
            'kills_opponent_mean_per_episode': combat['kills_opponent'] / n,
            'deaths_agent_mean_per_episode': combat['deaths_agent'] / n,
            'deaths_opponent_mean_per_episode': combat['deaths_opponent'] / n,
        },
    }


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    checkpoint: str | Path
    output_dir: str | Path = 'evaluations/dqn'
    device: str = 'cpu'
    num_envs: int = 8
    num_workers: int = 4
    episodes_per_context: int = 100
    eval_seed: int = 20261015
    mode: str = 'greedy'          # greedy oppure epsilon_greedy
    observation: str = 'belief'  # full, mask, belief o qmdp
    fog_radius: int = 4
    epsilon: float = 0.0        # ammesso solo con epsilon_greedy
    pin_memory: bool = False
    shaping_lambda: float | None = None  # None -> letto dalla firma di training
    confidence: float = 0.95
    bootstrap_replicates: int = 2000
    contexts: tuple[tuple[int, int], ...] = field(default_factory=lambda: _ALL_PAIRS)
    verbose: bool = False

    def __post_init__(self) -> None:
        for name in ('num_envs', 'num_workers', 'episodes_per_context', 'eval_seed'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'eval_seed' else 1):
                raise ValueError(f'{name} deve essere un intero valido, non bool')
        if self.eval_seed >= 2**64:
            raise ValueError('eval_seed deve essere uint64')
        if self.num_workers > self.num_envs:
            raise ValueError('num_workers non puo\' superare num_envs')
        if not str(self.checkpoint).strip() or not str(self.output_dir).strip():
            raise ValueError('Percorso checkpoint/output_dir vuoto')
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError('device deve essere una stringa PyTorch')
        selected_device = torch.device(self.device)
        if selected_device.type not in ('cpu', 'cuda'):
            raise ValueError('device deve essere cpu o cuda[:indice]')
        if type(self.pin_memory) is not bool or type(self.verbose) is not bool:
            raise TypeError('pin_memory e verbose devono essere bool')
        if self.pin_memory and selected_device.type != 'cuda':
            raise ValueError('pin_memory richiede CUDA')
        if self.observation not in ('full', 'mask', 'belief', 'qmdp'):
            raise ValueError('observation deve essere full, mask, belief o qmdp')
        if self.observation != 'full':
            validate_fog_radius(self.fog_radius)
        if self.mode not in ('greedy', 'epsilon_greedy'):
            raise ValueError('mode deve essere greedy o epsilon_greedy')
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (float, int)):
            raise TypeError('epsilon deve essere numerico')
        if not isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('epsilon deve appartenere a [0,1]')
        if self.mode == 'greedy' and self.epsilon != 0:
            raise ValueError('Greedy richiede epsilon=0 (nessuna esplorazione)')
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (float, int)):
            raise TypeError('confidence deve essere numerico')
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError('confidence deve essere in (0,1)')
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates < 100:
            raise ValueError('bootstrap_replicates deve essere >= 100')
        if self.shaping_lambda is not None:
            if (isinstance(self.shaping_lambda, bool)
                    or not isinstance(self.shaping_lambda, (int, float))
                    or not isfinite(self.shaping_lambda) or self.shaping_lambda < 0):
                raise ValueError('shaping_lambda deve essere >= 0 e finito')
        if not isinstance(self.contexts, tuple) or not self.contexts:
            raise ValueError('contexts deve essere una tupla non vuota')
        if (any(type(p) is not tuple or len(p) != 2 or
                any(type(v) is not int for v in p) or p not in _ALL_PAIRS
                for p in self.contexts) or len(self.contexts) != len(set(self.contexts))):
            raise ValueError('Contesti non validi o duplicati')


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    summary: dict[str, Any]
    summary_path: Path
    episodes_path: Path


def wilson_interval(wins: int, episodes: int, confidence: float = 0.95) -> tuple[float, float]:
    """IC binomiale Wilson (approssimato), robusto anche con 0/N vittorie."""
    if type(wins) is not int or type(episodes) is not int or episodes < 1 or not 0 <= wins <= episodes:
        raise ValueError('Richiesti interi 0 <= wins <= episodes, episodes >= 1')
    if (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
            or not isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError('confidence deve essere in (0,1)')
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = wins / episodes
    denominator = 1.0 + z * z / episodes
    centre = (p + z * z / (2.0 * episodes)) / denominator
    half = z * sqrt(p * (1 - p) / episodes + z * z / (4.0 * episodes**2)) / denominator
    lower = 0.0 if wins == 0 else max(0.0, centre - half)
    upper = 1.0 if wins == episodes else min(1.0, centre + half)
    return lower, upper


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA non disponibile: installare PyTorch con supporto CUDA')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError('Indice GPU CUDA non disponibile')
        return torch.device(f'cuda:{index}')
    return device


def load_evaluation_network(
    checkpoint: str | Path, device: str | torch.device = 'cpu',
) -> tuple[QNetwork, dict[str, Any]]:
    """Estrae SOLO la rete online da un checkpoint DQN standard valido.

    Ritorna (QNetwork in eval() sul device scelto, training_signature).
    Non modifica RNG globale, replay buffer, optimizer o pesi del checkpoint.
    """
    selected_device = _resolve_device(str(device))
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('format') != 'Giulio2-masked-dqn'
            or payload.get('version') != 1):
        raise ValueError('Checkpoint non riconosciuto come Giulio2 DQN versione 1')
    trainer = payload.get('trainer')
    metadata = payload.get('metadata')
    if not isinstance(trainer, dict) or not isinstance(metadata, dict):
        raise ValueError('Checkpoint incompleto: trainer/metadata')
    signature = metadata.get('training_signature')
    if not isinstance(signature, dict) or signature.get('version') != 1:
        raise ValueError('Checkpoint senza firma di training compatibile')
    dqn_conf = signature.get('dqn')
    if (not isinstance(dqn_conf, dict) or dqn_conf.get('gamma') != 1.0
            or trainer.get('config') != dqn_conf):
        raise ValueError('Il checkpoint non descrive il DQN standard gamma=1 previsto')
    # Validazione load_state_dict(strict=True): shape e tutte le chiavi.
    weights = trainer.get('online_network')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Checkpoint senza online_network')
    if any(not isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError('Online network non valida')
    # Inizializzare su CPU non richiede VRAM. Solo la rete finisce su GPU.
    # L'inizializzazione di QNetwork consuma RNG CPU: preservarlo in eval.
    with torch.random.fork_rng(devices=[]):
        model = QNetwork()
    model.load_state_dict(weights, strict=True)
    model.to(selected_device)
    model.eval()
    # Le attivazioni vengono dal checkpoint, non da un nome-file ipotizzato.
    progress = payload.get('progress')
    if not isinstance(progress, dict) or type(progress.get('environment_activations')) is not int:
        raise ValueError('Checkpoint senza contatore cumulativo delle attivazioni')
    if progress['environment_activations'] < 0:
        raise ValueError('Attivazioni negative nel checkpoint')
    signature = dict(signature)
    signature['_checkpoint_activations'] = progress['environment_activations']
    return model, signature


def fixed_context_distribution(map_id: int, opponent_id: int) -> ContextDistribution:
    """Distribuzione delta ufficiale: campionamento produce sempre (m,k)."""
    if (type(map_id) is not int or type(opponent_id) is not int or
            (map_id, opponent_id) not in _ALL_PAIRS):
        raise ValueError('Contesto non ufficiale')
    return ContextDistribution(
        q_maps=tuple(float(i == map_id) for i in MAP_IDS),
        q_opponents=tuple(float(i == opponent_id) for i in OPPONENT_IDS),
    )


def _validate_observation_signature(signature: dict[str, Any], config: EvaluationConfig) -> None:
    """Evita di usare accidentalmente la rete addestrata per un'altra osservazione.

    La baseline epsilon=1 campiona soltanto dalla maschera legale e non usa
    i pesi: per questa misura la firma dell'osservazione e' irrilevante.
    """
    if config.mode == 'epsilon_greedy' and config.epsilon == 1.0:
        return
    trained_fog = bool(signature.get('fog_of_war', False))
    trained_belief = bool(signature.get('belief_filter', False)) if trained_fog else False
    expected_fog = config.observation in ('mask', 'belief')
    expected_belief = config.observation == 'belief'
    if config.observation == 'qmdp':
        expected_fog = False  # QMDP richiede la Q-network full
    if trained_fog != expected_fog or trained_belief != expected_belief:
        raise ValueError(
            f"Checkpoint addestrato con fog={trained_fog}, belief={trained_belief}; "
            f"valutazione richiesta con observation={config.observation}. "
            "Per QMDP serve un checkpoint addestrato senza fog."
        )
    if trained_fog and signature.get('fog_radius') != config.fog_radius:
        raise ValueError('Raggio fog della valutazione diverso da quello del checkpoint')


def _qmdp_values(model: QNetwork, observation: ObservationBatch,
                 device: torch.device, *, batch_size: int = 256) -> torch.Tensor:
    """QMDP sulle belief per ambiente, senza accesso alla posizione reale nascosta.

    Le distribuzioni vengono prodotte dai BeliefTracker nei worker. La rete
    full riceve, per ogni cella nel supporto, un'osservazione one-hot ipotetica.
    """
    spatial, features = observation.spatial, observation.features
    result = torch.empty((spatial.shape[0], NUM_ACTIONS), device=device, dtype=torch.float32)
    for env in range(spatial.shape[0]):
        probabilities = spatial[env, SpatialChannel.OPPONENT]
        rows, cols = np.nonzero(probabilities > 0)
        if not len(rows):
            raise RuntimeError('Belief QMDP senza supporto')
        weights = np.asarray(probabilities[rows, cols], dtype=np.float64)
        weights /= weights.sum()
        weighted = torch.zeros(NUM_ACTIONS, device=device, dtype=torch.float32)
        for start in range(0, len(rows), batch_size):
            rr = rows[start:start + batch_size]
            cc = cols[start:start + batch_size]
            k = len(rr)
            hypotheses = np.repeat(spatial[env:env + 1], k, axis=0)
            hypotheses[:, SpatialChannel.OPPONENT] = 0
            hypotheses[np.arange(k), SpatialChannel.OPPONENT, rr, cc] = 1
            feats = np.repeat(features[env:env + 1], k, axis=0)
            values = model(torch.from_numpy(np.ascontiguousarray(hypotheses)).to(device),
                           torch.from_numpy(np.ascontiguousarray(feats)).to(device))
            w = torch.from_numpy(weights[start:start + batch_size].astype(np.float32)).to(device)
            weighted += (values * w[:, None]).sum(dim=0)
        result[env] = weighted
    return result


def _evaluate_pool(
    model: QNetwork,
    config: EvaluationConfig,
    device: torch.device,
    map_id: int,
    opponent_id: int,
    *,
    phase: int,
    env_count: int,
    waves: int,
    shaping_lambda: float,
) -> list[dict[str, Any]]:
    """Esegue soltanto episodi COMPLETI: ogni pool produce env_count*waves righe.

    La seconda fase opzionale (resto modulo num_envs) usa un altro SeedPlan,
    per evitare collisioni di seed tra pool con diversi numeri di ambienti.
    """
    context_seed = derive_seed(config.eval_seed, 'dqn-evaluation', map_id, opponent_id)
    plan = SeedPlan(context_seed, experiment_id=phase)
    generator = (
        make_torch_generator(
            derive_seed(context_seed, 'policy', phase), device=str(device),
        ) if config.mode == 'epsilon_greedy' and config.epsilon > 0 else None
    )
    saved: list[dict[str, Any]] = []
    distribution = fixed_context_distribution(map_id, opponent_id)
    with ParallelGridGames(
        num_envs=env_count,
        num_workers=min(config.num_workers, env_count),
        seed_plan=plan,
        shaping_lambda=shaping_lambda,
        context_distribution=distribution,
        collect_episode_metrics=True,
        fog_radius=(config.fog_radius if config.observation != 'full' else None),
        belief_filter=config.observation in ('belief', 'qmdp'),
    ) as envs:
        with torch.inference_mode():
            for wave in range(waves):
                returns = np.zeros(env_count, dtype=np.float64)
                decisions = np.zeros(env_count, dtype=np.int64)
                action_counts = np.zeros((env_count, NUM_ACTIONS), dtype=np.int64)
                legal_opportunities = np.zeros((env_count, NUM_ACTIONS), dtype=np.int64)
                seeds = envs.observation.episode_seeds
                for tick in range(HORIZON):
                    obs = envs.observation
                    if (not np.all(obs.activations == tick + 1)
                            or not np.all(obs.map_ids == map_id)
                            or not np.all(obs.opponent_ids == opponent_id)):
                        raise RuntimeError('Violato contesto fisso o orizzonte da 100 attivazioni')
                    network_obs = to_torch_batch(
                        ObservationBatch(obs.spatial, obs.features), device,
                        pin_memory=config.pin_memory, non_blocking=config.pin_memory,
                    )
                    network_masks = to_torch_action_masks(
                        obs.action_masks, device,
                        pin_memory=config.pin_memory, non_blocking=config.pin_memory,
                    )
                    if config.mode == 'epsilon_greedy' and config.epsilon == 1.0:
                        # Baseline Random: non eseguire inferenza inutile della rete.
                        q_values = torch.zeros((env_count, NUM_ACTIONS), device=device)
                        actions = epsilon_greedy_actions(
                            q_values, network_masks, 1.0, generator=generator,
                        ).cpu().numpy()
                    elif config.observation == 'qmdp':
                        q_values = _qmdp_values(
                            model, ObservationBatch(obs.spatial, obs.features), device,
                        )
                        actions = epsilon_greedy_actions(
                            q_values, network_masks, config.epsilon, generator=generator,
                        ).cpu().numpy()
                    else:
                        actions = select_actions(
                            model, network_obs.spatial, network_obs.features,
                            network_masks, config.epsilon, generator=generator,
                        ).cpu().numpy()
                    # Valida l'applicazione del masking prima dell'interazione MDP.
                    if not np.all(obs.action_masks[np.arange(env_count), actions]):
                        raise RuntimeError('La policy ha selezionato un\'azione illegale')
                    # Conteggi pre-step: la maschera e' quella dello stato scelto.
                    action_counts[np.arange(env_count), actions] += 1
                    legal_opportunities += obs.action_masks.astype(np.int64)
                    result = envs.step(actions)
                    terminal = tick == HORIZON - 1
                    if (not np.all(result.dones == terminal)
                            or (not terminal and not np.all(result.terminal_utilities == -1))
                            or (terminal and not np.isin(result.terminal_utilities, (0, 1)).all())):
                        raise RuntimeError('Violata semantica terminale delle 100 attivazioni')
                    returns += result.rewards.astype(np.float64)
                    decisions += obs.is_agent_decision.astype(np.int64)
                # Riepiloghi inviati dal worker PRIMA del reset automatico.
                terminal_metrics = result.terminal_episode_metrics
                if terminal_metrics is None or len(terminal_metrics) != env_count:
                    raise RuntimeError('Metriche terminali assenti nella valutazione')
                # Fine di un episodio completo per ogni ambiente, mai parziale.
                for env_id in range(env_count):
                    event_metrics = terminal_metrics[env_id]
                    if event_metrics is None or event_metrics['activations'] != HORIZON:
                        raise RuntimeError('Metriche terminali episodio mancanti')
                    n_decisions = int(decisions[env_id])
                    forced = HORIZON - n_decisions
                    action_stats = _episode_action_stats(action_counts[env_id],
                                                         legal_opportunities[env_id],
                                                         n_decisions, forced)
                    saved.append({
                        'map_id': map_id,
                        'opponent_id': opponent_id,
                        'episode_index': len(saved),  # rinumerato dal chiamante
                        'episode_seed': int(seeds[env_id]),
                        'win': bool(result.terminal_utilities[env_id] == 1),
                        'utility': int(result.terminal_utilities[env_id]),
                        'return': float(returns[env_id]),
                        'agent_decisions': n_decisions,
                        'forced_dummy_activations': forced,
                        'activations': HORIZON,
                        'behavior': {
                            'actions': action_stats,
                            'objective_captures': event_metrics['objective_captures'],
                            'combat': event_metrics['combat'],
                            'final_vp': event_metrics['final_vp'],
                        },
                    })
    return saved


def _summarize(episodes: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    count = len(episodes)
    wins = sum(int(ep['win']) for ep in episodes)
    lo, hi = wilson_interval(wins, count, confidence)
    result = {
        'episodes': count,
        'wins': wins,
        'non_wins': count - wins,  # perdite e pareggi indistinguibili da utility=0
        'win_rate': wins / count,
        'wilson_ci': [lo, hi],
        'mean_episode_return': sum(ep['return'] for ep in episodes) / count,
        'mean_agent_decisions': sum(ep['agent_decisions'] for ep in episodes) / count,
        'mean_forced_dummy_activations': sum(ep['forced_dummy_activations'] for ep in episodes) / count,
        'activations': HORIZON * count,
    }
    behavior = _aggregate_behavior(episodes)
    if behavior is not None:
        result['behavior'] = behavior
    return result


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='\n',
            prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
            delete=False,
        ) as out:
            temp = out.name
            out.write(text)
            out.flush()
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)


def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    """Valuta la policy congelata. Ritorna summary e due percorsi JSON/JSONL.

    Per un sottogruppo di contesti, la statistica globale e' relativa SOLO
    ai contesti selezionati; il protocollo finale richiede tutti e 24.
    """
    if not isinstance(config, EvaluationConfig):
        raise TypeError('config deve essere EvaluationConfig')
    device = _resolve_device(config.device)
    model, signature = load_evaluation_network(config.checkpoint, device)
    _validate_observation_signature(signature, config)
    recorded_shaping = signature.get('shaping_lambda')
    if (isinstance(recorded_shaping, bool) or not isinstance(recorded_shaping, (int, float))
            or not isfinite(recorded_shaping) or recorded_shaping < 0):
        raise ValueError('Reward shaping assente o invalido nel checkpoint')
    shaping = float(recorded_shaping)
    if config.shaping_lambda is not None and config.shaping_lambda != shaping:
        raise ValueError('shaping_lambda valutazione differente dal training')
    output_dir = Path(config.output_dir)
    if output_dir.resolve() == Path(config.checkpoint).resolve():
        raise ValueError('La directory di output non puo\' coincidere col checkpoint')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    per_context: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for map_id, opponent_id in config.contexts:
        complete_waves, remainder = divmod(config.episodes_per_context, config.num_envs)
        episodes: list[dict[str, Any]] = []
        if complete_waves:
            episodes.extend(_evaluate_pool(
                model, config, device, map_id, opponent_id,
                phase=0, env_count=config.num_envs,
                waves=complete_waves, shaping_lambda=shaping,
            ))
        if remainder:
            episodes.extend(_evaluate_pool(
                model, config, device, map_id, opponent_id,
                phase=1, env_count=remainder,
                waves=1, shaping_lambda=shaping,
            ))
        if len(episodes) != config.episodes_per_context:
            raise RuntimeError('Numero di episodi valutati non corretto')
        for i, ep in enumerate(episodes):
            ep['episode_index'] = i
        per_context.append({
            'map_id': map_id,
            'opponent_id': opponent_id,
            **_summarize(episodes, config.confidence),
        })
        all_episodes.extend(episodes)
        if config.verbose:
            stats = per_context[-1]
            print(f"[evaluate] m={map_id} k={opponent_id} "
                  f"episodes={stats['episodes']} wins={stats['wins']} "
                  f"P(win)={stats['win_rate']:.4f}", flush=True)
    overall = _summarize(all_episodes, config.confidence)
    # IC bootstrap stratificato senza alterare i dati grezzi: per ogni contesto
    # il resampling conserva il numero di episodi, poi media equipesata.
    performance = snapshot_metrics(
        per_context, confidence=config.confidence,
        replicates=config.bootstrap_replicates, seed=config.eval_seed,
    )
    if abs(overall['win_rate'] - performance['overall_win_rate_equal_context_weight']) > 1e-12:
        raise RuntimeError('Valutazione non bilanciata tra contesti')
    overall['stratified_bootstrap_ci'] = performance['stratified_bootstrap_ci']
    summary = {
        'format': 'Giulio2-dqn-evaluation',
        'version': 1,
        'checkpoint': str(Path(config.checkpoint).resolve()),
        'checkpoint_activations': signature['_checkpoint_activations'],
        'device': str(device),
        'mode': config.mode,
        'observation': config.observation,
        'fog_radius': config.fog_radius if config.observation != 'full' else None,
        'epsilon': float(config.epsilon),
        'eval_seed': config.eval_seed,
        'shaping_lambda': shaping,
        'num_envs': config.num_envs,
        'num_workers': config.num_workers,
        'episodes_per_context': config.episodes_per_context,
        'num_contexts': len(config.contexts),
        'confidence': float(config.confidence),
        'confidence_method': 'Wilson per contesto/aggregati (pooled overall descrittivo); bootstrap stratificato per globale',
        'bootstrap_replicates': config.bootstrap_replicates,
        'horizon': HORIZON,
        'total_wall_seconds': float(perf_counter() - start),
        'overall': overall,
        'contexts': per_context,
        'per_map': performance['per_map'],
        'per_opponent': performance['per_opponent'],
        'robustness': performance['robustness'],
    }
    episodes_path = output_dir / 'episodes.jsonl'
    summary_path = output_dir / 'summary.json'
    _atomic_write(episodes_path, ''.join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in all_episodes
    ))
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    return EvaluationResult(summary=summary, summary_path=summary_path, episodes_path=episodes_path)


def main(argv: list[str] | None = None) -> int:
    """CLI spawn-safe, priva di side effect in fase di import."""
    parser = argparse.ArgumentParser(description='Valutazione independent masked DQN su 24 contesti')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--episodes-per-context', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=None,
                        help='Compat FOW: episodi totali, multiplo del numero dei contesti')
    parser.add_argument('--json', type=Path, default=None,
                        help='Compat FOW: salva ANCHE un report aggregato per avversario')
    parser.add_argument('--eval-seed', '--seed', dest='eval_seed', type=int, default=20261015)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--mode', choices=('greedy', 'epsilon_greedy', 'full', 'mask', 'belief', 'qmdp'), default='greedy',
                        help='Policy: greedy/epsilon_greedy; full/mask/belief/qmdp abbreviano greedy + observation')
    parser.add_argument('--observation', choices=('full', 'mask', 'belief', 'qmdp'), default=None)
    parser.add_argument('--fog-radius', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--contexts', type=str, default=None,
                        help='Subset per smoke test, es. 1:1,1:2; default tutti i 24')
    parser.add_argument('--shaping-lambda', type=float, default=None)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--quiet', action='store_true',
                        help='Non stampare gli avanzamenti per contesto')
    args = parser.parse_args(argv)
    try:
        contexts = _ALL_PAIRS if args.contexts is None else tuple(
            tuple(int(v) for v in item.split(':')) for item in args.contexts.split(',')
        )
    except ValueError as exc:
        parser.error(f'--contexts malformato: {exc}')
    mode = args.mode
    observation = args.observation
    if mode in ('full', 'mask', 'belief', 'qmdp'):
        if observation is not None and observation != mode:
            parser.error('--mode e --observation sono incompatibili')
        observation, mode = mode, 'greedy'
    if observation is None:
        observation = 'belief'  # default per FOW; usare --observation full per MDP
    episodes_per_context = args.episodes_per_context
    if args.episodes is not None:
        if args.episodes <= 0 or args.episodes % len(contexts):
            parser.error('--episodes deve essere un multiplo positivo del numero dei contesti')
        if args.episodes_per_context != 100:
            parser.error('Usare --episodes oppure --episodes-per-context, non entrambi')
        episodes_per_context = args.episodes // len(contexts)
    output_dir = args.output_dir if args.output_dir is not None else (
        args.json.parent if args.json else Path('evaluations/dqn')
    )
    config = EvaluationConfig(
        checkpoint=args.checkpoint, device=args.device,
        num_envs=args.num_envs, num_workers=args.num_workers,
        episodes_per_context=episodes_per_context, eval_seed=args.eval_seed,
        output_dir=output_dir, mode=mode, observation=observation,
        fog_radius=args.fog_radius, epsilon=args.epsilon,
        confidence=args.confidence, bootstrap_replicates=args.bootstrap_replicates,
        pin_memory=args.pin_memory,
        shaping_lambda=args.shaping_lambda, contexts=contexts,
        verbose=not args.quiet,
    )
    result = run_evaluation(config)
    overall = result.summary['overall']
    print(f"[evaluate] mode={config.mode} observation={config.observation} episodes={overall['episodes']} "
          f"wins={overall['wins']} P(win)={overall['win_rate']:.4f} "
          f"Wilson-{config.confidence:.0%}={overall['wilson_ci']} ")
    print(f'[evaluate] report: {result.summary_path}')
    if args.json is not None:
        # Per i vecchi comandi --json: formato aggregato compatibile col FOW
        # precedente, calcolato ESCLUSIVAMENTE da episodes.jsonl.
        from collections import defaultdict
        from math import sqrt
        totals = defaultdict(lambda: {'n': 0, 'win': 0, 'draw': 0, 'diff': 0.0})
        with result.episodes_path.open(encoding='utf-8') as src:
            for line in src:
                ep = json.loads(line)
                vp = ep['behavior']['final_vp']
                diff = vp['agent'] - vp['opponent']
                for key in ('all', str(ep['opponent_id'])):
                    block = totals[key]
                    block['n'] += 1
                    block['win'] += int(ep['win'])
                    block['draw'] += int(diff == 0)
                    block['diff'] += diff
        old_report = {}
        for key, block in totals.items():
            n = block['n']
            rate = block['win'] / n
            old_report[key] = {
                'episodes': n,
                'win_rate': rate,
                'win_rate_stderr': sqrt(rate * (1 - rate) / n),
                'draw_rate': block['draw'] / n,
                'mean_score_diff': block['diff'] / n,
            }
        _atomic_write(args.json, json.dumps(old_report, ensure_ascii=False, indent=2) + '\n')
        print(f'[evaluate] report compatibilita: {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'EvaluationConfig', 'EvaluationResult', 'wilson_interval',
    'load_evaluation_network', 'fixed_context_distribution',
    'run_evaluation', 'main',
]
