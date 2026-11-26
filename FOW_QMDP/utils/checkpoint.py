"""Checkpoint versionati per Context-Conditioned Masked DQN standard.


Il formato contiene soltanto tensori CPU e tipi Python ammessi da
``torch.load(weights_only=True)``. Il ReplayBuffer rimane in RAM CPU e viene
serializzato a partire dai suoi state_dict(), senza rifare le regole MDP.

Per riprendere esattamente le traiettorie occorre salvare a un CONFINE
EPISODIO DI TUTTI GLI AMBIENTI (activation == 1) e passare il generatore
esplicito usato da ``collect_parallel_experience``. Gli ambienti non vengono
serializzati a meta' partita: il checkpoint registra seed, configurazione e
indici degli episodi per ricreare ParallelGridGames con ``episode_indices``.

L'esattezza numerica su CUDA richiede anche versioni/software e impostazioni
deterministiche compatibili; non e' garantita tra macchine diverse.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Final, Mapping

import numpy as np
import torch

from agent.dqn import DQNTrainer
from agent.parallel_env import ParallelGridGames, SeedPlanLike
from agent.replay_buffer import ReplayBuffer
from environment.context import ContextDistribution, DEFAULT_CONTEXT_DISTRIBUTION
from environment.game import HORIZON

CHECKPOINT_VERSION: Final[int] = 1
_REPLAY_ARRAYS: Final[tuple[str, ...]] = (
    'spatial', 'features', 'action_masks', 'actions', 'rewards',
    'next_spatial', 'next_features', 'next_action_masks', 'dones',
)
_REQUIRED: Final[frozenset[str]] = frozenset({
    'format', 'version', 'trainer', 'replay', 'progress', 'epsilon_schedule_state',
    'environments', 'rng', 'metadata', 'exact_resume_possible', 'training_device',
})


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} deve essere un int non negativo (non bool)')
    return value


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} deve essere un numero reale, non bool')
    number = float(value)
    if not isfinite(number):
        raise ValueError(f'{name} deve essere finito')
    return number


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Contatori GLOBALI, in attivazioni MDP (anche DUMMY), non gradient step.

    ``epsilon`` e' quello da usare per la prossima raccolta. I dettagli
    dello schedule, se necessari, sono in ``epsilon_schedule_state``.
    Gli update del gradiente sono gia' in DQNTrainer.update_steps.
    """

    environment_activations: int
    episodes_completed: int
    collections_completed: int
    epsilon: float

    def __post_init__(self) -> None:
        for name in ('environment_activations', 'episodes_completed',
                     'collections_completed'):
            _nonnegative_int(getattr(self, name), name)
        eps = _finite_real(self.epsilon, 'epsilon')
        if not 0 <= eps <= 1:
            raise ValueError('epsilon deve essere in [0, 1]')
        object.__setattr__(self, 'epsilon', eps)


@dataclass(frozen=True, slots=True)
class EnvironmentResume:
    """Dati sufficienti a ricreare ParallelGridGames *al confine episodio*.

    I seed derivati consentono di verificare lo stesso SeedPlan o base_seed
    senza serializzare oggetti arbitrari, processi, pipe o GameState.
    """

    num_envs: int
    episode_indices: tuple[int, ...]
    shaping_lambda: float
    q_maps: tuple[float, ...]
    q_opponents: tuple[float, ...]
    environment_seeds: tuple[int, ...]
    next_episode_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError('num_envs deve essere > 0')
        if len(self.episode_indices) != self.num_envs or any(
            type(i) is not int or i < 0 for i in self.episode_indices
        ):
            raise ValueError('episode_indices non validi')
        if len(self.environment_seeds) != self.num_envs or len(self.next_episode_seeds) != self.num_envs:
            raise ValueError('Lunghezza dei seed ambientali errata')
        for seed in (*self.environment_seeds, *self.next_episode_seeds):
            if type(seed) is not int or not 0 <= seed < (1 << 64):
                raise ValueError('Seed ambiente/episodio fuori intervallo uint64')
        value = _finite_real(self.shaping_lambda, 'shaping_lambda')
        if value < 0:
            raise ValueError('shaping_lambda deve essere >= 0')
        object.__setattr__(self, 'shaping_lambda', value)
        ContextDistribution(self.q_maps, self.q_opponents)


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Metadati ripristinati; i pesi/replay/RNG sono applicati agli oggetti."""

    progress: TrainingProgress
    epsilon_schedule_state: dict[str, Any]
    environments: EnvironmentResume | None
    metadata: dict[str, Any]
    exact_resume_possible: bool


def _plain(value: Any, name: str = 'value') -> Any:
    """Accetta solo tipi primitive JSON-like; no NaN, oggetti pickle o tensor."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (int, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f'{name} contiene NaN/Inf')
        return value
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise TypeError(f'{name}: tutte le chiavi devono essere stringhe')
        return {k: _plain(v, f'{name}.{k}') for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_plain(v, f'{name}[{i}]') for i, v in enumerate(value)]
    raise TypeError(f'{name}: tipo non consentito ({type(value).__name__})')


def _cpu_tree(value: Any) -> Any:
    """Stato PyTorch -> tensori CPU indipendenti; altri tipi safe per weights_only."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone()
    if isinstance(value, np.ndarray):
        # ReplayBuffer.state_dict() ha gia' prodotto copie autonome.
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError('Uno stato del checkpoint contiene NaN/Inf')
        return value
    if isinstance(value, Mapping):
        return { _cpu_tree(k): _cpu_tree(v) for k, v in value.items() }
    if type(value) is list:
        return [_cpu_tree(v) for v in value]
    if type(value) is tuple:
        return tuple(_cpu_tree(v) for v in value)
    raise TypeError(f'Tipo non supportato per checkpoint weights_only: {type(value).__name__}')


def _rng_snapshot(policy_generator: torch.Generator | None) -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        'torch_cpu': torch.get_rng_state().clone(),
        # Su CPU non inizializzare CUDA soltanto per il checkpoint.
        'torch_cuda': ([s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_initialized() else None),
        'policy_device': str(policy_generator.device) if policy_generator is not None else None,
        'policy': (policy_generator.get_state().cpu().clone()
                   if policy_generator is not None else None),
    }


def _restore_rng(snapshot: Mapping[str, Any], policy_generator: torch.Generator | None) -> None:
    # Usare soltanto checkpoint fidati. La validazione principale avviene in load.
    random.setstate(snapshot['python'])
    name, keys, pos, has_gauss, cached = snapshot['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(snapshot['torch_cpu'])
    if snapshot['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(snapshot['torch_cuda'])
    if policy_generator is not None and snapshot['policy'] is not None:
        policy_generator.set_state(snapshot['policy'])


def _make_environment_resume(environments: ParallelGridGames) -> EnvironmentResume:
    if not isinstance(environments, ParallelGridGames):
        raise TypeError('environments deve essere ParallelGridGames')
    indices = environments.checkpoint_episode_indices()  # rifiuta meta' episodio
    n = environments.num_envs
    return EnvironmentResume(
        num_envs=n,
        episode_indices=indices,
        shaping_lambda=environments.shaping_lambda,
        q_maps=tuple(environments.context_distribution.q_maps),
        q_opponents=tuple(environments.context_distribution.q_opponents),
        environment_seeds=tuple(environments.seed_plan.environment_seed(i) for i in range(n)),
        next_episode_seeds=tuple(
            environments.seed_plan.episode_seed(i + indices[i] * n) for i in range(n)
        ),
    )


def validate_environment_resume(
    resume: EnvironmentResume,
    *,
    num_envs: int,
    seed_plan: SeedPlanLike,
    shaping_lambda: float = 0.1,
    context_distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
) -> None:
    """Fallisce se il nuovo setup cambierebbe seed, mappe o reward shaping.

    Dopo la verifica costruire ``ParallelGridGames(...,
    episode_indices=resume.episode_indices)`` e poi chiamare ``start()``.
    Cambiare il solo ``num_workers`` e' consentito.
    """
    if not isinstance(resume, EnvironmentResume):
        raise TypeError('resume deve essere EnvironmentResume')
    if type(num_envs) is not int or num_envs != resume.num_envs:
        raise ValueError('Numero di ambienti diverso dal checkpoint')
    if not isinstance(context_distribution, ContextDistribution):
        raise TypeError('context_distribution deve essere ContextDistribution')
    if (tuple(context_distribution.q_maps) != resume.q_maps or
            tuple(context_distribution.q_opponents) != resume.q_opponents):
        raise ValueError('Distribuzione dei 24 contesti differente dal checkpoint')
    if _finite_real(shaping_lambda, 'shaping_lambda') != resume.shaping_lambda:
        raise ValueError('Reward shaping incompatibile con il checkpoint')
    if not all(callable(getattr(seed_plan, name, None)) for name in ('environment_seed', 'episode_seed')):
        raise TypeError('seed_plan non valido')
    expected = tuple(seed_plan.environment_seed(i) for i in range(num_envs))
    episodes = tuple(seed_plan.episode_seed(i + j * num_envs)
                     for i, j in enumerate(resume.episode_indices))
    if expected != resume.environment_seeds or episodes != resume.next_episode_seeds:
        raise ValueError('SeedPlan/base_seed incompatibile con il checkpoint')


def save_checkpoint(
    path: str | Path,
    *,
    trainer: DQNTrainer,
    replay_buffer: ReplayBuffer,
    progress: TrainingProgress,
    epsilon_schedule_state: Mapping[str, Any] | None = None,
    policy_generator: torch.Generator | None = None,
    environments: ParallelGridGames | None = None,
    metadata: Mapping[str, Any] | None = None,
    require_exact_resume: bool = False,
    fsync: bool = False,
) -> Path:
    """Salva in modo atomico un checkpoint DQN compatibile CPU/CUDA.

    ``require_exact_resume=True`` esige sia ambienti al confine episodio
    sia policy_generator; salva inoltre gli RNG globali. Anche cosi' CUDA
    non garantisce bit-exact fra diverse piattaforme e versioni software.
    """
    if not isinstance(trainer, DQNTrainer):
        raise TypeError('trainer deve essere DQNTrainer')
    if not isinstance(replay_buffer, ReplayBuffer):
        raise TypeError('replay_buffer deve essere ReplayBuffer')
    if not isinstance(progress, TrainingProgress):
        raise TypeError('progress deve essere TrainingProgress')
    if policy_generator is not None and not isinstance(policy_generator, torch.Generator):
        raise TypeError('policy_generator deve essere torch.Generator')
    if type(require_exact_resume) is not bool or type(fsync) is not bool:
        raise TypeError('require_exact_resume e fsync devono essere bool')
    if require_exact_resume and (environments is None or policy_generator is None):
        raise ValueError('Resume esatto richiede environments e policy_generator')
    if policy_generator is not None and policy_generator.device.type != trainer.device.type:
        raise ValueError('Generatore policy e trainer devono usare lo stesso tipo di device')
    if policy_generator is not None and trainer.device.type == 'cuda':
        expected_index = (trainer.device.index if trainer.device.index is not None
                          else torch.cuda.current_device())
        if policy_generator.device.index not in (None, expected_index):
            raise ValueError('Il generatore policy deve usare la GPU del trainer')

    env_info = _make_environment_resume(environments) if environments is not None else None
    if env_info is not None:
        if progress.episodes_completed != sum(env_info.episode_indices):
            raise ValueError('episodes_completed non coincide con i contatori degli ambienti')
        if progress.environment_activations != progress.episodes_completed * HORIZON:
            raise ValueError('Al confine episodio le attivazioni devono essere 100 per episodio')
    schedule = _plain({} if epsilon_schedule_state is None else epsilon_schedule_state,
                      'epsilon_schedule_state')
    notes = _plain({} if metadata is None else metadata, 'metadata')
    if not isinstance(schedule, dict) or not isinstance(notes, dict):
        raise TypeError('schedule e metadata devono essere mapping')

    checkpoint_path = Path(path)
    if not checkpoint_path.name or checkpoint_path.is_dir():
        raise ValueError('path deve indicare un file, non una directory')
    # Snapshot del trainer e del replay nel main. I pesi vengono portati su CPU
    # e il replay rimane completamente fuori dalla VRAM della RTX.
    payload = {
        'format': 'Giulio2-masked-dqn',
        'version': CHECKPOINT_VERSION,
        'training_device': str(next(trainer.online_network.parameters()).device),
        'trainer': _cpu_tree(trainer.state_dict()),
        'replay': _cpu_tree(replay_buffer.state_dict()),
        'progress': asdict(progress),
        'epsilon_schedule_state': schedule,
        'environments': asdict(env_info) if env_info is not None else None,
        'rng': _rng_snapshot(policy_generator),
        'metadata': notes,
        'exact_resume_possible': env_info is not None and policy_generator is not None,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        # NamedTemporaryFile viene chiuso prima di os.replace (Windows).
        with tempfile.NamedTemporaryFile(
            mode='wb', prefix=f'.{checkpoint_path.name}.', suffix='.tmp',
            dir=checkpoint_path.parent, delete=False,
        ) as handle:
            temporary_path = handle.name
            torch.save(payload, handle)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    trainer: DQNTrainer,
    replay_buffer: ReplayBuffer,
    policy_generator: torch.Generator | None = None,
    restore_rng: bool = True,
    require_exact_resume: bool = False,
) -> CheckpointInfo:
    """Ricarica lo stato e restituisce le istruzioni per riprendere l'ambiente.

    Istanziare trainer con uguale DQNConfig e replay con uguale capacity.
    I parametri sono caricati nel device gia' scelto per il trainer (CUDA
    compresa). Il checkpoint NON avvia i worker e NON ricrea train.py.
    Checkpoint con formato/parametri errati falliscono prima del restore.
    """
    if not isinstance(trainer, DQNTrainer):
        raise TypeError('trainer deve essere DQNTrainer')
    if not isinstance(replay_buffer, ReplayBuffer):
        raise TypeError('replay_buffer deve essere ReplayBuffer')
    if policy_generator is not None and not isinstance(policy_generator, torch.Generator):
        raise TypeError('policy_generator deve essere torch.Generator')
    if type(restore_rng) is not bool or type(require_exact_resume) is not bool:
        raise TypeError('restore_rng e require_exact_resume devono essere bool')
    if require_exact_resume and not restore_rng:
        raise ValueError('Resume esatto richiede restore_rng=True')
    # weights_only=True non carica classi Python arbitrarie/pickle non consentiti.
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _REQUIRED:
        raise ValueError('Formato del checkpoint non valido')
    if payload['format'] != 'Giulio2-masked-dqn' or payload['version'] != CHECKPOINT_VERSION:
        raise ValueError('Versione del checkpoint DQN incompatibile')
    if type(payload['exact_resume_possible']) is not bool:
        raise ValueError('Campo exact_resume_possible errato')
    if not isinstance(payload['training_device'], str):
        raise ValueError('Campo training_device errato')
    if require_exact_resume:
        current_device = str(next(trainer.online_network.parameters()).device)
        if current_device != payload['training_device']:
            raise ValueError('Resume esatto richiede lo stesso device di addestramento')
    if not isinstance(payload['trainer'], dict) or not isinstance(payload['replay'], dict):
        raise ValueError('Stato trainer/replay non valido')
    if not isinstance(payload['progress'], dict):
        raise ValueError('Contatori del checkpoint non validi')
    progress = TrainingProgress(**payload['progress'])
    schedule = _plain(payload['epsilon_schedule_state'], 'epsilon_schedule_state')
    notes = _plain(payload['metadata'], 'metadata')
    if not isinstance(schedule, dict) or not isinstance(notes, dict):
        raise ValueError('schedule e metadata devono essere mapping')
    env_state = payload['environments']
    env_info = EnvironmentResume(**env_state) if isinstance(env_state, dict) else None
    if env_state is not None and env_info is None:
        raise ValueError('Stato ambientale non valido')
    if env_info is not None:
        if progress.episodes_completed != sum(env_info.episode_indices):
            raise ValueError('Contatori episodi incoerenti nel checkpoint')
        if progress.environment_activations != progress.episodes_completed * HORIZON:
            raise ValueError('Contatori attivazioni incoerenti nel checkpoint')
    rng = payload['rng']
    if not isinstance(rng, dict) or set(rng) != {
        'python', 'numpy', 'torch_cpu', 'torch_cuda', 'policy_device', 'policy'
    }:
        raise ValueError('Stato RNG del checkpoint invalido')
    is_exact = env_info is not None and rng['policy'] is not None
    if payload['exact_resume_possible'] != is_exact:
        raise ValueError('Indicatore di resume esatto incoerente')
    if require_exact_resume and not is_exact:
        raise ValueError('Checkpoint privo di stato per un resume esatto')
    if restore_rng:
        if rng['policy'] is not None:
            if policy_generator is None and require_exact_resume:
                raise ValueError('Manca policy_generator per il resume esatto')
            if policy_generator is not None and str(policy_generator.device) != rng['policy_device']:
                raise ValueError('Il device del policy_generator non coincide')
        elif policy_generator is not None and require_exact_resume:
            raise ValueError('Lo stato del policy_generator manca nel checkpoint')
        if rng['torch_cuda'] is not None:
            if not torch.cuda.is_available():
                raise RuntimeError('Checkpoint con RNG CUDA: GPU richiesta o restore_rng=False')
            if len(rng['torch_cuda']) != torch.cuda.device_count():
                raise ValueError('Numero di GPU incompatibile con lo stato RNG salvato')
    # Prevalidare la configurazione dello stato DQN e del replay prima di
    # iniziare a modificare gli oggetti in memoria.
    from dataclasses import asdict as _asdict
    if payload['trainer'].get('config') != _asdict(trainer.config):
        raise ValueError('DQNConfig differente dal checkpoint')
    if payload['replay'].get('capacity') != replay_buffer.capacity:
        raise ValueError('Replay capacity differente dal checkpoint')
    replay_state = dict(payload['replay'])
    for name in _REPLAY_ARRAYS:
        tensor = replay_state.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            raise ValueError(f'Array replay {name} non presente come tensore CPU')
        replay_state[name] = tensor.detach().numpy().copy()
    # Il replay ha un load transazionale con verifiche sulle maschere PASS/DUMMY
    # e sui terminali: si sfrutta quello, senza duplicare le regole.
    replay_buffer.load_state_dict(replay_state)
    trainer.load_state_dict(payload['trainer'])
    if restore_rng:
        _restore_rng(rng, policy_generator)
    return CheckpointInfo(progress, schedule, env_info, notes, is_exact)


__all__ = [
    'CHECKPOINT_VERSION', 'TrainingProgress', 'EnvironmentResume',
    'CheckpointInfo', 'validate_environment_resume',
    'save_checkpoint', 'load_checkpoint',
]
