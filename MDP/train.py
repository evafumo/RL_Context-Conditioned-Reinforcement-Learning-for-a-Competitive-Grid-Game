"""Fase 5.5: training completo del Context-Conditioned Masked DQN standard.

Eseguire dalla radice::

    python train.py --device cuda:0 --num-envs 8 --num-workers 4 \
        --total-activations 80000 --output-dir runs/pilot_dqn

Il simulatore esegue tutte le 100 attivazioni MDP, DUMMY comprese. Il
collector inserisce ogni transizione nel replay CPU e separa esplicitamente
il terminale dalla nuova osservazione generata dall'auto-reset. I Q-values
vengono inferiti in batch sulla GPU principale; SOLO il DQNTrainer aggiorna
la Q-network con target network e masked Bellman maximum (non Double DQN).

Ogni finestra di raccolta copre un multiplo di 100 attivazioni per ambiente:
all'inizio e alla fine TUTTI gli ambienti sono al confine di episodio. Questo
consente checkpoint ambientali riavviabili tramite episode_indices e seed.
Gli update avvengono DOPO ogni finestra, non durante le attivazioni: epsilon
rimane costante all'interno di ciascuna finestra e segue uno schedule per
numero globale di attivazioni, non per numero di gradient step.

Replay e checkpoint restano in RAM/SSD CPU, mai tutti e 16 GB in VRAM.
Usare --pin-memory sulla GPU solo dopo benchmark. BF16 AMP e' facoltativo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from agent.collector import CollectionStats, collect_parallel_experience
from agent.dqn import DQNConfig, DQNTrainer
from agent.network import QNetwork
from agent.parallel_env import ParallelGridGames
from agent.replay_buffer import ReplayBuffer
from environment.game import HORIZON
from utils.checkpoint import (
    TrainingProgress, load_checkpoint, save_checkpoint,
    validate_environment_resume,
)
from utils.metrics import MetricsLogger
from utils.seeds import SeedPlan, derive_seed, make_torch_generator, seed_everything


@dataclass(frozen=True, slots=True)
class LinearEpsilonSchedule:
    """Decay lineare, misurato in attivazioni MDP globali (anche DUMMY)."""

    start: float = 1.0
    end: float = 0.05
    decay_activations: int = 160_000

    def __post_init__(self) -> None:
        for key in ('start', 'end'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'epsilon {key} deve essere numerico')
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'epsilon {key} deve essere in [0, 1]')
        if self.start < self.end:
            raise ValueError('epsilon start deve essere >= end')
        if type(self.decay_activations) is not int or self.decay_activations < 1:
            raise ValueError('decay_activations deve essere un intero positivo')

    def value(self, activations: int) -> float:
        if type(activations) is not int or activations < 0:
            raise ValueError('activations deve essere un intero >= 0')
        fraction = min(activations / self.decay_activations, 1.0)
        return float(self.start + fraction * (self.end - self.start))

    def state_dict(self) -> dict[str, Any]:
        return {'kind': 'linear_by_activation', 'version': 1,
                'start': float(self.start), 'end': float(self.end),
                'decay_activations': self.decay_activations}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configurazione del coordinatore (non del simulatore o della loss).

    Per il resume esatto, total_activations e steps_per_env DEVONO essere
    multipli di num_envs * HORIZON e HORIZON rispettivamente. Il numero di
    worker e il totale desiderato possono variare al resume; tutti gli altri
    parametri che determinano le transizioni o gli update devono coincidere.
    """

    output_dir: str | Path = 'runs/dqn_pilot'
    device: str = 'cpu'
    seed: int = 20261026
    experiment_id: int = 0
    num_envs: int = 8
    num_workers: int = 4
    total_activations: int = 80_000
    steps_per_env: int = HORIZON
    shaping_lambda: float = 0.1
    replay_capacity: int = 50_000
    batch_size: int = 256
    warmup_activations: int = 8_000
    updates_per_collection: int = 32
    epsilon: LinearEpsilonSchedule = field(default_factory=LinearEpsilonSchedule)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    checkpoint_every_activations: int = 20_000  # 0: solo checkpoint finale
    log_update_every: int = 10  # 0: non scrivere singoli update (solo raccolte)
    pin_memory: bool = False
    deterministic_algorithms: bool = False
    resume: str | Path | None = None

    def __post_init__(self) -> None:
        ints = ('seed', 'experiment_id', 'num_envs', 'num_workers',
                'total_activations', 'steps_per_env', 'replay_capacity',
                'batch_size', 'warmup_activations', 'updates_per_collection',
                'checkpoint_every_activations', 'log_update_every')
        for key in ints:
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f'{key} deve essere un intero >= 0 (non bool)')
        if self.seed >= 2**64 or self.experiment_id >= 2**64:
            raise ValueError('seed e experiment_id devono essere uint64')
        if self.num_envs < 1 or not 1 <= self.num_workers <= self.num_envs:
            raise ValueError('num_workers deve appartenere a [1, num_envs]')
        if self.total_activations == 0 or self.total_activations % (HORIZON * self.num_envs):
            raise ValueError('total_activations deve essere multiplo di 100 * num_envs')
        if self.steps_per_env < HORIZON or self.steps_per_env % HORIZON:
            raise ValueError('steps_per_env deve essere multiplo positivo di 100')
        if self.replay_capacity < self.batch_size or self.batch_size < 1:
            raise ValueError('replay_capacity deve essere >= batch_size >= 1')
        if self.updates_per_collection < 1:
            raise ValueError('updates_per_collection deve essere >= 1')
        if (isinstance(self.shaping_lambda, bool)
                or not isinstance(self.shaping_lambda, (int, float))
                or not isfinite(self.shaping_lambda) or self.shaping_lambda < 0):
            raise ValueError('shaping_lambda deve essere finito e >= 0')
        if not isinstance(self.epsilon, LinearEpsilonSchedule):
            raise TypeError('epsilon deve essere LinearEpsilonSchedule')
        if not isinstance(self.dqn, DQNConfig):
            raise TypeError('dqn deve essere DQNConfig')
        if self.dqn.gamma != 1.0:
            raise ValueError('Il progetto impone gamma=1: non modificare il DQNConfig.gamma')
        if type(self.pin_memory) is not bool or type(self.deterministic_algorithms) is not bool:
            raise TypeError('pin_memory e deterministic_algorithms devono essere bool')
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError('device deve essere una stringa PyTorch')
        device = torch.device(self.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('device deve essere cpu o cuda[:indice]')
        if self.pin_memory and device.type != 'cuda':
            raise ValueError('pin_memory=True richiede CUDA')
        if not str(self.output_dir).strip():
            raise ValueError('output_dir vuota')
        if self.resume is not None and not str(self.resume).strip():
            raise ValueError('resume vuoto')


@dataclass(frozen=True, slots=True)
class TrainingResult:
    progress: TrainingProgress
    gradient_updates: int
    replay_size: int
    checkpoint_path: Path
    metrics_path: Path
    elapsed_seconds: float


def training_signature(config: TrainConfig) -> dict[str, Any]:
    """Parametri che devono restare identici per proseguire una traiettoria.

    Non include budget finale, percorso output, worker o frequenza dei log.
    Include la granularita' delle raccolte (epsilon e timing degli update).
    """
    return {
        'version': 1, 'seed': config.seed, 'experiment_id': config.experiment_id,
        'num_envs': config.num_envs, 'steps_per_env': config.steps_per_env,
        'shaping_lambda': float(config.shaping_lambda),
        'replay_capacity': config.replay_capacity, 'batch_size': config.batch_size,
        'warmup_activations': config.warmup_activations,
        'updates_per_collection': config.updates_per_collection,
        'pin_memory': config.pin_memory,
        'deterministic_algorithms': config.deterministic_algorithms,
        'epsilon': config.epsilon.state_dict(), 'dqn': asdict(config.dqn),
    }


def _resolved_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA indisponibile: installare una build PyTorch CUDA')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'Device CUDA inesistente: cuda:{index}')
        return torch.device(f'cuda:{index}')
    return device


def _progress(activations: int, episodes: int, collections: int,
              schedule: LinearEpsilonSchedule) -> TrainingProgress:
    return TrainingProgress(
        environment_activations=activations,
        episodes_completed=episodes,
        collections_completed=collections,
        epsilon=schedule.value(activations),
    )


def run_training(config: TrainConfig) -> TrainingResult:
    """Esegue il ciclo completo e restituisce riepilogo/percorso checkpoint.

    La rete CUDA e i dati ottimizzati restano nel processo principale;
    ParallelGridGames avvia solo worker CPU spawn. Checkpoint esatti soltanto
    ai confini episodici. Resume richiede trainer/replay/config compatibili.
    """
    if not isinstance(config, TrainConfig):
        raise TypeError('config deve essere TrainConfig')
    device = _resolved_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError('output_dir non e\' una cartella')
    seed_plan = SeedPlan(config.seed, config.experiment_id)
    seed_everything(
        seed_plan.experiment_seed, seed_cuda=device.type == 'cuda',
        deterministic_algorithms=config.deterministic_algorithms,
    )
    # L'inizializzazione della rete puo' consumare RNG: load_checkpoint()
    # ripristina POI gli RNG dal file prima di eseguire una nuova raccolta.
    trainer = DQNTrainer(QNetwork(), config=config.dqn, device=device)
    replay = ReplayBuffer(
        capacity=config.replay_capacity,
        seed=derive_seed(seed_plan.experiment_seed, 'dqn-replay'),
    )
    generator = make_torch_generator(seed_plan.policy_seed(), device=str(device))
    signature = training_signature(config)
    episode_indices = None
    progress = _progress(0, 0, 0, config.epsilon)

    if config.resume is not None:
        info = load_checkpoint(
            config.resume, trainer=trainer, replay_buffer=replay,
            policy_generator=generator, require_exact_resume=True,
        )
        if info.metadata.get('training_signature') != signature:
            raise ValueError('Impostazioni incompatibili con il training salvato')
        if info.epsilon_schedule_state != config.epsilon.state_dict():
            raise ValueError('Schedule epsilon diverso dal checkpoint')
        if info.environments is None:
            raise ValueError('Checkpoint privo degli ambienti ripristinabili')
        validate_environment_resume(
            info.environments, num_envs=config.num_envs, seed_plan=seed_plan,
            shaping_lambda=config.shaping_lambda,
        )
        progress = info.progress
        if progress.epsilon != config.epsilon.value(progress.environment_activations):
            raise ValueError('Valore epsilon incoerente nel checkpoint')
        if progress.environment_activations % (HORIZON * config.num_envs):
            raise ValueError('Il checkpoint non e\' al confine episodico')
        if replay.total_added != progress.environment_activations:
            raise ValueError('Il replay non coincide con il numero di transizioni raccolte')
        if progress.environment_activations >= config.total_activations:
            raise ValueError('Il budget nuovo deve superare le attivazioni gia\' completate')
        episode_indices = info.environments.episode_indices

    if config.resume is None:
        metrics_path = output_dir / 'metrics.jsonl'
    else:
        # Sessione separata: nessuna duplicazione di eventi se si riparte da
        # un checkpoint anteriore a un crash. La sessione e' sovrascrivibile.
        metrics_path = output_dir / f'metrics_resume_from_{progress.environment_activations}.jsonl'
    checkpoint_path = output_dir / f'checkpoint_{progress.environment_activations}.pt'
    wall_start = perf_counter()
    with ParallelGridGames(
        num_envs=config.num_envs, num_workers=config.num_workers,
        seed_plan=seed_plan, shaping_lambda=config.shaping_lambda,
        episode_indices=episode_indices,
    ) as environments, MetricsLogger(metrics_path, append=False) as logger:
        if environments.checkpoint_episode_indices() != (
            episode_indices if episode_indices is not None else (0,) * config.num_envs
        ):
            raise RuntimeError('Gli ambienti non sono partiti dal confine atteso')
        while progress.environment_activations < config.total_activations:
            remaining_per_env = (config.total_activations - progress.environment_activations) // config.num_envs
            steps = min(config.steps_per_env, remaining_per_env)
            if steps < HORIZON or steps % HORIZON:
                raise RuntimeError('Una finestra deve coprire episodi completi')
            epsilon_used = config.epsilon.value(progress.environment_activations)
            old_indices = environments.checkpoint_episode_indices()
            old_activations = progress.environment_activations
            collect_start = perf_counter()
            collected = collect_parallel_experience(
                environments, trainer.online_network, replay,
                steps_per_env=steps, epsilon=epsilon_used, device=device,
                pin_memory=config.pin_memory, generator=generator,
            )
            collect_seconds = max(perf_counter() - collect_start, 1e-12)
            expected_transitions = config.num_envs * steps
            expected_episodes = config.num_envs * (steps // HORIZON)
            if not isinstance(collected, CollectionStats):
                raise RuntimeError('Collector ha restituito statistiche invalide')
            if (collected.num_envs != config.num_envs or collected.steps_per_env != steps
                    or collected.transitions != expected_transitions
                    or collected.completed_episodes != expected_episodes
                    or collected.agent_decisions + collected.forced_activations != expected_transitions
                    or not 0 <= collected.wins <= collected.completed_episodes
                    or not isfinite(collected.rewards_sum)):
                raise RuntimeError('Collector: contatori MDP incoerenti')
            new_indices = environments.checkpoint_episode_indices()
            if any(new - old != steps // HORIZON
                   for old, new in zip(old_indices, new_indices)):
                raise RuntimeError('Ogni ambiente deve aver completato gli episodi attesi')
            activations = old_activations + expected_transitions
            if replay.total_added != activations:
                raise RuntimeError('Mancano transizioni nel replay (incluse DUMMY)')
            progress = _progress(
                activations, progress.episodes_completed + collected.completed_episodes,
                progress.collections_completed + 1, config.epsilon,
            )
            if progress.episodes_completed != sum(new_indices):
                raise RuntimeError('Conteggio episodi incoerente con gli ambienti')
            logger.log_collection(
                progress.collections_completed, collected,
                epsilon=epsilon_used, collection_seconds=collect_seconds,
                replay_size=len(replay), replay_capacity=replay.capacity,
                other={
                    'environment_activations': progress.environment_activations,
                    'episodes_completed_total': progress.episodes_completed,
                    'epsilon_next': progress.epsilon,
                    'gradient_updates_before': trainer.update_steps,
                },
            )

            # Warm-up riguarda gli update, NON interrompe la raccolta MDP.
            if activations >= config.warmup_activations and len(replay) >= config.batch_size:
                for _ in range(config.updates_per_collection):
                    update_start = perf_counter()
                    update = trainer.train_on_replay(replay, config.batch_size)
                    update_seconds = max(perf_counter() - update_start, 1e-12)
                    if (config.log_update_every and
                            (trainer.update_steps % config.log_update_every == 0)):
                        logger.log_update(
                            trainer.update_steps, update, update_seconds=update_seconds,
                            epsilon=epsilon_used, replay_size=len(replay),
                            replay_capacity=replay.capacity,
                            other={'environment_activations': activations},
                        )

            checkpoint_due = (
                config.checkpoint_every_activations > 0 and
                (old_activations // config.checkpoint_every_activations <
                 activations // config.checkpoint_every_activations)
            )
            last_collection = activations == config.total_activations
            if checkpoint_due or last_collection:
                checkpoint_path = output_dir / f'checkpoint_{activations}.pt'
                save_checkpoint(
                    checkpoint_path, trainer=trainer, replay_buffer=replay,
                    progress=progress, policy_generator=generator,
                    environments=environments,
                    epsilon_schedule_state=config.epsilon.state_dict(),
                    metadata={'training_signature': signature},
                    require_exact_resume=True,
                )
                logger.log({
                    'kind': 'checkpoint', 'path': checkpoint_path.name,
                    'environment_activations': activations,
                    'gradient_updates': trainer.update_steps,
                })

    return TrainingResult(
        progress=progress, gradient_updates=trainer.update_steps,
        replay_size=len(replay), checkpoint_path=checkpoint_path,
        metrics_path=metrics_path, elapsed_seconds=perf_counter() - wall_start,
    )


def _config_from_yaml(path: str | Path) -> TrainConfig:
    """Schema YAML: train: {..parametri TrainConfig..}, dqn: {..DQNConfig..}.

    Se PyYAML manca, e' sempre possibile utilizzare gli argomenti CLI.
    Le chiavi non riconosciute causano errore invece di essere ignorate.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('Per --config installare PyYAML (pip install pyyaml)') from exc
    with Path(path).open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not set(payload) <= {'train', 'dqn'}:
        raise ValueError('Il file YAML richiede le sezioni train e/o dqn')
    args = payload.get('train', {})
    dqn_args = payload.get('dqn', {})
    if not isinstance(args, dict) or not isinstance(dqn_args, dict):
        raise ValueError('train e dqn nel YAML devono essere mapping')
    allowed = set(TrainConfig.__dataclass_fields__) - {'dqn', 'epsilon', 'resume'}
    if not set(args) <= allowed | {'epsilon'}:
        raise ValueError(f'Chiavi train sconosciute: {sorted(set(args) - allowed - {"epsilon"})}')
    if 'epsilon' in args:
        eps = args['epsilon']
        if not isinstance(eps, dict):
            raise ValueError('train.epsilon deve essere un mapping')
        args = dict(args, epsilon=LinearEpsilonSchedule(**eps))
    return TrainConfig(**args, dqn=DQNConfig(**dqn_args))


def main(argv: list[str] | None = None) -> int:
    """CLI invocabile anche dai test; NON avvia worker all'importazione."""
    parser = argparse.ArgumentParser(description='Context-Conditioned Masked DQN (standard)')
    parser.add_argument('--config', type=Path, default=None, help='YAML opzionale')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'cuda:0'), default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--resume', type=Path, default=None)
    for name in ('seed', 'experiment-id', 'num-envs', 'num-workers',
                 'total-activations', 'steps-per-env', 'replay-capacity',
                 'batch-size', 'warmup-activations', 'updates-per-collection',
                 'checkpoint-every-activations', 'log-update-every'):
        parser.add_argument(f'--{name}', type=int, default=None)
    for name in ('shaping-lambda', 'epsilon-start', 'epsilon-end'):
        parser.add_argument(f'--{name}', type=float, default=None)
    parser.add_argument('--epsilon-decay-activations', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--target-sync-interval', type=int, default=None)
    parser.add_argument('--huber-delta', type=float, default=None)
    parser.add_argument('--grad-clip-norm', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--pin-memory', action='store_true', default=None)
    parser.add_argument('--bf16', action='store_true', default=None)
    parser.add_argument('--deterministic-algorithms', action='store_true', default=None)
    args = vars(parser.parse_args(argv))
    conf_path = args.pop('config')
    base = _config_from_yaml(conf_path) if conf_path is not None else TrainConfig()
    epsilon_opts = {}
    for key in ('epsilon_start', 'epsilon_end', 'epsilon_decay_activations'):
        value = args.pop(key)
        if value is not None:
            epsilon_opts[key.removeprefix('epsilon_').replace('decay_activations', 'decay_activations')] = value
    dqn_opts = {}
    for key in ('learning_rate', 'target_sync_interval', 'huber_delta',
                'grad_clip_norm', 'weight_decay'):
        value = args.pop(key)
        if value is not None:
            dqn_opts[key] = value
    if args.pop('bf16'):
        dqn_opts['use_bfloat16_amp'] = True
    updates = {key: val for key, val in args.items() if val is not None}
    if epsilon_opts:
        updates['epsilon'] = replace(base.epsilon, **epsilon_opts)
    if dqn_opts:
        updates['dqn'] = replace(base.dqn, **dqn_opts)
    conf = replace(base, **updates)
    result = run_training(conf)
    print(f'[train] device={conf.device} envs={conf.num_envs} workers={conf.num_workers}')
    print(f'[train] attivazioni={result.progress.environment_activations} '
          f'episodi={result.progress.episodes_completed} '
          f'update={result.gradient_updates} replay={result.replay_size} '
          f'epsilon_prossimo={result.progress.epsilon:.4f}')
    print(f'[train] checkpoint={result.checkpoint_path} metriche={result.metrics_path}')
    return 0


if __name__ == '__main__':
    # Necessario su Windows 11 per multiprocessing spawn; nessun processo
    # viene creato automaticamente quando pytest importa train.py.
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
