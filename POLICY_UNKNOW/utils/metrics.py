"""Metriche e logging JSONL per Context-Conditioned Masked DQN.

Moduli:
  - agent.collector.CollectionStats: statistiche di raccolta, DUMMY inclusi;
  - agent.dqn.DQNUpdateStats: statistiche di UN optimizer step.

Il modulo non esegue step MDP, non campiona il replay e non calcola
Bellman targets. Non importa PyTorch all'avvio: CUDA viene utilizzata
soltanto da Stopwatch(device='cuda:0') per sincronizzare i tempi.

Esempio (processo principale, mai nei worker spawn)::

    from utils.metrics import MetricsLogger, Stopwatch

    with MetricsLogger('runs/exp_0/metrics.jsonl') as log:
        with Stopwatch() as timer:
            collection = collect_parallel_experience(...)
        log.log_collection(0, collection, collection_seconds=timer.elapsed_seconds,
                           epsilon=0.5, replay_size=len(replay),
                           replay_capacity=replay.capacity)

        with Stopwatch(device='cuda:0') as timer:
            update = trainer.train_on_replay(replay, 256)
        log.log_update(0, update, update_seconds=timer.elapsed_seconds,
                       replay_size=len(replay), epsilon=0.5)

Nota: collection.wins/completed_episodes NON e' un win rate di valutazione
indipendente (policy esplorativa e finestra di raccolta parziale).
I contatori di CollectionStats si riferiscono a tutte le attivazioni MDP.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from agent.collector import CollectionStats
    from agent.dqn import DQNUpdateStats


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} deve essere un int non negativo (non bool)')
    return value


def _finite_real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} deve essere un numero reale, non bool')
    number = float(value)
    if not isfinite(number) or (number <= 0 if positive else number < 0):
        constraint = 'positivo' if positive else 'non negativo'
        raise ValueError(f'{name} deve essere finito e {constraint}')
    return number


def _epsilon(value: float | None) -> float | None:
    if value is None:
        return None
    probability = _finite_real(value, 'epsilon')
    if probability > 1:
        raise ValueError('epsilon deve appartenere a [0, 1]')
    return probability


def _replay_metrics(replay_size: int | None, replay_capacity: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if replay_size is not None:
        result['replay_size'] = _nonnegative_int(replay_size, 'replay_size')
    if replay_capacity is not None:
        capacity = _nonnegative_int(replay_capacity, 'replay_capacity')
        if capacity == 0:
            raise ValueError('replay_capacity deve essere > 0')
        result['replay_capacity'] = capacity
        if replay_size is not None:
            if replay_size > capacity:
                raise ValueError('replay_size non puo superare replay_capacity')
            result['replay_fill_fraction'] = replay_size / capacity
    return result


def throughput(count: int, seconds: float) -> float:
    """Operazioni al secondo; chiamante specifica se sono attivazioni o update.

    Un tempo nullo o negativo non produce una misura di throughput valida.
    """
    _nonnegative_int(count, 'count')
    duration = _finite_real(seconds, 'seconds', positive=True)
    return count / duration


def summarize_collection(stats: CollectionStats) -> dict[str, Any]:
    """Riassume esattamente una CollectionStats, senza ricostruire l'MDP.

    Episodi avviati e distribuzioni per contesto non sono presenti in
    CollectionStats; non si possono dedurre da questa struttura.
    Il rapporto wins/completed_episodes e' SOLO descrittivo del batch.
    """
    from agent.collector import CollectionStats

    if not isinstance(stats, CollectionStats):
        raise TypeError('stats deve essere CollectionStats')
    for name in ('num_envs', 'steps_per_env', 'transitions', 'agent_decisions',
                 'forced_activations', 'completed_episodes', 'wins'):
        _nonnegative_int(getattr(stats, name), name)
    if stats.num_envs == 0 or stats.steps_per_env == 0:
        raise ValueError('num_envs e steps_per_env devono essere > 0')
    if stats.transitions != stats.num_envs * stats.steps_per_env:
        raise ValueError('transitions deve essere num_envs * steps_per_env')
    if stats.agent_decisions + stats.forced_activations != stats.transitions:
        raise ValueError('Ogni attivazione deve essere decisione o DUMMY')
    if stats.completed_episodes > stats.transitions:
        raise ValueError('completed_episodes supera le transizioni')
    if stats.wins > stats.completed_episodes:
        raise ValueError('wins supera completed_episodes')
    # Reward negativo ammesso: puo' essere un reward shaping legittimo.
    reward_sum = stats.rewards_sum
    if isinstance(reward_sum, (bool, np.bool_)) or not isinstance(reward_sum, Real):
        raise TypeError('rewards_sum deve essere reale')
    reward_sum = float(reward_sum)
    if not isfinite(reward_sum):
        raise ValueError('rewards_sum deve essere finito')

    n = stats.transitions
    return {
        'total_activations': n,
        'num_envs': stats.num_envs,
        'steps_per_env': stats.steps_per_env,
        'agent_decisions': stats.agent_decisions,
        'forced_dummy_activations': stats.forced_activations,
        'decision_fraction': stats.agent_decisions / n,
        'episodes_finished': stats.completed_episodes,
        'wins': stats.wins,
        'win_fraction_completed': (
            stats.wins / stats.completed_episodes if stats.completed_episodes else None
        ),
        'reward_sum': reward_sum,
        'reward_mean_per_activation': reward_sum / n,
    }


def _json_ready(value: Any) -> Any:
    """Converte soltanto strutture JSON-safe; rifiuta NaN/Inf e tensori.

    Gli scalari NumPy sono ammessi. Gli array e i tensori PyTorch NON sono
    serializzati implicitamente: farlo potrebbe introdurre grandi copie o
    sincronizzazioni CUDA nascoste nel loop di training.
    """
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError('Le metriche non ammettono NaN o infinito')
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError('Le chiavi JSON devono essere stringhe')
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    raise TypeError(f'Tipo non serializzabile nelle metriche: {type(value).__name__}')


def _update_dict(stats: DQNUpdateStats) -> dict[str, Any]:
    """Valida il contratto scalare del trainer DQN prima del logging."""
    from agent.dqn import DQNUpdateStats

    if not isinstance(stats, DQNUpdateStats):
        raise TypeError('dqn_metrics deve essere DQNUpdateStats')
    _nonnegative_int(stats.update_steps, 'update_steps')
    if stats.update_steps == 0:
        raise ValueError('DQNUpdateStats rappresenta un update gia eseguito')
    if type(stats.target_synced) is not bool:
        raise TypeError('target_synced deve essere bool')
    # TD error, deviazioni standard, loss e grad_norm sono non negativi.
    for name in ('loss', 'grad_norm', 'td_mae', 'td_p90',
                 'std_q', 'std_max_legal_q'):
        _finite_real(getattr(stats, name), name)
    # I Q-value e i target possono essere negativi.
    for name in ('mean_q', 'mean_target', 'mean_max_legal_q'):
        value = getattr(stats, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f'{name} deve essere reale')
        if not isfinite(float(value)):
            raise ValueError(f'{name} deve essere finito')
    return _json_ready(asdict(stats))


class MetricsLogger:
    """Writer JSON Lines, solo nel processo coordinatore (mai nei worker).

    Una riga = un record completo. append=True conserva i record esistenti;
    fsync=False evita una sync su disco ad ogni evento (log resta flushato).
    """

    def __init__(self, path: str | Path, *, append: bool = True, fsync: bool = False) -> None:
        self.path = Path(path)
        if not self.path.name or self.path.is_dir():
            raise ValueError('path deve indicare un file, non una directory')
        if type(append) is not bool or type(fsync) is not bool:
            raise TypeError('append e fsync devono essere bool')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open('a' if append else 'w', encoding='utf-8', newline='\n')
        self._fsync = fsync
        self.records_written = 0

    def log(self, record: Mapping[str, Any]) -> None:
        """Valida il record e lo scrive; nessuna mutazione dell'input."""
        if self._handle.closed:
            raise RuntimeError('MetricsLogger e gia chiuso')
        if not isinstance(record, Mapping):
            raise TypeError('record deve essere una mappa')
        data = _json_ready(record)
        if 'timestamp_utc' not in data:
            data = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **data}
        line = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._handle.write(line + '\n')
        self._handle.flush()
        if self._fsync:
            os.fsync(self._handle.fileno())
        self.records_written += 1

    def log_collection(
        self,
        collection_index: int,
        collection: CollectionStats,
        *,
        collection_seconds: float | None = None,
        epsilon: float | None = None,
        replay_size: int | None = None,
        replay_capacity: int | None = None,
        other: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Registra una finestra di interazioni MDP (DUMMY inclusi)."""
        _nonnegative_int(collection_index, 'collection_index')
        summary = summarize_collection(collection)
        record: dict[str, Any] = {
            'kind': 'dqn_collection',
            'collection_index': collection_index,
            'collection': summary,
        }
        if collection_seconds is not None:
            seconds = _finite_real(collection_seconds, 'collection_seconds', positive=True)
            record['collection_seconds'] = seconds
            record['collection_activations_per_second'] = throughput(
                summary['total_activations'], seconds
            )
        if epsilon is not None:
            record['epsilon'] = _epsilon(epsilon)
        record.update(_replay_metrics(replay_size, replay_capacity))
        if other is not None:
            if not isinstance(other, Mapping):
                raise TypeError('other deve essere una mappa')
            record['other'] = _json_ready(other)
        self.log(record)
        return record

    def log_update(
        self,
        update_index: int,
        dqn_metrics: DQNUpdateStats,
        *,
        collection: CollectionStats | None = None,
        collection_seconds: float | None = None,
        update_seconds: float | None = None,
        epsilon: float | None = None,
        replay_size: int | None = None,
        replay_capacity: int | None = None,
        other: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Registra UN optimizer step.

        collection e' facoltativo perche' N update possono riusare lo stesso
        replay senza raccogliere alcuna nuova attivazione. Per questo NON si
        definisce un ambiguo 'update_activations_per_second'.
        """
        _nonnegative_int(update_index, 'update_index')
        record: dict[str, Any] = {
            'kind': 'dqn_update',
            'update_index': update_index,
            'dqn': _update_dict(dqn_metrics),
        }
        if collection is not None:
            summary = summarize_collection(collection)
            record['collection'] = summary
        else:
            summary = None
        if collection_seconds is not None:
            seconds = _finite_real(collection_seconds, 'collection_seconds', positive=True)
            record['collection_seconds'] = seconds
            if summary is not None:
                record['collection_activations_per_second'] = throughput(
                    summary['total_activations'], seconds
                )
        if update_seconds is not None:
            seconds = _finite_real(update_seconds, 'update_seconds', positive=True)
            record['update_seconds'] = seconds
            record['updates_per_second'] = throughput(1, seconds)
        if collection_seconds is not None and update_seconds is not None and summary is not None:
            record['cycle_activations_per_second'] = throughput(
                summary['total_activations'], collection_seconds + update_seconds
            )
        if epsilon is not None:
            record['epsilon'] = _epsilon(epsilon)
        record.update(_replay_metrics(replay_size, replay_capacity))
        if other is not None:
            if not isinstance(other, Mapping):
                raise TypeError('other deve essere una mappa')
            record['other'] = _json_ready(other)
        self.log(record)
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class Stopwatch:
    """Cronometro monotono con sincronizzazione CUDA opzionale.

    device='cpu' non importa torch. 'cuda' e 'cuda:<index>' sincronizzano
    prima e dopo l'intervallo. Per benchmark GPU usare 'cuda:0'; per un
    training lungo evitare una sincronizzazione CUDA ad ogni micro-evento.
    """

    def __init__(self, device: str = 'cpu') -> None:
        if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::\d+)?', device):
            raise ValueError("device deve essere 'cpu', 'cuda' o 'cuda:N'")
        self.device = device
        self.elapsed_seconds: float | None = None
        self._start: float | None = None

    def _sync(self) -> None:
        if self.device.startswith('cuda'):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError('Sincronizzazione CUDA richiesta ma CUDA non disponibile')
            torch.cuda.synchronize(device=self.device)

    def __enter__(self) -> Stopwatch:
        if self._start is not None:
            raise RuntimeError('Stopwatch non puo essere riavviato mentre e in esecuzione')
        self._sync()
        self.elapsed_seconds = None
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self._sync()
            if self._start is not None:
                self.elapsed_seconds = perf_counter() - self._start
        finally:
            self._start = None


__all__ = ['throughput', 'summarize_collection', 'MetricsLogger', 'Stopwatch']
