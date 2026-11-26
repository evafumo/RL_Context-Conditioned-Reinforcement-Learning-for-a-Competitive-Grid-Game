"""Test di utils.metrics, dalla radice Giulio2: pytest tests/test_metrics.py -q."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from agent.collector import CollectionStats
from agent.dqn import DQNUpdateStats
from utils.metrics import MetricsLogger, Stopwatch, summarize_collection, throughput


def collection(**overrides):
    values = dict(num_envs=4, steps_per_env=100, transitions=400,
                  agent_decisions=150, forced_activations=250,
                  completed_episodes=4, wins=2, rewards_sum=-20.0)
    values.update(overrides)
    return CollectionStats(**values)


def update(**overrides):
    values = dict(loss=0.5, mean_q=-2.0, mean_target=-1.0,
                  grad_norm=5.0, update_steps=10, target_synced=False,
                  td_mae=3.0, td_p90=4.0, std_q=1.5,
                  mean_max_legal_q=-0.25, std_max_legal_q=0.75)
    values.update(overrides)
    return DQNUpdateStats(**values)


@pytest.mark.parametrize('count,seconds,expected', [(0, 2, 0), (100, 2, 50), (8, 0.5, 16)])
def test_throughput(count, seconds, expected):
    assert throughput(count, seconds) == expected


@pytest.mark.parametrize('count,seconds', [(-1, 1), (True, 1), (1, 0), (1, -1), (1, math.nan), (1, math.inf)])
def test_throughput_rejects_invalid(count, seconds):
    with pytest.raises((TypeError, ValueError)):
        throughput(count, seconds)


def test_summary_includes_all_activations_including_dummy():
    result = summarize_collection(collection())
    assert result['total_activations'] == 400
    assert result['agent_decisions'] == 150
    assert result['forced_dummy_activations'] == 250
    assert result['decision_fraction'] == 0.375
    assert result['episodes_finished'] == 4
    assert result['win_fraction_completed'] == 0.5
    assert result['reward_sum'] == -20
    assert result['reward_mean_per_activation'] == -0.05
    assert 'context_activation_counts' not in result


def test_summary_without_complete_episodes_does_not_invent_win_rate():
    result = summarize_collection(collection(completed_episodes=0, wins=0))
    assert result['win_fraction_completed'] is None


@pytest.mark.parametrize('change', [
    {'num_envs': 0},
    {'steps_per_env': 0},
    {'transitions': 399},
    {'forced_activations': 249},
    {'agent_decisions': -1},
    {'completed_episodes': 401},
    {'wins': 5},
    {'rewards_sum': math.nan},
    {'rewards_sum': math.inf},
])
def test_summary_rejects_inconsistent_stats(change):
    with pytest.raises((TypeError, ValueError)):
        summarize_collection(collection(**change))


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        summarize_collection({'transitions': 1})


def test_logger_collection_records_and_does_not_modify_stats(tmp_path):
    stats = collection()
    filename = tmp_path / 'nested' / 'log.jsonl'
    with MetricsLogger(filename) as logger:
        record = logger.log_collection(0, stats, collection_seconds=0.25,
                                       epsilon=0.6, replay_size=128,
                                       replay_capacity=1000,
                                       other={'phase': 'warmup', 'np': np.float32(3)})
        assert logger.records_written == 1
    loaded = json.loads(filename.read_text(encoding='utf8'))
    assert loaded['kind'] == 'dqn_collection'
    assert loaded['collection_index'] == 0
    assert loaded['collection_activations_per_second'] == 1600
    assert loaded['replay_fill_fraction'] == 0.128
    assert loaded['collection'] == record['collection']
    assert loaded['timestamp_utc'].endswith('+00:00')
    assert stats.rewards_sum == -20


def test_logger_append_preserves_previous_lines(tmp_path):
    filename = tmp_path / 'metrics.jsonl'
    with MetricsLogger(filename, append=False) as logger:
        logger.log({'kind': 'first'})
    with MetricsLogger(filename, append=True) as logger:
        logger.log({'kind': 'second'})
    rows = [json.loads(line) for line in filename.read_text().splitlines()]
    assert [r['kind'] for r in rows] == ['first', 'second']


def test_logger_update_without_collection(tmp_path):
    filename = tmp_path / 'metrics.jsonl'
    with MetricsLogger(filename) as logger:
        record = logger.log_update(9, update(), update_seconds=0.2,
                                   replay_size=400, epsilon=0.1)
    assert record['updates_per_second'] == 5
    assert record['dqn']['mean_q'] == -2
    assert record['dqn']['td_mae'] == 3.0
    assert record['dqn']['td_p90'] == 4.0
    assert record['dqn']['std_q'] == 1.5
    assert record['dqn']['mean_max_legal_q'] == -0.25
    assert record['dqn']['std_max_legal_q'] == 0.75
    assert record['dqn']['target_synced'] is False
    assert 'collection_activations_per_second' not in record
    assert 'cycle_activations_per_second' not in record
    assert json.loads(filename.read_text())['kind'] == 'dqn_update'


def test_logger_update_with_collection_and_durations(tmp_path):
    with MetricsLogger(tmp_path / 'metrics.jsonl') as logger:
        record = logger.log_update(10, update(target_synced=True),
                                   collection=collection(), collection_seconds=1.0,
                                   update_seconds=1.0)
    assert record['collection_activations_per_second'] == 400
    assert record['cycle_activations_per_second'] == 200
    assert record['dqn']['target_synced']


@pytest.mark.parametrize('bad', [
    {'loss': math.nan}, {'grad_norm': -1.0}, {'mean_target': math.inf},
    {'update_steps': 0}, {'target_synced': 1},
    {'td_mae': -1.0}, {'td_p90': math.nan},
    {'std_q': -0.001}, {'mean_max_legal_q': math.inf},
    {'std_max_legal_q': -0.1}, {'td_mae': True},
])
def test_logger_bad_update_stats_not_logged(tmp_path, bad):
    filename = tmp_path / 'metrics.jsonl'
    with MetricsLogger(filename) as logger:
        with pytest.raises((TypeError, ValueError)):
            logger.log_update(0, update(**bad))
        assert logger.records_written == 0
    assert filename.read_text() == ''


@pytest.mark.parametrize('kwargs', [
    {'epsilon': 1.1}, {'epsilon': -0.1}, {'epsilon': float('nan')},
    {'collection_seconds': 0}, {'replay_size': -2},
    {'replay_size': 201, 'replay_capacity': 200},
    {'replay_capacity': 0}, {'other': {'bad': math.inf}},
])
def test_invalid_collection_metadata_is_not_written(tmp_path, kwargs):
    filename = tmp_path / 'metrics.jsonl'
    with MetricsLogger(filename) as logger:
        with pytest.raises((TypeError, ValueError)):
            logger.log_collection(0, collection(), **kwargs)
        assert logger.records_written == 0
    assert filename.read_text() == ''


def test_logger_rejects_tensor_nan_and_array(tmp_path):
    import torch
    with MetricsLogger(tmp_path / 'metrics.jsonl') as logger:
        for item in (torch.tensor(1.0), np.array([1, 2]), math.nan):
            with pytest.raises((TypeError, ValueError)):
                logger.log({'value': item})
        assert logger.records_written == 0


def test_logger_close_is_idempotent_and_rejects_new_logs(tmp_path):
    logger = MetricsLogger(tmp_path / 'metrics.jsonl')
    logger.close()
    logger.close()
    with pytest.raises(RuntimeError):
        logger.log({'kind': 'x'})


def test_stopwatch_cpu_elapsed_and_restart():
    timer = Stopwatch('cpu')
    with timer:
        time.sleep(0.001)
    assert timer.elapsed_seconds is not None and timer.elapsed_seconds > 0
    with timer:
        pass
    assert timer.elapsed_seconds is not None and timer.elapsed_seconds >= 0


def test_stopwatch_times_even_when_user_code_raises():
    timer = Stopwatch()
    with pytest.raises(RuntimeError, match='work'):
        with timer:
            raise RuntimeError('work')
    assert timer.elapsed_seconds is not None


def test_stopwatch_invalid_device():
    for value in ('cuda:-1', 'cpu:0', 'xpu', 'cuda:a', None):
        with pytest.raises(ValueError):
            Stopwatch(value)


def test_stopwatch_cuda_validation():
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            with Stopwatch('cuda:0'):
                pass
    else:
        with Stopwatch('cuda:0') as timer:
            a = torch.ones((128,), device='cuda:0')
            _ = a * 2
        assert timer.elapsed_seconds is not None and timer.elapsed_seconds >= 0
