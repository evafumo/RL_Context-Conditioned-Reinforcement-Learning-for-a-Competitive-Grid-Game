"""Test di Fase 5.5 per train.py: orchestrazione, MDP, resume e CLI.

Collocare in Giulio2/tests/test_train.py e lanciare dalla radice::

    python -m pytest tests/test_train.py -v

I test spawn sono reali (Windows incluso); non si avviano worker all'import.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.dqn import DQNConfig, DQNTrainer
from agent.replay_buffer import ReplayBuffer
from environment.game import HORIZON
from train import (
    LinearEpsilonSchedule, TrainConfig, TrainingResult, main,
    run_training, training_signature,
)
from utils.checkpoint import TrainingProgress, load_checkpoint
from utils.seeds import SeedPlan, make_torch_generator


@pytest.fixture(autouse=True)
def _fast_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def config_for(tmp_path: Path, **overrides) -> TrainConfig:
    config = TrainConfig(
        output_dir=tmp_path, device='cpu', seed=23911, experiment_id=3,
        num_envs=1, num_workers=1, total_activations=HORIZON,
        steps_per_env=HORIZON, replay_capacity=220, batch_size=8,
        warmup_activations=0, updates_per_collection=2,
        epsilon=LinearEpsilonSchedule(start=0.8, end=0.2, decay_activations=300),
        dqn=DQNConfig(target_sync_interval=2, learning_rate=1e-4),
        checkpoint_every_activations=0, log_update_every=1,
    )
    return replace(config, **overrides)


def _read_metrics(path: Path) -> list[dict]:
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize('step,expected', [
    (0, 1.0), (25, 0.75), (50, 0.5), (100, 0.0), (110, 0.0),
])
def test_schedule(step, expected):
    schedule = LinearEpsilonSchedule(1, 0, 100)
    assert schedule.value(step) == pytest.approx(expected)
    assert schedule.state_dict() == {
        'kind': 'linear_by_activation', 'version': 1,
        'start': 1., 'end': 0., 'decay_activations': 100,
    }


@pytest.mark.parametrize('kwargs', [
    {'start': -0.1}, {'end': 1.1}, {'start': 0.1, 'end': 0.2},
    {'decay_activations': 0}, {'decay_activations': True},
    {'start': float('nan')}, {'end': float('inf')},
])
def test_invalid_schedule(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LinearEpsilonSchedule(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'num_envs': 0}, {'num_envs': True}, {'num_workers': 0},
    {'num_workers': 2}, {'total_activations': 99},
    {'total_activations': 101}, {'steps_per_env': 99},
    {'steps_per_env': 101}, {'batch_size': 0},
    {'replay_capacity': 7}, {'warmup_activations': -1},
    {'updates_per_collection': 0}, {'checkpoint_every_activations': -1},
    {'seed': 2**64}, {'experiment_id': 2**64},
    {'shaping_lambda': -0.1}, {'shaping_lambda': float('nan')},
    {'pin_memory': True}, {'dqn': DQNConfig(gamma=0.95)},
])
def test_bad_training_settings(tmp_path, kwargs):
    with pytest.raises((TypeError, ValueError)):
        config_for(tmp_path, **kwargs)


def test_schedule_affects_signature_and_budget_does_not(tmp_path):
    conf = config_for(tmp_path)
    signature = training_signature(conf)
    extended = replace(conf, total_activations=200, checkpoint_every_activations=123,
                       log_update_every=0, output_dir=tmp_path/'new')
    assert training_signature(extended) == signature
    assert training_signature(replace(conf, steps_per_env=200)) != signature
    assert training_signature(replace(conf, warmup_activations=100)) != signature
    assert training_signature(replace(conf, epsilon=LinearEpsilonSchedule())) != signature


def test_training_cpu_checkpoint_and_real_replay(tmp_path):
    conf = config_for(tmp_path)
    result = run_training(conf)
    assert result.progress.environment_activations == HORIZON
    assert result.progress.episodes_completed == 1
    assert result.progress.collections_completed == 1
    assert result.progress.epsilon == pytest.approx(conf.epsilon.value(HORIZON))
    assert result.gradient_updates == 2
    assert result.replay_size == HORIZON
    assert result.checkpoint_path.is_file()
    assert result.metrics_path.is_file()
    assert result.elapsed_seconds > 0

    records = _read_metrics(result.metrics_path)
    assert [rec['kind'] for rec in records] == [
        'dqn_collection', 'dqn_update', 'dqn_update', 'checkpoint',
    ]
    collection = records[0]['collection']
    assert collection['total_activations'] == HORIZON
    assert collection['agent_decisions'] + collection['forced_dummy_activations'] == HORIZON
    assert collection['episodes_finished'] == 1
    assert records[0]['other']['environment_activations'] == 100
    assert records[0]['epsilon'] == pytest.approx(conf.epsilon.start)
    assert records[1]['dqn']['update_steps'] == 1
    assert records[2]['dqn']['target_synced'] is True

    # Le transizioni vere sono esaminate senza modificare lo stato del file.
    saved = torch.load(result.checkpoint_path, map_location='cpu', weights_only=True)
    assert saved['exact_resume_possible'] is True
    assert saved['progress']['environment_activations'] == 100
    assert saved['trainer']['update_steps'] == 2
    assert saved['replay']['total_added'] == 100
    assert int(saved['replay']['dones'][:100].sum()) == 1
    assert bool(saved['replay']['dones'][99])
    assert saved['replay']['next_action_masks'][99].sum() == 0
    assert not bool(saved['replay']['next_spatial'][99].any())
    assert not bool(saved['replay']['next_features'][99].any())
    masks = saved['replay']['action_masks'][:100]
    actions = saved['replay']['actions'][:100].to(torch.long)
    assert masks.gather(1, actions[:, None]).all()
    assert int((actions == 6).sum()) > 0  # DUMMY non vengono esclusi


def test_no_gradient_updates_during_warmup(tmp_path):
    conf = config_for(tmp_path, warmup_activations=201)
    result = run_training(conf)
    assert result.gradient_updates == 0
    assert result.replay_size == HORIZON
    assert [r['kind'] for r in _read_metrics(result.metrics_path)] == [
        'dqn_collection', 'checkpoint',
    ]


def test_resume_matches_uninterrupted_training(tmp_path):
    """Su CPU, replay, pesi, target, Adam e RNG devono coincidere."""
    one_dir = tmp_path / 'uninterrupted'
    split_dir = tmp_path / 'split'
    whole = config_for(one_dir, total_activations=200)
    separate = config_for(split_dir)

    whole_result = run_training(whole)
    first_result = run_training(separate)
    resumed = run_training(replace(separate, total_activations=200,
                                   resume=first_result.checkpoint_path))
    assert resumed.progress.environment_activations == 200
    assert resumed.progress.episodes_completed == 2
    assert resumed.progress.collections_completed == 2
    assert resumed.gradient_updates == whole_result.gradient_updates == 4
    assert resumed.replay_size == whole_result.replay_size == 200

    first = torch.load(whole_result.checkpoint_path, weights_only=True, map_location='cpu')
    second = torch.load(resumed.checkpoint_path, weights_only=True, map_location='cpu')
    for field in ('progress', 'epsilon_schedule_state', 'environments', 'metadata'):
        assert first[field] == second[field]
    for field in ('spatial', 'features', 'action_masks', 'actions', 'rewards',
                  'next_spatial', 'next_features', 'next_action_masks', 'dones'):
        assert torch.equal(first['replay'][field], second['replay'][field]), field
    for layer_name in ('online_network', 'target_network'):
        for name, tensor in first['trainer'][layer_name].items():
            assert torch.equal(tensor, second['trainer'][layer_name][name]), name
    assert first['trainer']['update_steps'] == second['trainer']['update_steps']
    for param_idx, optim_values in first['trainer']['optimizer']['state'].items():
        other = second['trainer']['optimizer']['state'][param_idx]
        for name, value in optim_values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other[name])
            else:
                assert value == other[name]
    assert resumed.metrics_path.name == 'metrics_resume_from_100.jsonl'
    assert [r['kind'] for r in _read_metrics(resumed.metrics_path)] == [
        'dqn_collection', 'dqn_update', 'dqn_update', 'checkpoint',
    ]


def test_resume_rejects_different_schedule(tmp_path):
    first = config_for(tmp_path)
    result = run_training(first)
    changed = replace(first, total_activations=200,
                      epsilon=LinearEpsilonSchedule(start=0.9, end=0.2,
                                                    decay_activations=300),
                      resume=result.checkpoint_path)
    with pytest.raises(ValueError, match='incompatibili|epsilon'):
        run_training(changed)


def test_partial_last_collection_uses_episode_boundaries(tmp_path):
    config = config_for(tmp_path, steps_per_env=200, total_activations=300,
                        updates_per_collection=1, log_update_every=0)
    result = run_training(config)
    assert result.progress.environment_activations == 300
    assert result.progress.collections_completed == 2
    assert result.progress.episodes_completed == 3
    assert result.gradient_updates == 2
    records = [r for r in _read_metrics(result.metrics_path) if r['kind'] == 'dqn_collection']
    assert [r['collection']['total_activations'] for r in records] == [200, 100]


def test_periodic_checkpoint_at_crossed_threshold(tmp_path):
    config = config_for(tmp_path, total_activations=200, checkpoint_every_activations=150,
                        updates_per_collection=1, log_update_every=0)
    result = run_training(config)
    assert not (tmp_path / 'checkpoint_100.pt').exists()
    assert (tmp_path / 'checkpoint_200.pt').is_file()
    assert result.checkpoint_path.name == 'checkpoint_200.pt'


def test_main_cli_configuration(monkeypatch, tmp_path, capsys):
    received = []

    def fake_run(conf):
        received.append(conf)
        return TrainingResult(
            progress=TrainingProgress(100, 1, 1, 0.5), gradient_updates=2,
            replay_size=100, checkpoint_path=tmp_path/'x.pt',
            metrics_path=tmp_path/'metrics.jsonl', elapsed_seconds=0.1,
        )

    monkeypatch.setattr('train.run_training', fake_run)
    assert main(['--device', 'cpu', '--num-envs', '1', '--num-workers', '1',
                 '--total-activations', '100', '--bf16',
                 '--epsilon-start', '0.8', '--epsilon-end', '0.2',
                 '--epsilon-decay-activations', '500', '--learning-rate', '0.0003']) == 0
    assert received[0].epsilon.value(0) == 0.8
    assert received[0].epsilon.decay_activations == 500
    assert received[0].dqn.use_bfloat16_amp
    assert received[0].dqn.learning_rate == pytest.approx(0.0003)
    assert 'checkpoint=' in capsys.readouterr().out


def test_yaml_configuration(tmp_path, monkeypatch):
    pytest.importorskip('yaml')
    file = tmp_path / 'training.yaml'
    file.write_text('''train:\n  num_envs: 2\n  num_workers: 1\n  total_activations: 200\n  epsilon:\n    start: 0.9\n    end: 0.2\n    decay_activations: 800\ndqn:\n  learning_rate: 0.0002\n''', encoding='utf-8')
    received = []
    monkeypatch.setattr('train.run_training', lambda conf: received.append(conf) or TrainingResult(
        TrainingProgress(200, 2, 1, 0.2), 2, 200,
        tmp_path / 'checkpoint.pt', tmp_path / 'metrics.jsonl', 1.0))
    assert main(['--config', str(file), '--num-workers', '2']) == 0
    assert received[0].num_envs == 2
    assert received[0].num_workers == 2
    assert received[0].epsilon.start == 0.9
    assert received[0].dqn.learning_rate == pytest.approx(2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='PyTorch CUDA non disponibile')
def test_gpu_smoke_cuda(tmp_path):
    config = config_for(tmp_path, device='cuda:0', pin_memory=True, batch_size=16,
                        dqn=DQNConfig(target_sync_interval=2, learning_rate=1e-4))
    result = run_training(config)
    assert result.gradient_updates == 2
    saved = torch.load(result.checkpoint_path, map_location='cpu', weights_only=True)
    assert saved['training_device'] == 'cuda:0'
    assert all(t.device.type == 'cpu' for t in saved['trainer']['online_network'].values())
    assert saved['rng']['torch_cuda'] is not None
