"""Test per utils.checkpoint: salvataggio/ripristino DQN, replay e RNG.

Eseguire dalla root Giulio2: ``python -m pytest tests/test_checkpoint.py -q``.
Il test multiprocessing usa spawn ed e' compatibile anche con Windows.
"""

from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from agent.dqn import DQNConfig, DQNTrainer
from agent.parallel_env import ParallelGridGames
from agent.replay_buffer import ReplayBuffer
from environment.action_mask import legal_actions_to_mask
from environment.context import ContextDistribution
from environment.observations import Observation, SPATIAL_SHAPE, FEATURE_SIZE
from environment.rules import Action
from utils.checkpoint import (
    CHECKPOINT_VERSION, CheckpointInfo, EnvironmentResume, TrainingProgress,
    load_checkpoint, save_checkpoint, validate_environment_resume,
)
from utils.seeds import SeedPlan


class TinyQ(nn.Module):
    """Rete compatta (7 Q-values) per verificare Adam e le due network."""
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.arange(7, dtype=torch.float32))

    def forward(self, spatial, features):
        return self.q.unsqueeze(0).expand(spatial.shape[0], -1)


def _trainer(interval=5):
    return DQNTrainer(
        TinyQ(), config=DQNConfig(target_sync_interval=interval, learning_rate=0.003)
    )


def _observation(n=0):
    x = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    x[0, 0, 0] = float(n % 2)
    f = np.zeros(FEATURE_SIZE, dtype=np.float32)
    f[0] = float(n) / 100.0
    return Observation(x, f)


def _fill(replay, count=7):
    for i in range(count):
        forced = (i % 2 == 0)
        done = (i % 3 == 0)
        mask = legal_actions_to_mask([Action.DUMMY if forced else Action.PASS])
        replay.add(
            observation=_observation(i),
            action_mask=mask,
            action=Action.DUMMY if forced else Action.PASS,
            reward=float(i) - 1.0,
            next_observation=None if done else _observation(i + 1),
            next_action_mask=None if done else legal_actions_to_mask([Action.PASS]),
            done=done,
        )


def _save(tmp_path, *, tr=None, replay=None, progress=None, gen=None, **kwargs):
    tr = _trainer() if tr is None else tr
    replay = ReplayBuffer(15, seed=123) if replay is None else replay
    progress = TrainingProgress(0, 0, 0, 1.0) if progress is None else progress
    path = tmp_path / 'subdir' / 'dqn.pt'
    save_checkpoint(path, trainer=tr, replay_buffer=replay,
                    progress=progress, policy_generator=gen, **kwargs)
    return path, tr, replay


def _recursive_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _recursive_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            _recursive_equal(aa, bb)
    else:
        assert a == b


def test_progress_valid():
    assert TrainingProgress(100, 1, 1, 0.25).epsilon == 0.25


@pytest.mark.parametrize('kwargs', [
    dict(environment_activations=-1),
    dict(episodes_completed=-1),
    dict(collections_completed=-1),
    dict(epsilon=-0.1), dict(epsilon=1.1),
    dict(epsilon=float('nan')), dict(epsilon=float('inf')),
    dict(epsilon=True), dict(episodes_completed=True),
    dict(environment_activations=1.0),
])
def test_invalid_progress(kwargs):
    params = dict(environment_activations=0, episodes_completed=0,
                  collections_completed=0, epsilon=0.9)
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        TrainingProgress(**params)


def test_empty_checkpoint_roundtrip(tmp_path):
    path, tr, replay = _save(tmp_path)
    assert path.exists()
    assert path.parent.name == 'subdir'
    info = load_checkpoint(path, trainer=tr, replay_buffer=replay)
    assert isinstance(info, CheckpointInfo)
    assert info.progress == TrainingProgress(0, 0, 0, 1)
    assert info.environments is None
    assert not info.exact_resume_possible


def test_weights_only_and_cpu_tensors(tmp_path):
    path, _, _ = _save(tmp_path)
    raw = torch.load(path, map_location='cpu', weights_only=True)
    assert raw['version'] == CHECKPOINT_VERSION
    assert raw['training_device'] == 'cpu'
    assert raw['format'] == 'Giulio2-masked-dqn'
    assert raw['trainer']['online_network']['q'].device.type == 'cpu'
    assert raw['trainer']['target_network']['q'].device.type == 'cpu'
    assert raw['replay']['actions'].device.type == 'cpu'
    assert not any(p.suffix == '.tmp' for p in path.parent.iterdir())


def test_resume_online_target_optimizer_and_update_count(tmp_path):
    src = _trainer(interval=5)
    rp = ReplayBuffer(15, seed=83)
    _fill(rp)
    src.train_on_replay(rp, 4)
    src.train_on_replay(rp, 4)
    assert src.update_steps == 2
    with torch.no_grad():
        src.target_network.q.add_(3.0)  # indipendenza online/target
    state_before = src.state_dict()
    path, _, _ = _save(tmp_path, tr=src, replay=rp)

    restored = _trainer(interval=5)
    other = ReplayBuffer(15, seed=1)
    info = load_checkpoint(path, trainer=restored, replay_buffer=other)
    assert info.progress.epsilon == 1
    _recursive_equal(state_before, restored.state_dict())
    assert restored.update_steps == 2
    assert not any(p.requires_grad for p in restored.target_network.parameters())
    assert not restored.target_network.training
    # Stesso minibatch e stesso stato Adam: prossimo step deve combaciare.
    orig_stats = src.train_on_replay(rp, 4)
    new_stats = restored.train_on_replay(other, 4)
    assert orig_stats == new_stats
    _recursive_equal(src.state_dict(), restored.state_dict())


def test_replay_ring_terminal_dummy_and_sampling_stream(tmp_path):
    rp = ReplayBuffer(5, seed=22)
    _fill(rp, count=12)
    state = rp.state_dict()
    path, _, _ = _save(tmp_path, replay=rp)
    restored = ReplayBuffer(5, seed=99)
    load_checkpoint(path, trainer=_trainer(), replay_buffer=restored)
    state_new = restored.state_dict()
    _recursive_equal(state, state_new)
    for _ in range(4):
        a = rp.sample(3)
        b = restored.sample(3)
        for name in ('spatial', 'features', 'actions', 'action_masks', 'rewards',
                     'next_spatial', 'next_features', 'next_action_masks', 'dones'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_rng_roundtrip_python_numpy_torch_and_policy(tmp_path):
    random.seed(184)
    np.random.seed(895)
    torch.manual_seed(76)
    policy = torch.Generator(device='cpu').manual_seed(1234)
    _ = torch.rand(2, generator=policy)
    path, tr, replay = _save(tmp_path, gen=policy)
    expected = (
        random.random(), np.random.rand(), torch.rand(3),
        torch.rand(4, generator=policy),
    )
    for _ in range(11):
        random.random(); np.random.rand(); torch.rand(3)
        torch.rand(4, generator=policy)
    load_checkpoint(path, trainer=tr, replay_buffer=replay, policy_generator=policy)
    actual = (random.random(), np.random.rand(), torch.rand(3),
              torch.rand(4, generator=policy))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    torch.testing.assert_close(expected[3], actual[3], atol=0, rtol=0)


def test_restore_rng_false_does_not_modify_global_rng(tmp_path):
    path, tr, rp = _save(tmp_path)
    random.seed(117); np.random.seed(42); torch.manual_seed(21)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(117); np.random.seed(42); torch.manual_seed(21)
    load_checkpoint(path, trainer=tr, replay_buffer=rp, restore_rng=False)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


def test_schedule_metadata_roundtrip_and_isolation(tmp_path):
    sched = {'kind': 'linear', 'position': 100, 'epsilon_end': 0.1,
             'params': [1, None, True, {'warmup': 5}]}
    metadata = {'label': 'trial', 'versions': {'torch': torch.__version__}, 'seeds': [2, 3]}
    path, tr, rp = _save(tmp_path, epsilon_schedule_state=sched, metadata=metadata)
    sched['position'] = 999
    metadata['label'] = 'changed'
    info = load_checkpoint(path, trainer=tr, replay_buffer=rp)
    assert info.epsilon_schedule_state['position'] == 100
    assert info.metadata['label'] == 'trial'
    assert info.epsilon_schedule_state['params'][3]['warmup'] == 5


@pytest.mark.parametrize('value', [float('nan'), float('inf'), torch.zeros(2), object()])
def test_reject_unsafe_schedule_and_metadata(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        _save(tmp_path, epsilon_schedule_state={'bad': value})
    with pytest.raises((TypeError, ValueError)):
        _save(tmp_path, metadata={'bad': value})
    assert not (tmp_path / 'subdir' / 'dqn.pt').exists()


def test_metadata_requires_string_keys(tmp_path):
    with pytest.raises(TypeError):
        _save(tmp_path, metadata={1: 'wrong'})


def test_exact_resume_requires_environment_and_generator(tmp_path):
    with pytest.raises(ValueError, match='richiede'):
        _save(tmp_path, require_exact_resume=True)
    with pytest.raises(ValueError, match='richiede'):
        _save(tmp_path, gen=torch.Generator(), require_exact_resume=True)


def test_unstarted_env_and_mid_episode_rejected(tmp_path):
    env = ParallelGridGames(num_envs=1, num_workers=1, base_seed=111)
    with pytest.raises(RuntimeError):
        _save(tmp_path, environments=env)
    with env:
        action = Action.DUMMY if env.observation.action_masks[0, int(Action.DUMMY)] else Action.PASS
        env.step([int(action)])
        assert int(env.observation.activations[0]) == 2
        with pytest.raises(RuntimeError, match='confine episodio'):
            _save(tmp_path, environments=env)
    assert not (tmp_path / 'subdir' / 'dqn.pt').exists()


def test_boundary_environment_exact_resume_and_seed_validation(tmp_path):
    tr = _trainer()
    rp = ReplayBuffer(11, seed=42)
    gen = torch.Generator().manual_seed(123)
    plan = SeedPlan(master_seed=12345, experiment_id=4)
    with ParallelGridGames(num_envs=2, num_workers=1, seed_plan=plan) as env:
        for _ in range(100):
            m = env.observation.action_masks
            actions = [int(Action.DUMMY) if row[int(Action.DUMMY)] else int(Action.PASS)
                       for row in m]
            env.step(actions)
        assert tuple(env.observation.activations) == (1, 1)
        assert env.episode_indices == (1, 1)
        snapshot = deepcopy(env.observation)
        p = TrainingProgress(200, 2, 1, 0.25)
        path, _, _ = _save(
            tmp_path, tr=tr, replay=rp, progress=p, gen=gen,
            epsilon_schedule_state={'position': 200},
            environments=env, require_exact_resume=True,
        )
    new_tr, new_rp, new_gen = _trainer(), ReplayBuffer(11, seed=1), torch.Generator()
    info = load_checkpoint(path, trainer=new_tr, replay_buffer=new_rp,
                           policy_generator=new_gen, require_exact_resume=True)
    assert info.exact_resume_possible
    assert info.environments is not None
    assert info.environments.episode_indices == (1, 1)
    assert info.epsilon_schedule_state == {'position': 200}
    validate_environment_resume(info.environments, num_envs=2, seed_plan=plan)
    # Modificare il numero di worker NON modifica i seed degli episodi.
    with ParallelGridGames(num_envs=2, num_workers=2, seed_plan=plan,
                           episode_indices=info.environments.episode_indices) as reloaded:
        np.testing.assert_array_equal(reloaded.observation.spatial, snapshot.spatial)
        np.testing.assert_array_equal(reloaded.observation.features, snapshot.features)
        np.testing.assert_array_equal(reloaded.observation.action_masks, snapshot.action_masks)
        assert reloaded.observation.episode_seeds == snapshot.episode_seeds
    with pytest.raises(ValueError, match='SeedPlan'):
        validate_environment_resume(info.environments, num_envs=2,
                                    seed_plan=SeedPlan(master_seed=54321))
    with pytest.raises(ValueError, match='ambienti'):
        validate_environment_resume(info.environments, num_envs=1, seed_plan=plan)
    with pytest.raises(ValueError, match='Reward shaping'):
        validate_environment_resume(info.environments, num_envs=2, seed_plan=plan,
                                    shaping_lambda=0.2)
    with pytest.raises(ValueError, match='Distribuzione'):
        validate_environment_resume(
            info.environments, num_envs=2, seed_plan=plan,
            context_distribution=ContextDistribution(q_maps=(1.,0.,0.,0.)),
        )


def test_environment_progress_mismatch_rejected(tmp_path):
    with ParallelGridGames(num_envs=1, num_workers=1, base_seed=23) as env:
        with pytest.raises(ValueError, match='episodes_completed'):
            _save(tmp_path, environments=env, progress=TrainingProgress(100, 1, 0, 0.3))
        with pytest.raises(ValueError, match='attivazioni'):
            _save(tmp_path, environments=env, progress=TrainingProgress(1, 0, 0, 0.3))


def test_exact_load_requires_generator_and_rng_restore(tmp_path):
    # Salvare checkpoint incompleto: rifiutato dal resume esatto.
    path, tr, rp = _save(tmp_path)
    with pytest.raises(ValueError, match='resume esatto'):
        load_checkpoint(path, trainer=tr, replay_buffer=rp, require_exact_resume=True)
    with pytest.raises(ValueError, match='restore_rng'):
        load_checkpoint(path, trainer=tr, replay_buffer=rp, require_exact_resume=True,
                        restore_rng=False)


def test_bad_format_version_rejected_without_mutating_states(tmp_path):
    path, tr, rp = _save(tmp_path)
    initial = deepcopy(tr.state_dict())
    raw = torch.load(path, weights_only=True)
    raw['version'] = CHECKPOINT_VERSION + 1
    torch.save(raw, path)
    with pytest.raises(ValueError, match='Versione'):
        load_checkpoint(path, trainer=tr, replay_buffer=rp)
    _recursive_equal(initial, tr.state_dict())


def test_bad_top_level_keys_rejected(tmp_path):
    path, tr, rp = _save(tmp_path)
    raw = torch.load(path, weights_only=True)
    raw['new_key'] = 5
    torch.save(raw, path)
    with pytest.raises(ValueError, match='Formato'):
        load_checkpoint(path, trainer=tr, replay_buffer=rp)


def test_config_mismatch_rejected(tmp_path):
    path, tr, rp = _save(tmp_path)
    diff = _trainer(interval=10)
    before = diff.state_dict()
    with pytest.raises(ValueError, match='DQNConfig'):
        load_checkpoint(path, trainer=diff, replay_buffer=rp)
    _recursive_equal(before, diff.state_dict())


def test_replay_capacity_mismatch_rejected(tmp_path):
    path, tr, rp = _save(tmp_path)
    with pytest.raises(ValueError, match='capacity'):
        load_checkpoint(path, trainer=tr, replay_buffer=ReplayBuffer(20))


def test_illegal_action_saved_replay_rejected(tmp_path):
    tr, rp = _trainer(), ReplayBuffer(15, seed=1)
    _fill(rp)
    path, _, _ = _save(tmp_path, tr=tr, replay=rp)
    raw = torch.load(path, weights_only=True)
    raw['replay']['actions'][0] = int(Action.SHOOT)
    torch.save(raw, path)
    with pytest.raises(ValueError, match='Azione illegale'):
        load_checkpoint(path, trainer=_trainer(), replay_buffer=ReplayBuffer(15))


def test_terminal_next_state_invalid_rejected(tmp_path):
    rp = ReplayBuffer(15)
    _fill(rp)
    path, _, _ = _save(tmp_path, replay=rp)
    raw = torch.load(path, weights_only=True)
    raw['replay']['next_features'][0, 0] = 1.0
    torch.save(raw, path)
    with pytest.raises(ValueError, match='Segnaposto terminale'):
        load_checkpoint(path, trainer=_trainer(), replay_buffer=ReplayBuffer(15))


def test_checkpoint_atomic_replace_and_failure_preserves_old_file(tmp_path, monkeypatch):
    path, tr, rp = _save(tmp_path)
    original = path.read_bytes()
    def failing_replace(src, dst):
        raise OSError('simulazione guasto nel rename')
    monkeypatch.setattr('utils.checkpoint.os.replace', failing_replace)
    with pytest.raises(OSError, match='rename'):
        save_checkpoint(path, trainer=tr, replay_buffer=rp,
                        progress=TrainingProgress(0, 0, 0, 0.1))
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_bool_validation_arguments(tmp_path):
    path, tr, rp = _save(tmp_path)
    for args in ({'fsync': 1}, {'require_exact_resume': 1}):
        with pytest.raises(TypeError):
            save_checkpoint(path, trainer=tr, replay_buffer=rp,
                            progress=TrainingProgress(0, 0, 0, 1), **args)
    with pytest.raises(TypeError):
        load_checkpoint(path, trainer=tr, replay_buffer=rp, restore_rng=1)


def test_cpu_load_optional_cuda_test(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA non disponibile su questa macchina')
    cfg = DQNConfig(target_sync_interval=2)
    trainer = DQNTrainer(TinyQ(), config=cfg, device='cuda:0')
    replay = ReplayBuffer(15, seed=12)
    _fill(replay)
    _ = trainer.train_on_replay(replay, 3)
    gen = torch.Generator(device='cuda:0').manual_seed(91)
    file = tmp_path / 'cuda.pt'
    save_checkpoint(file, trainer=trainer, replay_buffer=replay,
                    progress=TrainingProgress(0,0,0,1.0), policy_generator=gen)
    cpu_trainer = DQNTrainer(TinyQ(), config=cfg, device='cpu')
    cpu_replay = ReplayBuffer(15)
    # La Q-network e Adam tornano sul device del trainer corrente.
    load_checkpoint(file, trainer=cpu_trainer, replay_buffer=cpu_replay,
                    restore_rng=False)
    assert next(cpu_trainer.online_network.parameters()).device.type == 'cpu'
    gpu_trainer = DQNTrainer(TinyQ(), config=cfg, device='cuda:0')
    load_checkpoint(file, trainer=gpu_trainer, replay_buffer=ReplayBuffer(15),
                    policy_generator=torch.Generator(device='cuda:0'))
    assert next(gpu_trainer.online_network.parameters()).device.type == 'cuda'
    for state in gpu_trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.ndim:
                assert value.device.type == 'cuda'


def test_real_qnetwork_roundtrip_forward(tmp_path):
    from agent.network import QNetwork
    trainer = DQNTrainer(QNetwork(), config=DQNConfig(target_sync_interval=3))
    replay = ReplayBuffer(12, seed=5)
    _fill(replay)
    _ = trainer.train_on_replay(replay, batch_size=4)
    x = torch.zeros((2, *SPATIAL_SHAPE), dtype=torch.float32)
    f = torch.zeros((2, FEATURE_SIZE), dtype=torch.float32)
    with torch.no_grad():
        q_online = trainer.online_network(x, f).clone()
        q_target = trainer.target_network(x, f).clone()
    path = tmp_path / 'real_qnetwork.pt'
    save_checkpoint(path, trainer=trainer, replay_buffer=replay,
                    progress=TrainingProgress(0, 0, 0, 0.5))
    restored = DQNTrainer(QNetwork(), config=DQNConfig(target_sync_interval=3))
    load_checkpoint(path, trainer=restored, replay_buffer=ReplayBuffer(12))
    with torch.no_grad():
        torch.testing.assert_close(restored.online_network(x, f), q_online, atol=0, rtol=0)
        torch.testing.assert_close(restored.target_network(x, f), q_target, atol=0, rtol=0)
    assert restored.update_steps == 1


def test_strict_device_consistency(tmp_path):
    path, tr, rp = _save(tmp_path)
    raw = torch.load(path, weights_only=True)
    raw['training_device'] = 'cuda:0'
    torch.save(raw, path)
    with pytest.raises(ValueError, match='stesso device'):
        load_checkpoint(path, trainer=tr, replay_buffer=rp,
                        require_exact_resume=True)
