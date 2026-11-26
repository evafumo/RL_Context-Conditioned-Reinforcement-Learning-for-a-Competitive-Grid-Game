"""Verifiche del collector DQN: replay, masking, terminali, RNG, CPU/CUDA.

Dalla radice Giulio2:
    python -m pytest tests/test_collector.py -v

I test veloci usano GridGame REALI in un harness nello stesso processo;
i test 'spawn' controllano il reale contratto multiprocess Windows.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from agent.collector import CollectionStats, collect_parallel_experience
from agent.parallel_env import (
    ParallelGridGames, ParallelObservation, ParallelStep,
    ParallelTransitionNext, _local_snapshot,
)
from agent.replay_buffer import ReplayBuffer
from agent.network import QNetwork
from agent.dqn import DQNTrainer, DQNConfig
from environment.action_mask import DUMMY_ACTION_ID, NUM_ACTIONS
from environment.context import ContextDistribution, MAP_IDS, OPPONENT_IDS
from environment.game import GridGame, HORIZON
from environment.rules import Action


def _fixed_distribution(map_index=0, opponent_index=0):
    maps = [0.0] * len(MAP_IDS)
    opponents = [0.0] * len(OPPONENT_IDS)
    maps[map_index] = 1.0
    opponents[opponent_index] = 1.0
    return ContextDistribution(tuple(maps), tuple(opponents))


class LocalGames(ParallelGridGames):
    """Stesso GridGame + snapshot del worker, senza costi di spawn per test unitari."""

    def __init__(self, *, num_envs=2, base_seed=20, context_distribution=None):
        kw = {} if context_distribution is None else {"context_distribution": context_distribution}
        super().__init__(num_envs=num_envs, num_workers=1, base_seed=base_seed, **kw)
        self._games = [
            GridGame(seed=self.seed_plan.environment_seed(i),
                     context_distribution=self.context_distribution)
            for i in range(num_envs)
        ]
        for i, game in enumerate(self._games):
            game.reset(seed=self.seed_plan.episode_seed(i))
        self._observation = self._snapshot()
        self._started = True
        self.step_calls = 0
        self.history = []

    def _snapshot(self):
        return ParallelObservation(**_local_snapshot(self._games))

    def step(self, actions):
        before = self.observation
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs,)
        assert actions.dtype == np.int64
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        utilities = np.full(self.num_envs, -1, dtype=np.int8)
        for i, (game, a) in enumerate(zip(self._games, actions)):
            assert before.action_masks[i, a], "collector ha scelto un'azione illegale"
            result = game.step(int(a))
            rewards[i], dones[i] = result.reward, result.terminated
            if result.terminated:
                utilities[i] = result.outcome.utility
                self._episode_indices[i] += 1
                next_id = i + self._episode_indices[i] * self.num_envs
                game.reset(seed=self.seed_plan.episode_seed(next_id))
        following = self._snapshot()
        spatial = following.spatial.copy()
        features = following.features.copy()
        masks = following.action_masks.copy()
        spatial[dones] = 0
        features[dones] = 0
        masks[dones] = False
        self._observation = following
        self.history.append((before, actions.copy(), rewards.copy(), dones.copy(), utilities.copy()))
        self.step_calls += 1
        return ParallelStep(
            rewards=rewards, dones=dones, terminal_utilities=utilities,
            transition_next=ParallelTransitionNext(spatial, features, masks),
            next_observation=following,
        )


class FixedQ(nn.Module):
    """Q alto per DUMMY, poi PASS: il masking deve sempre prevalere."""

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor([0., 1., 2., 3., 4., 6., 100.]))
        self.observed = []

    def forward(self, spatial, features):
        self.observed.append((tuple(spatial.shape), tuple(features.shape),
                              torch.is_grad_enabled(), self.training, spatial.device.type))
        return self.q.unsqueeze(0).expand(spatial.shape[0], -1)


class BadQ(FixedQ):
    def forward(self, spatial, features):
        return torch.zeros((spatial.shape[0], NUM_ACTIONS - 1), device=spatial.device)


class RaiseQ(FixedQ):
    def forward(self, spatial, features):
        raise RuntimeError("errore intenzionale")


def _saved(replay):
    """Usare snapshot pubblici per ispezionare TUTTE le transizioni."""
    return replay.state_dict()


def test_small_cpu_collection_keeps_model_train_mode():
    env = LocalGames(num_envs=4)
    network = FixedQ().train()
    buffer = ReplayBuffer(32, seed=101)
    initial = env.observation
    stats = collect_parallel_experience(env, network, buffer,
                                        steps_per_env=5, epsilon=0.0)
    assert isinstance(stats, CollectionStats)
    assert stats.num_envs == 4 and stats.steps_per_env == 5
    assert stats.transitions == 20 and len(buffer) == 20
    assert stats.agent_decisions + stats.forced_activations == 20
    assert stats.completed_episodes == 0 and stats.wins == 0
    assert env.step_calls == 5 and np.all(env.observation.activations == 6)
    assert network.training
    assert len(network.observed) == 5
    assert all(shape == (4, 4, 9, 9) and feat == (4, 13)
               and not grad and not training and device == 'cpu'
               for shape, feat, grad, training, device in network.observed)
    data = _saved(buffer)
    np.testing.assert_array_equal(data['spatial'][0], initial.spatial[0])
    np.testing.assert_array_equal(data['features'][0], initial.features[0])
    assert data['action_masks'].shape == (20, 7)
    assert data['next_spatial'].shape == (20, 4, 9, 9)
    assert not data['dones'].any()


def test_uses_greedy_mask_and_dummy_each_activation():
    env = LocalGames(num_envs=3, base_seed=102)
    buffer = ReplayBuffer(512)
    stats = collect_parallel_experience(env, FixedQ(), buffer,
                                        steps_per_env=95, epsilon=0)
    data = _saved(buffer)
    selected = data['actions']
    is_forced = data['action_masks'][:, DUMMY_ACTION_ID]
    assert is_forced.any() and (~is_forced).any()
    assert (selected[is_forced] == DUMMY_ACTION_ID).all()
    assert (selected[~is_forced] == int(Action.PASS)).all()
    assert np.all(data['action_masks'][np.arange(len(selected)), selected])
    assert np.count_nonzero(is_forced) == stats.forced_activations
    assert np.count_nonzero(~is_forced) == stats.agent_decisions
    assert len(buffer) == 3 * 95


def test_next_state_is_correct_for_non_terminal_transitions():
    env = LocalGames(num_envs=2)
    replay = ReplayBuffer(20)
    collect_parallel_experience(env, FixedQ(), replay, steps_per_env=4, epsilon=0)
    data = _saved(replay)
    # Replay inserisce nello stesso ordine di environment_id per ogni step.
    for i in range(2):
        np.testing.assert_array_equal(data['next_spatial'][i], data['spatial'][2+i])
        np.testing.assert_array_equal(data['next_features'][i], data['features'][2+i])
        np.testing.assert_array_equal(data['next_action_masks'][i], data['action_masks'][2+i])
    assert np.all(data['next_features'][:, 1] > data['features'][:, 1])


def test_terminal_has_no_bootstrap_and_no_mix_with_reset():
    env = LocalGames(num_envs=1, base_seed=8)
    buffer = ReplayBuffer(110)
    initial_seed = env.observation.episode_seeds[0]
    stats = collect_parallel_experience(env, FixedQ(), buffer,
                                        steps_per_env=101, epsilon=0.0)
    data = _saved(buffer)
    assert stats.transitions == 101 and stats.completed_episodes == 1
    assert env.observation.activations[0] == 2
    assert env.observation.episode_seeds[0] != initial_seed
    assert np.count_nonzero(data['dones']) == 1
    assert bool(data['dones'][99])
    assert not bool(data['dones'][100])
    assert not data['next_spatial'][99].any()
    assert not data['next_features'][99].any()
    assert not data['next_action_masks'][99].any()
    assert data['features'][99, 1] == pytest.approx(1.0)
    assert data['features'][100, 1] == pytest.approx(0.0)
    assert not np.array_equal(data['spatial'][100], data['next_spatial'][99])
    assert data['next_action_masks'][100].any()
    assert data['action_masks'][99, data['actions'][99]]


def test_collect_across_multiple_calls_and_two_episodes():
    env = LocalGames(num_envs=2, base_seed=7)
    buffer = ReplayBuffer(450)
    s1 = collect_parallel_experience(env, FixedQ(), buffer,
                                     steps_per_env=99, epsilon=0)
    assert s1.completed_episodes == 0
    s2 = collect_parallel_experience(env, FixedQ(), buffer,
                                     steps_per_env=101, epsilon=0)
    assert s2.completed_episodes == 4
    assert s1.transitions + s2.transitions == 400
    assert len(buffer) == 400
    assert np.count_nonzero(_saved(buffer)['dones']) == 4
    assert np.all(env.observation.activations == 1)
    assert env.episode_indices == (2, 2)


def test_reward_telescopes_over_complete_episode():
    env = LocalGames(num_envs=1, base_seed=205)
    buffer = ReplayBuffer(100)
    stats = collect_parallel_experience(env, FixedQ(), buffer,
                                        steps_per_env=100, epsilon=0)
    assert stats.completed_episodes == 1
    terminal_utility = int(env.history[-1][4][0])
    assert terminal_utility in (0, 1)
    # Reward shaping potential-based telescopico per episodio completo.
    assert np.isfinite(stats.rewards_sum)
    assert stats.rewards_sum == pytest.approx(float(terminal_utility), abs=2e-5)


def test_replay_capacity_wrap_keeps_latest_steps():
    env = LocalGames(num_envs=2)
    buffer = ReplayBuffer(5)
    stats = collect_parallel_experience(env, FixedQ(), buffer,
                                        steps_per_env=7, epsilon=0)
    assert stats.transitions == 14 and len(buffer) == 5
    assert buffer.total_added == 14 and buffer.position == 4
    assert np.all(np.isfinite(_saved(buffer)['rewards']))
    assert np.all(_saved(buffer)['action_masks'][np.arange(5), _saved(buffer)['actions']])


@pytest.mark.parametrize('epsilon', [0.0, 0.05, 0.5, 1.0])
def test_epsilon_values_always_legal(epsilon):
    env = LocalGames(num_envs=3, base_seed=90)
    replay = ReplayBuffer(180)
    gen = torch.Generator(device='cpu').manual_seed(999)
    stats = collect_parallel_experience(env, FixedQ(), replay,
                                        steps_per_env=60, epsilon=epsilon, generator=gen)
    data = _saved(replay)
    assert stats.transitions == 180
    assert np.all(data['action_masks'][np.arange(180), data['actions']])
    forced = data['action_masks'][:, DUMMY_ACTION_ID]
    assert np.all(data['actions'][forced] == DUMMY_ACTION_ID)


def test_random_policy_does_not_always_choose_pass():
    env = LocalGames(num_envs=2, base_seed=2)
    replay = ReplayBuffer(100)
    collect_parallel_experience(env, FixedQ(), replay, steps_per_env=50,
                                epsilon=1, generator=torch.Generator().manual_seed(7))
    state = _saved(replay)
    actions = state['actions']
    is_agent = ~state['action_masks'][:, DUMMY_ACTION_ID]
    assert np.any(actions[is_agent] != int(Action.PASS))


def test_reproducibility_with_torch_generator_and_independent_rng():
    results = []
    for _ in range(2):
        env = LocalGames(num_envs=3, base_seed=88)
        replay = ReplayBuffer(128, seed=123)
        net = FixedQ()
        result = collect_parallel_experience(env, net, replay,
                                             steps_per_env=40, epsilon=0.6,
                                             generator=torch.Generator().manual_seed(5))
        results.append((result, _saved(replay)))
    assert results[0][0] == results[1][0]
    for key in ('spatial', 'features', 'actions', 'rewards', 'dones',
                'next_spatial', 'next_features', 'next_action_masks'):
        np.testing.assert_array_equal(results[0][1][key], results[1][1][key])


def test_epsilon_zero_does_not_consume_generator_rng():
    env = LocalGames(num_envs=1)
    replay = ReplayBuffer(8)
    g = torch.Generator().manual_seed(123)
    old = g.get_state().clone()
    collect_parallel_experience(env, FixedQ(), replay,
                                steps_per_env=8, epsilon=0, generator=g)
    assert torch.equal(g.get_state(), old)


@pytest.mark.parametrize('value', [0, -1, True, 2.5, None, '100'])
def test_invalid_steps_per_env_no_side_effects(value):
    env = LocalGames(num_envs=1)
    replay = ReplayBuffer(10)
    with pytest.raises((TypeError, ValueError)):
        collect_parallel_experience(env, FixedQ(), replay,
                                    steps_per_env=value, epsilon=0)
    assert env.step_calls == 0 and len(replay) == 0


@pytest.mark.parametrize('epsilon', [-0.1, 1.1, float('nan'), float('inf'), True, None, '0.5'])
def test_invalid_epsilon_no_side_effects(epsilon):
    env = LocalGames(num_envs=1)
    replay = ReplayBuffer(10)
    with pytest.raises((TypeError, ValueError)):
        collect_parallel_experience(env, FixedQ(), replay,
                                    steps_per_env=5, epsilon=epsilon)
    assert env.step_calls == 0 and len(replay) == 0


def test_wrong_types_fail_early():
    env = LocalGames(num_envs=1)
    net = FixedQ()
    replay = ReplayBuffer(10)
    tests = [
        lambda: collect_parallel_experience(None, net, replay, steps_per_env=1, epsilon=0),
        lambda: collect_parallel_experience(env, object(), replay, steps_per_env=1, epsilon=0),
        lambda: collect_parallel_experience(env, net, None, steps_per_env=1, epsilon=0),
        lambda: collect_parallel_experience(env, net, replay, steps_per_env=1, epsilon=0, pin_memory=1),
        lambda: collect_parallel_experience(env, net, replay, steps_per_env=1, epsilon=0, pin_memory=True),
        lambda: collect_parallel_experience(env, net, replay, steps_per_env=1, epsilon=0, device='cuda:0'),
        lambda: collect_parallel_experience(env, net, replay, steps_per_env=1, epsilon=0, generator='bad'),
    ]
    for case in tests:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            case()
    assert env.step_calls == 0 and len(replay) == 0


def test_unstarted_environment_errors_before_inference():
    env = ParallelGridGames(num_envs=1, num_workers=1, base_seed=4)
    with pytest.raises(RuntimeError, match='Avviare'):
        collect_parallel_experience(env, FixedQ(), ReplayBuffer(10),
                                    steps_per_env=1, epsilon=0)


def test_bad_model_output_preserves_model_mode_and_replay():
    env = LocalGames(num_envs=1)
    replay = ReplayBuffer(10)
    net = BadQ().train()
    with pytest.raises(ValueError):
        collect_parallel_experience(env, net, replay, steps_per_env=3, epsilon=0)
    assert env.step_calls == 0 and len(replay) == 0 and net.training


def test_model_exception_preserves_original_eval_mode():
    env = LocalGames(num_envs=1)
    replay = ReplayBuffer(10)
    net = RaiseQ().eval()
    with pytest.raises(RuntimeError, match='intenzionale'):
        collect_parallel_experience(env, net, replay, steps_per_env=1, epsilon=0)
    assert not net.training and len(replay) == 0


def test_no_gradients_generated_by_collection():
    env = LocalGames(num_envs=2)
    replay = ReplayBuffer(12)
    net = FixedQ()
    collect_parallel_experience(env, net, replay, steps_per_env=6, epsilon=0.2)
    assert all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize('map_idx, opp_idx', [(m, o) for m in range(4) for o in range(6)])
def test_context_one_hot_preserved(map_idx, opp_idx):
    distribution = _fixed_distribution(map_idx, opp_idx)
    env = LocalGames(num_envs=1, context_distribution=distribution)
    replay = ReplayBuffer(3)
    collect_parallel_experience(env, FixedQ(), replay, steps_per_env=3, epsilon=0)
    batch = _saved(replay)
    features = batch['features']
    assert np.all(features[:, 3+map_idx] == 1.0)
    assert np.all(features[:, 7+opp_idx] == 1.0)
    assert np.all(features[:, 3:7].sum(axis=1) == 1.0)
    assert np.all(features[:, 7:13].sum(axis=1) == 1.0)


def test_real_qnetwork_and_one_dqn_update_on_cpu():
    torch.manual_seed(2026)
    trainer = DQNTrainer(config=DQNConfig(), device='cpu')
    replay = ReplayBuffer(16, seed=9)
    env = LocalGames(num_envs=2)
    stats = collect_parallel_experience(env, trainer.online_network, replay,
                                        steps_per_env=6, epsilon=0.1)
    assert stats.transitions == 12
    update = trainer.train_on_replay(replay, batch_size=8)
    assert update.loss >= 0 and np.isfinite(update.loss)
    assert trainer.update_steps == 1


@pytest.mark.parametrize('workers', [1, 2])
def test_spawn_parallel_env_matches_reference(workers):
    # Test reale IPC e Windows spawn; il modello resta nel processo principale.
    with ParallelGridGames(num_envs=2, num_workers=workers, base_seed=2026) as env:
        buffer = ReplayBuffer(24)
        stats = collect_parallel_experience(env, FixedQ(), buffer,
                                            steps_per_env=12, epsilon=0)
        assert stats.transitions == 24
        assert len(buffer) == 24
        assert np.all(env.observation.activations == 13)
        assert np.all(_saved(buffer)['action_masks'][
            np.arange(24), _saved(buffer)['actions']])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Richiede RTX/CUDA')
def test_cuda_rtx_inference_with_pinned_memory():
    net = QNetwork().to('cuda:0')
    env = LocalGames(num_envs=4)
    replay = ReplayBuffer(32)
    rng = torch.Generator(device='cuda:0').manual_seed(222)
    stats = collect_parallel_experience(env, net, replay,
                                        steps_per_env=8, epsilon=0.2,
                                        device='cuda:0', pin_memory=True, generator=rng)
    assert stats.transitions == 32
    assert len(replay) == 32
    assert all(p.device.type == 'cuda' for p in net.parameters())
    assert all(p.grad is None for p in net.parameters())
