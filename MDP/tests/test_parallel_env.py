"""Test CPU reali di parallel_env: Windows spawn, MDP, maschere e replay DQN.

Da Giulio2/:
    python -m pytest tests/test_parallel_env.py -v

NON richiedono CUDA: il training sulla RTX e' responsabilita' del collector.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from agent.parallel_env import (
    ParallelGridGames, ParallelObservation, ParallelStep,
    ParallelTransitionNext, _local_snapshot,
)
from agent.replay_buffer import ReplayBuffer
from environment.action_mask import (
    DUMMY_ACTION_ID, NUM_ACTIONS, legal_action_mask,
)
from environment.context import ContextDistribution, get_all_contexts
from environment.game import GridGame, HORIZON
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE, Observation
from environment.rules import Action


def _choose(obs: ParallelObservation) -> np.ndarray:
    """Policy test deterministica: DUMMY se forzato, altrimenti PASS."""
    return np.asarray([
        DUMMY_ACTION_ID if mask[DUMMY_ACTION_ID] else int(Action.PASS)
        for mask in obs.action_masks
    ], dtype=np.int64)


def _check_snapshot(obs: ParallelObservation, n: int) -> None:
    assert obs.spatial.shape == (n, *SPATIAL_SHAPE)
    assert obs.features.shape == (n, FEATURE_SIZE)
    assert obs.action_masks.shape == (n, NUM_ACTIONS)
    assert obs.is_agent_decision.shape == (n,)
    assert obs.map_ids.shape == obs.opponent_ids.shape == obs.activations.shape == (n,)
    assert obs.spatial.dtype == obs.features.dtype == np.float32
    assert obs.action_masks.dtype == obs.is_agent_decision.dtype == np.bool_
    assert obs.map_ids.dtype == obs.opponent_ids.dtype == obs.activations.dtype == np.int64
    assert len(obs.episode_seeds) == n
    # Necessari per torch.from_numpy senza warning (inferenza batched CPU/CUDA).
    assert obs.spatial.flags.writeable
    assert obs.features.flags.writeable
    assert obs.action_masks.flags.writeable
    np.testing.assert_array_equal(obs.is_agent_decision, ~obs.action_masks[:, DUMMY_ACTION_ID])
    assert obs.action_masks.any(axis=1).all()
    assert np.isfinite(obs.spatial).all() and np.isfinite(obs.features).all()
    assert np.all((obs.activations >= 1) & (obs.activations <= HORIZON))
    assert np.all((obs.map_ids >= 1) & (obs.map_ids <= 4))
    assert np.all((obs.opponent_ids >= 1) & (obs.opponent_ids <= 6))


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True, None])
def test_invalid_num_envs(invalid):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=invalid, num_workers=1, base_seed=2)


@pytest.mark.parametrize("workers", [0, -1, 4, 2.5, True])
def test_invalid_num_workers(workers):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=3, num_workers=workers, base_seed=2)


@pytest.mark.parametrize("seed", [-1, 2**64, 1.5, True, "hi"])
def test_invalid_base_seed(seed):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=2, num_workers=1, base_seed=seed)


def test_requires_exactly_one_seed_source():
    with pytest.raises(ValueError):
        ParallelGridGames(num_envs=1, num_workers=1)
    with pytest.raises(ValueError):
        ParallelGridGames(num_envs=1, num_workers=1, base_seed=3, seed_plan=object())
    with pytest.raises(TypeError):
        ParallelGridGames(num_envs=1, num_workers=1, seed_plan=object())


@pytest.mark.parametrize("value", [-1, -0.1, float('nan'), float('inf'), True, 'x'])
def test_invalid_shaping_lambda(value):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=1, num_workers=1, base_seed=1, shaping_lambda=value)


@pytest.mark.parametrize("value", [0, -1, float('nan'), True, 'bad'])
def test_invalid_timeout(value):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=1, num_workers=1, base_seed=1, timeout_s=value)


@pytest.mark.parametrize("indices", [(1,), (-1, 0), (True, 0), (0.1, 0), 'abc'])
def test_invalid_episode_indices(indices):
    with pytest.raises((ValueError, TypeError)):
        ParallelGridGames(num_envs=2, num_workers=1, base_seed=10, episode_indices=indices)


def test_invalid_context_distribution():
    with pytest.raises(TypeError):
        ParallelGridGames(num_envs=1, num_workers=1, base_seed=1, context_distribution=None)


@pytest.mark.parametrize("ctx", get_all_contexts(), ids=lambda c: f"m{c.map_id}-o{c.opponent_id}")
def test_local_snapshots_all_24_contexts_respect_legal_actions(ctx):
    game = GridGame(seed=12)
    game.reset(seed=1234, context=ctx)
    payload = _local_snapshot([game])
    assert payload['spatial'].shape == (1, *SPATIAL_SHAPE)
    assert payload['features'].shape == (1, FEATURE_SIZE)
    assert payload['action_masks'].shape == (1, NUM_ACTIONS)
    assert payload['map_ids'].tolist() == [ctx.map_id]
    assert payload['opponent_ids'].tolist() == [ctx.opponent_id]
    assert payload['activations'].tolist() == [1]
    np.testing.assert_array_equal(payload['action_masks'][0], legal_action_mask(game))
    assert payload['is_agent_decision'][0] == (not payload['action_masks'][0, DUMMY_ACTION_ID])


def test_start_actions_legality_and_close_idempotent():
    env = ParallelGridGames(num_envs=3, num_workers=2, base_seed=2026)
    with pytest.raises(RuntimeError, match='Avviare'):
        _ = env.observation
    with env:
        first = env.observation
        _check_snapshot(first, 3)
        assert np.all(first.activations == 1)
        assert len(env.worker_pids) == 2
        assert len(set(env.worker_pids)) == 2
        with pytest.raises(RuntimeError, match='riavviabile'):
            env.start()
        with pytest.raises(ValueError, match='shape'):
            env.step(np.ones((3, 1), dtype=np.int64))
        with pytest.raises(ValueError):
            env.step([int(Action.PASS)])
        with pytest.raises(ValueError):
            env.step([True, False, True])
        with pytest.raises(ValueError):
            env.step([7, 7, 7])
        with pytest.raises(ValueError, match='illegale'):
            illegal = [next(a for a in range(NUM_ACTIONS) if not mask[a]) for mask in first.action_masks]
            env.step(illegal)
        assert env.observation is first
        assert env.episode_indices == (0, 0, 0)
        assert env.checkpoint_episode_indices() == (0, 0, 0)
        # Sono ammesse anche azioni np.int64 in sequenze Python.
        result = env.step([np.int64(x) for x in _choose(first)])
        assert isinstance(result, ParallelStep)
        assert isinstance(result.transition_next, ParallelTransitionNext)
        assert not result.dones.any()
        assert np.all(result.next_observation.activations == 2)
        np.testing.assert_array_equal(result.transition_next.action_masks,
                                      result.next_observation.action_masks)
        with pytest.raises(RuntimeError, match='confine episodio'):
            env.checkpoint_episode_indices()
        _check_snapshot(result.next_observation, 3)
    env.close()
    with pytest.raises(RuntimeError):
        env.step([5, 5, 5])
    with pytest.raises(RuntimeError):
        env.start()


def test_episode_100_steps_terminal_reset_and_replay_integration():
    n = 4
    buffer = ReplayBuffer(capacity=HORIZON*n, seed=808)
    forced_count = 0
    agent_count = 0
    with ParallelGridGames(num_envs=n, num_workers=2, base_seed=808, shaping_lambda=0.1) as env:
        totals = np.zeros(n, dtype=np.float64)
        initial_seeds = env.observation.episode_seeds
        map_ids = env.observation.map_ids.copy()
        opp_ids = env.observation.opponent_ids.copy()
        for activation in range(1, HORIZON+1):
            before = env.observation
            assert np.all(before.activations == activation)
            np.testing.assert_array_equal(before.map_ids, map_ids)
            np.testing.assert_array_equal(before.opponent_ids, opp_ids)
            assert before.episode_seeds == initial_seeds
            actions = _choose(before)
            forced_count += int(np.sum(actions == DUMMY_ACTION_ID))
            agent_count += int(np.sum(actions == int(Action.PASS)))
            after = env.step(actions)
            totals += after.rewards
            assert np.array_equal(after.dones, np.full(n, activation == HORIZON))
            assert np.all(after.terminal_utilities == -1) if activation < HORIZON else np.all(np.isin(after.terminal_utilities, (0, 1)))
            for env_id in range(n):
                terminal = bool(after.dones[env_id])
                if terminal:
                    assert not after.transition_next.action_masks[env_id].any()
                    assert not after.transition_next.spatial[env_id].any()
                    assert not after.transition_next.features[env_id].any()
                    next_observation = None
                    next_mask = None
                else:
                    next_observation = Observation(
                        after.transition_next.spatial[env_id],
                        after.transition_next.features[env_id],
                    )
                    next_mask = after.transition_next.action_masks[env_id]
                buffer.add(
                    observation=Observation(before.spatial[env_id], before.features[env_id]),
                    action_mask=before.action_masks[env_id], action=int(actions[env_id]),
                    reward=float(after.rewards[env_id]), next_observation=next_observation,
                    next_action_mask=next_mask, done=terminal,
                )
        assert len(buffer) == HORIZON*n
        assert forced_count > 0 and agent_count > 0
        np.testing.assert_allclose(totals, after.terminal_utilities.astype(float), atol=1e-5)
        assert env.episode_indices == (1,)*n
        assert np.all(after.next_observation.activations == 1)
        assert after.next_observation.episode_seeds != initial_seeds
        assert env.checkpoint_episode_indices() == (1,)*n
        sample = buffer.sample(HORIZON*n)
        assert sample.actions.shape == (HORIZON*n,)
        assert np.count_nonzero(sample.dones) == n
        assert np.any(sample.actions == DUMMY_ACTION_ID)
        assert np.any(sample.actions == int(Action.PASS))
        assert not sample.next_action_masks[sample.dones].any()
        assert not sample.next_spatial[sample.dones].any()
        assert not sample.next_features[sample.dones].any()
        assert np.any(sample.next_action_masks[~sample.dones])


def _trajectory(num_workers: int):
    """Replay di 101 step con azioni identiche, anche oltre reset."""
    trace = []
    with ParallelGridGames(num_envs=3, num_workers=num_workers, base_seed=123456) as env:
        for _ in range(HORIZON+1):
            before = env.observation
            actions = _choose(before)
            result = env.step(actions)
            trace.append((
                before.spatial.copy(), before.features.copy(), before.action_masks.copy(),
                before.map_ids.copy(), before.opponent_ids.copy(),
                before.activations.copy(), before.episode_seeds,
                result.rewards.copy(), result.dones.copy(), result.terminal_utilities.copy(),
                result.next_observation.episode_seeds,
            ))
    return trace


def test_deterministic_independent_of_worker_count_including_reset():
    sequential = _trajectory(1)
    parallel = _trajectory(3)
    assert len(sequential) == len(parallel) == HORIZON+1
    for a, b in zip(sequential, parallel):
        for v, w in zip(a, b):
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, w)
            else:
                assert v == w


def test_custom_context_distribution_and_shaping_zero():
    ctx = ContextDistribution(
        q_maps=(0.0, 0.0, 1.0, 0.0),
        q_opponents=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    with ParallelGridGames(num_envs=2, num_workers=1, base_seed=41,
                           context_distribution=ctx, shaping_lambda=0.0) as env:
        for _ in range(4):
            obs = env.observation
            assert np.all(obs.map_ids == 3)
            assert np.all(obs.opponent_ids == 6)
            result = env.step(_choose(obs))
            assert np.array_equal(result.rewards, np.zeros(2, dtype=np.float32))


def test_resume_seeds_at_episode_boundaries():
    with ParallelGridGames(num_envs=2, num_workers=1, base_seed=9,
                           episode_indices=(2, 7)) as env:
        previous = env.observation
        start_seeds = previous.episode_seeds
        assert env.checkpoint_episode_indices() == (2, 7)
        env.step(_choose(previous))
    with ParallelGridGames(num_envs=2, num_workers=2, base_seed=9,
                           episode_indices=(2, 7)) as other:
        assert other.observation.episode_seeds == start_seeds
    with ParallelGridGames(num_envs=2, num_workers=1, base_seed=9,
                           episode_indices=(2, 8)) as other:
        assert other.observation.episode_seeds[0] == start_seeds[0]
        assert other.observation.episode_seeds[1] != start_seeds[1]


def test_close_without_start_and_double_close():
    env = ParallelGridGames(num_envs=1, num_workers=1, base_seed=1)
    env.close()
    env.close()
    with pytest.raises(RuntimeError):
        env.start()


def test_spawn_context_selected():
    assert mp.get_context('spawn').get_start_method() == 'spawn'

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_centralized_batched_network_inference_cpu_or_cuda(device):
    """La Q-network gira nel main, i worker ricevono SOLO gli ID delle azioni."""
    import torch
    from agent.network import QNetwork
    from agent.action_selection import select_actions
    from environment.observations import ObservationBatch, to_torch_batch
    from environment.action_mask import to_torch_action_masks

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("Test CUDA da eseguire sul PC con RTX 5060 Ti")
    model = QNetwork().to(device).eval()
    pin_memory = device.startswith("cuda")
    with ParallelGridGames(num_envs=4, num_workers=2, base_seed=111) as env:
        obs = env.observation
        batch = to_torch_batch(
            ObservationBatch(obs.spatial, obs.features),
            device=device, pin_memory=pin_memory, non_blocking=pin_memory,
        )
        masks = to_torch_action_masks(
            obs.action_masks, device=device, pin_memory=pin_memory,
            non_blocking=pin_memory,
        )
        actions = select_actions(model, batch.spatial, batch.features, masks, epsilon=0.25)
        assert actions.shape == (4,) and actions.dtype == torch.long
        assert actions.device.type == torch.device(device).type
        result = env.step(actions.cpu().numpy())
        assert result.rewards.shape == (4,)
        assert np.all(result.next_observation.activations == 2)
