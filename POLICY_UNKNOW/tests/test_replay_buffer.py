"""Test del replay buffer uniforme per il Masked DQN standard.

Dalla root Giulio2:
    python -m pytest tests/test_replay_buffer.py -v

I test CPU sono indipendenti dalla disponibilita' di una GPU NVIDIA.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from agent.replay_buffer import (
    REPLAY_STATE_VERSION,
    ReplayBatch,
    ReplayBuffer,
    TorchReplayBatch,
    to_torch_replay_batch,
)
from environment.action_mask import (
    DUMMY_ACTION_ID,
    NUM_ACTIONS,
    legal_action_mask,
    legal_actions_to_mask,
)
from environment.context import get_all_contexts, get_context
from environment.game import GameState, GridGame, HORIZON, TERMINAL_STATE
from environment.observations import (
    FEATURE_SIZE,
    SPATIAL_SHAPE,
    Observation,
    encode_observation,
)
from environment.rules import Action, Player


PASS_MASK = legal_actions_to_mask((Action.PASS,))
DUMMY_MASK = legal_actions_to_mask((Action.DUMMY,))
NORTH_PASS_MASK = legal_actions_to_mask((Action.NORTH, Action.PASS))


def _observation(marker: int = 0) -> Observation:
    """Osservazione sintetica con interfacce e dtype esatti della Fase 2."""
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    spatial[0, 0, 0] = 1.0
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features[0] = marker
    return Observation(spatial=spatial, features=features)


def _add(
    buffer: ReplayBuffer, marker: int = 0, *,
    done: bool = False, mask: np.ndarray = PASS_MASK, action: Action = Action.PASS,
) -> None:
    buffer.add(
        observation=_observation(marker),
        action_mask=mask.copy(),
        action=action,
        reward=float(marker) / 2.0,
        next_observation=None if done else _observation(marker + 1),
        next_action_mask=None if done else DUMMY_MASK.copy(),
        done=done,
    )


def _assert_shapes_and_dtypes(batch: ReplayBatch, size: int) -> None:
    assert batch.batch_size == size
    for field in ("spatial", "next_spatial"):
        array = getattr(batch, field)
        assert array.shape == (size, *SPATIAL_SHAPE)
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
    for field in ("features", "next_features"):
        array = getattr(batch, field)
        assert array.shape == (size, FEATURE_SIZE)
        assert array.dtype == np.float32
    for field in ("action_masks", "next_action_masks"):
        array = getattr(batch, field)
        assert array.shape == (size, NUM_ACTIONS)
        assert array.dtype == np.bool_
    for field, dtype in (("actions", np.int64), ("rewards", np.float32),
                         ("dones", np.bool_)):
        array = getattr(batch, field)
        assert array.shape == (size,)
        assert array.dtype == dtype


def test_initialization_and_memory():
    buffer = ReplayBuffer(capacity=7, seed=101)
    assert len(buffer) == 0
    assert buffer.position == 0
    assert buffer.total_added == 0
    assert buffer.capacity == 7
    bytes_each = (
        2 * np.prod(SPATIAL_SHAPE) * np.dtype(np.float32).itemsize
        + 2 * FEATURE_SIZE * np.dtype(np.float32).itemsize
        + 2 * NUM_ACTIONS * np.dtype(np.bool_).itemsize
        + np.dtype(np.int64).itemsize
        + np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
    )
    assert buffer.nbytes == 7 * bytes_each


@pytest.mark.parametrize("invalid,exception", [
    (0, ValueError), (-4, ValueError), (1.5, TypeError),
    (True, TypeError), (None, TypeError),
])
def test_invalid_capacity(invalid, exception):
    with pytest.raises(exception):
        ReplayBuffer(invalid)


def test_seed_must_be_integer():
    with pytest.raises(TypeError, match="seed"):
        ReplayBuffer(10, seed=12.2)
    with pytest.raises(TypeError, match="seed"):
        ReplayBuffer(10, seed=True)


def test_insert_sample_shapes_and_values():
    buffer = ReplayBuffer(10, seed=43)
    _add(buffer, 6)
    assert len(buffer) == 1
    batch = buffer.sample(1)
    _assert_shapes_and_dtypes(batch, 1)
    assert batch.actions.item() == int(Action.PASS)
    assert batch.rewards.item() == pytest.approx(3.0)
    assert batch.features[0, 0] == 6.0
    assert batch.next_features[0, 0] == 7.0
    np.testing.assert_array_equal(batch.action_masks[0], PASS_MASK)
    np.testing.assert_array_equal(batch.next_action_masks[0], DUMMY_MASK)
    assert not batch.dones.item()


def test_circular_overwrite_and_total_added():
    buffer = ReplayBuffer(3, seed=13)
    for i in range(8):
        _add(buffer, i)
    assert len(buffer) == 3
    assert buffer.total_added == 8
    assert buffer.position == 2
    values = set(buffer.sample(3).features[:, 0].tolist())
    assert values == {5.0, 6.0, 7.0}


def test_one_slot_buffer_and_clear_without_rng_rewind():
    buffer = ReplayBuffer(1, seed=3)
    _add(buffer, 1)
    _add(buffer, 2, done=True, mask=DUMMY_MASK, action=Action.DUMMY)
    assert len(buffer) == 1
    assert buffer.position == 0
    batch = buffer.sample(1)
    assert batch.features[0, 0] == 2.0
    assert batch.dones[0]
    buffer.clear()
    assert len(buffer) == 0 and buffer.position == 0 and buffer.total_added == 0
    with pytest.raises(ValueError):
        buffer.sample(1)
    _add(buffer, 99)
    assert buffer.sample(1).features[0, 0] == 99.0


def test_uniform_without_replacement():
    buffer = ReplayBuffer(9, seed=4)
    for i in range(9):
        _add(buffer, i)
    for _ in range(20):
        batch = buffer.sample(9)
        assert set(batch.features[:, 0].tolist()) == set(range(9))


def test_sample_reproducible_independent_of_global_numpy_rng():
    a = ReplayBuffer(12, seed=99)
    b = ReplayBuffer(12, seed=99)
    for marker in range(12):
        _add(a, marker)
        _add(b, marker)
    for i in range(10):
        np.random.seed(i + 1)
        np.random.random(27)
        np.testing.assert_array_equal(a.sample(6).features, b.sample(6).features)


def test_sample_uniformity_approximate_with_fixed_seed():
    buffer = ReplayBuffer(5, seed=2026)
    for i in range(5):
        _add(buffer, i)
    counts = np.zeros(5, dtype=int)
    for _ in range(2_000):
        counts[int(buffer.sample(1).features[0, 0])] += 1
    # Verifica debole anti-bias (seed fisso): 20% atteso per slot.
    assert np.all((counts > 300) & (counts < 500)), counts


@pytest.mark.parametrize("batch_size,exception", [
    (0, ValueError), (-1, ValueError), (3, ValueError),
    (True, TypeError), (1.0, TypeError),
])
def test_sample_invalid_size(batch_size, exception):
    buffer = ReplayBuffer(5, seed=1)
    _add(buffer, 1)
    _add(buffer, 2)
    with pytest.raises(exception):
        buffer.sample(batch_size)


def test_terminal_next_fields_are_absent_by_convention():
    buffer = ReplayBuffer(2, seed=3)
    _add(buffer, 0, done=True, mask=DUMMY_MASK, action=Action.DUMMY)
    batch = buffer.sample(1)
    assert batch.dones.item()
    assert batch.actions.item() == DUMMY_ACTION_ID
    assert not batch.next_action_masks.any()
    assert not batch.next_spatial.any()
    assert not batch.next_features.any()
    # DUMMY e' memorizzata come transizione completa con reward, non filtrata.
    assert batch.action_masks[0, DUMMY_ACTION_ID]


def test_snapshot_input_does_not_alias_storage():
    buffer = ReplayBuffer(2, seed=15)
    obs = _observation(2)
    nxt = _observation(3)
    curr_mask = PASS_MASK.copy()
    next_mask = DUMMY_MASK.copy()
    buffer.add(
        observation=obs, action_mask=curr_mask, action=Action.PASS, reward=2.,
        next_observation=nxt, next_action_mask=next_mask, done=False,
    )
    obs.spatial.fill(7)
    obs.features.fill(7)
    nxt.spatial.fill(7)
    nxt.features.fill(7)
    curr_mask[:] = False
    next_mask[:] = False
    batch = buffer.sample(1)
    assert batch.spatial[0, 0, 0, 0] == 1.
    assert batch.features[0, 0] == 2.
    assert batch.next_features[0, 0] == 3.
    np.testing.assert_array_equal(batch.action_masks[0], PASS_MASK)
    np.testing.assert_array_equal(batch.next_action_masks[0], DUMMY_MASK)


def test_sampled_arrays_dont_alias_storage_or_other_samples():
    buffer = ReplayBuffer(2, seed=5)
    _add(buffer, 9)
    first = buffer.sample(1)
    first.spatial[:] = 123.
    first.features[:] = 123.
    first.action_masks[:] = False
    second = buffer.sample(1)
    assert second.features[0, 0] == 9.0
    assert second.spatial[0, 0, 0, 0] == 1.0
    assert second.action_masks[0, int(Action.PASS)]


@pytest.mark.parametrize("action", [Action.NORTH, 0, np.int64(0)])
def test_illegal_action_is_rejected_before_write(action):
    buffer = ReplayBuffer(4, seed=1)
    with pytest.raises(ValueError, match="non e' legale"):
        buffer.add(
            observation=_observation(), action_mask=PASS_MASK, action=action,
            reward=0., next_observation=_observation(1),
            next_action_mask=PASS_MASK, done=False,
        )
    assert len(buffer) == 0 and buffer.position == 0


@pytest.mark.parametrize("invalid,exception", [
    (7, ValueError), (-1, ValueError), (False, TypeError),
    (1.2, TypeError), ("5", TypeError),
])
def test_invalid_action_type_and_range(invalid, exception):
    buffer = ReplayBuffer(2)
    with pytest.raises(exception):
        buffer.add(
            observation=_observation(), action_mask=PASS_MASK, action=invalid,
            reward=1., next_observation=_observation(),
            next_action_mask=PASS_MASK, done=False,
        )
    assert len(buffer) == 0


@pytest.mark.parametrize("invalid,exception", [
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError), ("1.2", TypeError), (True, TypeError),
    (1e100, ValueError),
])
def test_invalid_reward(invalid, exception):
    buffer = ReplayBuffer(2)
    with pytest.raises(exception):
        buffer.add(
            observation=_observation(), action_mask=PASS_MASK,
            action=Action.PASS, reward=invalid,
            next_observation=_observation(), next_action_mask=PASS_MASK,
            done=False,
        )
    assert len(buffer) == 0


@pytest.mark.parametrize("invalid", [0, 1, None, "True"])
def test_invalid_done_type(invalid):
    buffer = ReplayBuffer(2)
    with pytest.raises(TypeError, match="done"):
        buffer.add(
            observation=_observation(), action_mask=PASS_MASK,
            action=Action.PASS, reward=1.,
            next_observation=_observation(), next_action_mask=PASS_MASK,
            done=invalid,
        )
    assert len(buffer) == 0


def test_nonterminal_requires_next_observation_and_mask():
    buffer = ReplayBuffer(2)
    for next_observation, next_action_mask in ((None, PASS_MASK), (_observation(), None)):
        with pytest.raises(ValueError, match="obbligatori"):
            buffer.add(
                observation=_observation(), action_mask=PASS_MASK,
                action=Action.PASS, reward=0.,
                next_observation=next_observation,
                next_action_mask=next_action_mask, done=False,
            )
    assert len(buffer) == 0


def test_terminal_rejects_next_observation_and_mask():
    buffer = ReplayBuffer(2)
    for next_observation, next_action_mask in ((_observation(), None), (None, PASS_MASK)):
        with pytest.raises(ValueError, match="devono essere None"):
            buffer.add(
                observation=_observation(), action_mask=PASS_MASK,
                action=Action.PASS, reward=0.,
                next_observation=next_observation,
                next_action_mask=next_action_mask, done=True,
            )
    assert len(buffer) == 0


@pytest.mark.parametrize("corrupt", [
    np.zeros(NUM_ACTIONS, dtype=np.bool_),
    np.ones(NUM_ACTIONS, dtype=np.bool_),
    np.array([1, 0, 0, 0, 0, 1, 0], dtype=np.int8),
    np.zeros(NUM_ACTIONS + 1, dtype=np.bool_),
])
def test_invalid_current_masks_rejected(corrupt):
    buffer = ReplayBuffer(2)
    with pytest.raises((ValueError, TypeError)):
        buffer.add(
            observation=_observation(), action_mask=corrupt,
            action=Action.PASS, reward=1., next_observation=_observation(),
            next_action_mask=PASS_MASK, done=False,
        )
    assert len(buffer) == 0


@pytest.mark.parametrize("corrupt", [
    np.zeros(NUM_ACTIONS, dtype=np.bool_),
    np.ones(NUM_ACTIONS, dtype=np.bool_),
    np.zeros(NUM_ACTIONS + 1, dtype=np.bool_),
])
def test_invalid_next_masks_rejected(corrupt):
    buffer = ReplayBuffer(2)
    with pytest.raises(ValueError):
        buffer.add(
            observation=_observation(), action_mask=PASS_MASK,
            action=Action.PASS, reward=1., next_observation=_observation(),
            next_action_mask=corrupt, done=False,
        )
    assert len(buffer) == 0


@pytest.mark.parametrize("corrupt", [
    Observation(np.zeros((4, 8, 9), dtype=np.float32),
                np.zeros(FEATURE_SIZE, dtype=np.float32)),
    Observation(np.zeros(SPATIAL_SHAPE, dtype=np.float64),
                np.zeros(FEATURE_SIZE, dtype=np.float32)),
    Observation(np.zeros(SPATIAL_SHAPE, dtype=np.float32),
                np.zeros(FEATURE_SIZE - 1, dtype=np.float32)),
    Observation(np.zeros(SPATIAL_SHAPE, dtype=np.float32),
                np.full(FEATURE_SIZE, np.nan, dtype=np.float32)),
])
def test_invalid_observations_rejected(corrupt):
    buffer = ReplayBuffer(3)
    for name in ("observation", "next_observation"):
        kwargs = {
            "observation": _observation(), "action_mask": PASS_MASK,
            "action": Action.PASS, "reward": 1.,
            "next_observation": _observation(), "next_action_mask": PASS_MASK,
            "done": False,
        }
        kwargs[name] = corrupt
        with pytest.raises((TypeError, ValueError)):
            buffer.add(**kwargs)
        assert len(buffer) == 0


def test_failed_insert_does_not_overwrite_oldest_transition():
    buffer = ReplayBuffer(2, seed=1)
    _add(buffer, 7)
    _add(buffer, 8)
    old_position = buffer.position
    with pytest.raises(ValueError):
        buffer.add(
            observation=_observation(999), action_mask=PASS_MASK,
            action=Action.PASS, reward=0.,
            next_observation=None, next_action_mask=None, done=False,
        )
    assert buffer.position == old_position
    assert buffer.total_added == 2
    assert set(buffer.sample(2).features[:, 0]) == {7, 8}


def test_state_dict_roundtrip_restores_rng_and_ring():
    original = ReplayBuffer(5, seed=2026)
    for i in range(12):
        _add(original, i, done=(i % 3 == 0))
    original.sample(2)  # Avanza il RNG prima del checkpoint.
    checkpoint = original.state_dict()
    assert checkpoint["version"] == REPLAY_STATE_VERSION
    assert checkpoint["size"] == 5
    restored = ReplayBuffer(5, seed=999)
    restored.load_state_dict(checkpoint)
    assert restored.position == original.position
    assert restored.total_added == original.total_added
    assert len(restored) == len(original)
    for _ in range(12):
        x = original.sample(3)
        y = restored.sample(3)
        for field in ReplayBatch.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
    _add(restored, 100)
    _add(original, 100)
    np.testing.assert_array_equal(restored.sample(5).features,
                                  original.sample(5).features)


def test_state_dict_is_independent_snapshot():
    buffer = ReplayBuffer(3, seed=10)
    _add(buffer, 3)
    snapshot = buffer.state_dict()
    _add(buffer, 4)
    assert snapshot["size"] == 1
    assert snapshot["features"].shape == (1, FEATURE_SIZE)
    assert snapshot["features"][0, 0] == 3
    snapshot["features"][0, 0] = -7
    assert 3 in buffer.sample(2).features[:, 0]


@pytest.mark.parametrize("field,corrupt", [
    ("version", 999), ("capacity", 9), ("size", 3),
    ("position", 88), ("total_added", -1),
])
def test_invalid_checkpoint_headers_do_not_change_buffer(field, corrupt):
    buffer = ReplayBuffer(4, seed=10)
    _add(buffer, 1)
    checkpoint = buffer.state_dict()
    bad = deepcopy(checkpoint)
    bad[field] = corrupt
    with pytest.raises((TypeError, ValueError)):
        buffer.load_state_dict(bad)
    assert len(buffer) == 1
    assert buffer.sample(1).features[0, 0] == 1


def test_invalid_checkpoint_mask_does_not_change_buffer():
    buffer = ReplayBuffer(4, seed=10)
    _add(buffer, 1)
    checkpoint = buffer.state_dict()
    checkpoint["action_masks"][0] = False
    with pytest.raises(ValueError):
        buffer.load_state_dict(checkpoint)
    assert len(buffer) == 1
    assert buffer.sample(1).features[0, 0] == 1


def test_checkpoint_empty_buffer_restoration():
    empty = ReplayBuffer(2, seed=20)
    snapshot = empty.state_dict()
    assert snapshot["features"].shape == (0, FEATURE_SIZE)
    buffer = ReplayBuffer(2, seed=30)
    _add(buffer, 1)
    buffer.load_state_dict(snapshot)
    assert len(buffer) == 0 and buffer.total_added == 0
    _add(buffer, 5)
    assert buffer.sample(1).features[0, 0] == 5


def test_to_torch_cpu_shapes_types_and_values():
    buffer = ReplayBuffer(4, seed=0)
    _add(buffer, 2, done=False)
    _add(buffer, 3, done=True, mask=DUMMY_MASK, action=Action.DUMMY)
    batch = buffer.sample(2)
    tensors = to_torch_replay_batch(batch, device="cpu")
    assert isinstance(tensors, TorchReplayBatch)
    assert tensors.batch_size == 2
    assert tensors.spatial.shape == (2, *SPATIAL_SHAPE)
    assert tensors.features.shape == (2, FEATURE_SIZE)
    assert tensors.next_action_masks.shape == (2, NUM_ACTIONS)
    assert tensors.actions.dtype == torch.long
    assert tensors.rewards.dtype == torch.float32
    assert tensors.dones.dtype == torch.bool
    assert tensors.action_masks.dtype == torch.bool
    assert tensors.spatial.dtype == torch.float32
    for name in ReplayBatch.__dataclass_fields__:
        actual = getattr(tensors, name)
        assert actual.device.type == "cpu"
        np.testing.assert_array_equal(actual.numpy(), getattr(batch, name))


def test_to_torch_invalid_inputs():
    with pytest.raises(TypeError):
        to_torch_replay_batch(None)
    if not torch.cuda.is_available():
        buffer = ReplayBuffer(1)
        _add(buffer)
        with pytest.raises(RuntimeError, match="CUDA"):
            to_torch_replay_batch(buffer.sample(1), pin_memory=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
def test_to_torch_cuda_pinned_nonblocking():
    buffer = ReplayBuffer(4, seed=8)
    for i in range(4):
        _add(buffer, i, done=(i == 3))
    batch = buffer.sample(4)
    cuda_batch = to_torch_replay_batch(
        batch, device="cuda:0", pin_memory=True, non_blocking=True,
    )
    assert cuda_batch.spatial.is_cuda
    assert cuda_batch.action_masks.is_cuda
    assert cuda_batch.actions.is_cuda
    torch.cuda.synchronize()
    np.testing.assert_array_equal(cuda_batch.features.cpu().numpy(), batch.features)
    np.testing.assert_array_equal(cuda_batch.dones.cpu().numpy(), batch.dones)
    assert cuda_batch.next_action_masks[cuda_batch.dones].sum().item() == 0


@pytest.mark.parametrize("context", get_all_contexts(),
                         ids=lambda c: f"m{c.map_id}-k{c.opponent_id}")
def test_real_game_first_activations_obey_legal_actions(context):
    """Le transizioni reali di tutti i 24 contesti sono memorizzabili."""
    game = GridGame(seed=17)
    game.reset(seed=17, context=context, starting_player=Player.AGENT)
    buffer = ReplayBuffer(10, seed=17)
    for _ in range(6):
        state = game.state
        assert isinstance(state, GameState)
        obs = encode_observation(state, game.context)
        mask = legal_action_mask(game)
        action = Action.DUMMY if mask[DUMMY_ACTION_ID] else Action.PASS
        assert action in game.legal_actions()
        result = game.step(action)
        assert not result.terminated
        buffer.add(
            observation=obs, action_mask=mask, action=action,
            reward=result.reward,
            next_observation=encode_observation(result.next_state, game.context),
            next_action_mask=legal_action_mask(game), done=False,
        )
    batch = buffer.sample(6)
    assert len(buffer) == 6
    assert np.all(batch.action_masks[np.arange(6), batch.actions])
    assert np.all(batch.next_action_masks.any(axis=1))


def test_complete_episode_100_activations_with_dummy_and_terminal():
    """Regressione dell'orizzonte: 100 attivazioni, DUMMY e unico terminale."""
    game = GridGame(seed=123)
    game.reset(seed=123, context=get_context(2, 6),
               starting_player=Player.AGENT)
    buffer = ReplayBuffer(HORIZON, seed=123)
    for activation in range(1, HORIZON + 1):
        state = game.state
        assert isinstance(state, GameState)
        assert state.activation == activation
        mask = legal_action_mask(game)
        action = Action.DUMMY if mask[DUMMY_ACTION_ID] else Action.PASS
        result = game.step(action)
        assert result.action == action
        assert result.terminated == (activation == HORIZON)
        next_obs = None if result.terminated else encode_observation(result.next_state, game.context)
        next_mask = None if result.terminated else legal_action_mask(game)
        buffer.add(
            observation=encode_observation(state, game.context),
            action_mask=mask, action=action, reward=result.reward,
            next_observation=next_obs, next_action_mask=next_mask,
            done=result.terminated,
        )
    assert game.state is TERMINAL_STATE
    assert len(buffer) == HORIZON
    batch = buffer.sample(HORIZON)
    assert batch.dones.sum() == 1
    assert np.count_nonzero(batch.actions == DUMMY_ACTION_ID) > 0
    assert np.all(batch.action_masks[np.arange(HORIZON), batch.actions])
    assert not batch.next_action_masks[batch.dones].any()
    assert batch.next_action_masks[~batch.dones].any(axis=1).all()
    assert np.array_equal(
        np.sort(batch.features[:, 1]),
        np.sort(np.array([(i - 1) / 99 for i in range(1, HORIZON + 1)],
                         dtype=np.float32)),
    )
    # Il primo stato del nuovo episodio non viene mai memorizzato come s'
    # della transizione terminale: esso non e' richiesto al buffer.

