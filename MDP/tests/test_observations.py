"""Test di regressione della codifica neurale in environment/observations.py.

Da eseguire dalla cartella Giulio:
    python -m pytest tests/test_observations.py -v

I test PyTorch/CUDA sono facoltativi: quelli CPU vengono saltati quando
PyTorch non e' installato; quelli CUDA vengono saltati senza GPU disponibile.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from environment.context import MAP_IDS, OPPONENT_IDS, get_all_contexts, get_context
from environment.game import HORIZON, TERMINAL_STATE, GameState, GridGame
from environment.maps import START_CELLS
from environment.observations import (
    FEATURE_SIZE,
    MAP_FEATURE_OFFSET,
    MAX_ABS_SCORE_DIFFERENTIAL,
    OPPONENT_FEATURE_OFFSET,
    SPATIAL_SHAPE,
    Observation,
    ObservationBatch,
    SpatialChannel,
    TorchObservationBatch,
    encode_observation,
    stack_observations,
    to_torch_batch,
)
from environment.rules import Action, GRID_SIZE, Player


CONTEXTS = get_all_contexts()


def _state(
    context,
    *,
    agent_position=(5, 1),
    opponent_position=(5, 9),
    objectives=None,
    agent_vp=0,
    opponent_vp=0,
    player_to_act=Player.AGENT,
    activation=1,
) -> GameState:
    """Costruisce uno stato sintetico; i candidati sono celle ammissibili."""
    if objectives is None:
        candidates = sorted(
            context.game_map.free_cells
            - START_CELLS
            - {agent_position, opponent_position}
        )
        objectives = candidates[:3]
    return GameState(
        agent_position=agent_position,
        opponent_position=opponent_position,
        active_objectives=frozenset(objectives),
        agent_vp=agent_vp,
        opponent_vp=opponent_vp,
        player_to_act=player_to_act,
        activation=activation,
    )


def _expected_plane(positions) -> np.ndarray:
    """Maschera geometrica indipendente dall'implementazione del modulo."""
    plane = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for x, y in positions:
        plane[GRID_SIZE - y, x - 1] = 1.0
    return plane


@pytest.fixture
def sample_observations():
    """Tre osservazioni distinte, ciascuna validata dal simulatore."""
    context = get_context(MAP_IDS[0], OPPONENT_IDS[0])
    game = GridGame(seed=2026)
    game.reset(seed=2026, context=context, starting_player=Player.AGENT)
    result = []
    for _ in range(3):
        result.append(encode_observation(game.state, game.context))
        game.step(game.legal_actions()[0])
    return result


# ---------------------------------------------------------------------------
# 1. Contratto dell'interfaccia, dimensioni e convenzioni della codifica.
# ---------------------------------------------------------------------------


def test_dimensions_and_channel_identifiers():
    assert HORIZON == 100
    assert GRID_SIZE == 9
    assert SPATIAL_SHAPE == (4, 9, 9)
    assert FEATURE_SIZE == 13
    assert MAX_ABS_SCORE_DIFFERENTIAL == 800
    assert MAP_FEATURE_OFFSET == 3
    assert OPPONENT_FEATURE_OFFSET == 7
    assert len(MAP_IDS) == 4
    assert len(OPPONENT_IDS) == 6
    assert tuple(int(channel) for channel in SpatialChannel) == (0, 1, 2, 3)
    assert SpatialChannel.OBSTACLES == 0
    assert SpatialChannel.AGENT == 1
    assert SpatialChannel.OPPONENT == 2
    assert SpatialChannel.OBJECTIVES == 3


@pytest.mark.parametrize("context", CONTEXTS, ids=lambda c: f"m{c.map_id}-k{c.opponent_id}")
@pytest.mark.parametrize("starting_player", (Player.AGENT, Player.OPPONENT))
def test_initial_observation_in_every_context(context, starting_player):
    game = GridGame(seed=125)
    state = game.reset(seed=125, context=context, starting_player=starting_player)
    observation = encode_observation(state, context)

    assert isinstance(observation, Observation)
    assert observation.spatial.shape == SPATIAL_SHAPE
    assert observation.features.shape == (FEATURE_SIZE,)
    assert observation.spatial.dtype == np.float32
    assert observation.features.dtype == np.float32
    assert observation.spatial.flags.c_contiguous
    assert observation.features.flags.c_contiguous
    assert np.isfinite(observation.spatial).all()
    assert np.isfinite(observation.features).all()

    expected = (
        context.game_map.obstacles,
        {state.agent_position},
        {state.opponent_position},
        state.active_objectives,
    )
    for channel, positions in enumerate(expected):
        np.testing.assert_array_equal(
            observation.spatial[channel], _expected_plane(positions)
        )

    assert np.isin(observation.spatial, (0.0, 1.0)).all()
    assert observation.features[0] == 0.0
    assert observation.features[1] == 0.0
    assert observation.features[2] == float(starting_player is Player.AGENT)

    expected_maps = np.eye(len(MAP_IDS), dtype=np.float32)[MAP_IDS.index(context.map_id)]
    expected_opponents = np.eye(len(OPPONENT_IDS), dtype=np.float32)[
        OPPONENT_IDS.index(context.opponent_id)
    ]
    np.testing.assert_array_equal(
        observation.features[MAP_FEATURE_OFFSET:OPPONENT_FEATURE_OFFSET],
        expected_maps,
    )
    np.testing.assert_array_equal(
        observation.features[OPPONENT_FEATURE_OFFSET:], expected_opponents
    )


def test_spatial_orientation_north_south_and_east_west():
    context = get_context(1, 1)
    observation = encode_observation(_state(context), context)

    # L'Agent parte a sud in (5,1), l'Opponent a nord in (5,9).
    assert observation.spatial[SpatialChannel.AGENT, 8, 4] == 1.0
    assert observation.spatial[SpatialChannel.OPPONENT, 0, 4] == 1.0
    # Controllo di un ostacolo non centrale della mappa 1, in (7,6).
    assert observation.spatial[SpatialChannel.OBSTACLES, 3, 6] == 1.0
    assert observation.spatial[SpatialChannel.OBSTACLES, 6, 3] == 0.0


@pytest.mark.parametrize("number_of_objectives", (1, 2, 3))
def test_objective_channel_supports_all_allowed_sizes(number_of_objectives):
    context = get_context(4, 6)
    possible = sorted(context.game_map.free_cells - START_CELLS)
    state = _state(context, objectives=possible[:number_of_objectives])
    observation = encode_observation(state, context)

    np.testing.assert_array_equal(
        observation.spatial[SpatialChannel.OBJECTIVES],
        _expected_plane(state.active_objectives),
    )
    assert int(observation.spatial[SpatialChannel.OBJECTIVES].sum()) == number_of_objectives


@pytest.mark.parametrize("player,expected", [(Player.AGENT, 1.0), (Player.OPPONENT, 0.0)])
def test_player_to_act_encoding(player, expected):
    context = get_context(2, 3)
    observation = encode_observation(_state(context, player_to_act=player), context)
    assert observation.features[2] == expected


@pytest.mark.parametrize("activation", (1, 2, 50, 99, 100))
def test_time_encoding(activation):
    context = get_context(2, 3)
    observation = encode_observation(_state(context, activation=activation), context)
    assert observation.features[1] == pytest.approx((activation - 1) / 99, abs=1e-7)


@pytest.mark.parametrize("differential", (-800, -25, -1, 0, 1, 25, 800))
def test_normalized_score_differential(differential):
    context = get_context(1, 6)
    state = _state(context, agent_vp=differential, opponent_vp=0)
    observation = encode_observation(state, context)
    assert observation.features[0] == pytest.approx(differential / 800, abs=1e-7)
    assert -1.0 <= observation.features[0] <= 1.0


def test_features_depend_on_score_differential_not_absolute_points():
    context = get_context(3, 4)
    state_a = _state(context, agent_vp=20, opponent_vp=9)
    state_b = _state(context, agent_vp=100, opponent_vp=89)
    np.testing.assert_array_equal(
        encode_observation(state_a, context).features,
        encode_observation(state_b, context).features,
    )


def test_changed_positions_are_correctly_encoded():
    context = get_context(2, 6)
    state = _state(
        context,
        agent_position=(5, 2),
        opponent_position=(6, 8),
        objectives={(1, 2), (4, 5)},
        agent_vp=20,
        opponent_vp=5,
        activation=100,
        player_to_act=Player.OPPONENT,
    )
    observation = encode_observation(state, context)
    np.testing.assert_array_equal(
        observation.spatial[SpatialChannel.AGENT], _expected_plane({(5, 2)})
    )
    np.testing.assert_array_equal(
        observation.spatial[SpatialChannel.OPPONENT], _expected_plane({(6, 8)})
    )
    np.testing.assert_array_equal(
        observation.spatial[SpatialChannel.OBJECTIVES],
        _expected_plane({(1, 2), (4, 5)}),
    )
    assert observation.features[0] == pytest.approx(15 / 800)
    assert observation.features[1] == 1.0
    assert observation.features[2] == 0.0


# ---------------------------------------------------------------------------
# 2. Determinismo, isolamento dei dati e non interferenza con l'MDP.
# ---------------------------------------------------------------------------


def test_encoding_is_deterministic_and_returns_independent_arrays():
    context = get_context(1, 2)
    state = _state(context)
    first = encode_observation(state, context)
    second = encode_observation(state, context)
    np.testing.assert_array_equal(first.spatial, second.spatial)
    np.testing.assert_array_equal(first.features, second.features)
    assert not np.shares_memory(first.spatial, second.spatial)
    assert not np.shares_memory(first.features, second.features)
    first.spatial.fill(-10.0)
    first.features.fill(-10.0)
    fresh = encode_observation(state, context)
    np.testing.assert_array_equal(second.spatial, fresh.spatial)
    np.testing.assert_array_equal(second.features, fresh.features)


def test_encoding_does_not_change_game_state_or_random_streams():
    context = get_context(2, 6)
    game = GridGame(seed=1701)
    state = game.reset(context=context, starting_player=Player.AGENT)
    initial_state = game.state
    rng_names = (
        "rng_context", "rng_initiative", "rng_objectives", "rng_agent",
        "rng_opponent", "rng_combat",
    )
    rng_before = {name: getattr(game, name).getstate() for name in rng_names}
    for _ in range(5):
        encode_observation(state, game.context)
    assert game.state is initial_state
    assert state == initial_state
    assert {name: getattr(game, name).getstate() for name in rng_names} == rng_before


def test_observation_dataclass_fields_cannot_be_reassigned():
    observation = encode_observation(_state(get_context(1, 1)), get_context(1, 1))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        observation.features = np.zeros(FEATURE_SIZE, dtype=np.float32)


@pytest.mark.parametrize("context", CONTEXTS, ids=lambda c: f"m{c.map_id}-k{c.opponent_id}")
def test_observations_during_complete_episode(context):
    """Controlla 100 stati iniziali di attivazione in ogni contesto (2400 stati)."""
    game = GridGame(seed=202610, shaping_lambda=0.1)
    game.reset(seed=202610, context=context, starting_player=Player.AGENT)

    for activation in range(1, HORIZON + 1):
        state = game.state
        assert isinstance(state, GameState)
        observation = encode_observation(state, game.context)
        assert observation.spatial.shape == SPATIAL_SHAPE
        assert observation.features.shape == (FEATURE_SIZE,)
        assert observation.features[0] == pytest.approx(
            state.score_differential / MAX_ABS_SCORE_DIFFERENTIAL, abs=2e-7
        )
        assert observation.features[1] == pytest.approx((activation - 1) / 99, abs=2e-7)
        assert observation.features[2] == float(state.player_to_act is Player.AGENT)
        assert int(observation.spatial[SpatialChannel.OBJECTIVES].sum()) == len(
            state.active_objectives
        )
        legal = game.legal_actions()
        action = Action.DUMMY if legal == (Action.DUMMY,) else legal[0]
        result = game.step(action)
        assert result.terminated == (activation == HORIZON)

    assert game.state is TERMINAL_STATE
    assert game.terminated
    with pytest.raises(TypeError):
        encode_observation(game.state, game.context)


# ---------------------------------------------------------------------------
# 3. Rifiuto di stati incoerenti e dello stato terminale.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_state", [None, TERMINAL_STATE, object(), 3, "terminal"])
def test_rejects_non_game_state(invalid_state):
    with pytest.raises(TypeError):
        encode_observation(invalid_state, get_context(1, 1))


@pytest.mark.parametrize("invalid_context", [None, object(), 3, "m1-k1"])
def test_rejects_invalid_context(invalid_context):
    context = get_context(1, 1)
    with pytest.raises(TypeError):
        encode_observation(_state(context), invalid_context)


@pytest.mark.parametrize("position", [(4, 5), (0, 1), (10, 1)])
def test_rejects_agent_on_obstacle_or_outside_grid(position):
    context = get_context(1, 1)
    with pytest.raises(ValueError):
        encode_observation(_state(context, agent_position=position), context)


@pytest.mark.parametrize("position", [(4, 5), (5, 0), (5, 10)])
def test_rejects_opponent_on_obstacle_or_outside_grid(position):
    context = get_context(1, 1)
    with pytest.raises(ValueError):
        encode_observation(_state(context, opponent_position=position), context)


def test_rejects_overlapping_players():
    context = get_context(1, 1)
    with pytest.raises(ValueError):
        encode_observation(_state(context, opponent_position=(5, 1)), context)


@pytest.mark.parametrize("objectives", [(), ((5, 1),), ((5, 9),), ((4, 5),), ((0, 3),)])
def test_rejects_invalid_objective_positions_or_empty_set(objectives):
    context = get_context(1, 1)
    with pytest.raises(ValueError):
        encode_observation(_state(context, objectives=objectives), context)


def test_rejects_objectives_on_start_cells_even_when_vacant():
    context = get_context(1, 1)
    state = _state(
        context,
        agent_position=(4, 1),
        opponent_position=(6, 9),
        objectives={(5, 1)},
    )
    with pytest.raises(ValueError):
        encode_observation(state, context)


def test_rejects_objective_on_current_player_position():
    context = get_context(1, 1)
    state = _state(context, agent_position=(4, 1), objectives={(4, 1)})
    with pytest.raises(ValueError):
        encode_observation(state, context)


def test_rejects_more_than_three_objectives():
    context = get_context(1, 1)
    objectives = sorted(context.game_map.free_cells - START_CELLS)[:4]
    with pytest.raises(ValueError):
        encode_observation(_state(context, objectives=objectives), context)


@pytest.mark.parametrize("differential", (-801, 801))
def test_rejects_score_difference_outside_conservative_bound(differential):
    context = get_context(1, 1)
    with pytest.raises(ValueError):
        encode_observation(_state(context, agent_vp=differential), context)


# ---------------------------------------------------------------------------
# 4. Aggregazione di osservazioni NumPy e trasferimento PyTorch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", (1, 2, 3))
def test_stacking_observations_preserves_order_and_shape(sample_observations, batch_size):
    source = sample_observations[:batch_size]
    batch = stack_observations(source)
    assert isinstance(batch, ObservationBatch)
    assert batch.spatial.shape == (batch_size, *SPATIAL_SHAPE)
    assert batch.features.shape == (batch_size, FEATURE_SIZE)
    assert batch.spatial.dtype == np.float32
    assert batch.features.dtype == np.float32
    assert batch.spatial.flags.c_contiguous
    assert batch.features.flags.c_contiguous
    for i, observation in enumerate(source):
        np.testing.assert_array_equal(batch.spatial[i], observation.spatial)
        np.testing.assert_array_equal(batch.features[i], observation.features)
        assert not np.shares_memory(batch.spatial[i], observation.spatial)
        assert not np.shares_memory(batch.features[i], observation.features)


def test_stacking_does_not_alias_input_arrays(sample_observations):
    batch = stack_observations(sample_observations)
    original = np.copy(batch.spatial)
    sample_observations[0].spatial.fill(-99)
    np.testing.assert_array_equal(batch.spatial, original)
    batch.features.fill(-99)
    assert sample_observations[1].features[1] != -99


def test_stacking_rejects_empty_list():
    with pytest.raises(ValueError):
        stack_observations([])


@pytest.mark.parametrize("invalid", [None, 12, "observation"])
def test_stacking_rejects_non_observations(invalid, sample_observations):
    with pytest.raises(TypeError):
        stack_observations([sample_observations[0], invalid])


@pytest.mark.parametrize(
    "spatial_shape,feature_shape,spatial_dtype,feature_dtype",
    [
        ((3, 9, 9), (13,), np.float32, np.float32),
        ((4, 9, 8), (13,), np.float32, np.float32),
        ((4, 9, 9), (12,), np.float32, np.float32),
        ((4, 9, 9), (13,), np.float64, np.float32),
        ((4, 9, 9), (13,), np.float32, np.int32),
    ],
)
def test_stacking_rejects_wrong_shapes_or_dtypes(
    spatial_shape, feature_shape, spatial_dtype, feature_dtype
):
    broken = Observation(
        spatial=np.zeros(spatial_shape, dtype=spatial_dtype),
        features=np.zeros(feature_shape, dtype=feature_dtype),
    )
    with pytest.raises(ValueError):
        stack_observations([broken])


def test_to_torch_batch_cpu(sample_observations):
    torch = pytest.importorskip("torch")
    batch = stack_observations(sample_observations)
    tensors = to_torch_batch(batch, device="cpu")
    assert isinstance(tensors, TorchObservationBatch)
    assert isinstance(tensors.spatial, torch.Tensor)
    assert isinstance(tensors.features, torch.Tensor)
    assert tensors.spatial.device.type == "cpu"
    assert tensors.features.device.type == "cpu"
    assert tensors.spatial.dtype == torch.float32
    assert tensors.features.dtype == torch.float32
    assert tuple(tensors.spatial.shape) == (3, 4, 9, 9)
    assert tuple(tensors.features.shape) == (3, 13)
    np.testing.assert_array_equal(tensors.spatial.numpy(), batch.spatial)
    np.testing.assert_array_equal(tensors.features.numpy(), batch.features)


def test_to_torch_batch_rejects_invalid_argument():
    pytest.importorskip("torch")
    with pytest.raises(TypeError):
        to_torch_batch(None)


@pytest.mark.parametrize(
    "spatial_shape,feature_shape,spatial_dtype,feature_dtype",
    [
        ((1, 3, 9, 9), (1, 13), np.float32, np.float32),
        ((1, 4, 9, 9), (2, 13), np.float32, np.float32),
        ((1, 4, 9, 9), (1, 12), np.float32, np.float32),
        ((1, 4, 9, 9), (1, 13), np.float64, np.float32),
        ((1, 4, 9, 9), (1, 13), np.float32, np.float64),
        ((0, 4, 9, 9), (0, 13), np.float32, np.float32),
    ],
)
def test_to_torch_batch_rejects_invalid_shapes_or_dtypes(
    spatial_shape, feature_shape, spatial_dtype, feature_dtype
):
    pytest.importorskip("torch")
    broken = ObservationBatch(
        spatial=np.zeros(spatial_shape, dtype=spatial_dtype),
        features=np.zeros(feature_shape, dtype=feature_dtype),
    )
    with pytest.raises(ValueError):
        to_torch_batch(broken)


def test_pinned_memory_requires_cuda_when_unavailable(sample_observations):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA disponibile: il caso di errore CPU non e' applicabile")
    batch = stack_observations(sample_observations)
    with pytest.raises(RuntimeError, match="CUDA"):
        to_torch_batch(batch, device="cpu", pin_memory=True)


def test_to_torch_batch_cuda_if_available(sample_observations):
    """Verifica opzionale direttamente sulla RTX 5060 Ti, se disponibile."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Nessuna GPU CUDA disponibile: test CUDA saltato")

    batch = stack_observations(sample_observations)
    tensors = to_torch_batch(
        batch, device="cuda", pin_memory=True, non_blocking=True
    )
    torch.cuda.synchronize()
    assert tensors.spatial.is_cuda
    assert tensors.features.is_cuda
    assert tensors.spatial.dtype == torch.float32
    assert tensors.features.dtype == torch.float32
    assert tuple(tensors.spatial.shape) == (3, 4, 9, 9)
    assert tuple(tensors.features.shape) == (3, 13)
    np.testing.assert_array_equal(tensors.spatial.cpu().numpy(), batch.spatial)
    np.testing.assert_array_equal(tensors.features.cpu().numpy(), batch.features)
