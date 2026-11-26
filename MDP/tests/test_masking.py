"""Regressione del masking per Context-Conditioned Masked DQN.

Esecuzione dalla root Giulio2:
    python -m pytest tests/test_masking.py -v

Le prove CUDA vengono saltate quando non e' disponibile una GPU NVIDIA.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.action_mask import (
    DUMMY_ACTION_ID,
    NUM_ACTIONS,
    legal_action_mask,
    legal_actions_to_mask,
    mask_q_values,
    stack_action_masks,
    to_torch_action_masks,
)
from environment.context import get_all_contexts, get_context
from environment.game import TERMINAL_STATE, GameState, GridGame
from environment.maps import START_CELLS
from environment.rules import Action, Player


CONTEXTS = get_all_contexts()


def _game_with_state(
    *,
    agent=(5, 1),
    opponent=(5, 9),
    player=Player.AGENT,
    map_id=1,
    opponent_id=1,
) -> GridGame:
    """Prepara uno stato sintetico per isolare bordi, tiro e melee."""
    context = get_context(map_id, opponent_id)
    game = GridGame(seed=1234)
    game.reset(seed=1234, context=context, starting_player=player)
    assert agent in context.game_map.free_cells
    assert opponent in context.game_map.free_cells
    assert agent != opponent
    objectives = sorted(
        context.game_map.free_cells - START_CELLS - {agent, opponent}
    )[:3]
    game._state = GameState(  # Solo test: non esiste un setter pubblico dello stato.
        agent_position=agent,
        opponent_position=opponent,
        active_objectives=frozenset(objectives),
        agent_vp=0,
        opponent_vp=0,
        player_to_act=player,
        activation=10,
    )
    return game


def _expected_mask(*actions: Action) -> np.ndarray:
    """Oracolo indipendente dalla funzione sotto test."""
    result = np.zeros(7, dtype=np.bool_)
    for action in actions:
        result[int(action)] = True
    return result


def test_action_ids_and_mask_contract():
    assert NUM_ACTIONS == 7
    assert DUMMY_ACTION_ID == 6
    assert tuple(int(action) for action in Action) == tuple(range(7))

    mask = legal_action_mask(_game_with_state())
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (7,)
    assert mask.dtype == np.bool_
    assert mask.flags.c_contiguous
    np.testing.assert_array_equal(
        mask,
        _expected_mask(Action.NORTH, Action.EAST, Action.WEST, Action.PASS),
    )


@pytest.mark.parametrize("context", CONTEXTS, ids=lambda c: f"m{c.map_id}-k{c.opponent_id}")
@pytest.mark.parametrize("starting_player", (Player.AGENT, Player.OPPONENT))
def test_initial_masks_in_every_context(context, starting_player):
    game = GridGame(seed=55)
    game.reset(seed=55, context=context, starting_player=starting_player)
    mask = legal_action_mask(game)
    assert set(np.flatnonzero(mask)) == {int(a) for a in game.legal_actions()}
    if starting_player is Player.OPPONENT:
        np.testing.assert_array_equal(mask, _expected_mask(Action.DUMMY))
    else:
        assert mask[int(Action.PASS)]
        assert not mask[DUMMY_ACTION_ID]


def test_obstacle_and_map_boundary():
    # In mappa 1 la casella (4,5) e' un ostacolo; da (4,4) NORTH e' illegale.
    game = _game_with_state(agent=(4, 4), opponent=(5, 9))
    mask = legal_action_mask(game)
    np.testing.assert_array_equal(
        mask,
        _expected_mask(Action.SOUTH, Action.EAST, Action.WEST, Action.PASS),
    )
    assert not mask[int(Action.NORTH)]


def test_shot_with_obstacles_in_between():
    # Il tiro usa distanza Manhattan 3; gli ostacoli (5,5) e (5,6)
    # NON bloccano la linea di tiro secondo il regolamento di Giulio2.
    game = _game_with_state(agent=(5, 4), opponent=(5, 7))
    mask = legal_action_mask(game)
    assert mask[int(Action.SHOOT)]
    assert not mask[int(Action.NORTH)]  # (5,5) e' un ostacolo.
    assert mask[int(Action.PASS)]


@pytest.mark.parametrize("agent,opponent,expected", [
    ((5, 1), (5, 9), False),  # distanza 8
    ((5, 4), (5, 7), True),   # distanza 3
    ((5, 4), (5, 2), True),   # distanza 2, caselle libere in mappa 1
    ((5, 4), (6, 4), False),  # distanza 1: melee obbligatorio
])
def test_shoot_distance_cases(agent, opponent, expected):
    game = _game_with_state(agent=agent, opponent=opponent)
    assert bool(legal_action_mask(game)[int(Action.SHOOT)]) is expected


@pytest.mark.parametrize("player", (Player.AGENT, Player.OPPONENT))
def test_adjacent_players_force_dummy(player):
    game = _game_with_state(agent=(5, 4), opponent=(6, 4), player=player)
    assert game.legal_actions() == (Action.DUMMY,)
    np.testing.assert_array_equal(legal_action_mask(game), _expected_mask(Action.DUMMY))


def test_opponent_turn_forces_dummy_even_without_melee():
    game = _game_with_state(player=Player.OPPONENT)
    np.testing.assert_array_equal(legal_action_mask(game), _expected_mask(Action.DUMMY))


def test_mask_uses_game_as_only_authority(monkeypatch):
    game = _game_with_state()
    monkeypatch.setattr(game, "legal_actions", lambda: (Action.PASS,))
    np.testing.assert_array_equal(legal_action_mask(game), _expected_mask(Action.PASS))


def test_terminal_does_not_produce_a_mask():
    game = _game_with_state()
    game._state = TERMINAL_STATE
    assert game.terminated
    with pytest.raises(ValueError, match="terminale"):
        legal_action_mask(game)


def test_uninitialized_game_and_invalid_game():
    with pytest.raises(RuntimeError, match="reset"):
        legal_action_mask(GridGame(seed=1))
    with pytest.raises(TypeError, match="GridGame"):
        legal_action_mask(None)


def test_does_not_change_game_state_or_rngs():
    game = _game_with_state()
    state_before = game.state
    states_before = tuple(
        rng.getstate()
        for rng in (game.rng_context, game.rng_initiative, game.rng_objectives,
                    game.rng_agent, game.rng_opponent, game.rng_combat)
    )
    first = legal_action_mask(game)
    second = legal_action_mask(game)
    first[:] = False  # Modificare l'output non deve contaminare la nuova maschera.
    assert second.any()
    assert game.state is state_before
    assert states_before == tuple(
        rng.getstate()
        for rng in (game.rng_context, game.rng_initiative, game.rng_objectives,
                    game.rng_agent, game.rng_opponent, game.rng_combat)
    )


@pytest.mark.parametrize("bad_actions", [
    (),
    (Action.NORTH,),                 # manca PASS nell'attivazione normale
    (Action.DUMMY, Action.PASS),     # DUMMY non puo' convivere con PASS
    (Action.PASS, Action.PASS),     # duplicati
    (Action.PASS, 7),               # ID fuori intervallo
    (Action.PASS, -1),
    (Action.PASS, True),            # booleani non sono ID azione
    (Action.PASS, 1.0),
    (Action.PASS, "SHOOT"),
])
def test_illegal_action_lists_rejected(bad_actions):
    with pytest.raises(ValueError):
        legal_actions_to_mask(bad_actions)


def test_non_iterable_action_list_rejected():
    with pytest.raises(TypeError):
        legal_actions_to_mask(None)


def test_action_ids_can_be_python_integers():
    np.testing.assert_array_equal(
        legal_actions_to_mask((0, 5)), _expected_mask(Action.NORTH, Action.PASS)
    )
    np.testing.assert_array_equal(
        legal_actions_to_mask((6,)), _expected_mask(Action.DUMMY)
    )


def test_stack_masks_preserves_order_and_copies_inputs():
    masks = [
        _expected_mask(Action.NORTH, Action.PASS),
        _expected_mask(Action.DUMMY),
        _expected_mask(Action.SHOOT, Action.PASS),
    ]
    batch = stack_action_masks(masks)
    assert batch.shape == (3, 7)
    assert batch.dtype == np.bool_
    assert batch.flags.c_contiguous
    for index, mask in enumerate(masks):
        np.testing.assert_array_equal(batch[index], mask)
        assert not np.shares_memory(batch[index], mask)
    masks[0][:] = False
    assert batch[0, int(Action.PASS)]


@pytest.mark.parametrize("invalid", [
    np.zeros(7, dtype=np.bool_),
    np.zeros(6, dtype=np.bool_),
    np.ones(7, dtype=np.float32),
    _expected_mask(Action.NORTH),
    _expected_mask(Action.DUMMY, Action.PASS),
    [False] * 7,
])
def test_invalid_numpy_masks_rejected(invalid):
    with pytest.raises((ValueError, TypeError)):
        stack_action_masks([invalid])
    with pytest.raises((ValueError, TypeError)):
        to_torch_action_masks(invalid)


def test_empty_or_malformed_batch_rejected():
    with pytest.raises(ValueError):
        stack_action_masks([])
    with pytest.raises(TypeError):
        stack_action_masks(None)
    with pytest.raises(ValueError):
        to_torch_action_masks(np.zeros((0, 7), dtype=np.bool_))
    with pytest.raises(ValueError):
        to_torch_action_masks(np.zeros((2, 6), dtype=np.bool_))
    with pytest.raises(TypeError):
        to_torch_action_masks([True] * 7)


def test_all_100_activations_for_each_of_24_contexts():
    """Ogni azione selezionata dalla maschera dev'essere accettata dall'MDP."""
    for index, context in enumerate(CONTEXTS):
        game = GridGame(seed=1000 + index)
        game.reset(seed=1000 + index, context=context)
        for step in range(100):
            assert game.state.activation == step + 1
            legal = game.legal_actions()
            mask = legal_action_mask(game)
            assert set(np.flatnonzero(mask)) == {int(a) for a in legal}
            assert mask.any()
            assert mask[DUMMY_ACTION_ID] == (legal == (Action.DUMMY,))
            action = Action(int(np.flatnonzero(mask)[0]))
            result = game.step(action)
            assert result.action == action
            assert result.terminated == (step == 99)
        assert game.terminated
        with pytest.raises(ValueError, match="terminale"):
            legal_action_mask(game)


def test_torch_cpu_conversion_and_masking():
    torch = pytest.importorskip("torch")
    masks = stack_action_masks([
        _expected_mask(Action.NORTH, Action.PASS),
        _expected_mask(Action.DUMMY),
    ])
    tensor_masks = to_torch_action_masks(masks, device="cpu")
    assert tensor_masks.shape == (2, 7)
    assert tensor_masks.device.type == "cpu"
    assert tensor_masks.dtype == torch.bool
    np.testing.assert_array_equal(tensor_masks.numpy(), masks)

    q = torch.tensor([
        [1.0, 999.0, 888.0, 777.0, 666.0, 2.0, 555.0],
        [999.0, 888.0, 777.0, 666.0, 555.0, 444.0, 0.5],
    ], dtype=torch.float32)
    original = q.clone()
    masked = mask_q_values(q, tensor_masks)
    assert torch.equal(q, original)  # Non in-place.
    assert masked.argmax(dim=1).tolist() == [int(Action.PASS), int(Action.DUMMY)]
    assert masked[0, 0].item() == 1.0
    assert masked[0, 5].item() == 2.0
    assert torch.isneginf(masked[0, 1])
    assert torch.isneginf(masked[1, :6]).all()


def test_torch_single_mask_and_gradient_flow():
    torch = pytest.importorskip("torch")
    action_mask = to_torch_action_masks(_expected_mask(Action.NORTH, Action.PASS))
    q = torch.arange(7, dtype=torch.float32, requires_grad=True)
    masked = mask_q_values(q, action_mask)
    masked[action_mask].sum().backward()
    assert q.grad is not None
    torch.testing.assert_close(q.grad, action_mask.to(torch.float32))
    assert torch.isneginf(masked[~action_mask]).all()


@pytest.mark.parametrize("bad_case", [
    "not_torch_q", "not_torch_mask", "wrong_shape", "wrong_batch",
    "integer_q", "float_mask", "broadcast_mask",
])
def test_mask_q_values_rejects_wrong_inputs(bad_case):
    torch = pytest.importorskip("torch")
    q = torch.zeros((2, 7), dtype=torch.float32)
    mask = torch.ones((2, 7), dtype=torch.bool)
    if bad_case == "not_torch_q":
        q = np.zeros((2, 7), dtype=np.float32)
    elif bad_case == "not_torch_mask":
        mask = np.ones((2, 7), dtype=np.bool_)
    elif bad_case == "wrong_shape":
        q = torch.zeros((2, 6), dtype=torch.float32)
        mask = torch.ones((2, 6), dtype=torch.bool)
    elif bad_case == "wrong_batch":
        mask = torch.ones((3, 7), dtype=torch.bool)
    elif bad_case == "integer_q":
        q = torch.zeros((2, 7), dtype=torch.int64)
    elif bad_case == "float_mask":
        mask = mask.to(torch.float32)
    elif bad_case == "broadcast_mask":
        mask = torch.ones((7,), dtype=torch.bool)
    with pytest.raises((TypeError, ValueError)):
        mask_q_values(q, mask)


def test_cuda_batch_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA non disponibile sul computer di test")

    masks = stack_action_masks([
        _expected_mask(Action.NORTH, Action.PASS),
        _expected_mask(Action.DUMMY),
    ])
    gpu_masks = to_torch_action_masks(
        masks, device="cuda", pin_memory=True, non_blocking=True
    )
    assert gpu_masks.device.type == "cuda"
    assert gpu_masks.dtype == torch.bool
    q = torch.tensor([
        [1.0, 100.0, 100.0, 100.0, 100.0, 2.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 3.0],
    ], device="cuda")
    filtered = mask_q_values(q, gpu_masks)
    assert filtered.device.type == "cuda"
    assert filtered.argmax(dim=1).cpu().tolist() == [5, 6]
    assert torch.isneginf(filtered[~gpu_masks]).all().item()


def test_cuda_mismatched_device_is_rejected_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA non disponibile sul computer di test")
    q = torch.ones(7, device="cuda")
    cpu_mask = to_torch_action_masks(_expected_mask(Action.PASS))
    with pytest.raises(ValueError, match="stesso device"):
        mask_q_values(q, cpu_mask)
