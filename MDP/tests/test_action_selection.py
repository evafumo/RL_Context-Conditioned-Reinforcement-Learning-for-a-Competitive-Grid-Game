"""Test della Fase 3.3: epsilon-greedy con action masking per DQN standard.

Eseguire dalla cartella Giulio2:
    python -m pytest tests/test_action_selection.py -v

I controlli CUDA vengono saltati se la GPU non e' disponibile.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from agent.action_selection import (
    epsilon_greedy_actions,
    greedy_actions,
    select_actions,
)
from agent.network import QNetwork
from environment.action_mask import (
    DUMMY_ACTION_ID,
    NUM_ACTIONS,
    legal_action_mask,
    legal_actions_to_mask,
    stack_action_masks,
    to_torch_action_masks,
)
from environment.context import get_all_contexts
from environment.game import GridGame
from environment.observations import (
    encode_observation,
    stack_observations,
    to_torch_batch,
)
from environment.rules import Action, Player


def _mask(*actions: Action) -> torch.Tensor:
    return torch.from_numpy(legal_actions_to_mask(actions))


def _examples() -> tuple[torch.Tensor, torch.Tensor]:
    masks = torch.stack([
        _mask(Action.NORTH, Action.PASS),
        _mask(Action.EAST, Action.SHOOT, Action.PASS),
        _mask(Action.DUMMY),
        _mask(Action.SOUTH, Action.WEST, Action.PASS),
    ])
    q = torch.tensor([
        [-3.0, 800.0, 900.0, 1000.0, 1000.0, -2.0, 999.0],
        [300.0, 200.0, -9.0, 400.0, -1.0, -2.0, 500.0],
        [999.0, 999.0, 999.0, 999.0, 999.0, 999.0, -1000.0],
        [100.0, -1.0, 100.0, -1.0, 100.0, -5.0, 100.0],
    ], dtype=torch.float32)
    return q, masks


def _assert_legal(actions: torch.Tensor, masks: torch.Tensor) -> None:
    assert actions.ndim == 1
    assert actions.dtype == torch.long
    assert actions.device == masks.device
    assert bool(masks.gather(1, actions.unsqueeze(1)).all().item())


def test_greedy_respects_legal_mask_and_negative_q_values() -> None:
    q, masks = _examples()
    before_q, before_masks = q.clone(), masks.clone()
    actions = greedy_actions(q, masks)
    assert actions.tolist() == [int(Action.PASS), int(Action.SHOOT),
                                DUMMY_ACTION_ID, int(Action.SOUTH)]
    _assert_legal(actions, masks)
    torch.testing.assert_close(q, before_q)
    torch.testing.assert_close(masks, before_masks)


def test_greedy_tie_breaks_to_lowest_legal_action_id() -> None:
    mask = _mask(Action.NORTH, Action.EAST, Action.PASS).unsqueeze(0)
    q = torch.tensor([[1.0, 1000.0, 1.0, 1000.0, 1000.0, 1.0, 1000.0]])
    assert greedy_actions(q, mask).item() == int(Action.NORTH)


@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.5, 1.0])
def test_dummy_always_chosen_in_forced_rows(epsilon: float) -> None:
    q, masks = _examples()
    generator = torch.Generator().manual_seed(555)
    for _ in range(15):
        actions = epsilon_greedy_actions(q, masks, epsilon, generator=generator)
        assert actions[2].item() == DUMMY_ACTION_ID
        _assert_legal(actions, masks)


def test_epsilon_zero_equals_greedy_and_consumes_no_rng() -> None:
    q, masks = _examples()
    generator = torch.Generator().manual_seed(2701)
    state = generator.get_state().clone()
    with torch.enable_grad():
        actions = epsilon_greedy_actions(q.clone().requires_grad_(True),
                                         masks, 0.0, generator=generator)
    torch.testing.assert_close(actions, greedy_actions(q, masks))
    torch.testing.assert_close(state, generator.get_state())
    assert not actions.requires_grad


def test_epsilon_one_selects_only_legal_and_is_reproducible() -> None:
    q, masks = _examples()
    q = q.repeat(60, 1)
    masks = masks.repeat(60, 1)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    first = epsilon_greedy_actions(q, masks, 1.0, generator=gen1)
    second = epsilon_greedy_actions(q, masks, 1.0, generator=gen2)
    torch.testing.assert_close(first, second)
    _assert_legal(first, masks)
    assert (first[2::4] == DUMMY_ACTION_ID).all()


def test_epsilon_one_samples_uniformly_among_legal_actions() -> None:
    mask = _mask(Action.NORTH, Action.SHOOT, Action.PASS)
    n = 9000
    q = torch.tensor([[-100.0, 99.0, 99.0, 99.0, 2.0, 1.0, 99.0]]).expand(n, -1)
    masks = mask.unsqueeze(0).expand(n, -1)
    actions = epsilon_greedy_actions(
        q, masks, 1.0, generator=torch.Generator().manual_seed(77)
    )
    _assert_legal(actions, masks)
    counts = torch.bincount(actions, minlength=NUM_ACTIONS)
    assert int(counts.sum()) == n
    for a in (Action.NORTH, Action.SHOOT, Action.PASS):
        assert abs(int(counts[int(a)]) / n - (1 / 3)) < 0.025
    for a in (Action.SOUTH, Action.EAST, Action.WEST, Action.DUMMY):
        assert counts[int(a)] == 0


def test_intermediate_epsilon_is_reproducible_and_explores() -> None:
    n = 512
    masks = _mask(Action.NORTH, Action.PASS).unsqueeze(0).expand(n, -1)
    q = torch.tensor([[3.0, 999.0, 999.0, 999.0, 999.0, 0.0, 999.0]]).expand(n, -1)
    a = epsilon_greedy_actions(
        q, masks, 0.3, generator=torch.Generator().manual_seed(2026)
    )
    b = epsilon_greedy_actions(
        q, masks, 0.3, generator=torch.Generator().manual_seed(2026)
    )
    torch.testing.assert_close(a, b)
    _assert_legal(a, masks)
    assert 20 < (a == int(Action.PASS)).sum().item() < 160


@pytest.mark.parametrize("bad", [
    -0.1, 1.1, float("nan"), float("inf"), -float("inf"), True,
    None, "0.1", torch.tensor(0.3),
])
def test_invalid_epsilon_rejected(bad: object) -> None:
    q, masks = _examples()
    with pytest.raises((TypeError, ValueError)):
        epsilon_greedy_actions(q, masks, bad)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        select_actions(_SpyQNetwork(), torch.zeros(4, 4, 9, 9),
                       torch.zeros(4, 13), masks, bad)  # type: ignore[arg-type]


def test_invalid_generator_rejected() -> None:
    q, masks = _examples()
    for probability in (0.0, 0.1):
        with pytest.raises(TypeError, match="generator"):
            epsilon_greedy_actions(q, masks, probability, generator="wrong")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_q,bad_mask,error", [
    (torch.zeros(7), torch.ones(7, dtype=torch.bool), ValueError),
    (torch.zeros(0, 7), torch.zeros(0, 7, dtype=torch.bool), ValueError),
    (torch.zeros(2, 6), torch.ones(2, 6, dtype=torch.bool), ValueError),
    (torch.zeros(2, 8), torch.ones(2, 8, dtype=torch.bool), ValueError),
    (torch.zeros(2, 7, dtype=torch.int64), torch.ones(2, 7, dtype=torch.bool), TypeError),
    (torch.zeros(2, 7), torch.ones(3, 7, dtype=torch.bool), ValueError),
    (torch.zeros(2, 7), torch.ones(2, 7, dtype=torch.float32), TypeError),
    (torch.zeros(2, 7), None, TypeError),
])
def test_invalid_q_values_and_masks_rejected(bad_q, bad_mask, error) -> None:
    with pytest.raises(error):
        epsilon_greedy_actions(bad_q, bad_mask, 0.0)
    with pytest.raises(error):
        greedy_actions(bad_q, bad_mask)


def test_q_values_must_be_tensor() -> None:
    with pytest.raises(TypeError):
        greedy_actions(None, torch.ones(2, NUM_ACTIONS, dtype=torch.bool))


def test_action_selector_does_not_backpropagate_or_modify_inputs() -> None:
    q, masks = _examples()
    q.requires_grad_()
    q_before = q.detach().clone()
    masks_before = masks.clone()
    with torch.enable_grad():
        actions = epsilon_greedy_actions(
            q, masks, 0.4, generator=torch.Generator().manual_seed(5)
        )
    assert not actions.requires_grad
    torch.testing.assert_close(q.detach(), q_before)
    torch.testing.assert_close(masks, masks_before)


class _SpyQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("base_q", torch.tensor([0., -1., 30., -3., 2., 1., 99.]))
        self.calls = 0
        self.grad_enabled_during_forward: bool | None = None

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return self.base_q.unsqueeze(0).expand(spatial.shape[0], -1)


def test_select_actions_infers_once_no_grad_and_does_not_change_mode() -> None:
    model = _SpyQNetwork().train()
    q, masks = _examples()
    spatial = torch.zeros((len(q), 4, 9, 9))
    features = torch.zeros((len(q), 13))
    actions = select_actions(model, spatial, features, masks, epsilon=0.0)
    assert model.calls == 1
    assert model.grad_enabled_during_forward is False
    assert model.training
    assert actions.tolist() == [int(Action.PASS), int(Action.EAST),
                                DUMMY_ACTION_ID, int(Action.PASS)]
    _assert_legal(actions, masks)


def test_select_actions_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        select_actions(None, torch.zeros(1, 4, 9, 9),
                       torch.zeros(1, 13), _mask(Action.PASS).unsqueeze(0), 0.0)


def test_end_to_end_qnetwork_all_24_contexts() -> None:
    games = []
    obs = []
    masks = []
    for i, context in enumerate(get_all_contexts()):
        game = GridGame(seed=99 + i)
        state = game.reset(seed=99 + i, context=context,
                           starting_player=Player.OPPONENT if i % 2 else Player.AGENT)
        games.append(game)
        obs.append(encode_observation(state, context))
        masks.append(legal_action_mask(game))
    batch = to_torch_batch(stack_observations(obs))
    mask_batch = to_torch_action_masks(stack_action_masks(masks))
    model = QNetwork().eval()
    actions = select_actions(model, batch.spatial, batch.features, mask_batch, 0.35,
                             generator=torch.Generator().manual_seed(6))
    assert actions.shape == (24,)
    _assert_legal(actions, mask_batch)
    for game, action in zip(games, actions.tolist()):
        assert Action(action) in game.legal_actions()
        result = game.step(action)
        assert result.action == Action(action)
        assert not result.terminated


@pytest.mark.parametrize("epsilon", [0.0, 0.3, 1.0])
def test_all_100_activations_in_24_contexts_respect_rules(epsilon: float) -> None:
    """Ogni selezione viene accettata da GridGame; DUMMY incluso."""
    q = torch.tensor([[6., 5., 4., 3., 2., 1., -7.]])
    gen = torch.Generator().manual_seed(20261026)
    games = []
    for i, context in enumerate(get_all_contexts()):
        game = GridGame(seed=1000 + i)
        game.reset(seed=1000 + i, context=context)
        games.append(game)
    # 100 inferenze batched sui 24 ambienti, non 2400 singole inferenze.
    for t in range(100):
        mask_np = stack_action_masks([legal_action_mask(g) for g in games])
        masks = to_torch_action_masks(mask_np)
        q_batch = q.expand(len(games), -1)
        actions = epsilon_greedy_actions(q_batch, masks, epsilon, generator=gen)
        _assert_legal(actions, masks)
        for i, (game, action) in enumerate(zip(games, actions.tolist())):
            assert mask_np[i, action]
            assert Action(action) in game.legal_actions()
            result = game.step(action)
            assert result.terminated == (t == 99)
            if mask_np[i, DUMMY_ACTION_ID]:
                assert result.forced
    assert all(game.terminated for game in games)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Richiede CUDA")
def test_cuda_batched_inference_masking_and_seed() -> None:
    contexts = get_all_contexts()
    games, observations, np_masks = [], [], []
    for i, context in enumerate(contexts):
        game = GridGame(seed=101 + i)
        state = game.reset(seed=101 + i, context=context,
                           starting_player=Player.OPPONENT if i % 2 else Player.AGENT)
        games.append(game)
        observations.append(encode_observation(state, context))
        np_masks.append(legal_action_mask(game))
    batch = to_torch_batch(stack_observations(observations), device="cuda",
                           pin_memory=True, non_blocking=True)
    masks = to_torch_action_masks(stack_action_masks(np_masks), device="cuda",
                                  pin_memory=True, non_blocking=True)
    model = QNetwork().to("cuda").eval()
    first = select_actions(model, batch.spatial, batch.features, masks, epsilon=0.25,
                           generator=torch.Generator(device="cuda").manual_seed(71))
    second = select_actions(model, batch.spatial, batch.features, masks, epsilon=0.25,
                            generator=torch.Generator(device="cuda").manual_seed(71))
    assert first.is_cuda and first.dtype == torch.long and first.shape == (24,)
    torch.testing.assert_close(first, second)
    _assert_legal(first, masks)
    cpu_actions = first.cpu().tolist()  # Unico trasferimento batched per step CPU.
    for game, action in zip(games, cpu_actions):
        assert Action(action) in game.legal_actions()
        game.step(action)
    with pytest.raises(ValueError, match="generatore RNG"):
        epsilon_greedy_actions(torch.zeros(1, 7, device="cuda"),
                               masks[:1], 0.1, generator=torch.Generator(device="cpu"))
