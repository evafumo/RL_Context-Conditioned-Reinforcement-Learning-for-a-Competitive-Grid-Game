"""Test di agent/dqn.py: Bellman masking, DQN standard, CUDA opzionale.

Dalla root Giulio2:
    python -m pytest tests/test_dqn.py -q
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from agent.dqn import (
    DQNConfig, DQNTrainer, DQNUpdateStats,
    compute_dqn_loss, compute_dqn_targets, hard_sync_target,
)
from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer, TorchReplayBatch, to_torch_replay_batch
from environment.action_mask import NUM_ACTIONS, legal_action_mask, legal_actions_to_mask
from environment.context import get_context
from environment.game import GridGame, HORIZON, GameState
from environment.observations import SPATIAL_SHAPE, FEATURE_SIZE, Observation, encode_observation
from environment.rules import Action, Player


class TableQ(nn.Module):
    """Modello test: un solo vettore Q addestrabile, uguale per tutte le righe."""

    def __init__(self, values: tuple[float, ...]):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.calls = 0

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.values.unsqueeze(0).expand(spatial.shape[0], -1)


ONLINE_Q = (-10.0, 2.0, -2.0, -3.0, 800.0, 20.0, -2.0)
TARGET_Q = (4.0, 2.0, -2.0, -3.0, 900.0, -3.0, -1.0)
PASS_MASK = legal_actions_to_mask((Action.PASS,))
DUMMY_MASK = legal_actions_to_mask((Action.DUMMY,))
NORTH_PASS_MASK = legal_actions_to_mask((Action.NORTH, Action.PASS))


def _batch() -> TorchReplayBatch:
    """Tre transizioni: normale, terminale, successiva forzata DUMMY."""
    spatial = torch.zeros((3, *SPATIAL_SHAPE), dtype=torch.float32)
    features = torch.zeros((3, FEATURE_SIZE), dtype=torch.float32)
    action_masks = torch.as_tensor(np.stack((PASS_MASK, DUMMY_MASK, PASS_MASK)))
    actions = torch.tensor((int(Action.PASS), int(Action.DUMMY), int(Action.PASS)), dtype=torch.int64)
    rewards = torch.tensor((1., 2., 3.), dtype=torch.float32)
    next_spatial = spatial.clone()
    next_features = features.clone()
    next_masks = torch.as_tensor(np.stack((NORTH_PASS_MASK, np.zeros(NUM_ACTIONS, dtype=bool), DUMMY_MASK)))
    dones = torch.tensor((False, True, False), dtype=torch.bool)
    return TorchReplayBatch(
        spatial, features, action_masks, actions, rewards,
        next_spatial, next_features, next_masks, dones,
    )


def _observation(t: int) -> Observation:
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    spatial[0, 0, 0] = 1.0
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features[1] = t / 100.
    return Observation(spatial, features)


def _replay(count: int = 6) -> ReplayBuffer:
    replay = ReplayBuffer(20, seed=42)
    for i in range(count):
        terminal = i % 3 == 0
        mask = DUMMY_MASK if i % 2 else PASS_MASK
        replay.add(
            observation=_observation(i),
            action_mask=mask.copy(),
            action=Action.DUMMY if i % 2 else Action.PASS,
            reward=float(i),
            next_observation=None if terminal else _observation(i + 1),
            next_action_mask=None if terminal else NORTH_PASS_MASK.copy(),
            done=terminal,
        )
    return replay


def test_targets_standard_dqn_mask_terminal_and_dummy():
    online = TableQ(ONLINE_Q)
    target = TableQ(TARGET_Q)
    expected = torch.tensor([5., 2., 2.])  # 1+max(4,-3), 2, 3+(-1)
    actual = compute_dqn_targets(target, _batch())
    torch.testing.assert_close(actual, expected)
    assert target.calls == 1
    assert online.calls == 0
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("gamma,expected", [
    (1., [5., 2., 2.]), (0.5, [3., 2., 2.5]), (0., [1., 2., 3.]),
])
def test_gamma_discount_factor(gamma, expected):
    actual = compute_dqn_targets(TableQ(TARGET_Q), _batch(), gamma=gamma)
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_loss_uses_raw_q_for_executed_action_and_huber():
    online = TableQ(ONLINE_Q)
    target = TableQ(TARGET_Q)
    loss, selected, td_targets = compute_dqn_loss(online, target, _batch())
    torch.testing.assert_close(selected, torch.tensor([20., -2., 20.]))
    torch.testing.assert_close(td_targets, torch.tensor([5., 2., 2.]))
    torch.testing.assert_close(loss, F.smooth_l1_loss(selected, td_targets, beta=1.))
    loss.backward()
    assert online.calls == 1 and target.calls == 1
    assert online.values.grad is not None
    assert online.values.grad[int(Action.PASS)] != 0
    assert online.values.grad[int(Action.DUMMY)] != 0
    assert online.values.grad[int(Action.SHOOT)] == 0
    assert target.values.grad is None


def test_no_double_dqn_online_prefers_pass_but_target_prefers_north():
    online = TableQ(ONLINE_Q)  # online: PASS=20, NORTH=-10
    target = TableQ(TARGET_Q)  # target: PASS=-3, NORTH=4
    y = compute_dqn_targets(target, _batch())[0]
    assert y.item() == 5.0  # standard DQN: 1 + max(4,-3)
    assert y.item() != -2.0  # Double DQN sarebbe: 1 + Q_target(PASS) = -2
    assert online.calls == 0


def test_terminal_row_cannot_bootstrap_even_if_target_huge():
    td = compute_dqn_targets(TableQ((1e8,) * NUM_ACTIONS), _batch())
    assert td[1].item() == 2.0
    assert torch.isfinite(td).all()


def test_target_hard_sync_includes_every_parameter_and_freezes_target():
    online = TableQ(ONLINE_Q)
    target = TableQ(TARGET_Q)
    hard_sync_target(online, target)
    torch.testing.assert_close(online.values, target.values)
    assert not target.training
    assert all(not p.requires_grad for p in target.parameters())
    with torch.no_grad():
        online.values[0].add_(10)
    assert online.values[0] != target.values[0]


def test_target_not_synced_until_interval_and_optimizer_updates():
    cfg = DQNConfig(learning_rate=0.005, target_sync_interval=2, grad_clip_norm=1.)
    trainer = DQNTrainer(TableQ(ONLINE_Q), config=cfg)
    original_online = trainer.online_network.values.detach().clone()
    initial_target = trainer.target_network.values.detach().clone()
    s1 = trainer.train_step(_batch())
    assert isinstance(s1, DQNUpdateStats)
    assert s1.update_steps == 1 and not s1.target_synced
    torch.testing.assert_close(trainer.target_network.values, initial_target)
    assert not torch.equal(trainer.online_network.values, original_online)
    assert s1.loss >= 0 and math.isfinite(s1.grad_norm)
    assert s1.grad_norm >= 0
    s2 = trainer.train_step(_batch())
    assert s2.update_steps == 2 and s2.target_synced
    torch.testing.assert_close(trainer.online_network.values, trainer.target_network.values)
    assert trainer.target_network.values.grad is None
    assert not trainer.target_network.training
    assert not trainer.target_network.values.requires_grad


def test_manual_sync_does_not_reset_step_counter():
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    trainer.train_step(_batch())
    trainer.sync_target()
    assert trainer.update_steps == 1
    torch.testing.assert_close(trainer.online_network.values, trainer.target_network.values)


def test_optimizer_uses_adam_and_float32():
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.config.gamma == 1.0
    assert trainer.device.type == "cpu"
    assert trainer.online_network.values.dtype == torch.float32


def test_train_on_replay_sampling_and_transfer_cpu():
    trainer = DQNTrainer(TableQ(ONLINE_Q), config=DQNConfig(target_sync_interval=5))
    replay = _replay(6)
    stats = trainer.train_on_replay(replay, 4)
    assert stats.update_steps == 1
    assert len(replay) == 6
    assert stats.loss >= 0
    with pytest.raises(ValueError, match="transizioni disponibili"):
        trainer.train_on_replay(replay, 7)


def test_train_step_numpy_replaybatch_cpu():
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    sample = _replay().sample(3)
    s = trainer.train_step(sample)
    assert s.update_steps == 1


def test_state_dict_roundtrip_preserves_unsynced_target_and_optimizer():
    cfg = DQNConfig(target_sync_interval=9)
    original = DQNTrainer(TableQ(ONLINE_Q), config=cfg)
    original.train_step(_batch())
    snapshot = original.state_dict()
    restored = DQNTrainer(TableQ((0.,)*NUM_ACTIONS), config=cfg)
    restored.load_state_dict(snapshot)
    assert restored.update_steps == 1
    torch.testing.assert_close(original.online_network.values, restored.online_network.values)
    torch.testing.assert_close(original.target_network.values, restored.target_network.values)
    assert not torch.equal(restored.online_network.values, restored.target_network.values)
    # Stato Adam completo: stesso update successivo e stessi momenti.
    original.train_step(_batch())
    restored.train_step(_batch())
    torch.testing.assert_close(original.online_network.values, restored.online_network.values)
    assert not restored.target_network.values.requires_grad


@pytest.mark.parametrize("broken", [
    {"update_steps": -1}, {"version": 999},
    {"config": {"gamma": 0.3}},
])
def test_invalid_checkpoint_rejected(broken):
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    snapshot = trainer.state_dict()
    snapshot.update(broken)
    with pytest.raises(ValueError):
        trainer.load_state_dict(snapshot)


@pytest.mark.parametrize("overrides,exception", [
    ({"gamma": -0.1}, ValueError), ({"gamma": 1.1}, ValueError),
    ({"gamma": math.nan}, ValueError), ({"gamma": True}, TypeError),
    ({"learning_rate": 0}, ValueError), ({"learning_rate": -1}, ValueError),
    ({"huber_delta": 0}, ValueError), ({"grad_clip_norm": 0}, ValueError),
    ({"target_sync_interval": 0}, ValueError),
    ({"target_sync_interval": True}, ValueError),
    ({"target_sync_interval": 1.5}, ValueError),
    ({"weight_decay": -1}, ValueError),
    ({"use_bfloat16_amp": "yes"}, TypeError),
])
def test_invalid_configuration(overrides, exception):
    with pytest.raises(exception):
        DQNConfig(**overrides)


def test_cpu_bfloat16_is_rejected_explicitly():
    with pytest.raises(ValueError, match="CUDA"):
        DQNTrainer(TableQ(ONLINE_Q), config=DQNConfig(use_bfloat16_amp=True))


def test_invalid_network_type_is_rejected():
    with pytest.raises(TypeError):
        DQNTrainer(5)
    with pytest.raises(TypeError):
        hard_sync_target(5, TableQ(TARGET_Q))
    with pytest.raises(TypeError):
        compute_dqn_targets(5, _batch())


@pytest.mark.parametrize("field,new_value,exception", [
    ("actions", torch.tensor([6, 6, 5]), ValueError), # illegale nella prima maschera
    ("actions", torch.tensor([5, 6, 15]), ValueError),
    ("actions", torch.tensor([5., 6., 5.]), TypeError),
    ("features", torch.zeros(2, FEATURE_SIZE), ValueError),
    ("action_masks", torch.zeros((3, 7), dtype=torch.float32), TypeError),
    ("next_action_masks", torch.ones(3, 7, dtype=torch.bool), ValueError),
    ("rewards", torch.tensor([float("nan"), 2., 3.]), ValueError),
    ("next_spatial", torch.zeros(3, *SPATIAL_SHAPE, dtype=torch.float64), TypeError),
])
def test_invalid_batch_rejected(field, new_value, exception):
    batch = replace(_batch(), **{field: new_value})
    with pytest.raises(exception):
        compute_dqn_targets(TableQ(TARGET_Q), batch)


def test_nonterminal_no_legal_action_rejected():
    batch = _batch()
    masks = batch.next_action_masks.clone()
    masks[0] = False
    with pytest.raises(ValueError, match="non terminale"):
        compute_dqn_targets(TableQ(TARGET_Q), replace(batch, next_action_masks=masks))


def test_dummy_cannot_coexist_with_pass_or_movement():
    batch = _batch()
    masks = batch.action_masks.clone()
    masks[1, int(Action.PASS)] = True
    with pytest.raises(ValueError, match="PASS/DUMMY"):
        compute_dqn_targets(TableQ(TARGET_Q), replace(batch, action_masks=masks))


def test_terminal_placeholder_must_be_zero():
    batch = _batch()
    next_spatial = batch.next_spatial.clone()
    next_spatial[1, 0, 0, 0] = 1.
    with pytest.raises(ValueError, match="segnaposto zero"):
        compute_dqn_targets(TableQ(TARGET_Q), replace(batch, next_spatial=next_spatial))


def test_real_qnetwork_update_cpu_forward_and_gradients():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261026)
        trainer = DQNTrainer(QNetwork(), config=DQNConfig(target_sync_interval=4))
    data = _replay(4).sample(4)
    first = trainer.online_network.q_head.weight.detach().clone()
    result = trainer.train_step(data)
    assert result.update_steps == 1 and math.isfinite(result.loss)
    assert not torch.equal(first, trainer.online_network.q_head.weight)
    assert all(p.grad is None for p in trainer.target_network.parameters())


def test_genuine_gridgame_transition_dqn_loss_and_masking():
    game = GridGame(seed=42)
    game.reset(seed=42, context=get_context(1, 6), starting_player=Player.AGENT)
    memory = ReplayBuffer(HORIZON, seed=42)
    dummy_count = 0
    for activation in range(1, HORIZON + 1):
        state = game.state
        assert isinstance(state, GameState)
        mask = legal_action_mask(game)
        chosen = Action.DUMMY if mask[int(Action.DUMMY)] else Action.PASS
        dummy_count += int(chosen == Action.DUMMY)
        observation = encode_observation(state, game.context)
        transition = game.step(chosen)
        memory.add(
            observation=observation,
            action_mask=mask,
            action=chosen,
            reward=transition.reward,
            next_observation=None if transition.terminated else encode_observation(transition.next_state, game.context),
            next_action_mask=None if transition.terminated else legal_action_mask(game),
            done=transition.terminated,
        )
    assert len(memory) == HORIZON and dummy_count > 0
    sample = memory.sample(HORIZON)
    assert sample.dones.sum() == 1
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    torch_sample = to_torch_replay_batch(sample)
    targets = compute_dqn_targets(trainer.target_network, torch_sample)
    assert torch.isfinite(targets).all()
    torch.testing.assert_close(targets[torch_sample.dones], torch_sample.rewards[torch_sample.dones])
    stats = trainer.train_step(torch_sample)
    assert stats.update_steps == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
def test_cuda_float32_minibatch_gradient_and_sync():
    cfg = DQNConfig(target_sync_interval=2)
    trainer = DQNTrainer(QNetwork(), config=cfg, device="cuda:0")
    data = _replay(4).sample(4)
    result = trainer.train_step(data)  # conversione CPU -> pinned -> cuda
    assert result.update_steps == 1 and math.isfinite(result.loss)
    assert next(trainer.online_network.parameters()).device.type == "cuda"
    assert next(trainer.target_network.parameters()).device.type == "cuda"
    assert not any(p.requires_grad for p in trainer.target_network.parameters())
    assert all(p.grad is None for p in trainer.target_network.parameters())
    trainer.train_step(data)
    for a, b in zip(trainer.online_network.parameters(), trainer.target_network.parameters()):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                    reason="BF16 CUDA non disponibile")
def test_cuda_optional_bfloat16_autocast():
    cfg = DQNConfig(use_bfloat16_amp=True)
    trainer = DQNTrainer(QNetwork(), config=cfg, device="cuda:0")
    result = trainer.train_step(_replay(4).sample(4))
    assert result.update_steps == 1 and math.isfinite(result.loss)
    assert next(trainer.online_network.parameters()).dtype == torch.float32


def test_dqn_diagnostics_are_pre_update_masked_and_single_forward():
    """Statistiche esatte su tre transizioni, senza influenzare Bellman/loss.

    SHOOT ha Q=800 ma e' illegale in ogni stato corrente. Nella prima
    transizione NORTH e PASS sono legali, con NORTH piu' vantaggioso.
    """
    online = TableQ((30., 2., -2., -3., 800., 20., -2.))
    batch = _batch()
    masks = batch.action_masks.clone()
    masks[0] = torch.as_tensor(NORTH_PASS_MASK)
    batch = replace(batch, action_masks=masks)
    trainer = DQNTrainer(online, config=DQNConfig(target_sync_interval=50))

    # Q(s,a) scelti = [20,-2,20]; target = [31,2,1], poiche' la
    # target network iniziale e' una copia dell'online (standard DQN).
    result = trainer.train_step(batch)
    assert trainer.online_network.calls == 1
    assert trainer.target_network.calls == 1
    chosen = torch.tensor([20., -2., 20.])
    targets = torch.tensor([31., 2., 1.])  # 1+max(30,20), 2, 3+(-2)
    abs_td = (targets - chosen).abs()
    max_legal = torch.tensor([30., -2., 20.])
    assert result.loss == pytest.approx(F.smooth_l1_loss(chosen, targets).item())
    assert result.mean_q == pytest.approx(chosen.mean().item())
    assert result.mean_target == pytest.approx(targets.mean().item())
    assert result.td_mae == pytest.approx(abs_td.mean().item())
    assert result.td_p90 == pytest.approx(torch.quantile(abs_td, 0.9).item())
    assert result.std_q == pytest.approx(chosen.std(unbiased=False).item())
    assert result.mean_max_legal_q == pytest.approx(max_legal.mean().item())
    assert result.std_max_legal_q == pytest.approx(max_legal.std(unbiased=False).item())
    assert result.mean_max_legal_q < 800.0  # Q illegale ignorato
    assert all(math.isfinite(getattr(result, field)) for field in (
        'loss', 'mean_q', 'mean_target', 'grad_norm', 'td_mae', 'td_p90',
        'std_q', 'mean_max_legal_q', 'std_max_legal_q',
    ))


def test_dqn_diagnostics_single_sample_have_zero_population_std():
    batch = _batch()
    one = replace(batch, **{
        name: getattr(batch, name)[:1].clone() for name in (
            'spatial', 'features', 'action_masks', 'actions', 'rewards',
            'next_spatial', 'next_features', 'next_action_masks', 'dones',
        )
    })
    trainer = DQNTrainer(TableQ(ONLINE_Q))
    result = trainer.train_step(one)
    assert result.std_q == 0.0
    assert result.std_max_legal_q == 0.0
    assert result.td_mae == pytest.approx(result.td_p90)
