"""Test della Q-Network della Fase 3.2 (DQN standard, non Double DQN).

Eseguire dalla radice Giulio2: python -m pytest tests/test_network.py -q
Il test CUDA si attiva soltanto se PyTorch trova una GPU utilizzabile.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
from torch import nn

from agent.network import (
    SHARED_EMBEDDING_SIZE,
    SPATIAL_EMBEDDING_SIZE,
    VECTOR_EMBEDDING_SIZE,
    QNetwork,
)
from environment.action_mask import (
    NUM_ACTIONS,
    legal_action_mask,
    legal_actions_to_mask,
    mask_q_values,
    stack_action_masks,
    to_torch_action_masks,
)
from environment.context import get_all_contexts, get_context
from environment.game import GridGame
from environment.observations import (
    FEATURE_SIZE,
    SPATIAL_SHAPE,
    encode_observation,
    stack_observations,
    to_torch_batch,
)
from environment.rules import Action, Player


@pytest.fixture(scope="module")
def model() -> QNetwork:
    # I test di sola lettura riutilizzano la stessa rete per essere più rapidi.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261026)
        return QNetwork().eval()


def _batch(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.zeros((batch_size, *SPATIAL_SHAPE), dtype=torch.float32)
    features = torch.zeros((batch_size, FEATURE_SIZE), dtype=torch.float32)
    spatial[:, 1, 8, 4] = 1.0  # posizione Agent nell'input di prova
    features[:, 3] = 1.0       # una mappa
    features[:, 7] = 1.0       # un avversario
    return spatial, features


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_forward_shapes_finite_float32_cpu(model: QNetwork, batch_size: int) -> None:
    spatial, features = _batch(batch_size)
    with torch.no_grad():
        q_values = model(spatial, features)
    assert q_values.shape == (batch_size, NUM_ACTIONS)
    assert q_values.dtype == torch.float32
    assert q_values.device.type == "cpu"
    assert torch.isfinite(q_values).all()
    assert not q_values.requires_grad


def test_encoder_dimensions_and_q_head(model: QNetwork) -> None:
    spatial, features = _batch(3)
    with torch.no_grad():
        spatial_embedding = model.encoder.spatial_encoder(spatial)
        vector_embedding = model.encoder.vector_encoder(features)
        embedding = model.encoder(spatial, features)
        q_values = model(spatial, features)
    assert spatial_embedding.shape == (3, SPATIAL_EMBEDDING_SIZE)
    assert vector_embedding.shape == (3, VECTOR_EMBEDDING_SIZE)
    assert embedding.shape == (3, SHARED_EMBEDDING_SIZE)
    assert q_values.shape == (3, NUM_ACTIONS)
    assert isinstance(model.q_head, nn.Linear)
    assert model.q_head.out_features == NUM_ACTIONS
    assert not any(isinstance(module, (nn.Softmax, nn.Sigmoid, nn.Tanh))
                   for module in model.q_head.modules())


def test_forward_does_not_apply_action_mask(model: QNetwork) -> None:
    spatial, features = _batch(2)
    with torch.no_grad():
        q_values = model(spatial, features)
    mask = torch.from_numpy(stack_action_masks([
        legal_actions_to_mask([Action.PASS]),
        legal_actions_to_mask([Action.DUMMY]),
    ]))
    original = q_values.clone()
    masked = mask_q_values(q_values, mask)
    torch.testing.assert_close(q_values, original)
    assert torch.isfinite(q_values).all()  # anche gli ID illegali hanno un Q grezzo
    assert torch.isneginf(masked[~mask]).all()
    torch.testing.assert_close(masked[mask], q_values[mask])
    assert masked.argmax(dim=1).tolist() == [int(Action.PASS), int(Action.DUMMY)]


def test_batch_matches_individual_evaluation(model: QNetwork) -> None:
    spatial, features = _batch(4)
    spatial[1, 2, 0, 4] = 1.0
    features[2, 1] = 0.8
    features[3, 3] = 0.0
    features[3, 4] = 1.0
    with torch.no_grad():
        q_batch = model(spatial, features)
        q_singles = torch.cat([
            model(spatial[i:i + 1], features[i:i + 1])
            for i in range(spatial.shape[0])
        ])
    torch.testing.assert_close(q_batch, q_singles, rtol=1e-5, atol=1e-6)


def test_q_head_can_output_negative_values(model: QNetwork) -> None:
    """Verifica l'uscita di forward(), non solo quella di nn.Linear."""
    spatial, features = _batch(2)
    with torch.no_grad():
        original_weight = model.q_head.weight.clone()
        original_bias = model.q_head.bias.clone()
        try:
            model.q_head.weight.zero_()
            model.q_head.bias.fill_(-2.0)
            q_values = model(spatial, features)
            assert q_values.shape == (2, NUM_ACTIONS)
            assert torch.all(q_values == -2.0)
        finally:
            # Il modello della fixture è condiviso fra test di sola lettura.
            model.q_head.weight.copy_(original_weight)
            model.q_head.bias.copy_(original_bias)


def test_gradients_and_adam_update() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        model = QNetwork().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    spatial, features = _batch(4)
    spatial[0, 0, 2, 2] = 1.0
    features[1, 0] = 0.2
    actions = torch.tensor([0, 5, 6, 3], dtype=torch.long)
    targets = torch.tensor([-1.5, 0.0, 2.0, -0.2], dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    q = model(spatial, features)
    # Q_online(s,a) si estrae SENZA action masking; importante per DQN.
    selected_q = q.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = nn.functional.smooth_l1_loss(selected_q, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.encoder.spatial_encoder.features[0].weight.grad.abs().sum() > 0
    assert model.encoder.vector_encoder.features[0].weight.grad.abs().sum() > 0
    assert model.q_head.weight.grad.abs().sum() > 0
    old_weight = model.q_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.q_head.weight, old_weight)


def test_state_dict_roundtrip(model: QNetwork) -> None:
    spatial, features = _batch(2)
    with torch.no_grad():
        expected = model(spatial, features)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    assert state
    restored = copy.deepcopy(model)
    with torch.no_grad():
        restored.q_head.bias.add_(1.0)
        assert not torch.allclose(restored(spatial, features), expected)
        restored.load_state_dict(state, strict=True)
        actual = restored(spatial, features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initialization_finite_and_zero_bias(model: QNetwork) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            assert torch.isfinite(module.weight).all()
            assert torch.count_nonzero(module.weight) > 0
            assert module.bias is not None
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))


@pytest.mark.parametrize("invalid", [None, 123, "tensor", np.zeros((1, 4, 9, 9))])
def test_rejects_non_tensor_spatial(model: QNetwork, invalid: object) -> None:
    _, features = _batch(1)
    with pytest.raises(TypeError):
        model(invalid, features)


def test_rejects_non_tensor_features(model: QNetwork) -> None:
    spatial, _ = _batch(1)
    with pytest.raises(TypeError):
        model(spatial, [[0.0] * FEATURE_SIZE])


@pytest.mark.parametrize("bad_spatial", [(4, 9, 9), (1, 3, 9, 9), (1, 4, 8, 9),
                                         (1, 4, 9, 8), (0, 4, 9, 9)])
def test_rejects_wrong_spatial_shape(model: QNetwork, bad_spatial: tuple[int, ...]) -> None:
    _, features = _batch(1)
    with pytest.raises(ValueError, match="spatial"):
        model(torch.zeros(bad_spatial), features)


@pytest.mark.parametrize("bad_features", [(13,), (1, 12), (1, 14), (2, 13)])
def test_rejects_wrong_feature_shape(model: QNetwork, bad_features: tuple[int, ...]) -> None:
    spatial, _ = _batch(1)
    with pytest.raises(ValueError, match="features"):
        model(spatial, torch.zeros(bad_features))


def test_rejects_integer_features(model: QNetwork) -> None:
    spatial, features = _batch(1)
    with pytest.raises(TypeError, match="floating"):
        model(spatial, features.long())


def test_rejects_integer_spatial(model: QNetwork) -> None:
    spatial, features = _batch(1)
    with pytest.raises(TypeError, match="floating"):
        model(spatial.long(), features)


def test_rejects_mismatched_input_dtypes(model: QNetwork) -> None:
    spatial, features = _batch(1)
    with pytest.raises(TypeError, match="dtype"):
        model(spatial.double(), features)


def test_rejects_dtype_not_matching_model(model: QNetwork) -> None:
    spatial, features = _batch(1)
    with pytest.raises(TypeError, match="dtype"):
        model(spatial.double(), features.double())


def test_all_24_real_contexts_and_legal_action_masking(model: QNetwork) -> None:
    """Un batch delle 24 combinazioni (mappa, policy) con stati veri."""
    observations = []
    masks = []
    legal_action_sets = []
    contexts = get_all_contexts()
    assert len(contexts) == 24

    for i, context in enumerate(contexts):
        game = GridGame(seed=2026 + i)
        game.reset(seed=2026 + i, context=context, starting_player=Player.AGENT)
        observations.append(encode_observation(game.state, game.context))
        mask = legal_action_mask(game)
        masks.append(mask)
        legal_action_sets.append({int(action) for action in game.legal_actions()})
        assert mask.dtype == np.bool_

    batch = to_torch_batch(stack_observations(observations), device="cpu")
    action_masks = to_torch_action_masks(stack_action_masks(masks), device="cpu")
    with torch.no_grad():
        q_values = model(batch.spatial, batch.features)
        selected = mask_q_values(q_values, action_masks).argmax(dim=-1)
    assert q_values.shape == (24, NUM_ACTIONS)
    assert action_masks.shape == (24, NUM_ACTIONS)
    for index, action_id in enumerate(selected.tolist()):
        assert action_id in legal_action_sets[index]

    # La one-hot mappa + avversario resta nel vettore delle 13 feature.
    assert len({(c.map_id, c.opponent_id) for c in contexts}) == 24
    assert torch.all(batch.features[:, 3:7].sum(dim=-1) == 1)
    assert torch.all(batch.features[:, 7:13].sum(dim=-1) == 1)


def test_forced_turn_uses_only_dummy(model: QNetwork) -> None:
    game = GridGame(seed=9)
    game.reset(seed=9, context=get_context(1, 1), starting_player=Player.OPPONENT)
    assert game.legal_actions() == (Action.DUMMY,)
    observation = encode_observation(game.state, game.context)
    batch = to_torch_batch(stack_observations([observation]))
    mask = to_torch_action_masks(legal_action_mask(game))
    with torch.no_grad():
        q = model(batch.spatial, batch.features)
        selected = mask_q_values(q.squeeze(0), mask).argmax().item()
    assert selected == int(Action.DUMMY)
    assert game.state.player_to_act is Player.OPPONENT


def test_network_does_not_modify_game(model: QNetwork) -> None:
    game = GridGame(seed=42)
    game.reset(seed=42, context=get_context(2, 5), starting_player=Player.AGENT)
    before_state = game.state
    before_context = game.context
    before_actions = game.legal_actions()
    obs = encode_observation(game.state, game.context)
    batch = to_torch_batch(stack_observations([obs]))
    with torch.no_grad():
        _ = model(batch.spatial, batch.features)
    assert game.state == before_state
    assert game.context == before_context
    assert game.legal_actions() == before_actions
    assert not game.terminated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
def test_cuda_batched_forward_backward_and_masking() -> None:
    device = torch.device("cuda:0")
    net = QNetwork().to(device).train()
    examples = []
    masks = []
    for i, context in enumerate(get_all_contexts()):
        game = GridGame(seed=800 + i)
        game.reset(seed=800 + i, context=context,
                   starting_player=Player.AGENT if i % 2 == 0 else Player.OPPONENT)
        examples.append(encode_observation(game.state, game.context))
        masks.append(legal_action_mask(game))

    batch = to_torch_batch(stack_observations(examples), device=device,
                           pin_memory=True, non_blocking=True)
    action_masks = to_torch_action_masks(stack_action_masks(masks), device=device,
                                         pin_memory=True, non_blocking=True)
    q = net(batch.spatial, batch.features)
    assert q.shape == (24, NUM_ACTIONS)
    assert q.device == device and q.dtype == torch.float32
    assert torch.isfinite(q).all()
    actions = mask_q_values(q, action_masks).argmax(dim=-1)
    assert bool(action_masks.gather(1, actions[:, None]).all())
    selected = q.gather(1, actions[:, None]).squeeze(1)
    loss = (selected.square()).mean()
    loss.backward()
    assert net.q_head.weight.grad is not None
    assert torch.isfinite(net.q_head.weight.grad).all()

    # Il supporto BF16 dipende da installazione PyTorch/driver; opzionale.
    if torch.cuda.is_bf16_supported():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q_amp = net(batch.spatial, batch.features)
        assert q_amp.shape == (24, NUM_ACTIONS)
        assert torch.isfinite(q_amp.float()).all()
