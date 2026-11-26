"""Q-network CNN + MLP per Context-Conditioned Masked DQN.

    observations.py: spatial (B,4,9,9), features (B,13), float32.
    action_mask.py: maschere booleane (B,7), True = azione legale.
    rules.py: sette azioni con ID 0..6, inclusa DUMMY = 6.

Questa rete condivide i parametri fra le quattro mappe e i sei avversari.
forward() restituisce ESCLUSIVAMENTE i sette Q-value GREZZI per stato:
non applica softmax, argmax o maschere e non implementa la loss DQN.

Il masking va applicato all'esterno con environment.action_mask.mask_q_values()
solo quando si sceglie un'azione o si calcola il massimo del target Bellman.
Il Q_online(s,a) della loss si estrae dai Q-value non mascherati.

Uso su GPU (inferenza centralizzata, ambienti su CPU):
    model = QNetwork().to('cuda')
    q_values = model(spatial_batch_cuda, feature_batch_cuda)

Input e parametri float32 sono raccomandati anche con torch.autocast.
Il modulo non inizializza CUDA all'importazione e non trasferisce dati
fra CPU e GPU all'interno di forward().
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from environment.action_mask import NUM_ACTIONS
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE

SPATIAL_EMBEDDING_SIZE: Final[int] = 128
VECTOR_EMBEDDING_SIZE: Final[int] = 64
SHARED_EMBEDDING_SIZE: Final[int] = 128


def _init_hidden_layer(layer: nn.Module) -> None:
    """Inizializza Conv2d e Linear nascosti in modo ortogonale (gain ReLU)."""
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SpatialEncoder(nn.Module):
    """CNN sui quattro piani 9x9; output (B,128).

    Tre convoluzioni 3x3 senza downsampling preservano le coordinate.
    Come nella versione PPO, il layer fully connected fonde le posizioni.
    """

    def __init__(self) -> None:
        super().__init__()
        channels, height, width = SPATIAL_SHAPE
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32 * height * width, SPATIAL_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.apply(_init_hidden_layer)

    def forward(self, spatial: Tensor) -> Tensor:
        """Riceve (B,4,9,9) e restituisce (B,128)."""
        return self.features(spatial)


class VectorEncoder(nn.Module):
    """MLP per punteggio, tempo, turno, mappa e avversario; output (B,64)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, VECTOR_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.apply(_init_hidden_layer)

    def forward(self, features: Tensor) -> Tensor:
        """Riceve (B,13) e restituisce (B,64)."""
        return self.features(features)


class SharedEncoder(nn.Module):
    """Concatena 128+64 feature e le fonde in un embedding (B,128)."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.vector_encoder = VectorEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(SPATIAL_EMBEDDING_SIZE + VECTOR_EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, SHARED_EMBEDDING_SIZE),
            nn.ReLU(),
        )
        self.fusion.apply(_init_hidden_layer)

    def forward(self, spatial: Tensor, features: Tensor) -> Tensor:
        """Riceve (B,4,9,9) e (B,13); restituisce (B,128)."""
        spatial_embedding = self.spatial_encoder(spatial)
        vector_embedding = self.vector_encoder(features)
        return self.fusion(torch.cat((spatial_embedding, vector_embedding), dim=-1))


class QNetwork(nn.Module):
    """Q_theta(s,a;m,k): un vettore di sette valori per osservazione.

    forward(spatial, features) -> Tensor (B,7), non mascherato.
    Il batch B deve essere >= 1. Il modello e gli input devono trovarsi
    sullo stesso device e avere lo stesso dtype floating point (float32
    inizialmente). La validazione non legge valori CUDA e non forza sync.

    L'ultima testa è lineare, senza sigmoid/tanh/softmax, poiché i ritorni
    attesi possono assumere anche valori negativi.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.q_head = nn.Linear(SHARED_EMBEDDING_SIZE, NUM_ACTIONS)
        # In DQN non si tratta di logits di policy PPO: gain=1 per la testa Q.
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def _validate_inputs(self, spatial: Tensor, features: Tensor) -> None:
        """Convalida tipi, batch, geometria, dtype e device senza sync GPU."""
        if not isinstance(spatial, Tensor) or not isinstance(features, Tensor):
            raise TypeError("spatial e features devono essere tensori PyTorch")
        if (
            spatial.ndim != 4
            or tuple(spatial.shape[1:]) != SPATIAL_SHAPE
            or features.ndim != 2
            or features.shape[1] != FEATURE_SIZE
            or spatial.shape[0] != features.shape[0]
            or spatial.shape[0] < 1
        ):
            raise ValueError(
                "Attesi spatial (B,4,9,9) e features (B,13), con B >= 1"
            )
        if not spatial.is_floating_point() or not features.is_floating_point():
            raise TypeError("spatial e features devono essere floating point")
        if spatial.dtype != features.dtype:
            raise TypeError("spatial e features devono avere lo stesso dtype")
        if spatial.device != features.device:
            raise ValueError("spatial e features devono stare sullo stesso device")
        reference_parameter = self.q_head.weight
        if spatial.device != reference_parameter.device:
            raise ValueError(
                "Gli input devono stare sul device del modello: usare model.to(device) "
                "e to_torch_batch(..., device=device)"
            )
        if spatial.dtype != reference_parameter.dtype:
            raise TypeError(
                "Il dtype degli input deve coincidere con quello dei parametri "
                "(float32 consigliato, anche con autocast)"
            )

    def forward(self, spatial: Tensor, features: Tensor) -> Tensor:
        """Calcola Q non mascherati, shape (B,7), con gradienti se abilitati."""
        self._validate_inputs(spatial, features)
        embedding = self.encoder(spatial, features)
        return self.q_head(embedding)


__all__ = [
    "SpatialEncoder",
    "VectorEncoder",
    "SharedEncoder",
    "QNetwork",
    "SPATIAL_EMBEDDING_SIZE",
    "VECTOR_EMBEDDING_SIZE",
    "SHARED_EMBEDDING_SIZE",
]
