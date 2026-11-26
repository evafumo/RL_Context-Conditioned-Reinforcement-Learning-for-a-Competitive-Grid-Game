"""Codifica neurale delle osservazioni del Context-Conditioned MDP.

Questo modulo e' un adattatore puro tra GameState + EpisodeContext e la
futura architettura CNN + MLP. NON modifica lo stato, NON estrae numeri
casuali e NON calcola le azioni legali (responsabilita' di action_mask.py).

Convenzioni:
    spatial: float32, shape (4, 9, 9), ordine [canale, riga, colonna]
        0: ostacoli; 1: Agent; 2: Opponent; 3: obiettivi attivi.
    features: float32, shape (13,)
        0: D / 800
        1: (activation - 1) / 99
        2: 1.0 se e' il turno Agent, 0.0 altrimenti
        3:7: one-hot mappa (ID 1..4)
        7:13: one-hot Opponent (ID 1..6)

Le coordinate della griglia partono da (1,1) in basso a sinistra. Per la CNN,
la prima riga dell'array e' y=9: (x, y) -> [9-y, x-1].

Il terminale s_dagger NON ha un'osservazione standard: viene rappresentato
dal flag terminated di StepResult, non da una griglia inventata.

Per utilizzare CUDA, codificare inizialmente su CPU, raggruppare piu'
osservazioni e trasferire il batch in un'unica operazione. PyTorch viene
importato soltanto quando e' chiamato to_torch_batch(), cosi' il simulatore
resta utilizzabile indipendentemente dall'installazione di torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final, Sequence

import numpy as np

from .context import MAP_IDS, OPPONENT_IDS, EpisodeContext
from .game import HORIZON, GameState
from .maps import START_CELLS
from .rules import (
    GRID_SIZE,
    KILLER_VP,
    OBJECTIVE_VP,
    VICTIM_VP,
    Player,
    Position,
)

if TYPE_CHECKING:
    import torch


class SpatialChannel(IntEnum):
    """Indici costanti dei quattro piani per la CNN."""

    OBSTACLES = 0
    AGENT = 1
    OPPONENT = 2
    OBJECTIVES = 3


SPATIAL_SHAPE: Final[tuple[int, int, int]] = (
    len(SpatialChannel), GRID_SIZE, GRID_SIZE
)
MAP_FEATURE_OFFSET: Final[int] = 3
OPPONENT_FEATURE_OFFSET: Final[int] = MAP_FEATURE_OFFSET + len(MAP_IDS)
FEATURE_SIZE: Final[int] = OPPONENT_FEATURE_OFFSET + len(OPPONENT_IDS)

# Per ogni singola attivazione: al piu' una cattura (+/-1) e una morte
# (+/-7 sul differenziale). E' un bound conservativo per 100 attivazioni.
MAX_ABS_SCORE_DIFFERENTIAL: Final[int] = (
    HORIZON * (OBJECTIVE_VP + KILLER_VP - VICTIM_VP)
)

_MAP_TO_INDEX: Final[dict[int, int]] = {
    map_id: index for index, map_id in enumerate(MAP_IDS)
}
_OPPONENT_TO_INDEX: Final[dict[int, int]] = {
    opponent_id: index for index, opponent_id in enumerate(OPPONENT_IDS)
}


@dataclass(frozen=True, slots=True)
class Observation:
    """Osservazione di uno stato NON terminale, prodotta come array NumPy.

    Gli array restituiti sono nuovi e indipendenti dallo stato originale.
    `frozen=True` impedisce la riassegnazione dei campi, non la modifica
    esplicita degli elementi degli array NumPy.
    """

    spatial: np.ndarray  # (4, 9, 9), float32
    features: np.ndarray  # (13,), float32


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Batch CPU pronto per l'eventuale trasferimento su CUDA."""

    spatial: np.ndarray  # (B, 4, 9, 9), float32
    features: np.ndarray  # (B, 13), float32


@dataclass(frozen=True, slots=True)
class TorchObservationBatch:
    """Batch PyTorch sul device scelto (per esempio 'cuda')."""

    spatial: torch.Tensor
    features: torch.Tensor


def _array_indices(position: Position) -> tuple[int, int]:
    """Da coordinate (x,y) 1-based, con nord in alto, a (riga,colonna)."""
    x, y = position
    return GRID_SIZE - y, x - 1


def _validate_observable_state(state: GameState, context: EpisodeContext) -> None:
    """Rifiuta stati incoerenti, senza alterare il simulatore."""
    free = context.game_map.free_cells
    if state.agent_position not in free:
        raise ValueError("Posizione Agent non attraversabile nella mappa corrente")
    if state.opponent_position not in free:
        raise ValueError("Posizione Opponent non attraversabile nella mappa corrente")
    if state.agent_position == state.opponent_position:
        raise ValueError("Agent e Opponent non possono occupare la stessa cella")

    objectives = state.active_objectives
    if not 1 <= len(objectives) <= 3:
        raise ValueError("Uno stato non terminale deve avere da 1 a 3 obiettivi")
    if not objectives.issubset(free.difference(
        START_CELLS, (state.agent_position, state.opponent_position)
    )):
        raise ValueError("Obiettivi su celle vietate o occupate")
    if abs(state.score_differential) > MAX_ABS_SCORE_DIFFERENTIAL:
        raise ValueError("Differenziale VP fuori dal limite delle 100 attivazioni")


def encode_observation(state: GameState, context: EpisodeContext) -> Observation:
    """Codifica (S_t,m,k) senza perdere le informazioni dell'MDP non terminale.

    Args:
        state: GameState prima dell'attivazione (t=1,...,100).
        context: EpisodeContext fissato al reset (mappa e Opponent).

    Returns:
        Observation(spatial, features), con array C-contiguous float32.

    Raises:
        TypeError: per tipi sbagliati, incluso lo stato terminale s_dagger.
        ValueError: per stato incompatibile con la geometria o le regole.

    Il flag terminated di StepResult segnala lo stato assorbente: NON
    chiamare questa funzione con TERMINAL_STATE.
    """
    if not isinstance(state, GameState):
        raise TypeError("state deve essere un GameState non terminale")
    if not isinstance(context, EpisodeContext):
        raise TypeError("context deve essere un EpisodeContext")
    _validate_observable_state(state, context)

    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)

    for position in context.game_map.obstacles:
        spatial[SpatialChannel.OBSTACLES, *_array_indices(position)] = 1.0

    spatial[SpatialChannel.AGENT, *_array_indices(state.agent_position)] = 1.0
    spatial[SpatialChannel.OPPONENT, *_array_indices(state.opponent_position)] = 1.0
    for position in state.active_objectives:
        spatial[SpatialChannel.OBJECTIVES, *_array_indices(position)] = 1.0

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features[0] = state.score_differential / MAX_ABS_SCORE_DIFFERENTIAL
    features[1] = (state.activation - 1) / (HORIZON - 1)
    features[2] = float(state.player_to_act is Player.AGENT)
    features[MAP_FEATURE_OFFSET + _MAP_TO_INDEX[context.map_id]] = 1.0
    features[OPPONENT_FEATURE_OFFSET + _OPPONENT_TO_INDEX[context.opponent_id]] = 1.0

    return Observation(spatial=spatial, features=features)


def stack_observations(observations: Sequence[Observation]) -> ObservationBatch:
    """Raggruppa B osservazioni in un singolo batch NumPy C-contiguous.

    Il raggruppamento su CPU evita B piccoli trasferimenti CUDA. Il batch
    e' pronto per una CNN: [B,C,H,W] e un MLP: [B,FEATURE_SIZE].
    """
    if not observations:
        raise ValueError("Occorre almeno un'osservazione per creare un batch")
    for index, observation in enumerate(observations):
        if not isinstance(observation, Observation):
            raise TypeError(f"Elemento {index} non e' una Observation")
        if (
            observation.spatial.shape != SPATIAL_SHAPE
            or observation.features.shape != (FEATURE_SIZE,)
            or observation.spatial.dtype != np.float32
            or observation.features.dtype != np.float32
        ):
            raise ValueError(f"Shape o dtype non valido nell'osservazione {index}")

    spatial = np.stack([obs.spatial for obs in observations], axis=0)
    features = np.stack([obs.features for obs in observations], axis=0)
    return ObservationBatch(spatial=spatial, features=features)


def to_torch_batch(
    batch: ObservationBatch,
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> TorchObservationBatch:
    """Converte e trasferisce un intero batch in PyTorch, anche su GPU CUDA.

    PyTorch e' importato in maniera lazy; fino a qui la simulazione usa solo
    NumPy e non crea contesti CUDA. torch.from_numpy evita copie CPU inutili.

    Per la RTX 5060 Ti, se torch e CUDA sono correttamente configurati:
        gpu_batch = to_torch_batch(
            stack_observations(observations),
            device="cuda",
            pin_memory=True,
            non_blocking=True,
        )

    Una copia asincrona H2D richiede memoria CPU pinned; l'eventuale beneficio
    va misurato sulle dimensioni reali del batch. Non trasferire una singola
    osservazione ad ogni attivazione se e' possibile raggrupparne molte.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Installare PyTorch per usare to_torch_batch()") from exc

    if not isinstance(batch, ObservationBatch):
        raise TypeError("batch deve essere un ObservationBatch")
    if (
        batch.spatial.ndim != 4
        or batch.spatial.shape[1:] != SPATIAL_SHAPE
        or batch.features.shape != (batch.spatial.shape[0], FEATURE_SIZE)
        or batch.spatial.dtype != np.float32
        or batch.features.dtype != np.float32
    ):
        raise ValueError("Il batch deve avere shape (B,4,9,9), (B,13) e dtype float32")
    if batch.spatial.shape[0] == 0:
        raise ValueError("Un batch di dimensione zero non e' ammesso")

    cpu_spatial = torch.from_numpy(np.ascontiguousarray(batch.spatial))
    cpu_features = torch.from_numpy(np.ascontiguousarray(batch.features))
    if pin_memory:
        if not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True richiede PyTorch con CUDA disponibile")
        cpu_spatial = cpu_spatial.pin_memory()
        cpu_features = cpu_features.pin_memory()

    return TorchObservationBatch(
        spatial=cpu_spatial.to(device=device, non_blocking=non_blocking),
        features=cpu_features.to(device=device, non_blocking=non_blocking),
    )
