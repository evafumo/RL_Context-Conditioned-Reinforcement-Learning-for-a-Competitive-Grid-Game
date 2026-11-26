"""Codifica delle osservazioni neurali MDP e POMDP (CNN + MLP).

Nel POMDP lo stato del gioco X_t e la mappa m sono visibili, mentre
l'identita' K della policy avversaria rimane nascosta. Le ultime sei feature
contengono il posterior bayesiano b_t(k) anziche' la one-hot della vera K.

Convenzioni condivise:
    spatial: float32, (4,9,9), canali ostacoli/Agent/Opponent/obiettivi.
    features: float32, (13,)
      0       : differenziale VP / 800
      1       : (activation - 1) / 99
      2       : indicatore del turno Agent
      3:7     : one-hot della mappa nota (map ID 1..4)
      7:13    : POMDP -> belief sulle 6 policy (ordine OPPONENT_IDS)
                MDP   -> one-hot dell'identita' avversaria (compatibilita')

Interfacce:
    encode_belief_observation(state, game_map, belief)
        Interfaccia POMDP preferita: NON riceve EpisodeContext ne' K.
    encode_pomdp_observation(pomdp_observation)
        Per l'oggetto pubblico ritornato da POMDPGame.observe()/reset().
    encode_observation(state, context, belief=...)
        Compatibilita': con belief esplicito codifica il POMDP ignorando K;
        SENZA belief conserva la codifica MDP originale con one-hot di K.
        Non usare la forma senza belief nel training/evaluation POMDP.

Tutte le funzioni sono pure e non avanzano RNG, non cambiano lo stato del
simulatore e non aggiornano il belief (compito di environment/belief.py).
Il terminale s_dagger non ha un'osservazione neurale: usare terminated.
Le coordinate (x,y) sono 1-based, y verso nord; negli array la riga e' 9-y.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final, Sequence

import numpy as np

from .context import MAP_IDS, OPPONENT_IDS, EpisodeContext
from .game import HORIZON, GameState
from .maps import GameMap, START_CELLS
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
    from .pomdp import POMDPObservation


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
BELIEF_FEATURE_OFFSET: Final[int] = OPPONENT_FEATURE_OFFSET
BELIEF_SIZE: Final[int] = len(OPPONENT_IDS)
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


def _validate_observable_state(state: GameState, game_map: GameMap) -> None:
    """Rifiuta stati incoerenti, senza alterare il simulatore."""
    free = game_map.free_cells
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


def _encode_visible_state(
    state: GameState, game_map: GameMap
) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce i quattro piani e le prime sette feature, senza accedere a K."""
    if not isinstance(state, GameState):
        raise TypeError("state deve essere un GameState non terminale")
    if not isinstance(game_map, GameMap):
        raise TypeError("game_map deve essere una GameMap")
    if game_map.map_id not in _MAP_TO_INDEX:
        raise ValueError("La mappa non appartiene alla pool delle mappe")
    _validate_observable_state(state, game_map)

    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)
    for position in game_map.obstacles:
        spatial[SpatialChannel.OBSTACLES, *_array_indices(position)] = 1.0
    spatial[SpatialChannel.AGENT, *_array_indices(state.agent_position)] = 1.0
    spatial[SpatialChannel.OPPONENT, *_array_indices(state.opponent_position)] = 1.0
    for position in state.active_objectives:
        spatial[SpatialChannel.OBJECTIVES, *_array_indices(position)] = 1.0

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features[0] = state.score_differential / MAX_ABS_SCORE_DIFFERENTIAL
    features[1] = (state.activation - 1) / (HORIZON - 1)
    features[2] = float(state.player_to_act is Player.AGENT)
    features[MAP_FEATURE_OFFSET + _MAP_TO_INDEX[game_map.map_id]] = 1.0
    return spatial, features


def _validated_belief(belief: Sequence[float] | np.ndarray) -> np.ndarray:
    """Valida b_t come distribuzione sulle policy nell'ordine OPPONENT_IDS.

    La tolleranza 1e-6 ammette l'arrotondamento del belief a float32;
    internamente normalizziamo in float64 prima della codifica float32.
    Non viene mai utilizzata l'identita' reale della policy.
    """
    try:
        raw = np.asarray(belief)
        if raw.dtype.kind not in ("i", "u", "f"):
            raise ValueError("belief deve contenere soltanto numeri reali")
        vector = np.asarray(belief, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("belief deve essere una sequenza numerica") from exc
    if vector.shape != (BELIEF_SIZE,):
        raise ValueError(
            f"belief deve avere shape ({BELIEF_SIZE},), ricevuta {vector.shape}"
        )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("belief deve contenere probabilita' finite e non negative")
    total = float(vector.sum(dtype=np.float64))
    if not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("La somma delle probabilita' del belief deve valere 1")
    return vector / total


def encode_belief_observation(
    state: GameState,
    game_map: GameMap,
    belief: Sequence[float] | np.ndarray,
) -> Observation:
    """Codifica (X_t, m, b_t) senza passare l'identita' vera dell'Opponent.

    ``belief`` deve essere il posterior aggiornato PRIMA dell'attivazione
    corrente; viene fornito da POMDPGame/BayesianOpponentBelief e NON
    calcolato qui. Ordine: Random, Aggressive, Hunter, Hybrid, Shooter, ASH.

    Non esiste un default uniforme: questo impedisce di perdere per errore
    l'informazione accumulata durante l'episodio.
    """
    probabilities = _validated_belief(belief)
    spatial, features = _encode_visible_state(state, game_map)
    features[BELIEF_FEATURE_OFFSET:] = probabilities.astype(np.float32)
    return Observation(spatial=spatial, features=features)


def encode_pomdp_observation(observation: POMDPObservation) -> Observation:
    """Codifica l'osservazione pubblica restituita da POMDPGame.

    Per sicurezza richiede il tipo POMDPObservation; l'oggetto non espone K.
    L'eventuale ultima azione osservata e' gia' riassunta nel posterior.
    """
    # Import lazy: il vecchio MDP resta utilizzabile anche senza pomdp.py.
    from .pomdp import POMDPObservation

    if not isinstance(observation, POMDPObservation):
        raise TypeError("observation deve essere una POMDPObservation")
    return encode_belief_observation(
        observation.state, observation.game_map, observation.belief
    )


def encode_observation(
    state: GameState,
    context: EpisodeContext,
    *,
    belief: Sequence[float] | np.ndarray | None = None,
) -> Observation:
    """Codifica l'osservazione originale MDP o, con belief, quella POMDP.

    Compatibilita' con il progetto originale:
        encode_observation(state, context)       -> one-hot della vera K (MDP)

    Nel POMDP:
        encode_observation(state, context, belief=b_t)
    oppure, preferibilmente (senza passare K):
        encode_belief_observation(state, game_map, b_t)
        encode_pomdp_observation(pomdp_game.observe())

    Se il belief e' presente, ``context.opponent_id`` non viene letto.
    Non chiamare questa funzione sullo stato terminale.
    """
    if not isinstance(state, GameState):
        raise TypeError("state deve essere un GameState non terminale")
    if not isinstance(context, EpisodeContext):
        raise TypeError("context deve essere un EpisodeContext")

    if belief is not None:
        return encode_belief_observation(state, context.game_map, belief)

    # Solo percorso storico MDP: non usare in training/evaluation POMDP.
    spatial, features = _encode_visible_state(state, context.game_map)
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
