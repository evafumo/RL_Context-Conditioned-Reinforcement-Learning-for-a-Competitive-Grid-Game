"""Campionamento del contesto episodico (mappa, policy avversaria).

Un episodio utilizza una sola coppia (m, k), scelta all'inizio con
q_C(m, k) = q_M(m) * q_O(k). Il modulo non possiede stato mutabile e non
modifica mappe, avversari, dinamica o generatore casuale globale.

La scelta dell'iniziativa e la generazione iniziale degli obiettivi sono
responsabilita' di game.py, non di questo modulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from random import Random
from typing import Final, Sequence

from .maps import GameMap, get_all_maps, get_map
from .opponents import OpponentKind, get_all_opponents, get_opponent_kind


MAP_IDS: Final[tuple[int, ...]] = tuple(
    game_map.map_id for game_map in get_all_maps()
)
OPPONENT_IDS: Final[tuple[int, ...]] = tuple(
    int(kind) for kind in get_all_opponents()
)


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """Coppia (m, k) immutabile, valida per l'intero episodio.

    game_map: la mappa della pool (con geometria e ostacoli fissi).
    opponent: identita' di una delle sei policy avversarie.
    """

    game_map: GameMap
    opponent: OpponentKind

    def __post_init__(self) -> None:
        if not isinstance(self.game_map, GameMap):
            raise ValueError("game_map deve essere una GameMap")
        try:
            expected_map = get_map(self.game_map.map_id)
        except ValueError as exc:
            raise ValueError("La mappa non appartiene alla pool") from exc
        if self.game_map != expected_map:
            raise ValueError("La mappa non corrisponde alla pool ufficiale")
        if type(self.opponent) is not OpponentKind:
            raise ValueError("opponent deve essere un OpponentKind")

    @property
    def map_id(self) -> int:
        """Identificativo della mappa (da 1 a 4)."""
        return self.game_map.map_id

    @property
    def opponent_id(self) -> int:
        """Identificativo della policy avversaria (da 1 a 6)."""
        return int(self.opponent)


@dataclass(frozen=True, slots=True)
class ContextDistribution:
    """Distribuzioni marginali q_M e q_O, indipendenti.

    q_maps[i] si riferisce a MAP_IDS[i]; q_opponents[j] si riferisce a
    OPPONENT_IDS[j]. Entrambe devono essere distribuzioni normalizzate.
    Sono ammesse probabilita' nulle, ma non negative o non finite.
    """

    q_maps: tuple[float, ...] = (0.25,) * 4
    q_opponents: tuple[float, ...] = (1.0 / 6.0,) * 6

    def __post_init__(self) -> None:
        maps = _validate_probabilities(self.q_maps, len(MAP_IDS), "q_maps")
        opponents = _validate_probabilities(
            self.q_opponents, len(OPPONENT_IDS), "q_opponents"
        )
        object.__setattr__(self, "q_maps", maps)
        object.__setattr__(self, "q_opponents", opponents)


def _validate_probabilities(
    values: Sequence[float], expected_count: int, name: str
) -> tuple[float, ...]:
    """Converte e controlla una distribuzione discreta di probabilita'."""
    try:
        values_tuple = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} deve essere una sequenza") from exc

    if len(values_tuple) != expected_count:
        raise ValueError(
            f"{name} richiede {expected_count} probabilita', "
            f"ricevute {len(values_tuple)}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
        for value in values_tuple
    ):
        raise ValueError(f"{name} ammette solo probabilita' finite e >= 0")

    probabilities = tuple(float(value) for value in values_tuple)
    if not isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"Le probabilita' di {name} devono sommare a 1")
    return probabilities


DEFAULT_CONTEXT_DISTRIBUTION: Final[ContextDistribution] = ContextDistribution()


def get_context(
    map_id: int, opponent_id: int | OpponentKind
) -> EpisodeContext:
    """Costruisce un contesto specifico (senza alcun campionamento).

    Utile per evaluation e test di tutte le 24 combinazioni.
    Genera ValueError per identificativi non appartenenti alle pool.
    """
    return EpisodeContext(
        game_map=get_map(map_id),
        opponent=get_opponent_kind(opponent_id),
    )


def get_all_contexts() -> tuple[EpisodeContext, ...]:
    """Restituisce le 24 coppie (m, k), ordinate prima per m poi per k."""
    return ALL_CONTEXTS


def _sample_index(probabilities: tuple[float, ...], rng: Random) -> int:
    """Estrae un indice categoriale tramite un singolo numero uniforme.

    Il fallback finale corregge esclusivamente l'arrotondamento numerico
    e non seleziona mai una categoria con probabilita' nulla.
    """
    u = rng.random()
    cumulative = 0.0
    last_positive = 0
    for index, probability in enumerate(probabilities):
        if probability > 0.0:
            last_positive = index
        cumulative += probability
        if u < cumulative:
            return index
    return last_positive


def sample_context(
    rng: Random,
    distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
) -> EpisodeContext:
    """Campiona indipendentemente m ~ q_M e k ~ q_O per un nuovo episodio.

    Usa due estrazioni indipendenti dal RNG fornito, in ordine mappa poi
    avversario. L'oggetto restituito deve essere conservato da game.py
    senza ricampionarlo durante le 100 attivazioni.
    """
    map_index = _sample_index(distribution.q_maps, rng)
    opponent_index = _sample_index(distribution.q_opponents, rng)
    return get_context(MAP_IDS[map_index], OPPONENT_IDS[opponent_index])


def context_probability(
    context: EpisodeContext,
    distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
) -> float:
    """Probabilita' teorica q_C(m, k) = q_M(m) * q_O(k)."""
    return (
        distribution.q_maps[MAP_IDS.index(context.map_id)]
        * distribution.q_opponents[OPPONENT_IDS.index(context.opponent_id)]
    )


ALL_CONTEXTS: Final[tuple[EpisodeContext, ...]] = tuple(
    get_context(map_id, opponent_id)
    for map_id in MAP_IDS
    for opponent_id in OPPONENT_IDS
)
