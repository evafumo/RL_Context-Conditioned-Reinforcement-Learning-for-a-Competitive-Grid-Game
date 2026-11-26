"""Pool di mappe fisse 9x9 per il gioco competitivo.

Coordinate cartesiane a base 1: x aumenta verso est, y verso nord.
Le mappe contengono solo geometria statica; il campionamento dell'episodio
spetta a context.py, mentre obiettivi e posizioni dinamiche spettano a game.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping

from .rules import (
    AGENT_START,
    GRID_SIZE,
    OPPONENT_START,
    Position,
    is_inside_grid,
)


ALL_CELLS: Final[frozenset[Position]] = frozenset(
    (x, y)
    for x in range(1, GRID_SIZE + 1)
    for y in range(1, GRID_SIZE + 1)
)
START_CELLS: Final[frozenset[Position]] = frozenset(
    (AGENT_START, OPPONENT_START)
)


@dataclass(frozen=True, slots=True)
class GameMap:
    """Mappa immutabile, validata alla creazione.

    `free_cells` è calcolato una sola volta e riutilizzabile nei moduli futuri.
    """

    map_id: int
    obstacles: frozenset[Position]
    free_cells: frozenset[Position] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.map_id) is not int or self.map_id < 1:
            raise ValueError("map_id deve essere un intero positivo")

        # Copia difensiva: anche se viene passato un set, la mappa resta immutabile.
        obstacles = frozenset(self.obstacles)
        for position in obstacles:
            if (
                not isinstance(position, tuple)
                or len(position) != 2
                or any(type(coord) is not int for coord in position)
                or not is_inside_grid(position)
            ):
                raise ValueError(f"Ostacolo con coordinate non valide: {position!r}")

        if obstacles & START_CELLS:
            raise ValueError("Le due posizioni iniziali devono essere libere")

        free_cells = ALL_CELLS - obstacles
        if len(free_cells - START_CELLS) < 5:
            raise ValueError(
                "La mappa richiede almeno cinque celle attraversabili "
                "oltre alle posizioni iniziali"
            )

        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "free_cells", free_cells)


def is_traversable(game_map: GameMap, position: Position) -> bool:
    """True solo per celle dentro la griglia e prive di ostacoli."""
    return position in game_map.free_cells


def get_map(map_id: int) -> GameMap:
    """Restituisce una delle quattro mappe tramite il suo identificativo."""
    if type(map_id) is not int:
        raise ValueError(f"map_id deve essere un intero: {map_id!r}")
    try:
        return MAP_POOL[map_id]
    except KeyError as exc:
        raise ValueError(f"Mappa sconosciuta: {map_id!r}. ID validi: 1, 2, 3, 4") from exc


def get_all_maps() -> tuple[GameMap, ...]:
    """Restituisce l'intera pool in ordine deterministico (1, 2, 3, 4)."""
    return tuple(MAP_POOL[map_id] for map_id in sorted(MAP_POOL))


def render_map(
    game_map: GameMap,
    agent_position: Position = AGENT_START,
    opponent_position: Position = OPPONENT_START,
) -> str:
    """Visualizza la mappa nel terminale, con y=9 in alto e x=1 a sinistra.

    Legenda: A = Agent, O = Opponent, # = ostacolo, . = cella libera.
    Utile per il debug; non viene usata nelle transizioni dell'MDP.
    """
    if agent_position == opponent_position:
        raise ValueError("I giocatori non possono occupare la stessa casella")
    if not is_traversable(game_map, agent_position):
        raise ValueError("Posizione dell'Agent non attraversabile")
    if not is_traversable(game_map, opponent_position):
        raise ValueError("Posizione dell'Opponent non attraversabile")

    lines = ["   " + " ".join(str(x) for x in range(1, GRID_SIZE + 1))]
    for y in range(GRID_SIZE, 0, -1):
        symbols: list[str] = []
        for x in range(1, GRID_SIZE + 1):
            position = (x, y)
            if position == agent_position:
                symbols.append("A")
            elif position == opponent_position:
                symbols.append("O")
            elif position in game_map.obstacles:
                symbols.append("#")
            else:
                symbols.append(".")
        lines.append(f"{y:>2} " + " ".join(symbols))
    return "\n".join(lines)


MAP_POOL: Final[Mapping[int, GameMap]] = MappingProxyType(
    {
        1: GameMap(
            map_id=1,
            obstacles=frozenset(
                {(4, 5), (5, 5), (6, 5), (4, 6), (5, 6), (7, 6)}
            ),
        ),
        2: GameMap(
            map_id=2,
            obstacles=frozenset(
                {(4, 3), (5, 3), (6, 3), (4, 7), (5, 7), (6, 7)}
            ),
        ),
        3: GameMap(
            map_id=3,
            obstacles=frozenset(
                {(2, 6), (3, 6), (4, 6), (7, 3), (8, 3), (9, 3)}
            ),
        ),
        4: GameMap(
            map_id=4,
            obstacles=frozenset(
                {(2, 4), (2, 5), (2, 6), (5, 5), (8, 4), (8, 5), (8, 6)}
            ),
        ),
    }
)
