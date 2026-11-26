"""Generazione, conquista e rigenerazione degli obiettivi del gioco 9x9.

Coordinate (x, y) a base 1. Il modulo non conserva stato globale e non
modifica direttamente i Victory Points. Le estrazioni usano un RNG esplicito.

Ordine degli eventi durante un movimento:
    Move -> Capture -> eventuale rigenerazione -> eventuale Melee.

Il modulo game.py e' responsabile di chiamare resolve_objective_capture()
soltanto dopo un movimento legale e prima dell'eventuale combattimento.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import AbstractSet, Final

from .maps import GameMap, START_CELLS
from .rules import Player, Position


OBJECTIVES_PER_GROUP: Final[int] = 3
ObjectiveSet = frozenset[Position]


@dataclass(frozen=True, slots=True)
class ObjectiveUpdate:
    """Esito immutabile del controllo di conquista dopo un movimento.

    active_objectives: obiettivi presenti alla fine dell'evento; se l'ultimo
        e' stato catturato, contiene gia' i tre nuovi obiettivi.
    captured_position: posizione conquistata, oppure None.
    regenerated: True soltanto se l'ultimo obiettivo e' stato catturato.
    """

    active_objectives: ObjectiveSet
    captured_position: Position | None
    regenerated: bool

    @property
    def captured(self) -> bool:
        return self.captured_position is not None


def _validate_player_positions(
    game_map: GameMap,
    agent_position: Position,
    opponent_position: Position,
) -> None:
    """Rifiuta posizioni incompatibili con la geometria dell'MDP."""
    if agent_position not in game_map.free_cells:
        raise ValueError(f"Posizione dell'Agent non attraversabile: {agent_position!r}")
    if opponent_position not in game_map.free_cells:
        raise ValueError(
            f"Posizione dell'Opponent non attraversabile: {opponent_position!r}"
        )
    if agent_position == opponent_position:
        raise ValueError("Agent e Opponent non possono occupare la stessa casella")


def admissible_objective_cells(
    game_map: GameMap,
    agent_position: Position,
    opponent_position: Position,
) -> tuple[Position, ...]:
    """Celle di generazione ammissibili, ordinate per riproducibilita'.

    E_m(pA, pO) = F_m \\ {pA_start, pO_start, pA, pO}.
    Le posizioni iniziali sono escluse anche quando non sono occupate.
    """
    _validate_player_positions(game_map, agent_position, opponent_position)
    eligible = game_map.free_cells.difference(
        START_CELLS, (agent_position, opponent_position)
    )
    return tuple(sorted(eligible))


def generate_objectives(
    game_map: GameMap,
    agent_position: Position,
    opponent_position: Position,
    rng: Random,
) -> ObjectiveSet:
    """Genera uniformemente un sottoinsieme NON ordinato di tre celle distinte.

    Con N celle ammissibili, ogni insieme di tre celle ha probabilita'
    1 / comb(N, 3). random.Random.sample() estrae senza reinserimento.
    """
    eligible = admissible_objective_cells(
        game_map, agent_position, opponent_position
    )
    if len(eligible) < OBJECTIVES_PER_GROUP:
        raise ValueError("Meno di tre celle ammissibili per gli obiettivi")
    return frozenset(rng.sample(eligible, OBJECTIVES_PER_GROUP))


def resolve_objective_capture(
    game_map: GameMap,
    active_objectives: AbstractSet[Position],
    moving_player: Player,
    agent_position: Position,
    opponent_position: Position,
    rng: Random,
) -> ObjectiveUpdate:
    """Risolve l'eventuale conquista DOPO un movimento gia' eseguito.

    Le posizioni fornite sono quelle DOPO il movimento e PRIMA del melee.
    Non chiamare questa funzione per Shoot, Pass o melee obbligatorio.

    - Nessuna conquista: il gruppo rimane invariato, senza estrazioni RNG.
    - Conquista non finale: si rimuove soltanto la casella conquistata.
    - Conquista dell'ultimo obiettivo: si generano subito altri tre obiettivi
      escludendo le posizioni attuali e quelle iniziali, prima del melee.

    Non vengono applicati VP: game.py usera' score_for_objective() da rules.py
    quando l'attributo `captured` e' True.
    """
    _validate_player_positions(game_map, agent_position, opponent_position)
    if moving_player is Player.AGENT:
        moved_position = agent_position
        other_position = opponent_position
    elif moving_player is Player.OPPONENT:
        moved_position = opponent_position
        other_position = agent_position
    else:
        raise ValueError(f"Giocatore non valido: {moving_player!r}")

    objectives = frozenset(active_objectives)
    if not 1 <= len(objectives) <= OBJECTIVES_PER_GROUP:
        raise ValueError("Il gruppo attivo deve contenere da 1 a 3 obiettivi")
    if not objectives.issubset(game_map.free_cells.difference(START_CELLS)):
        raise ValueError("Obiettivi su ostacoli, fuori griglia o sugli spawn fissi")
    if other_position in objectives:
        raise ValueError("Un obiettivo non puo' essere occupato dall'altro giocatore")

    if moved_position not in objectives:
        return ObjectiveUpdate(objectives, None, False)

    remaining = objectives.difference((moved_position,))
    if remaining:
        return ObjectiveUpdate(remaining, moved_position, False)

    regenerated = generate_objectives(
        game_map, agent_position, opponent_position, rng
    )
    return ObjectiveUpdate(regenerated, moved_position, True)
