"""Regole elementari del gioco 9x9.

Questo modulo non conserva né modifica lo stato di una partita.
Le funzioni stocastiche ricevono un RNG esplicito, gestito da game.py.
Coordinate: x cresce verso est, y cresce verso nord, entrambe da 1 a 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import AbstractSet, Final, Protocol


Position = tuple[int, int]

GRID_SIZE: Final[int] = 9
AGENT_START: Final[Position] = (5, 1)
OPPONENT_START: Final[Position] = (5, 9)

SHOOT_MIN_DISTANCE: Final[int] = 2
SHOOT_MAX_DISTANCE: Final[int] = 3
SHOOT_KILL_PROBABILITY: Final[float] = 0.20
MELEE_ATTACK_KILL_PROBABILITY: Final[float] = 0.75
MELEE_COUNTERATTACK_KILL_PROBABILITY: Final[float] = 0.50

OBJECTIVE_VP: Final[int] = 1
KILLER_VP: Final[int] = 2
VICTIM_VP: Final[int] = -5


class Player(Enum):
    """Identità del giocatore (non indica chi ha l'iniziativa)."""

    AGENT = "A"
    OPPONENT = "O"


class Action(IntEnum):
    """ID stabili: 0-5 sono azioni di gioco, 6 è l'azione MDP forzata."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    SHOOT = 4
    PASS = 5
    DUMMY = 6  # ⊥: turno avversario oppure melee obbligatorio


MOVEMENT_ACTIONS: Final[tuple[Action, ...]] = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)
PRIMITIVE_ACTIONS: Final[tuple[Action, ...]] = (
    *MOVEMENT_ACTIONS,
    Action.SHOOT,
    Action.PASS,
)

_MOVEMENT_DELTAS: Final[dict[Action, Position]] = {
    Action.NORTH: (0, 1),
    Action.SOUTH: (0, -1),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
}


class ShotOutcome(Enum):
    TARGET_KILLED = "target_killed"
    MISS = "miss"


class MeleeOutcome(Enum):
    ATTACKER_KILLS = "attacker_kills"
    DEFENDER_KILLS = "defender_kills"
    BOTH_SURVIVE = "both_survive"


@dataclass(frozen=True, slots=True)
class ScoreDelta:
    """Variazioni dei punteggi dei due giocatori per un singolo evento."""

    agent_vp: int = 0
    opponent_vp: int = 0

    @property
    def differential(self) -> int:
        """Variazione di D = VP_Agent - VP_Opponent."""
        return self.agent_vp - self.opponent_vp


class RandomSource(Protocol):
    """Compatibile con random.Random e numpy.random.Generator."""

    def random(self) -> float: ...


def other_player(player: Player) -> Player:
    """Restituisce l'altro giocatore."""
    if player is Player.AGENT:
        return Player.OPPONENT
    if player is Player.OPPONENT:
        return Player.AGENT
    raise ValueError(f"Giocatore non valido: {player!r}")


def manhattan_distance(first: Position, second: Position) -> int:
    """Calcola |x1-x2| + |y1-y2|."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def are_adjacent(first: Position, second: Position) -> bool:
    """Il melee è obbligatorio quando la distanza Manhattan è esattamente 1."""
    return manhattan_distance(first, second) == 1


def is_inside_grid(position: Position, grid_size: int = GRID_SIZE) -> bool:
    """Controlla i limiti della griglia usando coordinate a base 1."""
    x, y = position
    return 1 <= x <= grid_size and 1 <= y <= grid_size


def move_destination(position: Position, action: Action) -> Position:
    """Restituisce la destinazione geometrica, SENZA verificare la legalità."""
    try:
        dx, dy = _MOVEMENT_DELTAS[action]
    except KeyError as exc:
        raise ValueError(f"L'azione {action!r} non è un movimento") from exc
    return position[0] + dx, position[1] + dy


def is_legal_move(
    position: Position,
    action: Action,
    obstacles: AbstractSet[Position],
    opponent_position: Position,
    grid_size: int = GRID_SIZE,
) -> bool:
    """Un movimento è legale se la destinazione è interna, libera e non occupata."""
    if action not in _MOVEMENT_DELTAS:
        return False
    destination = move_destination(position, action)
    return (
        is_inside_grid(destination, grid_size)
        and destination not in obstacles
        and destination != opponent_position
    )


def legal_movement_actions(
    position: Position,
    obstacles: AbstractSet[Position],
    opponent_position: Position,
    grid_size: int = GRID_SIZE,
) -> tuple[Action, ...]:
    """Movimenti consentiti, nell'ordine costante N, S, E, W.

    La verifica del melee obbligatorio e il mask completo spettano a game.py
    e, successivamente, ad action_mask.py.
    """
    return tuple(
        action
        for action in MOVEMENT_ACTIONS
        if is_legal_move(position, action, obstacles, opponent_position, grid_size)
    )


def can_shoot(attacker_position: Position, defender_position: Position) -> bool:
    """Tiro legale a distanza Manhattan 2 o 3; gli ostacoli non bloccano il tiro."""
    distance = manhattan_distance(attacker_position, defender_position)
    return SHOOT_MIN_DISTANCE <= distance <= SHOOT_MAX_DISTANCE


def resolve_shot(rng: RandomSource) -> ShotOutcome:
    """Estrae l'esito di un tiro già verificato come legale.

    Il colpo uccide con probabilità 0.20, altrimenti non produce effetti.
    Non modifica posizioni né punti vittoria.
    """
    if rng.random() < SHOOT_KILL_PROBABILITY:
        return ShotOutcome.TARGET_KILLED
    return ShotOutcome.MISS


def resolve_melee(rng: RandomSource) -> MeleeOutcome:
    """Risolve un singolo melee; l'attaccante è chi ha mosso o ha il turno.

    Pr(attaccante uccide) = 0.75.
    Solo se fallisce, il difensore contrattacca con Pr(kill) = 0.50.
    Distribuzione finale: 0.75, 0.125, 0.125.
    """
    if rng.random() < MELEE_ATTACK_KILL_PROBABILITY:
        return MeleeOutcome.ATTACKER_KILLS
    if rng.random() < MELEE_COUNTERATTACK_KILL_PROBABILITY:
        return MeleeOutcome.DEFENDER_KILLS
    return MeleeOutcome.BOTH_SURVIVE


def score_for_objective(player: Player) -> ScoreDelta:
    """Conquista: +1 VP soltanto al giocatore che cattura l'obiettivo."""
    if player is Player.AGENT:
        return ScoreDelta(agent_vp=OBJECTIVE_VP)
    if player is Player.OPPONENT:
        return ScoreDelta(opponent_vp=OBJECTIVE_VP)
    raise ValueError(f"Giocatore non valido: {player!r}")


def score_for_kill(killer: Player) -> ScoreDelta:
    """Morte: +2 VP all'uccisore e -5 VP alla vittima (|ΔD| = 7)."""
    if killer is Player.AGENT:
        return ScoreDelta(agent_vp=KILLER_VP, opponent_vp=VICTIM_VP)
    if killer is Player.OPPONENT:
        return ScoreDelta(agent_vp=VICTIM_VP, opponent_vp=KILLER_VP)
    raise ValueError(f"Giocatore non valido: {killer!r}")


def respawn_positions() -> tuple[Position, Position]:
    """Posizioni dopo QUALSIASI morte, sempre nell'ordine (Agent, Opponent).

    Il ripristino effettivo dei giocatori è responsabilità di game.py.
    """
    return AGENT_START, OPPONENT_START
