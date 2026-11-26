"""Sei policy avversarie Markoviane per l'ambiente competitivo 9x9.

Random, Aggressive, Hunter, Hybrid, Shooter e ASH seguono e_Opponents.pdf.
Le policy scelgono solo azioni: game.py risolve movimenti, tiro, melee,
conquista, punteggio, respawn e transizioni forzate DUMMY.

Tutte le distribuzioni dipendono solo dallo stato corrente e dalla mappa.
La selezione stocastica riceve un generatore random.Random esplicito.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from math import inf
from random import Random
from typing import AbstractSet, Final, Mapping

from .maps import GameMap, START_CELLS
from .rules import (
    Action,
    MOVEMENT_ACTIONS,
    Player,
    Position,
    are_adjacent,
    can_shoot,
    legal_movement_actions,
    manhattan_distance,
    move_destination,
)


class OpponentKind(IntEnum):
    """Identificativi stabili delle sei policy, da usare nel contesto (m, k)."""

    RANDOM = 1
    AGGRESSIVE = 2
    HUNTER = 3
    HYBRID = 4
    SHOOTER = 5
    ASH = 6


OPPONENT_POOL: Final[tuple[OpponentKind, ...]] = tuple(OpponentKind)
SHOOTER_SHOOT_PROBABILITY: Final[float] = 0.60
HYBRID_PREFER_PROBABILITY: Final[float] = 0.98
HYBRID_OTHER_PROBABILITY: Final[float] = 0.02
HYBRID_TIE_PROBABILITY: Final[float] = 0.50


@dataclass(frozen=True, slots=True)
class OpponentState:
    """Vista immutabile dello stato MDP prima di un'attivazione.

    Contiene anche differenziale e tempo, benché le sei policy qui definite
    non ne facciano uso nella propria distribuzione delle azioni.
    """

    agent_position: Position
    opponent_position: Position
    active_objectives: frozenset[Position]
    score_differential: int
    player_to_act: Player
    activation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "active_objectives", frozenset(self.active_objectives)
        )
        if type(self.score_differential) is not int:
            raise ValueError("score_differential deve essere un intero")
        if type(self.activation) is not int or not 1 <= self.activation <= 100:
            raise ValueError("activation deve essere compresa fra 1 e 100")
        if not isinstance(self.player_to_act, Player):
            raise ValueError("player_to_act deve essere Player.AGENT o Player.OPPONENT")


@dataclass(frozen=True, slots=True)
class _PathPlan:
    """Distanza BFS minima e prime mosse di uno dei cammini più brevi."""

    distance: int | float
    moves: tuple[Action, ...]


def get_opponent_kind(opponent_id: int | OpponentKind) -> OpponentKind:
    """Converte l'identificativo 1..6 nell'enumerazione corrispondente."""
    if isinstance(opponent_id, OpponentKind):
        return opponent_id
    if type(opponent_id) is not int:
        raise ValueError(f"Identificativo Opponent non valido: {opponent_id!r}")
    try:
        return OpponentKind(opponent_id)
    except ValueError as exc:
        raise ValueError(f"Opponent sconosciuto: {opponent_id!r}; ID validi: 1..6") from exc


def get_all_opponents() -> tuple[OpponentKind, ...]:
    """Restituisce i sei opponent nell'ordine dei rispettivi ID."""
    return OPPONENT_POOL


def _validate_decision_state(state: OpponentState, game_map: GameMap) -> None:
    """Accetta solo stati validi in cui Opponent deve scegliere un'azione."""
    if state.player_to_act is not Player.OPPONENT:
        raise ValueError("La policy Opponent va interrogata solo nel suo turno")
    if state.agent_position not in game_map.free_cells:
        raise ValueError("Posizione Agent non attraversabile")
    if state.opponent_position not in game_map.free_cells:
        raise ValueError("Posizione Opponent non attraversabile")
    if state.agent_position == state.opponent_position:
        raise ValueError("I giocatori non possono occupare la stessa casella")
    if are_adjacent(state.agent_position, state.opponent_position):
        raise ValueError("Melee obbligatorio: game.py deve risolverlo senza policy")
    goals = state.active_objectives
    if not 1 <= len(goals) <= 3:
        raise ValueError("Devono essere attivi da 1 a 3 obiettivi")
    allowed_goals = game_map.free_cells.difference(
        START_CELLS, (state.agent_position, state.opponent_position)
    )
    if not goals.issubset(allowed_goals):
        raise ValueError("Obiettivi su celle proibite o occupate")


def legal_opponent_actions(
    state: OpponentState, game_map: GameMap
) -> tuple[Action, ...]:
    """B_m(s): movimenti legali, Shoot se disponibile, Pass sempre.

    Non restituisce DUMMY: l'attivazione forzata viene gestita da game.py.
    """
    _validate_decision_state(state, game_map)
    moves = legal_movement_actions(
        state.opponent_position, game_map.obstacles, state.agent_position
    )
    shoot = (Action.SHOOT,) if can_shoot(
        state.opponent_position, state.agent_position
    ) else ()
    return moves + shoot + (Action.PASS,)


def _uniform_distribution(actions: tuple[Action, ...]) -> dict[Action, float]:
    """Distribuzione uniforme sulle azioni fornite, che devono essere non vuote."""
    if not actions:
        raise ValueError("Non è possibile campionare da un insieme vuoto")
    probability = 1.0 / len(actions)
    return {action: probability for action in actions}


def _shortest_path_plan(
    state: OpponentState,
    game_map: GameMap,
    targets: AbstractSet[Position],
) -> _PathPlan:
    """BFS multi-source inversa verso le destinazioni, escludendo Agent.

    La distanza è infinita quando nessun bersaglio è raggiungibile.
    Restituisce tutte le prime mosse legali che diminuiscono la distanza di 1.
    """
    accessible = game_map.free_cells.difference((state.agent_position,))
    valid_targets = sorted(targets.intersection(accessible))
    if not valid_targets:
        return _PathPlan(inf, ())

    distances: dict[Position, int] = {cell: 0 for cell in valid_targets}
    queue: deque[Position] = deque(valid_targets)

    while queue:
        position = queue.popleft()
        for move in MOVEMENT_ACTIONS:
            neighbor = move_destination(position, move)
            if neighbor in accessible and neighbor not in distances:
                distances[neighbor] = distances[position] + 1
                queue.append(neighbor)

    distance = distances.get(state.opponent_position)
    if distance is None:
        return _PathPlan(inf, ())
    if distance == 0:
        return _PathPlan(0, ())

    legal_moves = legal_movement_actions(
        state.opponent_position, game_map.obstacles, state.agent_position
    )
    first_moves = tuple(
        action
        for action in legal_moves
        if distances.get(move_destination(state.opponent_position, action))
        == distance - 1
    )
    return _PathPlan(distance, first_moves)


def _aggressive_plan(state: OpponentState, game_map: GameMap) -> _PathPlan:
    """Percorsi minimi verso una casella libera adiacente all'Agent."""
    targets = frozenset(
        cell for action in MOVEMENT_ACTIONS
        if (cell := move_destination(state.agent_position, action))
        in game_map.free_cells
    )
    return _shortest_path_plan(state, game_map, targets)


def _hunter_plan(state: OpponentState, game_map: GameMap) -> _PathPlan:
    """Percorsi minimi verso uno qualsiasi degli obiettivi attivi."""
    return _shortest_path_plan(state, game_map, state.active_objectives)


def _path_policy(
    plan: _PathPlan, legal_actions: tuple[Action, ...]
) -> dict[Action, float]:
    """Uniforme tra prime mosse ottimali, o Random se irraggiungibile."""
    return _uniform_distribution(plan.moves if plan.moves else legal_actions)


def _hunter_weight(hunter_distance: int | float, attack_distance: int | float) -> float:
    """Peso di Hunter per Hybrid/ASH; include il caso inf == inf."""
    if hunter_distance < attack_distance:
        return HYBRID_PREFER_PROBABILITY
    if hunter_distance > attack_distance:
        return HYBRID_OTHER_PROBABILITY
    return HYBRID_TIE_PROBABILITY


def _mixture(
    first: Mapping[Action, float],
    second: Mapping[Action, float],
    first_weight: float,
) -> dict[Action, float]:
    """Distribuzione marginale di una miscela stocastica di due policy."""
    return {
        action: probability
        for action in sorted(first.keys() | second.keys())
        if (probability := (
            first_weight * first.get(action, 0.0)
            + (1.0 - first_weight) * second.get(action, 0.0)
        )) > 0.0
    }


def _shooter_policy(
    state: OpponentState,
    legal_actions: tuple[Action, ...],
    aggressive: _PathPlan,
) -> dict[Action, float]:
    """0.60 Shoot + 0.40 avvicinamento quando il tiro è legale.

    Se la meta melee è irraggiungibile, nel ramo del 40% si usa
    esclusivamente un movimento legale, oppure Pass se non ve ne sono.
    """
    if Action.SHOOT not in legal_actions:
        return _path_policy(aggressive, legal_actions)

    if aggressive.moves:
        approach_actions = aggressive.moves
    else:
        movements = tuple(
            action for action in legal_actions if action in MOVEMENT_ACTIONS
        )
        approach_actions = movements if movements else (Action.PASS,)

    return _mixture(
        {Action.SHOOT: 1.0},
        _uniform_distribution(approach_actions),
        SHOOTER_SHOOT_PROBABILITY,
    )


def opponent_action_probabilities(
    opponent: int | OpponentKind,
    state: OpponentState,
    game_map: GameMap,
) -> dict[Action, float]:
    """Restituisce π_k^O(b|s;m) esatta, senza estrazioni casuali.

    Il dizionario contiene solo azioni con probabilità positiva;
    tutte le azioni assenti, compreso DUMMY, hanno probabilità zero.
    """
    kind = get_opponent_kind(opponent)
    legal_actions = legal_opponent_actions(state, game_map)

    if kind is OpponentKind.RANDOM:
        return _uniform_distribution(legal_actions)

    if kind is OpponentKind.AGGRESSIVE:
        return _path_policy(_aggressive_plan(state, game_map), legal_actions)

    if kind is OpponentKind.HUNTER:
        return _path_policy(_hunter_plan(state, game_map), legal_actions)

    aggressive = _aggressive_plan(state, game_map)

    if kind is OpponentKind.SHOOTER:
        return _shooter_policy(state, legal_actions, aggressive)

    hunter = _hunter_plan(state, game_map)
    weight = _hunter_weight(hunter.distance, aggressive.distance)
    hunter_policy = _path_policy(hunter, legal_actions)

    if kind is OpponentKind.HYBRID:
        offensive_policy = _path_policy(aggressive, legal_actions)
    else:  # ASH: fuori gittata equivale ad Aggressive, in gittata a Shooter
        offensive_policy = _shooter_policy(state, legal_actions, aggressive)

    return _mixture(hunter_policy, offensive_policy, weight)


def choose_opponent_action(
    opponent: int | OpponentKind,
    state: OpponentState,
    game_map: GameMap,
    rng: Random,
) -> Action:
    """Campiona una singola azione legale secondo la policy scelta.

    L'estrazione è indipendente dalla storia dato lo stato corrente.
    Il generatore va creato/gestito da game.py, non da questo modulo.
    """
    distribution = opponent_action_probabilities(opponent, state, game_map)
    u = rng.random()
    cumulative = 0.0
    last_action = Action.PASS
    for action, probability in distribution.items():
        cumulative += probability
        last_action = action
        if u < cumulative:
            return action
    # Protezione contro gli errori di arrotondamento floating-point.
    return last_action
