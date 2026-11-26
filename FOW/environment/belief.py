"""Filtro bayesiano ESATTO sulla posizione nascosta dell'Opponent (belief-MDP).

Unica variabile non osservata: la cella dell'Opponent quando dista piu' di
`radius` (Manhattan) dall'Agent. Tutto il resto (posizioni dell'Agent, obiettivi,
punteggio, turno, mappa, tipo di avversario) e' pubblico.

La belief b(cella) e' una matrice 9x9 con somma 1, indicizzata come il piano
Opponent della CNN: (x, y) -> [9 - y, x - 1]. Aggiornamento dopo ogni attivazione:

  1. morte            -> b = delta(OPPONENT_START)            (respawn pubblico)
  2. cattura Opponent -> b = delta(cella catturata)           (l'obiettivo sparisce)
  3. turno Opponent   -> predizione con la policy ESATTA pi_k(a|s) di opponents.py
  4. osservazione     -> se visibile b = delta(vera); altrimenti si azzerano le
                         celle in vista, gli obiettivi attivi e la cella dell'Agent,
                         poi si rinormalizza (informazione "negativa").

Il filtro usa soltanto informazione disponibile all'Agent. La posizione vera serve
unicamente a decidere se l'Opponent e' visibile e a verificare la coerenza.
Se la cella vera ha massa zero il filtro solleva RuntimeError: significa che il
modello dell'Opponent usato qui non coincide con quello di game.py.
"""

from __future__ import annotations

import numpy as np

from .fog_of_war import validate_fog_radius
from .game import GameState, StepResult
from .opponents import OpponentState, opponent_action_probabilities
from .rules import (
    GRID_SIZE,
    MOVEMENT_ACTIONS,
    OPPONENT_START,
    Player,
    Position,
    are_adjacent,
    manhattan_distance,
    move_destination,
)


def _rc(position: Position) -> tuple[int, int]:
    """(x, y) 1-based con nord in alto -> (riga, colonna) dell'array."""
    x, y = position
    return GRID_SIZE - y, x - 1


def _pos(row: int, col: int) -> Position:
    return col + 1, GRID_SIZE - row


class BeliefTracker:
    """Una istanza per ambiente. Chiamare reset() a inizio episodio e
    update(previous, result, context) dopo OGNI step non terminale."""

    def __init__(self, radius: int, *, cache_limit: int = 200_000) -> None:
        validate_fog_radius(radius)
        if type(cache_limit) is not int or cache_limit < 1:
            raise ValueError("cache_limit deve essere un intero positivo")
        self.radius = radius
        self._cache_limit = cache_limit
        self._transition_cache: dict[tuple, tuple[tuple[Position, float], ...]] = {}
        self._visible_cache: dict[Position, np.ndarray] = {}
        self._belief = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Nuovo episodio: l'Opponent parte sempre da OPPONENT_START."""
        self._set_point(OPPONENT_START)

    def plane(self) -> np.ndarray:
        """Piano (9,9) float32 da scrivere nel canale OPPONENT dell'osservazione."""
        return self._belief.astype(np.float32)

    def support(self) -> tuple[list[Position], np.ndarray]:
        """Celle con massa positiva e relativi pesi (somma 1), per QMDP."""
        rows, cols = np.nonzero(self._belief > 0.0)
        positions = [_pos(int(r), int(c)) for r, c in zip(rows, cols)]
        weights = self._belief[rows, cols].astype(np.float64)
        return positions, weights / weights.sum()

    def update(self, previous: GameState, result: StepResult, context) -> None:
        """Incorpora una attivazione. `context` e' game.context dell'episodio."""
        if result.terminated:
            self.reset()
            return
        nxt = result.next_state
        events = result.events
        actor = previous.player_to_act

        if events.killer is not None:
            self._set_point(OPPONENT_START)
        elif actor is Player.OPPONENT and events.captured_objective is not None:
            self._set_point(events.captured_objective)
        elif actor is Player.OPPONENT:
            self._predict(previous, context)
        # Turno Agent: l'Opponent non si muove, la belief resta invariata.
        self._observe(previous, nxt)

    # ------------------------------------------------------------ interni
    def _set_point(self, position: Position) -> None:
        self._belief.fill(0.0)
        self._belief[_rc(position)] = 1.0

    def _visible_mask(self, center: Position) -> np.ndarray:
        mask = self._visible_cache.get(center)
        if mask is None:
            mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if manhattan_distance(center, _pos(row, col)) <= self.radius:
                        mask[row, col] = True
            self._visible_cache[center] = mask
        return mask

    def _transitions(self, p: Position, previous: GameState, context):
        """Distribuzione esatta della cella di arrivo dell'Opponent partendo da p."""
        if are_adjacent(previous.agent_position, p):
            return ((p, 1.0),)  # melee obbligatorio: nessuna policy, non si muove
        key = (context.opponent_id, context.map_id, previous.agent_position,
               previous.active_objectives, p)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        hypothetical = OpponentState(
            agent_position=previous.agent_position,
            opponent_position=p,
            active_objectives=previous.active_objectives,
            score_differential=previous.score_differential,
            player_to_act=Player.OPPONENT,
            activation=previous.activation,
        )
        distribution = opponent_action_probabilities(
            context.opponent, hypothetical, context.game_map
        )
        arrivals: dict[Position, float] = {}
        for action, probability in distribution.items():
            destination = move_destination(p, action) if action in MOVEMENT_ACTIONS else p
            arrivals[destination] = arrivals.get(destination, 0.0) + probability
        value = tuple(arrivals.items())
        if len(self._transition_cache) >= self._cache_limit:
            self._transition_cache.clear()
        self._transition_cache[key] = value
        return value

    def _predict(self, previous: GameState, context) -> None:
        new = np.zeros_like(self._belief)
        for row, col in np.argwhere(self._belief > 0.0):
            weight = self._belief[row, col]
            p = _pos(int(row), int(col))
            for q, probability in self._transitions(p, previous, context):
                new[_rc(q)] += weight * probability
        self._belief = new

    def _observe(self, previous: GameState, nxt: GameState) -> None:
        true_position = nxt.opponent_position
        if self._belief[_rc(true_position)] <= 0.0:
            raise RuntimeError(
                "Belief incoerente: la cella vera dell'Opponent ha probabilita' 0 "
                f"(vera={true_position}, Agent={nxt.agent_position}, "
                f"attivazione={previous.activation}). Il modello dell'avversario "
                "nel filtro non coincide con game.py."
            )
        if manhattan_distance(nxt.agent_position, true_position) <= self.radius:
            self._set_point(true_position)  # visibile
            return
        belief = np.where(self._visible_mask(nxt.agent_position), 0.0, self._belief)
        for goal in nxt.active_objectives:
            belief[_rc(goal)] = 0.0  # se ci fosse stato, lo avrebbe catturato
        total = belief.sum()
        if not total > 0.0:
            raise RuntimeError("Belief collassata: nessuna cella compatibile")
        self._belief = belief / total


__all__ = ["BeliefTracker"]
