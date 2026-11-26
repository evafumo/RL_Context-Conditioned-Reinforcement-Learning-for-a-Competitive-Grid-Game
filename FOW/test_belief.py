"""Test del BeliefTracker: la cella vera deve avere sempre massa > 0.

Lanciare dalla radice del progetto:  python test_belief.py [episodi] [raggio]
Il tracker solleva RuntimeError se il filtro diverge dal simulatore.
"""
import sys
from collections import Counter

import numpy as np

from environment.belief import BeliefTracker
from environment.game import GridGame
from environment.rules import Action, MOVEMENT_ACTIONS, manhattan_distance, move_destination

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
RADIUS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def pick_action(game, state, legal):
    """Agent che si avvicina all'Opponent meta' delle volte: genera molti
    passaggi visibile <-> nascosto, che e' il caso difficile per il filtro."""
    if legal == (Action.DUMMY,):
        return Action.DUMMY
    rng = game.rng_agent
    if rng.random() < 0.5:
        moves = [a for a in legal if a in MOVEMENT_ACTIONS]
        if moves:
            return min(moves, key=lambda a: manhattan_distance(
                move_destination(state.agent_position, a), state.opponent_position))
    return rng.choice(legal)


per_opponent = Counter()
hidden_steps = visible_steps = 0
max_entropy = 0.0
for episode in range(EPISODES):
    game = GridGame(seed=episode)
    game.reset(seed=10_000 + episode)
    tracker = BeliefTracker(RADIUS)
    kind = game.context.opponent_id
    while not game.terminated:
        state = game.state
        action = pick_action(game, state, game.legal_actions())
        result = game.step(action)
        tracker.update(state, result, game.context)  # solleva se incoerente
        if result.terminated:
            break
        plane = tracker.plane()
        assert plane.shape == (9, 9) and plane.min() >= 0
        assert abs(float(plane.sum()) - 1.0) < 1e-4, plane.sum()
        nxt = result.next_state
        near = manhattan_distance(nxt.agent_position, nxt.opponent_position) <= RADIUS
        if near:
            visible_steps += 1
            assert plane.max() == 1.0 and plane.sum() == 1.0   # one-hot sulla cella vera
        else:
            hidden_steps += 1
            positions, weights = tracker.support()
            assert all(manhattan_distance(nxt.agent_position, p) > RADIUS for p in positions)
            max_entropy = max(max_entropy, float(-(weights * np.log(weights)).sum()))
    per_opponent[kind] += 1

print(f"OK: {EPISODES} episodi, raggio {RADIUS}")
print(f"passi con Opponent visibile: {visible_steps}, nascosto: {hidden_steps}")
print(f"episodi per avversario (ID): {dict(sorted(per_opponent.items()))}")
print(f"entropia massima osservata della belief: {max_entropy:.2f} nat")
