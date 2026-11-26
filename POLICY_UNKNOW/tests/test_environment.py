"""Test unitari di rules.py (Fase 1.1). Estendere con i test di game.py."""

import random

import pytest

from environment.rules import (
    Action,
    MeleeOutcome,
    Player,
    ScoreDelta,
    ShotOutcome,
    are_adjacent,
    can_shoot,
    is_inside_grid,
    is_legal_move,
    legal_movement_actions,
    manhattan_distance,
    move_destination,
    other_player,
    resolve_melee,
    resolve_shot,
    respawn_positions,
    score_for_kill,
    score_for_objective,
)


class FixedRandom:
    """RNG controllato per verificare gli esiti sui valori di frontiera."""

    def __init__(self, *values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def test_grid_coordinates_and_distances():
    assert is_inside_grid((1, 1))
    assert is_inside_grid((9, 9))
    assert not is_inside_grid((0, 1))
    assert not is_inside_grid((10, 1))
    assert manhattan_distance((1, 1), (4, 5)) == 7
    assert are_adjacent((5, 5), (5, 6))
    assert not are_adjacent((5, 5), (6, 6))


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.NORTH, (5, 6)),
        (Action.SOUTH, (5, 4)),
        (Action.EAST, (6, 5)),
        (Action.WEST, (4, 5)),
    ],
)
def test_movement_directions(action, expected):
    assert move_destination((5, 5), action) == expected


def test_move_legality_obstacles_bounds_and_occupied_cells():
    assert is_legal_move((5, 5), Action.NORTH, set(), (9, 9))
    assert not is_legal_move((5, 9), Action.NORTH, set(), (9, 9))
    assert not is_legal_move((5, 5), Action.NORTH, {(5, 6)}, (9, 9))
    assert not is_legal_move((5, 5), Action.NORTH, set(), (5, 6))
    assert not is_legal_move((5, 5), Action.SHOOT, set(), (9, 9))
    assert legal_movement_actions((1, 1), {(1, 2)}, (9, 9)) == (Action.EAST,)
    with pytest.raises(ValueError):
        move_destination((5, 5), Action.PASS)


def test_shooting_distances_and_obstacle_independence():
    assert not can_shoot((5, 5), (5, 6))  # melee obbligatorio
    assert can_shoot((5, 5), (5, 7))
    assert can_shoot((5, 5), (8, 5))
    assert not can_shoot((5, 5), (9, 5))
    # can_shoot non riceve ostacoli: la line-of-sight non è prevista.


def test_shot_probability_threshold():
    assert resolve_shot(FixedRandom(0.0)) is ShotOutcome.TARGET_KILLED
    assert resolve_shot(FixedRandom(0.199999)) is ShotOutcome.TARGET_KILLED
    assert resolve_shot(FixedRandom(0.20)) is ShotOutcome.MISS


def test_melee_sequential_attack_and_counterattack():
    # Attaccante uccide: NON deve essere estratto il secondo numero.
    assert resolve_melee(FixedRandom(0.749999)) is MeleeOutcome.ATTACKER_KILLS
    # Primo attacco fallisce, contrattacco riesce.
    assert resolve_melee(FixedRandom(0.75, 0.49999)) is MeleeOutcome.DEFENDER_KILLS
    # Falliscono entrambi: il turno si chiude e nessuno muore.
    assert resolve_melee(FixedRandom(0.75, 0.50)) is MeleeOutcome.BOTH_SURVIVE


def test_score_updates_and_respawn():
    assert score_for_objective(Player.AGENT) == ScoreDelta(agent_vp=1)
    assert score_for_objective(Player.OPPONENT) == ScoreDelta(opponent_vp=1)
    assert score_for_kill(Player.AGENT) == ScoreDelta(agent_vp=2, opponent_vp=-5)
    assert score_for_kill(Player.OPPONENT) == ScoreDelta(agent_vp=-5, opponent_vp=2)
    assert score_for_kill(Player.AGENT).differential == 7
    assert score_for_kill(Player.OPPONENT).differential == -7
    assert respawn_positions() == ((5, 1), (5, 9))
    assert other_player(Player.AGENT) is Player.OPPONENT
    assert other_player(Player.OPPONENT) is Player.AGENT


def test_seed_reproducibility():
    rng1 = random.Random(2026)
    rng2 = random.Random(2026)
    assert [resolve_melee(rng1) for _ in range(500)] == [
        resolve_melee(rng2) for _ in range(500)
    ]


def test_shot_empirical_distribution():
    rng = random.Random(7)
    trials = 50_000
    kills = sum(resolve_shot(rng) is ShotOutcome.TARGET_KILLED for _ in range(trials))
    assert abs(kills / trials - 0.20) < 0.01


def test_melee_empirical_distribution():
    rng = random.Random(11)
    trials = 50_000
    outcomes = [resolve_melee(rng) for _ in range(trials)]
    for outcome, expected in (
        (MeleeOutcome.ATTACKER_KILLS, 0.75),
        (MeleeOutcome.DEFENDER_KILLS, 0.125),
        (MeleeOutcome.BOTH_SURVIVE, 0.125),
    ):
        assert abs(outcomes.count(outcome) / trials - expected) < 0.01
