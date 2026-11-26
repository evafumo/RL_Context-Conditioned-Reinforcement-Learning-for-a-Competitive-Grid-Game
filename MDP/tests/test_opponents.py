"""Test delle sei policy: legalita', BFS, fallback e probabilita' esatte."""

import math
import random
from collections import Counter

import pytest

from environment.maps import GameMap, START_CELLS, get_all_maps, get_map
from environment.opponents import (
    OpponentKind,
    OpponentState,
    choose_opponent_action,
    get_all_opponents,
    get_opponent_kind,
    legal_opponent_actions,
    opponent_action_probabilities,
)
from environment.rules import Action, Player, manhattan_distance


def state(a=(5, 2), o=(5, 4), goals=((7, 4),), t=2):
    return OpponentState(a, o, frozenset(goals), -4, Player.OPPONENT, t)


def test_pool_ids_and_names():
    assert [x.value for x in get_all_opponents()] == list(range(1, 7))
    assert [x.name for x in get_all_opponents()] == [
        'RANDOM', 'AGGRESSIVE', 'HUNTER', 'HYBRID', 'SHOOTER', 'ASH'
    ]
    assert get_opponent_kind(6) is OpponentKind.ASH
    for invalid in (0, 7, True, '1', None):
        with pytest.raises(ValueError):
            get_opponent_kind(invalid)


def test_full_state_is_immutable_and_owns_goals():
    goals = {(7, 4)}
    s = OpponentState((5, 2), (5, 4), goals, 0, Player.OPPONENT, 2)
    goals.clear()
    assert s.active_objectives == frozenset({(7, 4)})
    with pytest.raises(AttributeError):
        s.activation = 3


def test_normal_legal_actions_and_shot():
    s = state()
    legal = legal_opponent_actions(s, get_map(1))
    assert legal == (
        Action.SOUTH, Action.EAST, Action.WEST, Action.SHOOT, Action.PASS
    )
    assert Action.DUMMY not in legal


def test_random_uniform_over_ALL_actions():
    legal = legal_opponent_actions(state(), get_map(1))
    probs = opponent_action_probabilities(OpponentKind.RANDOM, state(), get_map(1))
    assert set(probs) == set(legal)
    assert all(p == pytest.approx(1 / len(legal)) for p in probs.values())


def test_aggressive_shortest_path_not_manhattan_only():
    probs = opponent_action_probabilities(2, state(), get_map(1))
    assert probs == {Action.SOUTH: 1.0}


def test_hunter_shortest_path():
    probs = opponent_action_probabilities(3, state(), get_map(1))
    assert probs == {Action.EAST: 1.0}


def test_hybrid_when_aggression_closer():
    probs = opponent_action_probabilities(4, state(), get_map(1))
    assert probs[Action.EAST] == pytest.approx(.02)
    assert probs[Action.SOUTH] == pytest.approx(.98)
    assert len(probs) == 2


def test_hybrid_when_hunting_closer():
    s = state(a=(5, 1), o=(5, 9), goals=((6, 9),))
    probs = opponent_action_probabilities(4, s, get_map(1))
    assert probs[Action.EAST] == pytest.approx(.98)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_hybrid_tie_equal_weights():
    s = state(goals=((4, 4),))
    probs = opponent_action_probabilities(4, s, get_map(1))
    assert probs == {Action.SOUTH: pytest.approx(.50), Action.WEST: pytest.approx(.50)}


def test_shooter_in_range():
    probs = opponent_action_probabilities(5, state(), get_map(1))
    assert probs == {Action.SOUTH: pytest.approx(.40), Action.SHOOT: pytest.approx(.60)}


def test_shooter_outside_range_same_as_aggressive():
    s = state(a=(5, 1), o=(5, 9), goals=((6, 9),))
    assert opponent_action_probabilities(5, s, get_map(2)) == (
        opponent_action_probabilities(2, s, get_map(2))
    )


def test_ash_probability_with_shoot_legal():
    probs = opponent_action_probabilities(6, state(), get_map(1))
    assert probs[Action.EAST] == pytest.approx(.02)
    assert probs[Action.SHOOT] == pytest.approx(.588)
    assert probs[Action.SOUTH] == pytest.approx(.392)


def test_ash_tie_with_shoot_legal():
    probs = opponent_action_probabilities(6, state(goals=((4, 4),)), get_map(1))
    assert probs[Action.WEST] == pytest.approx(.50)
    assert probs[Action.SHOOT] == pytest.approx(.30)
    assert probs[Action.SOUTH] == pytest.approx(.20)


def test_ash_without_shoot_same_as_hybrid():
    s = state(a=(5, 1), o=(5, 9), goals=((6, 9),))
    for game_map in get_all_maps():
        assert opponent_action_probabilities(6, s, game_map) == (
            opponent_action_probabilities(4, s, game_map)
        )


def test_shooter_unreachable_melee_keeps_shot_at_sixty_percent():
    # Opponent circondato da ostacoli; il muro interposto NON blocca Shoot.
    custom = GameMap(20, frozenset({(4, 5), (5, 4), (6, 5), (5, 6)}))
    s = state(a=(5, 3), o=(5, 5), goals=((1, 1),))
    assert legal_opponent_actions(s, custom) == (Action.SHOOT, Action.PASS)
    assert opponent_action_probabilities(5, s, custom) == {
        Action.PASS: pytest.approx(.4), Action.SHOOT: pytest.approx(.6)
    }


def test_bfs_avoids_agent_as_blocking_cell():
    # Unico varco del muro: la casella (5,5) e' occupata da Agent.
    custom = GameMap(21, frozenset((x, 5) for x in range(1, 10) if x != 5))
    s = state(a=(5, 5), o=(5, 7), goals=((5, 4),))
    random_probs = opponent_action_probabilities(1, s, custom)
    hunter_probs = opponent_action_probabilities(3, s, custom)
    assert hunter_probs == random_probs  # obiettivo irraggiungibile
    assert Action.SHOOT in random_probs  # tiro attraverso ostacoli ammissibile


def test_random_fallback_when_aggressive_unreachable():
    custom = GameMap(22, frozenset((x, 5) for x in range(1, 10)))
    s = state(a=(5, 3), o=(5, 7), goals=((1, 7),))
    assert opponent_action_probabilities(2, s, custom) == (
        opponent_action_probabilities(1, s, custom)
    )


def test_infinity_tie_hybrid_uses_random():
    custom = GameMap(22, frozenset((x, 5) for x in range(1, 10)))
    s = state(a=(5, 3), o=(5, 7), goals=((1, 2),))
    assert opponent_action_probabilities(4, s, custom) == (
        opponent_action_probabilities(1, s, custom)
    )


def test_ash_hunter_fallback_can_shoot():
    custom = GameMap(22, frozenset((x, 5) for x in range(1, 10)))
    s = state(a=(5, 6), o=(5, 8), goals=((2, 3),))
    legal = legal_opponent_actions(s, custom)
    assert Action.SHOOT in legal
    probs = opponent_action_probabilities(6, s, custom)
    # Hunter = fallback Random con peso .02; Shooter con peso .98.
    assert probs[Action.SHOOT] == pytest.approx(.98 * .60 + .02 / len(legal))


def test_shooter_when_unreachable_but_legal_movements_no_second_shot():
    custom = GameMap(22, frozenset((x, 5) for x in range(1, 10)))
    s = state(a=(5, 4), o=(5, 6), goals=((1, 6),))
    probs = opponent_action_probabilities(5, s, custom)
    assert probs.get(Action.SHOOT, 0.0) == pytest.approx(.6)
    assert probs.get(Action.PASS, 0.0) == pytest.approx(0.0)
    assert sum(probs[a] for a in probs if a in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)) == pytest.approx(.4)


@pytest.mark.parametrize('who', list(OpponentKind))
def test_reject_melee_forced(who):
    s = state(a=(5, 3), o=(5, 4), goals=((1, 1),))
    with pytest.raises(ValueError, match='Melee obbligatorio'):
        opponent_action_probabilities(who, s, get_map(1))


def test_reject_agent_turn():
    s = OpponentState((5, 2), (5, 4), frozenset({(7, 4)}), 0, Player.AGENT, 3)
    with pytest.raises(ValueError, match='solo nel suo turno'):
        choose_opponent_action(1, s, get_map(1), random.Random(9))


@pytest.mark.parametrize('goals', [frozenset(), frozenset({(5,1)}), frozenset({(5,4)})])
def test_reject_invalid_objectives(goals):
    with pytest.raises(ValueError):
        opponent_action_probabilities(1, state(goals=goals), get_map(1))


def test_invalid_time_rejected():
    with pytest.raises(ValueError):
        state(t=101)


def test_seed_reproducible():
    s = state()
    game_map = get_map(1)
    def trajectory(seed):
        rng = random.Random(seed)
        return [choose_opponent_action(6, s, game_map, rng) for _ in range(100)]
    assert trajectory(1234) == trajectory(1234)


def test_sampling_shooter_empirical_shoot_rate():
    s = state()
    rng = random.Random(123)
    count = Counter(choose_opponent_action(5, s, get_map(1), rng) for _ in range(12000))
    assert count[Action.SHOOT] / 12000 == pytest.approx(.6, abs=.02)


def test_sampling_random_empirical_uniformity():
    s = state()
    rng = random.Random(321)
    count = Counter(choose_opponent_action(1, s, get_map(1), rng) for _ in range(12000))
    for action in legal_opponent_actions(s, get_map(1)):
        assert count[action] / 12000 == pytest.approx(.2, abs=.02)


def test_equal_optimal_moves_are_equiprobable():
    # Obiettivo direttamente due caselle a sud-est della posizione Opponent.
    s = state(a=(1, 1), o=(5, 9), goals=((6, 8),))
    probs = opponent_action_probabilities(3, s, get_map(1))
    assert probs == {Action.SOUTH: pytest.approx(.5), Action.EAST: pytest.approx(.5)}


def test_many_reachable_and_unreachable_states_all_policies_legal():
    rng = random.Random(492)
    for game_map in get_all_maps():
        accessible = sorted(game_map.free_cells)
        for _ in range(75):
            agent, opponent = rng.sample(accessible, 2)
            if manhattan_distance(agent, opponent) <= 1:
                continue
            eligible = sorted(game_map.free_cells.difference(START_CELLS, (agent, opponent)))
            goals = frozenset(rng.sample(eligible, rng.randint(1, 3)))
            s = OpponentState(agent, opponent, goals, rng.randint(-60, 60), Player.OPPONENT, rng.randint(1, 100))
            legal = set(legal_opponent_actions(s, game_map))
            for who in OpponentKind:
                probabilities = opponent_action_probabilities(who, s, game_map)
                assert probabilities.keys() <= legal
                assert all(p > 0.0 for p in probabilities.values())
                assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)
                assert choose_opponent_action(who, s, game_map, rng) in legal
