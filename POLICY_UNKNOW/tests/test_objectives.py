"""Test unitari del modulo objectives.py, indipendenti da game.py."""

from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import combinations
import random

import pytest

from environment.maps import ALL_CELLS, START_CELLS, GameMap, get_all_maps, get_map
from environment.objectives import (
    OBJECTIVES_PER_GROUP,
    ObjectiveUpdate,
    admissible_objective_cells,
    generate_objectives,
    resolve_objective_capture,
)
from environment.rules import AGENT_START, OPPONENT_START, Player, score_for_objective


@pytest.mark.parametrize("game_map", get_all_maps(), ids=lambda m: f"map_{m.map_id}")
def test_initial_generation_on_all_maps(game_map):
    rng = random.Random(100 + game_map.map_id)
    goals = generate_objectives(game_map, AGENT_START, OPPONENT_START, rng)
    assert isinstance(goals, frozenset)
    assert len(goals) == OBJECTIVES_PER_GROUP
    assert goals <= game_map.free_cells
    assert not goals & START_CELLS
    assert not goals & game_map.obstacles


def test_admissible_cells_exclude_fixed_starts_and_current_positions():
    game_map = get_map(1)
    agent = (1, 1)
    opponent = (9, 9)
    admissible = admissible_objective_cells(game_map, agent, opponent)
    assert admissible == tuple(sorted(admissible))
    assert len(admissible) == len(game_map.free_cells) - 4
    assert not set(admissible) & {AGENT_START, OPPONENT_START, agent, opponent}
    assert not set(admissible) & game_map.obstacles


def test_same_seed_same_objective_sequence():
    rng_1, rng_2 = random.Random(2026), random.Random(2026)
    game_map = get_map(2)
    seq_1 = [generate_objectives(game_map, AGENT_START, OPPONENT_START, rng_1) for _ in range(20)]
    seq_2 = [generate_objectives(game_map, AGENT_START, OPPONENT_START, rng_2) for _ in range(20)]
    assert seq_1 == seq_2


def test_no_capture_does_not_consume_randomness_and_keeps_goals():
    game_map = get_map(1)
    goals = frozenset({(1, 1), (2, 2), (3, 3)})
    rng = random.Random(37)
    baseline = random.Random(37)
    update = resolve_objective_capture(
        game_map, goals, Player.AGENT, (4, 1), OPPONENT_START, rng
    )
    assert update == ObjectiveUpdate(goals, None, False)
    assert not update.captured
    assert rng.random() == baseline.random()


def test_single_capture_removes_exactly_one_and_keeps_remaining():
    game_map = get_map(1)
    goals = frozenset({(1, 1), (2, 2), (3, 3)})
    update = resolve_objective_capture(
        game_map, goals, Player.AGENT, (1, 1), OPPONENT_START, random.Random(4)
    )
    assert update.captured
    assert update.captured_position == (1, 1)
    assert not update.regenerated
    assert update.active_objectives == frozenset({(2, 2), (3, 3)})
    assert goals == frozenset({(1, 1), (2, 2), (3, 3)})
    assert score_for_objective(Player.AGENT).differential == 1


def test_second_objective_capture_leaves_only_last_objective():
    game_map = get_map(3)
    update = resolve_objective_capture(
        game_map, {(1, 1), (2, 2)}, Player.OPPONENT,
        AGENT_START, (2, 2), random.Random(5),
    )
    assert update.captured_position == (2, 2)
    assert update.active_objectives == frozenset({(1, 1)})
    assert not update.regenerated
    assert score_for_objective(Player.OPPONENT).differential == -1


def test_last_capture_regenerates_before_melee_on_current_positions():
    """Adiacenza post-move: il melee deve avvenire *dopo* questa operazione."""
    game_map = get_map(4)
    # Agent a (5,4), Opponent a (5,3): posizioni adiacenti.
    update = resolve_objective_capture(
        game_map, {(5, 4)}, Player.AGENT, (5, 4), (5, 3), random.Random(1)
    )
    assert update.captured
    assert update.captured_position == (5, 4)
    assert update.regenerated
    assert len(update.active_objectives) == 3
    assert not update.active_objectives & {
        (5, 4), (5, 3), AGENT_START, OPPONENT_START
    }
    assert update.active_objectives <= game_map.free_cells


def test_regeneration_excludes_current_positions_even_if_away_from_spawns():
    game_map = get_map(2)
    update = resolve_objective_capture(
        game_map, {(1, 1)}, Player.OPPONENT, (9, 9), (1, 1), random.Random(42)
    )
    assert update.regenerated
    assert not update.active_objectives & {
        AGENT_START, OPPONENT_START, (9, 9), (1, 1)
    }


def test_respawn_positions_are_never_goals():
    game_map = get_map(2)
    for seed in range(50):
        result = generate_objectives(game_map, (9, 9), (1, 1), random.Random(seed))
        assert not result & START_CELLS


def test_minimum_size_map_still_allows_three_goals():
    """Con cinque celle extra, due occupate, rimangono esattamente tre celle."""
    free_extra = {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)}
    game_map = GameMap(10, ALL_CELLS - START_CELLS - free_extra)
    goals = generate_objectives(game_map, (1, 1), (1, 2), random.Random(5))
    assert goals == frozenset({(1, 3), (1, 4), (1, 5)})


def test_uniform_generation_over_unordered_sets():
    """Mappa con 5 candidati -> C(5,3)=10 gruppi equiprobabili."""
    free_extra = {(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)}
    game_map = GameMap(11, ALL_CELLS - START_CELLS - free_extra)
    rng = random.Random(89)
    counts = Counter(
        generate_objectives(game_map, AGENT_START, OPPONENT_START, rng)
        for _ in range(20_000)
    )
    expected = {frozenset(c) for c in combinations(free_extra, 3)}
    assert set(counts) == expected
    assert all(abs(count / 20_000 - 0.1) < 0.015 for count in counts.values())


@pytest.mark.parametrize("a,o", [
    ((0, 1), OPPONENT_START),
    ((4, 5), OPPONENT_START),  # ostacolo mappa 1
    (AGENT_START, AGENT_START),
])
def test_reject_invalid_player_positions(a, o):
    with pytest.raises(ValueError):
        generate_objectives(get_map(1), a, o, random.Random(1))


@pytest.mark.parametrize("goals", [
    set(),
    {(1, 1), (2, 2), (3, 3), (4, 4)},
    {AGENT_START},
    {(4, 5)},  # ostacolo mappa 1
    {(0, 1)},
    {OPPONENT_START},
])
def test_reject_invalid_active_objectives(goals):
    with pytest.raises(ValueError):
        resolve_objective_capture(
            get_map(1), goals, Player.AGENT,
            (1, 1), OPPONENT_START, random.Random(2),
        )


def test_reject_goal_occupied_by_other_player():
    with pytest.raises(ValueError, match="altro giocatore"):
        resolve_objective_capture(
            get_map(1), {(2, 2)}, Player.AGENT,
            (1, 1), (2, 2), random.Random(4),
        )


def test_invalid_moving_player_rejected():
    with pytest.raises(ValueError, match="Giocatore non valido"):
        resolve_objective_capture(
            get_map(1), {(1, 1)}, "Agent", (1, 1), OPPONENT_START,
            random.Random(4),
        )


def test_result_is_immutable():
    update = resolve_objective_capture(
        get_map(1), {(1, 1)}, Player.AGENT, (1, 1), OPPONENT_START,
        random.Random(42),
    )
    assert isinstance(update.active_objectives, frozenset)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        update.regenerated = False
