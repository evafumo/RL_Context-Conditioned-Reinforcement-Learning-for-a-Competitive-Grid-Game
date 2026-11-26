"""Test del modulo maps.py: geometria statica, vincoli e orientamento."""

from collections import deque
from dataclasses import FrozenInstanceError

import pytest

from environment.maps import (
    ALL_CELLS,
    MAP_POOL,
    GameMap,
    get_all_maps,
    get_map,
    is_traversable,
    render_map,
)
from environment.rules import (
    AGENT_START,
    OPPONENT_START,
    Action,
    is_legal_move,
)


EXPECTED_OBSTACLES = {
    1: {(4, 5), (5, 5), (6, 5), (4, 6), (5, 6), (7, 6)},
    2: {(4, 3), (5, 3), (6, 3), (4, 7), (5, 7), (6, 7)},
    3: {(2, 6), (3, 6), (4, 6), (7, 3), (8, 3), (9, 3)},
    4: {(2, 4), (2, 5), (2, 6), (5, 5), (8, 4), (8, 5), (8, 6)},
}


@pytest.mark.parametrize("map_id", [1, 2, 3, 4])
def test_exact_obstacle_coordinates(map_id):
    game_map = get_map(map_id)
    assert game_map.map_id == map_id
    assert game_map.obstacles == EXPECTED_OBSTACLES[map_id]
    assert len(game_map.free_cells) == 81 - len(EXPECTED_OBSTACLES[map_id])
    assert game_map.free_cells | game_map.obstacles == ALL_CELLS
    assert not (game_map.free_cells & game_map.obstacles)


@pytest.mark.parametrize("map_id", [1, 2, 3, 4])
def test_spawn_cells_and_minimum_free_cells(map_id):
    game_map = get_map(map_id)
    assert is_traversable(game_map, AGENT_START)
    assert is_traversable(game_map, OPPONENT_START)
    assert len(game_map.free_cells - {AGENT_START, OPPONENT_START}) >= 5
    assert not is_traversable(game_map, (0, 1))
    assert not is_traversable(game_map, (10, 9))


def test_pool_order_and_immutability():
    assert tuple(MAP_POOL) == (1, 2, 3, 4)
    assert tuple(m.map_id for m in get_all_maps()) == (1, 2, 3, 4)
    with pytest.raises(TypeError):
        MAP_POOL[5] = get_map(1)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        get_map(1).map_id = 8
    with pytest.raises(AttributeError):
        get_map(1).obstacles.add((1, 1))


def test_defensive_copy_of_obstacles():
    source = {(3, 3)}
    game_map = GameMap(5, source)
    source.add((4, 4))
    assert game_map.obstacles == frozenset({(3, 3)})
    assert (4, 4) in game_map.free_cells


@pytest.mark.parametrize(
    "invalid_obstacles",
    [
        {(5, 1)},
        {(5, 9)},
        {(0, 5)},
        {(10, 5)},
        {(-1, 3)},
        {(1.5, 2)},
        {(True, 2)},
        {(3, 3, 3)},
        ALL_CELLS - {AGENT_START, OPPONENT_START},
    ],
)
def test_invalid_obstacles_rejected(invalid_obstacles):
    with pytest.raises(ValueError):
        GameMap(99, invalid_obstacles)


@pytest.mark.parametrize("invalid_id", [0, -2, True, "1"])
def test_invalid_map_id_rejected(invalid_id):
    with pytest.raises(ValueError):
        GameMap(invalid_id, frozenset())


def test_unknown_map_id():
    with pytest.raises(ValueError, match="Mappa sconosciuta"):
        get_map(5)
    with pytest.raises(ValueError, match="intero"):
        get_map(True)


def test_movement_uses_existing_rules_module():
    obstacles = get_map(1).obstacles
    assert not is_legal_move((4, 4), Action.NORTH, obstacles, (9, 9))
    assert is_legal_move((4, 4), Action.WEST, obstacles, (9, 9))


def test_render_map_cartesian_orientation():
    rows = render_map(get_map(1)).splitlines()
    assert len(rows) == 10
    assert rows[0] == "   1 2 3 4 5 6 7 8 9"
    assert rows[1].startswith(" 9 ")
    assert rows[9].startswith(" 1 ")
    assert rows[1].split()[5] == "O"  # coordinata x=5, y=9
    assert rows[9].split()[5] == "A"  # coordinata x=5, y=1
    assert rows[4].split()[4] == "#"  # coordinata x=4, y=6
    assert rows[5].split()[6] == "#"  # coordinata x=6, y=5
    with pytest.raises(ValueError):
        render_map(get_map(1), (5, 1), (5, 1))


def test_all_four_maps_are_connected_diagnostic():
    """Proprietà dei quattro dati forniti, NON vincolo di GameMap."""
    for game_map in get_all_maps():
        queue = deque([AGENT_START])
        visited = {AGENT_START}
        while queue:
            x, y = queue.popleft()
            for next_position in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if next_position in game_map.free_cells and next_position not in visited:
                    visited.add(next_position)
                    queue.append(next_position)
        assert visited == game_map.free_cells, f"Mappa {game_map.map_id} non connessa"
