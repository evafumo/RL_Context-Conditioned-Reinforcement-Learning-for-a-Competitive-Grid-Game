"""Test di context.py: pool, indipendenza, validazione e riproducibilita'."""

from collections import Counter
from dataclasses import FrozenInstanceError
from math import isclose
from random import Random

import pytest

from environment.context import (
    ALL_CONTEXTS,
    DEFAULT_CONTEXT_DISTRIBUTION,
    MAP_IDS,
    OPPONENT_IDS,
    ContextDistribution,
    EpisodeContext,
    context_probability,
    get_all_contexts,
    get_context,
    sample_context,
)
from environment.maps import GameMap, get_all_maps, get_map
from environment.opponents import OpponentKind


def test_pool_has_four_maps_six_opponents_and_24_unique_pairs():
    assert MAP_IDS == (1, 2, 3, 4)
    assert OPPONENT_IDS == (1, 2, 3, 4, 5, 6)
    contexts = get_all_contexts()
    assert contexts == ALL_CONTEXTS
    assert len(contexts) == 24
    assert len({(c.map_id, c.opponent_id) for c in contexts}) == 24
    assert [(c.map_id, c.opponent_id) for c in contexts] == [
        (m, k) for m in MAP_IDS for k in OPPONENT_IDS
    ]


def test_default_probability_is_uniform_and_sums_to_one():
    probabilities = [context_probability(c) for c in get_all_contexts()]
    assert all(p == pytest.approx(1 / 24) for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_explicit_context_matches_original_pools():
    for ctx in get_all_contexts():
        assert ctx.game_map is get_map(ctx.map_id)
        assert isinstance(ctx.opponent, OpponentKind)
        assert get_context(ctx.map_id, ctx.opponent_id) == ctx


def test_explicit_context_does_not_use_rng():
    assert get_context(4, OpponentKind.ASH).opponent_id == 6


@pytest.mark.parametrize('map_id, opponent_id', [(0, 1), (5, 2), (True, 1), (1, 0), (1, 7), (1, '1')])
def test_explicit_context_rejects_invalid_ids(map_id, opponent_id):
    with pytest.raises(ValueError):
        get_context(map_id, opponent_id)


def test_context_frozen_and_map_unchanged():
    ctx = get_context(1, 1)
    original = ctx.game_map.obstacles
    with pytest.raises(FrozenInstanceError):
        ctx.opponent = OpponentKind.ASH
    assert ctx.game_map.obstacles == original
    assert ctx.map_id == 1
    assert ctx.opponent_id == 1


def test_context_rejects_external_map_same_id_different_layout():
    custom = GameMap(1, frozenset({(1, 1)}))
    with pytest.raises(ValueError):
        EpisodeContext(custom, OpponentKind.RANDOM)


def test_context_rejects_raw_int_for_opponent_field():
    with pytest.raises(ValueError):
        EpisodeContext(get_map(1), 1)


def test_custom_distribution_is_immutable_and_defensively_copied():
    maps = [0.1, 0.2, 0.3, 0.4]
    opponents = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
    dist = ContextDistribution(maps, opponents)
    maps[0] = 0.9
    opponents[0] = 0.9
    assert dist.q_maps == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert dist.q_opponents == pytest.approx((0.1, 0.1, 0.1, 0.2, 0.2, 0.3))
    with pytest.raises(FrozenInstanceError):
        dist.q_maps = (0.25,) * 4


@pytest.mark.parametrize('kwargs', [
    {'q_maps': (0.5, 0.5)},
    {'q_maps': (0.5, 0.5, 0.0, 0.1)},
    {'q_maps': (-0.1, 0.3, 0.4, 0.4)},
    {'q_maps': (float('nan'), 0.0, 0.0, 1.0)},
    {'q_maps': (float('inf'), 0.0, 0.0, 1.0)},
    {'q_maps': (True, 0.0, 0.0, 0.0)},
    {'q_opponents': (0.2,) * 6},
    {'q_opponents': (0.25,) * 4},
    {'q_opponents': None},
])
def test_invalid_distribution_rejected(kwargs):
    with pytest.raises(ValueError):
        ContextDistribution(**kwargs)


def test_nonuniform_context_probability_factorizes():
    dist = ContextDistribution(
        q_maps=(0.1, 0.2, 0.3, 0.4),
        q_opponents=(0.1, 0.1, 0.1, 0.2, 0.2, 0.3),
    )
    assert context_probability(get_context(4, 6), dist) == pytest.approx(0.12)
    assert context_probability(get_context(1, 2), dist) == pytest.approx(0.01)
    assert sum(context_probability(ctx, dist) for ctx in ALL_CONTEXTS) == pytest.approx(1.0)


def test_forced_distribution_sample_only_specified_pair():
    dist = ContextDistribution(
        q_maps=(0.0, 0.0, 1.0, 0.0),
        q_opponents=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    rng = Random(42)
    for _ in range(100):
        ctx = sample_context(rng, dist)
        assert (ctx.map_id, ctx.opponent_id) == (3, 6)


def test_seed_reproducibility_and_no_global_rng_dependency():
    r1, r2 = Random(518), Random(518)
    s1 = [(x.map_id, x.opponent_id) for x in (sample_context(r1) for _ in range(200))]
    s2 = [(x.map_id, x.opponent_id) for x in (sample_context(r2) for _ in range(200))]
    assert s1 == s2


class FakeRNG:
    def __init__(self, numbers):
        self.numbers = iter(numbers)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.numbers)


def test_two_draws_one_for_each_marginal_in_defined_order():
    rng = FakeRNG((0.26, 0.99))
    result = sample_context(rng)
    assert (result.map_id, result.opponent_id) == (2, 6)
    assert rng.calls == 2


def test_empirical_uniform_joint_distribution():
    rng = Random(2026)
    samples = 48_000
    counter = Counter(
        (ctx.map_id, ctx.opponent_id)
        for ctx in (sample_context(rng) for _ in range(samples))
    )
    assert len(counter) == 24
    assert all(abs(counter[(m, k)] / samples - 1.0 / 24.0) < 0.005
               for m in MAP_IDS for k in OPPONENT_IDS)


def test_empirical_nonuniform_independent_marginals():
    distribution = ContextDistribution(
        q_maps=(0.1, 0.2, 0.3, 0.4),
        q_opponents=(0.1, 0.1, 0.1, 0.2, 0.2, 0.3),
    )
    rng = Random(700)
    samples = 60_000
    joint_counts = Counter(
        (ctx.map_id, ctx.opponent_id)
        for ctx in (sample_context(rng, distribution) for _ in range(samples))
    )
    for ctx in get_all_contexts():
        expected = context_probability(ctx, distribution)
        assert abs(joint_counts[(ctx.map_id, ctx.opponent_id)] / samples - expected) < 0.005


def test_context_sampling_does_not_affect_map_objects():
    before = {m.map_id: m.obstacles for m in get_all_maps()}
    rng = Random(99)
    for _ in range(100):
        sample_context(rng)
    after = {m.map_id: m.obstacles for m in get_all_maps()}
    assert before == after
