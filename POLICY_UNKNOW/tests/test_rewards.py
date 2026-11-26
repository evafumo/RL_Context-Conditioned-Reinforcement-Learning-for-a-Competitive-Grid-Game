"""Fase 2 — test rigorosi della funzione di reward del simulatore MDP.

Eseguire dalla radice del progetto (cartella Giulio):

    python -m pytest tests/test_rewards.py -v
    python -m pytest tests/ -v

Riferimento matematico (H = 100, gamma = 1):
    D_t = VP_A(t) - VP_O(t)
    Phi(s) = lambda * D(s) negli stati non terminali; Phi(s_dagger) = 0
    R_{t+1} = lambda * (D_{t+1} - D_t), per t < 100
    R_101 = U(D_final) - lambda * D_100, per t = 100
    U(D_final) = 1 se D_final > 0, 0 altrimenti
    sum(R) = U(D_final) per una partita completa con D_1 = 0.

I test agiscono sul simulatore esistente: non reimplementano il reward.
Alcuni stati vengono preparati artificialmente per testare esiti rari e
l'attivazione 100 senza dipendere dalla fortuna dei generatori casuali.
"""

from __future__ import annotations

from math import isclose

import pytest

import environment.game as game_module
from environment.context import get_all_contexts, get_context
from environment.game import (
    HORIZON,
    TERMINAL_STATE,
    GameState,
    GridGame,
)
from environment.rules import (
    AGENT_START,
    OPPONENT_START,
    Action,
    MeleeOutcome,
    Player,
    ShotOutcome,
)


# Posizioni libere nella mappa 1 e presenti anche nei test della Fase 1.
GOAL = (7, 7)
CAPTURE_GOAL = (4, 2)


def _game(lam: float, *, starter: Player = Player.AGENT, seed: int = 101) -> GridGame:
    """Crea una partita inizializzata con un contesto noto."""
    game = GridGame(seed=seed, shaping_lambda=lam)
    game.reset(context=get_context(1, 1), starting_player=starter)
    return game


def _set_state(
    game: GridGame,
    *,
    agent: tuple[int, int] = (3, 2),
    opponent: tuple[int, int] = (8, 8),
    goals: tuple[tuple[int, int], ...] = (GOAL,),
    agent_vp: int = 4,
    opponent_vp: int = 2,
    player: Player = Player.AGENT,
    activation: int = 20,
) -> GameState:
    """Colloca lo stato prima dello step, per testare esiti deterministici."""
    state = GameState(
        agent_position=agent,
        opponent_position=opponent,
        active_objectives=frozenset(goals),
        agent_vp=agent_vp,
        opponent_vp=opponent_vp,
        player_to_act=player,
        activation=activation,
    )
    game._state = state
    return state


def _force_opponent_action(monkeypatch: pytest.MonkeyPatch, action: Action) -> None:
    """Rende deterministica la sola scelta della policy avversaria."""
    monkeypatch.setattr(game_module, "choose_opponent_action", lambda *args: action)


def _post_differential(transition) -> int:
    """Ricava D dopo la risoluzione anche nell'ultimo turno."""
    if transition.terminated:
        assert transition.next_state is TERMINAL_STATE
        assert transition.outcome is not None
        return transition.outcome.score_differential
    assert isinstance(transition.next_state, GameState)
    return transition.next_state.score_differential


def _delta_from_events(transition) -> int:
    """Ricava indipendentemente Delta D dai soli eventi regolamentari."""
    delta = 0
    if transition.events.captured_objective is not None:
        delta += 1 if transition.state.player_to_act is Player.AGENT else -1
    if transition.events.killer is Player.AGENT:
        delta += 7
    elif transition.events.killer is Player.OPPONENT:
        delta -= 7
    return delta


def _assert_reward_formula(transition, lam: float) -> None:
    """Verifica formula e punteggio con due ricostruzioni indipendenti."""
    before = transition.state.score_differential
    after = _post_differential(transition)
    assert after - before == _delta_from_events(transition)
    if transition.terminated:
        expected_utility = int(after > 0)
        assert transition.outcome is not None
        assert transition.outcome.utility == expected_utility
        assert transition.reward == pytest.approx(expected_utility - lam * before)
    else:
        assert transition.reward == pytest.approx(lam * (after - before))


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1, 0.5, 1.0, 2.5])
@pytest.mark.parametrize("d", [-11, -1, 0, 1, 13])
def test_nonterminal_pass_does_not_change_score_or_reward(lam, d):
    game = _game(lam)
    _set_state(game, agent_vp=max(d, 0), opponent_vp=max(-d, 0))
    result = game.step(Action.PASS)

    assert not result.terminated
    assert result.events.captured_objective is None
    assert result.events.killer is None
    assert result.next_state.score_differential == d
    assert result.reward == pytest.approx(0.0)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("lam", [0.0, 0.07, 0.1, 0.75, 3.0])
@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
def test_nonterminal_objective_capture_reward(monkeypatch, lam, actor):
    game = _game(lam, starter=actor)
    if actor is Player.AGENT:
        _set_state(
            game, agent=(3, 2), opponent=(8, 8),
            goals=(CAPTURE_GOAL, GOAL), player=actor,
        )
        action = Action.EAST
        expected_delta = 1
    else:
        _set_state(
            game, agent=(8, 8), opponent=(5, 2),
            goals=(CAPTURE_GOAL, GOAL), player=actor,
        )
        _force_opponent_action(monkeypatch, Action.WEST)
        action = Action.DUMMY
        expected_delta = -1

    result = game.step(action)
    assert result.events.captured_objective == CAPTURE_GOAL
    assert not result.events.objectives_regenerated
    assert result.events.killer is None
    assert _post_differential(result) - result.state.score_differential == expected_delta
    assert result.reward == pytest.approx(expected_delta * lam)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("lam", [0.0, 0.1, 0.4, 1.0])
def test_last_objective_regeneration_adds_no_extra_reward(monkeypatch, actor, lam):
    game = _game(lam, starter=actor)
    if actor is Player.AGENT:
        _set_state(game, agent=(3, 2), opponent=(8, 8),
                   goals=(CAPTURE_GOAL,), player=actor)
        action = Action.EAST
        expected = lam
    else:
        _set_state(game, agent=(8, 8), opponent=(5, 2),
                   goals=(CAPTURE_GOAL,), player=actor)
        _force_opponent_action(monkeypatch, Action.WEST)
        action = Action.DUMMY
        expected = -lam

    result = game.step(action)
    assert result.events.captured_objective == CAPTURE_GOAL
    assert result.events.objectives_regenerated
    assert len(result.next_state.active_objectives) == 3
    assert result.reward == pytest.approx(expected)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("outcome", [ShotOutcome.MISS, ShotOutcome.TARGET_KILLED])
@pytest.mark.parametrize("lam", [0.0, 0.1, 0.4, 1.3])
def test_nonterminal_shoot_reward_and_respawn(monkeypatch, actor, outcome, lam):
    game = _game(lam, starter=actor)
    _set_state(game, agent=(3, 2), opponent=(5, 2), player=actor)
    monkeypatch.setattr(game_module, "resolve_shot", lambda rng: outcome)
    if actor is Player.OPPONENT:
        _force_opponent_action(monkeypatch, Action.SHOOT)
    action = Action.SHOOT if actor is Player.AGENT else Action.DUMMY

    result = game.step(action)
    expected_delta = (
        (7 if actor is Player.AGENT else -7)
        if outcome is ShotOutcome.TARGET_KILLED else 0
    )
    assert result.events.shot_outcome is outcome
    assert result.reward == pytest.approx(lam * expected_delta)
    if expected_delta:
        assert result.events.killer is actor
        assert (result.next_state.agent_position, result.next_state.opponent_position) == (
            AGENT_START, OPPONENT_START,
        )
    else:
        assert result.events.killer is None
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("outcome", list(MeleeOutcome))
@pytest.mark.parametrize("lam", [0.0, 0.1, 0.6, 1.7])
def test_forced_melee_reward_for_every_outcome(monkeypatch, actor, outcome, lam):
    game = _game(lam, starter=actor)
    _set_state(game, agent=(3, 2), opponent=(4, 2), player=actor)
    monkeypatch.setattr(game_module, "resolve_melee", lambda rng: outcome)
    assert game.legal_actions() == (Action.DUMMY,)

    result = game.step(Action.DUMMY)
    sign = 1 if actor is Player.AGENT else -1
    expected_delta = {
        MeleeOutcome.ATTACKER_KILLS: 7 * sign,
        MeleeOutcome.DEFENDER_KILLS: -7 * sign,
        MeleeOutcome.BOTH_SURVIVE: 0,
    }[outcome]

    assert result.forced
    assert result.events.melee_attacker is actor
    assert result.events.melee_outcome is outcome
    assert result.reward == pytest.approx(lam * expected_delta)
    if outcome is not MeleeOutcome.BOTH_SURVIVE:
        assert (result.next_state.agent_position, result.next_state.opponent_position) == (
            AGENT_START, OPPONENT_START,
        )
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("outcome", list(MeleeOutcome))
@pytest.mark.parametrize("lam", [0.0, 0.1, 1.1])
def test_capture_before_melee_combines_score_changes(monkeypatch, actor, outcome, lam):
    game = _game(lam, starter=actor)
    if actor is Player.AGENT:
        _set_state(game, agent=(3, 2), opponent=(5, 2),
                   goals=(CAPTURE_GOAL, GOAL), player=actor)
        action = Action.EAST
        sign = 1
    else:
        _set_state(game, agent=(5, 2), opponent=(3, 2),
                   goals=(CAPTURE_GOAL, GOAL), player=actor)
        _force_opponent_action(monkeypatch, Action.EAST)
        action = Action.DUMMY
        sign = -1

    monkeypatch.setattr(game_module, "resolve_melee", lambda rng: outcome)
    result = game.step(action)
    expected_delta = sign + {
        MeleeOutcome.ATTACKER_KILLS: 7 * sign,
        MeleeOutcome.DEFENDER_KILLS: -7 * sign,
        MeleeOutcome.BOTH_SURVIVE: 0,
    }[outcome]

    assert result.events.captured_objective == CAPTURE_GOAL
    assert result.events.melee_outcome is outcome
    assert result.reward == pytest.approx(lam * expected_delta)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("lam", [0.0, 0.05, 0.2, 1.0, 2.0])
@pytest.mark.parametrize("d_before", [-12, -1, 0, 1, 17])
def test_terminal_pass_uses_pre_activation_score(lam, d_before):
    game = _game(lam)
    _set_state(game, agent_vp=max(d_before, 0),
               opponent_vp=max(-d_before, 0), activation=HORIZON)
    result = game.step(Action.PASS)
    utility = int(d_before > 0)

    assert result.state.activation == HORIZON
    assert result.terminated
    assert result.next_state is TERMINAL_STATE
    assert game.terminated
    assert result.outcome == game.outcome
    assert result.outcome.utility == utility
    assert result.reward == pytest.approx(utility - lam * d_before)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize(
    "starting_d, expected_d, expected_utility",
    [
        (-2, -1, 0),  # La cattura non basta a vincere.
        (-1, 0, 0),   # Il pareggio vale zero.
        (0, 1, 1),    # Una cattura all'ultimo turno decide la vittoria.
        (2, 3, 1),
    ],
)
@pytest.mark.parametrize("lam", [0.0, 0.15, 1.5])
def test_terminal_capture_uses_post_capture_utility_and_pre_capture_potential(
    starting_d, expected_d, expected_utility, lam
):
    game = _game(lam)
    _set_state(
        game, agent=(3, 2), opponent=(8, 8),
        goals=(CAPTURE_GOAL, GOAL),
        agent_vp=max(starting_d, 0), opponent_vp=max(-starting_d, 0),
        activation=HORIZON,
    )
    result = game.step(Action.EAST)

    assert result.events.captured_objective == CAPTURE_GOAL
    assert result.outcome.score_differential == expected_d
    assert result.outcome.utility == expected_utility
    assert result.reward == pytest.approx(expected_utility - lam * starting_d)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.3, 2.0])
@pytest.mark.parametrize(
    "actor, starting_d, expected_d, expected_utility",
    [
        (Player.AGENT, -2, 5, 1),  # Uccisione: perdita -> vittoria.
        (Player.AGENT, -7, 0, 0),  # Uccisione: perdita -> pareggio.
        (Player.AGENT, 2, 9, 1),
        (Player.OPPONENT, 2, -5, 0),  # Morte: vantaggio -> sconfitta.
        (Player.OPPONENT, 7, 0, 0),   # Morte: vantaggio -> pareggio.
    ],
)
def test_terminal_shoot_kill_does_not_add_last_delta_shaping(
    monkeypatch, lam, actor, starting_d, expected_d, expected_utility
):
    game = _game(lam, starter=actor)
    _set_state(
        game, agent=(3, 2), opponent=(5, 2), player=actor,
        agent_vp=max(starting_d, 0), opponent_vp=max(-starting_d, 0),
        activation=HORIZON,
    )
    monkeypatch.setattr(game_module, "resolve_shot", lambda rng: ShotOutcome.TARGET_KILLED)
    if actor is Player.OPPONENT:
        _force_opponent_action(monkeypatch, Action.SHOOT)
    action = Action.SHOOT if actor is Player.AGENT else Action.DUMMY

    result = game.step(action)
    assert result.terminated
    assert result.events.killer is actor
    assert result.outcome.score_differential == expected_d
    assert result.outcome.utility == expected_utility
    assert result.reward == pytest.approx(expected_utility - lam * starting_d)
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("lam", [0.0, 0.1, 1.25])
@pytest.mark.parametrize("outcome", list(MeleeOutcome))
def test_terminal_capture_and_melee_are_resolved_before_terminal_reward(
    monkeypatch, lam, outcome
):
    game = _game(lam)
    _set_state(
        game, agent=(3, 2), opponent=(5, 2),
        goals=(CAPTURE_GOAL, GOAL), agent_vp=0, opponent_vp=0,
        activation=HORIZON,
    )
    monkeypatch.setattr(game_module, "resolve_melee", lambda rng: outcome)

    result = game.step(Action.EAST)
    expected_final_d = {
        MeleeOutcome.ATTACKER_KILLS: 8,
        MeleeOutcome.DEFENDER_KILLS: -6,
        MeleeOutcome.BOTH_SURVIVE: 1,
    }[outcome]

    assert result.events.captured_objective == CAPTURE_GOAL
    assert result.events.melee_outcome is outcome
    assert result.outcome.score_differential == expected_final_d
    assert result.reward == pytest.approx(int(expected_final_d > 0))  # D_100 = 0
    _assert_reward_formula(result, lam)


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.8])
def test_terminal_forced_melee_uses_dummy_but_terminal_reward_is_not_zero(
    monkeypatch, lam
):
    game = _game(lam, starter=Player.OPPONENT)
    _set_state(
        game, agent=(3, 2), opponent=(4, 2),
        agent_vp=4, opponent_vp=2, player=Player.OPPONENT,
        activation=HORIZON,
    )
    monkeypatch.setattr(game_module, "resolve_melee", lambda rng: MeleeOutcome.ATTACKER_KILLS)
    result = game.step(Action.DUMMY)

    assert result.forced
    assert result.outcome.score_differential == -5  # 2 - 7
    assert result.outcome.utility == 0
    assert result.reward == pytest.approx(-lam * 2)
    _assert_reward_formula(result, lam)


def test_nonterminal_and_terminal_formulas_differ_even_for_same_state_and_action():
    lam = 0.2
    game_at_99 = _game(lam)
    _set_state(game_at_99, agent_vp=6, opponent_vp=1, activation=99)
    result_99 = game_at_99.step(Action.PASS)

    game_at_100 = _game(lam)
    _set_state(game_at_100, agent_vp=6, opponent_vp=1, activation=100)
    result_100 = game_at_100.step(Action.PASS)

    assert result_99.reward == pytest.approx(0.0)
    assert result_100.reward == pytest.approx(1.0 - lam * 5)
    assert result_99.next_state.score_differential == 5
    assert result_100.next_state is TERMINAL_STATE


@pytest.mark.parametrize("lam", [0.0, 0.2, 1.0])
def test_reward_depends_on_differential_not_absolute_victory_points(monkeypatch, lam):
    first = _game(lam)
    second = _game(lam)
    for game, common_offset in ((first, 0), (second, 30)):
        _set_state(game, agent=(3, 2), opponent=(5, 2),
                   agent_vp=3 + common_offset, opponent_vp=1 + common_offset)
    monkeypatch.setattr(game_module, "resolve_shot", lambda rng: ShotOutcome.TARGET_KILLED)
    one = first.step(Action.SHOOT)
    two = second.step(Action.SHOOT)
    assert one.reward == pytest.approx(two.reward)
    assert one.next_state.score_differential == two.next_state.score_differential
    _assert_reward_formula(one, lam)
    _assert_reward_formula(two, lam)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("lam", [0.0, 0.15, 1.0])
def test_expected_nonterminal_shot_reward_equals_probability_times_seven(
    monkeypatch, actor, lam
):
    rewards = {}
    for outcome in ShotOutcome:
        game = _game(lam, starter=actor)
        _set_state(game, agent=(3, 2), opponent=(5, 2), player=actor)
        monkeypatch.setattr(game_module, "resolve_shot", lambda rng, x=outcome: x)
        if actor is Player.OPPONENT:
            _force_opponent_action(monkeypatch, Action.SHOOT)
        rewards[outcome] = game.step(
            Action.SHOOT if actor is Player.AGENT else Action.DUMMY
        ).reward
    expected = 1.4 * lam * (1 if actor is Player.AGENT else -1)
    assert 0.20 * rewards[ShotOutcome.TARGET_KILLED] + (
        0.80 * rewards[ShotOutcome.MISS]
    ) == pytest.approx(expected)


@pytest.mark.parametrize("actor", [Player.AGENT, Player.OPPONENT])
@pytest.mark.parametrize("lam", [0.0, 0.15, 1.0])
def test_expected_nonterminal_melee_reward_is_4_375_lambda(monkeypatch, actor, lam):
    rewards = {}
    for outcome in MeleeOutcome:
        game = _game(lam, starter=actor)
        _set_state(game, agent=(3, 2), opponent=(4, 2), player=actor)
        monkeypatch.setattr(game_module, "resolve_melee", lambda rng, x=outcome: x)
        rewards[outcome] = game.step(Action.DUMMY).reward

    expected = 4.375 * lam * (1 if actor is Player.AGENT else -1)
    actual = (
        0.75 * rewards[MeleeOutcome.ATTACKER_KILLS]
        + 0.125 * rewards[MeleeOutcome.DEFENDER_KILLS]
        + 0.125 * rewards[MeleeOutcome.BOTH_SURVIVE]
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.4])
def test_expected_nonterminal_capture_plus_melee_reward_is_5_375_lambda(monkeypatch, lam):
    rewards = {}
    for outcome in MeleeOutcome:
        game = _game(lam)
        _set_state(game, agent=(3, 2), opponent=(5, 2),
                   goals=(CAPTURE_GOAL, GOAL))
        monkeypatch.setattr(game_module, "resolve_melee", lambda rng, x=outcome: x)
        rewards[outcome] = game.step(Action.EAST).reward
    expectation = (
        0.75 * rewards[MeleeOutcome.ATTACKER_KILLS]
        + 0.125 * rewards[MeleeOutcome.DEFENDER_KILLS]
        + 0.125 * rewards[MeleeOutcome.BOTH_SURVIVE]
    )
    assert expectation == pytest.approx(5.375 * lam)


@pytest.mark.parametrize("context", get_all_contexts(), ids=lambda c: f"m{c.map_id}_k{c.opponent.value}")
@pytest.mark.parametrize("lam", [0.0, 0.13, 0.8])
@pytest.mark.parametrize("starter", [Player.AGENT, Player.OPPONENT])
def test_complete_episode_reward_identity_every_context(context, lam, starter):
    """Ogni attivazione, ogni evento e telescoping su tutte le 24 coppie (m,k)."""
    game = GridGame(seed=173, shaping_lambda=lam)
    trace = game.play_episode(context=context, starting_player=starter)

    assert len(trace.transitions) == HORIZON
    assert game.terminated
    assert game.outcome == trace.outcome
    assert trace.transitions[0].state.score_differential == 0

    for index, transition in enumerate(trace.transitions, start=1):
        assert transition.state.activation == index
        assert transition.terminated is (index == HORIZON)
        _assert_reward_formula(transition, lam)
        if index < HORIZON:
            assert transition.next_state == trace.transitions[index].state
        else:
            assert transition.next_state is TERMINAL_STATE
            assert transition.outcome == trace.outcome

    # Somma telescopica: lambda * (D_100 - D_1) + U - lambda * D_100 = U.
    assert sum(step.reward for step in trace.transitions) == pytest.approx(
        trace.outcome.utility, abs=1e-9
    )
    assert trace.total_reward == pytest.approx(trace.outcome.utility, abs=1e-9)
    assert trace.outcome.utility == int(trace.outcome.agent_vp > trace.outcome.opponent_vp)


@pytest.mark.parametrize("context_id", [(1, 1), (2, 3), (4, 6)])
def test_shaping_scale_does_not_change_gameplay_trajectory_or_winner(context_id):
    """Il parametro lambda influenza il reward, non le dinamiche del gioco."""
    context = get_context(*context_id)
    episodes = [
        GridGame(seed=913, shaping_lambda=lam).play_episode(
            context=context, starting_player=Player.AGENT
        )
        for lam in (0.0, 0.1, 0.7, 2.0)
    ]
    reference = episodes[0]
    for episode, lam in zip(episodes[1:], (0.1, 0.7, 2.0)):
        assert episode.outcome == reference.outcome
        assert episode.context == reference.context
        for expected, actual in zip(reference.transitions, episode.transitions, strict=True):
            assert actual.state == expected.state
            assert actual.action == expected.action
            assert actual.next_state == expected.next_state
            assert actual.events == expected.events
            assert actual.terminated == expected.terminated
            _assert_reward_formula(actual, lam)
        assert episode.total_reward == pytest.approx(reference.outcome.utility, abs=1e-9)


@pytest.mark.parametrize("d_start", [-9, 0, 7])
@pytest.mark.parametrize("lam", [0.0, 0.1, 0.9])
def test_telescoping_with_artificial_nonzero_initial_differential(d_start, lam):
    """Forma generale: sum(R) = U(D_final) - lambda * D_initial."""
    game = _game(lam, seed=312)
    initial = game.state
    _set_state(
        game,
        agent=initial.agent_position,
        opponent=initial.opponent_position,
        goals=tuple(initial.active_objectives),
        agent_vp=max(d_start, 0),
        opponent_vp=max(-d_start, 0),
        player=initial.player_to_act,
        activation=1,
    )

    rewards = []
    for _ in range(HORIZON):
        legal = game.legal_actions()
        action = Action.DUMMY if legal == (Action.DUMMY,) else game.rng_agent.choice(legal)
        step = game.step(action)
        _assert_reward_formula(step, lam)
        rewards.append(step.reward)

    assert game.outcome is not None
    assert game.terminated
    assert isclose(
        sum(rewards), game.outcome.utility - lam * d_start,
        rel_tol=0.0, abs_tol=1e-9,
    )
