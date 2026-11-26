"""Fase 1.6: test delle 100 attivazioni del simulatore e del reward."""

from dataclasses import FrozenInstanceError
from math import isclose
from random import Random

import pytest

import environment.game as game_module
from environment.context import get_all_contexts, get_context
from environment.game import (
    HORIZON, TERMINAL_STATE, GameState, GridGame, GameOutcome,
)
from environment.opponents import OpponentKind
from environment.rules import (
    AGENT_START, OPPONENT_START, Action, MeleeOutcome, Player, ShotOutcome,
)


def new_game(*, seed=123, lam=0.1, map_id=1, opponent_id=1, starter=Player.AGENT):
    game = GridGame(seed=seed, shaping_lambda=lam)
    game.reset(context=get_context(map_id, opponent_id), starting_player=starter)
    return game


def inject(game, *, a=(3, 2), o=(5, 2), goals=((7, 7),), avp=0, ovp=0,
           actor=Player.AGENT, t=1):
    # Stato costruito per collaudare uno specifico ramo della dinamica.
    game._state = GameState(a, o, frozenset(goals), avp, ovp, actor, t)


def test_not_initialized_or_illegal_lambda():
    with pytest.raises(RuntimeError):
        GridGame().step(Action.PASS)
    with pytest.raises(RuntimeError):
        GridGame().legal_actions()
    for bad in (-0.001, float('nan'), float('inf'), True, '0.2'):
        with pytest.raises(ValueError):
            GridGame(shaping_lambda=bad)


@pytest.mark.parametrize('map_id', (1, 2, 3, 4))
def test_reset_has_valid_initial_state(map_id):
    game = new_game(map_id=map_id)
    s = game.state
    assert s.activation == 1
    assert s.agent_position == AGENT_START
    assert s.opponent_position == OPPONENT_START
    assert s.agent_vp == s.opponent_vp == s.score_differential == 0
    assert len(s.active_objectives) == 3
    assert not (s.active_objectives & {AGENT_START, OPPONENT_START})
    assert s.active_objectives <= game.context.game_map.free_cells
    assert game.outcome is None and not game.terminated
    with pytest.raises(FrozenInstanceError):
        s.agent_vp = 900


def test_reset_seed_repeatability_and_start_override():
    game = GridGame(seed=20)
    a = game.reset(seed=321)
    context_a = game.context
    b = game.reset(seed=321)
    assert (a, context_a) == (b, game.context)
    for starter in Player:
        assert game.reset(context=get_context(3, 6), starting_player=starter).player_to_act is starter
    with pytest.raises(TypeError):
        game.reset(context='bad')
    with pytest.raises(TypeError):
        game.reset(starting_player='O')


def test_agent_legal_actions_and_invalid_action_is_atomic():
    game = new_game()
    s = game.state
    assert set(game.legal_actions()) == {Action.NORTH, Action.EAST, Action.WEST, Action.PASS}
    for illegal in (Action.SOUTH, Action.SHOOT, Action.DUMMY, 99, True, 'N'):
        with pytest.raises(ValueError):
            game.step(illegal)
        assert game.state is s
    accepted = game.step(Action.PASS)
    assert accepted.state is s
    assert accepted.action == Action.PASS
    assert accepted.reward == 0.0
    assert accepted.events.effective_action == Action.PASS
    assert accepted.next_state.player_to_act is Player.OPPONENT
    assert accepted.next_state.activation == 2


def test_opponent_turn_consumes_only_dummy_and_executes_its_policy():
    game = new_game(opponent_id=2, starter=Player.OPPONENT)
    assert game.legal_actions() == (Action.DUMMY,)
    with pytest.raises(ValueError):
        game.step(Action.SOUTH)
    result = game.step(Action.DUMMY)
    assert result.forced
    assert result.events.effective_action in (Action.SOUTH, Action.EAST, Action.WEST)
    assert result.state.player_to_act is Player.OPPONENT
    assert result.next_state.player_to_act is Player.AGENT
    assert result.next_state.activation == 2
    assert result.state.agent_position == result.next_state.agent_position


def test_agent_move_captures_objective_without_melee():
    game = new_game()
    inject(game, a=(3, 2), o=(8, 8), goals=((4, 2), (7, 7)))
    result = game.step(Action.EAST)
    assert result.events.captured_objective == (4, 2)
    assert not result.events.objectives_regenerated
    assert result.events.melee_outcome is None
    assert result.next_state.agent_position == (4, 2)
    assert result.next_state.active_objectives == frozenset({(7, 7)})
    assert (result.next_state.agent_vp, result.next_state.opponent_vp) == (1, 0)
    assert result.reward == pytest.approx(0.1)


def test_moving_into_melee_capture_then_attacker_kill(monkeypatch):
    game = new_game()
    inject(game, a=(3, 2), o=(5, 2), goals=((4, 2), (7, 7)))
    monkeypatch.setattr(game_module, 'resolve_melee', lambda rng: MeleeOutcome.ATTACKER_KILLS)
    result = game.step(Action.EAST)
    ev = result.events
    assert ev.captured_objective == (4, 2)
    assert ev.melee_attacker is Player.AGENT
    assert ev.melee_outcome is MeleeOutcome.ATTACKER_KILLS
    assert ev.killer is Player.AGENT
    assert (result.next_state.agent_vp, result.next_state.opponent_vp) == (3, -5)
    assert result.next_state.score_differential == 8
    assert result.reward == pytest.approx(0.8)
    assert (result.next_state.agent_position, result.next_state.opponent_position) == (
        AGENT_START, OPPONENT_START
    )
    assert result.next_state.active_objectives == frozenset({(7, 7)})
    assert result.next_state.player_to_act is Player.OPPONENT


def test_last_objective_regenerates_before_melee_and_persists_after_death(monkeypatch):
    game = new_game(seed=31)
    inject(game, a=(3, 2), o=(5, 2), goals=((4, 2),))
    monkeypatch.setattr(game_module, 'resolve_melee', lambda rng: MeleeOutcome.DEFENDER_KILLS)
    result = game.step(Action.EAST)
    assert result.events.captured_objective == (4, 2)
    assert result.events.objectives_regenerated
    assert result.events.killer is Player.OPPONENT
    assert (result.next_state.agent_vp, result.next_state.opponent_vp) == (-4, 2)
    assert result.reward == pytest.approx(-0.6)
    assert result.next_state.agent_position == AGENT_START
    assert result.next_state.opponent_position == OPPONENT_START
    assert len(result.next_state.active_objectives) == 3
    assert (4, 2) not in result.next_state.active_objectives
    assert not ({AGENT_START, OPPONENT_START, (5, 2)} & result.next_state.active_objectives)


def test_melee_at_start_is_forced_and_does_not_invoke_opponent_policy(monkeypatch):
    game = new_game(starter=Player.OPPONENT)
    inject(game, a=(3, 2), o=(4, 2), goals=((7, 7),), actor=Player.OPPONENT)
    def forbidden(*args, **kwargs):
        raise AssertionError('Opponent should not be called during forced melee')
    monkeypatch.setattr(game_module, 'choose_opponent_action', forbidden)
    monkeypatch.setattr(game_module, 'resolve_melee', lambda rng: MeleeOutcome.BOTH_SURVIVE)
    assert game.legal_actions() == (Action.DUMMY,)
    transition = game.step(Action.DUMMY)
    assert transition.events.effective_action is None
    assert transition.events.melee_attacker is Player.OPPONENT
    assert transition.events.melee_outcome is MeleeOutcome.BOTH_SURVIVE
    assert transition.next_state.agent_position == (3, 2)
    assert transition.next_state.opponent_position == (4, 2)
    assert transition.next_state.player_to_act is Player.AGENT
    assert transition.reward == 0.0
    # Continua obbligatoriamente in melee, ma ora Agent e' attaccante.
    assert game.legal_actions() == (Action.DUMMY,)
    second = game.step(Action.DUMMY)
    assert second.events.melee_attacker is Player.AGENT
    assert second.next_state.player_to_act is Player.OPPONENT


def test_counterattack_kill_awards_points_to_defender(monkeypatch):
    game = new_game()
    inject(game, a=(3, 2), o=(4, 2), goals=((7, 7),))
    monkeypatch.setattr(game_module, 'resolve_melee', lambda rng: MeleeOutcome.DEFENDER_KILLS)
    tr = game.step(Action.DUMMY)
    assert tr.events.killer is Player.OPPONENT
    assert (tr.next_state.agent_vp, tr.next_state.opponent_vp) == (-5, 2)
    assert tr.reward == pytest.approx(-0.7)


def test_shoot_hit_and_miss(monkeypatch):
    game = new_game()
    inject(game, a=(3, 2), o=(5, 2), goals=((7, 7),))
    assert Action.SHOOT in game.legal_actions()
    monkeypatch.setattr(game_module, 'resolve_shot', lambda rng: ShotOutcome.MISS)
    tr = game.step(Action.SHOOT)
    assert tr.events.shot_outcome is ShotOutcome.MISS
    assert tr.events.killer is None
    assert tr.reward == 0.0
    assert tr.next_state.agent_position == (3, 2)
    assert tr.next_state.opponent_position == (5, 2)
    assert tr.next_state.active_objectives == frozenset({(7, 7)})
    game2 = new_game()
    inject(game2, a=(3, 2), o=(5, 2), goals=((7, 7),))
    monkeypatch.setattr(game_module, 'resolve_shot', lambda rng: ShotOutcome.TARGET_KILLED)
    tr2 = game2.step(Action.SHOOT)
    assert tr2.events.killer is Player.AGENT
    assert tr2.next_state.score_differential == 7
    assert tr2.reward == pytest.approx(0.7)
    assert (tr2.next_state.agent_position, tr2.next_state.opponent_position) == (
        AGENT_START, OPPONENT_START
    )
    assert tr2.next_state.active_objectives == frozenset({(7, 7)})


def test_no_out_of_bounds_or_obstacle_moves():
    game = new_game(map_id=1)
    inject(game, a=(3, 5), o=(7, 9), goals=((7, 7),))
    assert Action.EAST not in game.legal_actions()  # (4,5) obstacle
    with pytest.raises(ValueError):
        game.step(Action.EAST)


def test_terminal_reward_uses_d_before_100_even_if_last_action_kills(monkeypatch):
    game = new_game(lam=0.2)
    inject(game, a=(3, 2), o=(5, 2), goals=((7, 7),), avp=3, ovp=1, t=100)
    monkeypatch.setattr(game_module, 'resolve_shot', lambda rng: ShotOutcome.TARGET_KILLED)
    tr = game.step(Action.SHOOT)
    assert tr.terminated and tr.next_state is TERMINAL_STATE
    assert game.state is TERMINAL_STATE
    assert tr.outcome == game.outcome
    assert tr.outcome.agent_vp == 5 and tr.outcome.opponent_vp == -4
    assert tr.outcome.score_differential == 9
    assert tr.outcome.winner is Player.AGENT and tr.outcome.utility == 1
    # R_101 = U(D_final) - lambda * D_100, dove D_100=2.
    assert tr.reward == pytest.approx(1 - 0.2 * 2)
    with pytest.raises(RuntimeError):
        game.step(Action.DUMMY)


def test_terminal_tie_loss_and_final_capture():
    game = new_game(lam=0.1)
    inject(game, a=(3, 2), o=(8, 8), goals=((7, 7),), t=100)
    tr = game.step(Action.PASS)
    assert tr.outcome.winner is None and tr.outcome.utility == 0
    assert tr.reward == 0.0

    game = new_game(lam=0.25)
    inject(game, a=(3, 2), o=(8, 8), goals=((7, 7),), avp=0, ovp=1, t=100)
    tr = game.step(Action.PASS)
    assert tr.outcome.winner is Player.OPPONENT
    assert tr.reward == pytest.approx(0.25)  # old D=-1

    game = new_game(lam=0.1)
    inject(game, a=(3, 2), o=(8, 8), goals=((4, 2), (7, 7)), t=100)
    tr = game.step(Action.EAST)
    assert tr.events.captured_objective == (4, 2)
    assert tr.outcome.winner is Player.AGENT and tr.outcome.utility == 1
    assert tr.outcome.agent_vp == 1
    assert tr.reward == pytest.approx(1.0)  # old D=0


def test_play_episode_uses_a_single_fixed_context_and_correct_turn_count():
    context = get_context(4, 6)
    game = GridGame(seed=50, shaping_lambda=0.1)
    trace = game.play_episode(context=context, starting_player=Player.OPPONENT)
    assert trace.context == context
    assert len(trace.transitions) == HORIZON == 100
    assert trace.transitions[0].state.activation == 1
    assert trace.transitions[-1].state.activation == 100
    assert all(tr.state.player_to_act is (
        Player.OPPONENT if i % 2 == 0 else Player.AGENT
    ) for i, tr in enumerate(trace.transitions))
    assert all(not tr.terminated for tr in trace.transitions[:-1])
    assert trace.transitions[-1].terminated
    assert game.terminated
    assert trace.transitions[-1].outcome == trace.outcome
    assert trace.total_reward == pytest.approx(trace.outcome.utility, abs=1e-10)


def test_custom_agent_policy_can_play_full_episode():
    calls = 0
    def always_pass(state, legal, rng):
        nonlocal calls
        calls += 1
        assert state.player_to_act is Player.AGENT
        return Action.PASS

    game = GridGame(seed=9)
    trace = game.play_episode(agent_policy=always_pass, context=get_context(2, 3))
    assert len(trace.transitions) == 100
    assert calls <= 50
    assert trace.total_reward == pytest.approx(trace.outcome.utility)


@pytest.mark.parametrize('context', get_all_contexts())
def test_24_contexts_finish_without_illegal_transitions_and_shaping_identity(context):
    for seed in (0, 1, 17):
        game = GridGame(seed=seed, shaping_lambda=0.13)
        trace = game.play_episode(context=context)
        assert len(trace.transitions) == 100
        assert trace.outcome.utility in (0, 1)
        assert isclose(trace.total_reward, trace.outcome.utility, abs_tol=1e-10)
        for i, tr in enumerate(trace.transitions):
            assert tr.state.activation == i + 1
            if i < 99:
                assert tr.next_state.activation == i + 2
                assert tr.next_state.player_to_act is (
                    Player.AGENT if tr.state.player_to_act is Player.OPPONENT
                    else Player.OPPONENT
                )
                assert tr.next_state.active_objectives <= context.game_map.free_cells
                assert 1 <= len(tr.next_state.active_objectives) <= 3
                assert tr.next_state.agent_position != tr.next_state.opponent_position
                assert tr.next_state.agent_position in context.game_map.free_cells
                assert tr.next_state.opponent_position in context.game_map.free_cells
                assert tr.reward == pytest.approx(
                    0.13 * (tr.next_state.score_differential - tr.state.score_differential)
                )
            else:
                assert tr.next_state is TERMINAL_STATE
                assert tr.reward == pytest.approx(
                    trace.outcome.utility - 0.13 * tr.state.score_differential
                )
            if tr.events.killer is not None and not tr.terminated:
                assert tr.next_state.agent_position == AGENT_START
                assert tr.next_state.opponent_position == OPPONENT_START


def test_reproducible_full_episode():
    def run():
        game = GridGame(seed=81)
        trace = game.play_episode()
        return trace
    assert run() == run()


def test_reset_after_termination():
    game = GridGame(seed=2)
    game.play_episode()
    assert game.terminated and game.outcome is not None
    s = game.reset(context=get_context(1, 1), starting_player=Player.AGENT)
    assert s.activation == 1 and not game.terminated and game.outcome is None


def test_malformed_opponent_policy_is_rejected_before_state_changes(monkeypatch):
    game = new_game(starter=Player.OPPONENT, opponent_id=1)
    original = game.state
    monkeypatch.setattr(game_module, 'choose_opponent_action', lambda *args: Action.NORTH)
    with pytest.raises(RuntimeError, match='illegale'):
        game.step(Action.DUMMY)
    assert game.state is original
