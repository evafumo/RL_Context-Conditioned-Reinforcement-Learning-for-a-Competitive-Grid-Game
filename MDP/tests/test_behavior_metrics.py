"""Metriche C: conteggi eventi MDP senza alterare gioco o policy."""
from __future__ import annotations

import pytest

from agent.behavior_metrics import EpisodeEventCounters
from agent.parallel_env import ParallelGridGames
from environment.game import GridGame, HORIZON
from environment.rules import Action, Player, ShotOutcome, MeleeOutcome


def _choose(obs):
    return [int(Action.DUMMY) if m[int(Action.DUMMY)] else int(Action.PASS)
            for m in obs.action_masks]


@pytest.mark.parametrize('seed', [3, 2026, 123456])
def test_event_counter_matches_real_grid_game_and_vp(seed):
    game = GridGame(seed=seed)
    game.reset(seed=seed)
    meter = EpisodeEventCounters()
    transitions = []
    for i in range(HORIZON):
        action = Action.DUMMY if game.legal_actions() == (Action.DUMMY,) else Action.PASS
        result = game.step(action)
        meter.record(result)
        transitions.append(result)
        assert result.terminated == (i == HORIZON - 1)
    out = meter.finish(transitions[-1].outcome)
    captures_agent = sum(int(t.events.captured_objective is not None and
                             t.state.player_to_act is Player.AGENT) for t in transitions)
    captures_opp = sum(int(t.events.captured_objective is not None and
                           t.state.player_to_act is Player.OPPONENT) for t in transitions)
    shots_agent = sum(int(t.events.shot_outcome is not None and
                          t.state.player_to_act is Player.AGENT) for t in transitions)
    shots_opp = sum(int(t.events.shot_outcome is not None and
                        t.state.player_to_act is Player.OPPONENT) for t in transitions)
    kills_agent = sum(int(t.events.killer is Player.AGENT) for t in transitions)
    kills_opp = sum(int(t.events.killer is Player.OPPONENT) for t in transitions)
    melees = sum(int(t.events.melee_outcome is not None) for t in transitions)
    assert out['objective_captures'] == {'agent': captures_agent, 'opponent': captures_opp}
    combat = out['combat']
    assert combat['shots_agent'] == shots_agent
    assert combat['shots_opponent'] == shots_opp
    assert combat['kills_agent'] == kills_agent
    assert combat['kills_opponent'] == kills_opp
    assert combat['deaths_agent'] == kills_opp
    assert combat['deaths_opponent'] == kills_agent
    assert combat['melee_encounters'] == melees
    assert combat['melee_initiated_agent'] + combat['melee_initiated_opponent'] == melees
    assert combat['shot_kills_agent'] == sum(int(t.events.shot_outcome is ShotOutcome.TARGET_KILLED
                                                 and t.state.player_to_act is Player.AGENT) for t in transitions)
    assert combat['shot_kills_opponent'] == sum(int(t.events.shot_outcome is ShotOutcome.TARGET_KILLED
                                                    and t.state.player_to_act is Player.OPPONENT) for t in transitions)
    assert out['final_vp'] == {'agent': transitions[-1].outcome.agent_vp,
                               'opponent': transitions[-1].outcome.opponent_vp}
    assert out['activations'] == HORIZON
    with pytest.raises(RuntimeError, match='consecutive'):
        meter.record(transitions[-1])


def test_opt_in_telemetry_terminal_before_reset_and_no_leak():
    with ParallelGridGames(num_envs=2, num_workers=1, base_seed=718,
                           collect_episode_metrics=True) as env:
        for wave in range(2):
            start_seeds = env.observation.episode_seeds
            for i in range(HORIZON):
                result = env.step(_choose(env.observation))
                assert result.terminal_episode_metrics is not None
                assert len(result.terminal_episode_metrics) == 2
                if i < HORIZON - 1:
                    assert result.terminal_episode_metrics == (None, None)
                else:
                    assert result.dones.all()
                    assert env.observation.episode_seeds != start_seeds
                    for item in result.terminal_episode_metrics:
                        assert item['activations'] == 100
                        assert isinstance(item['final_vp']['agent'], int)
                        assert item['combat']['deaths_agent'] == item['combat']['kills_opponent']
    with ParallelGridGames(num_envs=1, num_workers=1, base_seed=718) as env:
        assert env.step(_choose(env.observation)).terminal_episode_metrics is None


def test_episode_summaries_independent_of_worker_count():
    def run(workers):
        result = None
        with ParallelGridGames(num_envs=2, num_workers=workers, base_seed=912,
                               collect_episode_metrics=True) as env:
            for _ in range(HORIZON):
                result = env.step(_choose(env.observation))
        return result.terminal_episode_metrics
    assert run(1) == run(2)


def test_invalid_telemetry_flag():
    with pytest.raises(TypeError):
        ParallelGridGames(num_envs=1, num_workers=1, base_seed=1,
                          collect_episode_metrics=1)


@pytest.mark.parametrize('actor,events,expected', [
    (Player.AGENT,
     {'effective_action': Action.SHOOT,
      'shot_outcome': ShotOutcome.TARGET_KILLED, 'killer': Player.AGENT},
     {'shots_agent': 1, 'shot_kills_agent': 1, 'kills_agent': 1}),
    (Player.OPPONENT,
     {'effective_action': Action.SHOOT,
      'shot_outcome': ShotOutcome.TARGET_KILLED, 'killer': Player.OPPONENT},
     {'shots_opponent': 1, 'shot_kills_opponent': 1, 'kills_opponent': 1}),
    (Player.AGENT,
     {'melee_outcome': MeleeOutcome.DEFENDER_KILLS,
      'melee_attacker': Player.AGENT, 'killer': Player.OPPONENT},
     {'melee_encounters': 1, 'melee_initiated_agent': 1,
      'melee_kills_opponent': 1, 'kills_opponent': 1}),
    (Player.OPPONENT,
     {'melee_outcome': MeleeOutcome.DEFENDER_KILLS,
      'melee_attacker': Player.OPPONENT, 'killer': Player.AGENT},
     {'melee_encounters': 1, 'melee_initiated_opponent': 1,
      'melee_kills_agent': 1, 'kills_agent': 1}),
])
def test_combat_event_attribution(actor, events, expected):
    from dataclasses import replace
    from environment.game import ActivationEvents

    game = GridGame(seed=2)
    game.reset(seed=2)
    result = game.step(Action.DUMMY if game.legal_actions() == (Action.DUMMY,) else Action.PASS)
    simulated_result = replace(result,
                               state=replace(result.state, player_to_act=actor),
                               events=ActivationEvents(**events))
    counter = EpisodeEventCounters()
    counter.record(simulated_result)
    for key, value in expected.items():
        assert getattr(counter, key) == value


def test_collect_telemetry_does_not_change_mdp_trajectory():
    def run(with_metrics):
        records = []
        with ParallelGridGames(num_envs=2, num_workers=1, base_seed=1221,
                               collect_episode_metrics=with_metrics) as env:
            for _ in range(HORIZON + 1):
                obs = env.observation
                result = env.step(_choose(obs))
                records.append((result.rewards.copy(), result.dones.copy(),
                                result.terminal_utilities.copy(),
                                result.next_observation.spatial.copy(),
                                result.next_observation.episode_seeds))
        return records
    before, after = run(False), run(True)
    for a, b in zip(before, after):
        for x, y in zip(a, b):
            if hasattr(x, 'shape'):
                assert (x == y).all()
            else:
                assert x == y
