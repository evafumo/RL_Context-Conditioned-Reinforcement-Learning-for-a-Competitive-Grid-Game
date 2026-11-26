"""Fase 5.6 — test valutazione DQN senza training o replay durante l'evaluation.

Dalla root: python -m pytest tests/test_evaluate.py -q
Usa worker spawn reali e una rete QNetwork salvata nel formato di train.py.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.dqn import DQNConfig, DQNTrainer
from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer
from environment.context import MAP_IDS, OPPONENT_IDS, sample_context
from environment.game import HORIZON
from evaluate import (
    EvaluationConfig, EvaluationResult, fixed_context_distribution,
    load_evaluation_network, main, run_evaluation, wilson_interval,
)
from utils.checkpoint import TrainingProgress, save_checkpoint


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu():
    initial_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(initial_threads)


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    torch.manual_seed(42)
    trainer = DQNTrainer(QNetwork(), config=DQNConfig(gamma=1.0), device='cpu')
    replay = ReplayBuffer(capacity=16, seed=987)
    ckpt = tmp_path / 'dqn.pt'
    save_checkpoint(
        ckpt, trainer=trainer, replay_buffer=replay,
        progress=TrainingProgress(0, 0, 0, 1.0),
        metadata={'training_signature': {
            'version': 1, 'shaping_lambda': 0.1,
            'dqn': {'gamma': 1.0, 'learning_rate': 1e-4,
                    'target_sync_interval': 1000, 'huber_delta': 1.0,
                    'grad_clip_norm': 10., 'weight_decay': 0.,
                    'use_bfloat16_amp': False},
        }},
    )
    assert ckpt.is_file()
    return ckpt


def conf(checkpoint_file: Path, **kwargs) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=checkpoint_file,
        output_dir=checkpoint_file.parent / 'eval',
        num_envs=1, num_workers=1, episodes_per_context=1,
        eval_seed=20261015, contexts=((1, 1),),
        **kwargs,
    )


@pytest.mark.parametrize('wins,n', [(0, 1), (1, 1), (0, 10), (10, 10), (5, 10), (47, 100)])
def test_wilson_contains_observed_rate(wins, n):
    low, high = wilson_interval(wins, n)
    assert 0 <= low <= wins / n <= high <= 1
    assert low <= high


@pytest.mark.parametrize('arg', [(-1, 1), (2, 1), (0, 0), (1.0, 2), (True, 1), (0, True)])
def test_wilson_invalid(arg):
    with pytest.raises((TypeError, ValueError)):
        wilson_interval(*arg)


@pytest.mark.parametrize('confidence', [0, 1, -0.2, float('nan'), float('inf'), True])
def test_wilson_invalid_confidence(confidence):
    with pytest.raises((TypeError, ValueError)):
        wilson_interval(1, 2, confidence)


@pytest.mark.parametrize('kwargs', [
    {'num_envs': 0}, {'num_envs': True}, {'num_workers': 0},
    {'num_workers': 2}, {'episodes_per_context': 0},
    {'eval_seed': -1}, {'eval_seed': 2**64},
    {'mode': 'stochastic'}, {'mode': 'greedy', 'epsilon': 0.2},
    {'epsilon': -0.1}, {'epsilon': 1.1}, {'epsilon': float('nan')},
    {'pin_memory': True}, {'confidence': 0},
    {'shaping_lambda': -1}, {'shaping_lambda': float('inf')},
    {'contexts': ()}, {'contexts': ((1, 1), (1, 1))},
    {'contexts': ((0, 1),)}, {'contexts': ((1, 7),)},
    {'contexts': ((True, 1),)}, {'device': 'mps'},
])
def test_bad_config(checkpoint_file, kwargs):
    values = {'checkpoint': checkpoint_file, 'contexts': ((1, 1),),
              'num_envs': 1, 'num_workers': 1}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        EvaluationConfig(**values)


def test_all_24_contexts_and_delta_distributions():
    standard = EvaluationConfig(checkpoint='something.pt')
    assert len(standard.contexts) == 24
    assert len(set(standard.contexts)) == 24
    assert set(standard.contexts) == {(m, k) for m in MAP_IDS for k in OPPONENT_IDS}
    from random import Random
    for pair in standard.contexts:
        distribution = fixed_context_distribution(*pair)
        assert sum(distribution.q_maps) == 1
        assert sum(distribution.q_opponents) == 1
        for seed in range(4):
            ctx = sample_context(Random(seed), distribution)
            assert (ctx.map_id, ctx.opponent_id) == pair
    with pytest.raises(ValueError):
        fixed_context_distribution(3, 7)


def test_checkpoint_load_only_online_network(checkpoint_file):
    payload = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    before = torch.get_rng_state().clone()
    model, signature = load_evaluation_network(checkpoint_file, device='cpu')
    assert signature['shaping_lambda'] == 0.1
    assert not model.training
    assert next(model.parameters()).device.type == 'cpu'
    for name, value in model.state_dict().items():
        assert torch.equal(value, payload['trainer']['online_network'][name])
    assert torch.equal(torch.get_rng_state(), before), 'Il caricamento non deve consumare RNG globale'
    with pytest.raises(RuntimeError, match='CUDA'):
        if not torch.cuda.is_available():
            load_evaluation_network(checkpoint_file, 'cuda:0')
        else:
            raise RuntimeError('CUDA test non applicabile')


def test_reject_invalid_checkpoint(tmp_path, checkpoint_file):
    from copy import deepcopy
    payload = torch.load(checkpoint_file, weights_only=True, map_location='cpu')
    for change in ('format', 'gamma', 'weights', 'metadata'):
        broken = deepcopy(payload)
        if change == 'format':
            broken['format'] = 'wrong'
        elif change == 'gamma':
            broken['metadata']['training_signature']['dqn']['gamma'] = 0.99
        elif change == 'weights':
            del broken['trainer']['online_network']['q_head.weight']
        else:
            broken['metadata'].pop('training_signature')
        bad_path = tmp_path / (change + '.pt')
        torch.save(broken, bad_path)
        with pytest.raises((ValueError, RuntimeError)):
            load_evaluation_network(bad_path)


def test_single_context_full_episode_cpu(checkpoint_file):
    cfg = conf(checkpoint_file)
    # Nessun parametro deve essere aggiornato, nessun replay usato.
    before = checkpoint_file.read_bytes()
    result = run_evaluation(cfg)
    assert isinstance(result, EvaluationResult)
    assert result.summary_path.exists()
    assert result.episodes_path.exists()
    assert checkpoint_file.read_bytes() == before
    summary = json.loads(result.summary_path.read_text('utf-8'))
    assert summary['mode'] == 'greedy'
    assert summary['num_contexts'] == 1
    assert summary['horizon'] == HORIZON
    assert summary['overall']['episodes'] == 1
    assert summary['overall']['activations'] == HORIZON
    assert summary['contexts'][0]['map_id'] == 1
    assert summary['contexts'][0]['opponent_id'] == 1
    episodes = [json.loads(x) for x in result.episodes_path.read_text('utf-8').splitlines()]
    assert len(episodes) == 1
    row = episodes[0]
    assert row['episode_index'] == 0
    assert row['activations'] == HORIZON
    assert row['agent_decisions'] + row['forced_dummy_activations'] == HORIZON
    assert row['forced_dummy_activations'] > 0
    assert row['utility'] == int(row['win'])
    assert isinstance(row['episode_seed'], int)
    assert np.isfinite(row['return'])


def test_remainder_exact_and_seeds_nonoverlapping(checkpoint_file):
    cfg = replace(conf(checkpoint_file), num_envs=2, num_workers=1,
                  episodes_per_context=3, contexts=((1, 2), (4, 6)))
    r1 = run_evaluation(cfg)
    rows1 = [json.loads(x) for x in r1.episodes_path.read_text('utf-8').splitlines()]
    assert len(rows1) == 6
    assert r1.summary['overall']['activations'] == 600
    for map_id, opponent_id in cfg.contexts:
        rows = [r for r in rows1 if (r['map_id'], r['opponent_id']) == (map_id, opponent_id)]
        assert len(rows) == 3
        assert [r['episode_index'] for r in rows] == [0, 1, 2]
        assert len({r['episode_seed'] for r in rows}) == 3
    assert r1.summary['overall']['win_rate'] == pytest.approx(
        np.mean([c['win_rate'] for c in r1.summary['contexts']])
    )
    # Ripetibilita' bit-a-bit sullo stesso host e nella stessa build CPU.
    r2 = run_evaluation(replace(cfg, output_dir=checkpoint_file.parent / 'second'))
    rows2 = [json.loads(x) for x in r2.episodes_path.read_text('utf-8').splitlines()]
    assert rows1 == rows2
    assert r1.summary['overall'] == r2.summary['overall']
    assert r1.summary['contexts'] == r2.summary['contexts']


def test_epsilon_greedy_and_shaping_conflict(checkpoint_file):
    cfg = conf(checkpoint_file)
    with pytest.raises(ValueError, match='shaping_lambda'):
        run_evaluation(replace(cfg, shaping_lambda=0.25))
    result = run_evaluation(replace(cfg, mode='epsilon_greedy', epsilon=1.0))
    assert result.summary['mode'] == 'epsilon_greedy'
    assert result.summary['epsilon'] == 1.0
    assert result.summary['overall']['episodes'] == 1


def test_run_requires_eval_config():
    with pytest.raises(TypeError):
        run_evaluation({'checkpoint': 'bad'})


def test_cli_entrypoint_and_json_files(checkpoint_file):
    path = checkpoint_file.parent / 'cli_results'
    exit_status = main([
        '--checkpoint', str(checkpoint_file), '--num-envs', '1',
        '--num-workers', '1', '--episodes-per-context', '1',
        '--eval-seed', '42', '--output-dir', str(path),
        '--contexts', '1:1',
    ])
    assert exit_status == 0
    assert (path / 'summary.json').is_file()
    assert (path / 'episodes.jsonl').is_file()
    assert json.loads((path / 'summary.json').read_text())['num_contexts'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Richiede GPU CUDA')
def test_cuda_evaluation(checkpoint_file):
    result = run_evaluation(replace(
        conf(checkpoint_file), device='cuda:0', pin_memory=True,
    ))
    assert result.summary['device'] == 'cuda:0'
    assert result.summary['overall']['episodes'] == 1


def test_complete_24_context_summary_without_starting_workers(checkpoint_file, monkeypatch):
    """24 contesti, aggregation/JSON e esattamente N episodi per contesto."""
    import evaluate as evaluate_module
    calls = []

    def fake_pool(model, config, device, map_id, opponent_id, *, phase, env_count, waves, shaping_lambda):
        assert shaping_lambda == 0.1
        calls.append((map_id, opponent_id, env_count * waves))
        return [
            {
                'map_id': map_id, 'opponent_id': opponent_id,
                'episode_index': -1, 'episode_seed': 10000 + map_id * 10 + opponent_id,
                'win': map_id == 1, 'utility': int(map_id == 1),
                'return': 1. if map_id == 1 else 0.,
                'agent_decisions': 50, 'forced_dummy_activations': 50,
                'activations': HORIZON,
            } for _ in range(env_count * waves)
        ]

    monkeypatch.setattr(evaluate_module, '_evaluate_pool', fake_pool)
    config = EvaluationConfig(
        checkpoint=checkpoint_file, num_envs=4, num_workers=2,
        episodes_per_context=5, output_dir=checkpoint_file.parent / 'all_contexts',
    )
    result = run_evaluation(config)
    assert result.summary['num_contexts'] == 24
    assert result.summary['overall']['episodes'] == 120
    assert result.summary['overall']['wins'] == 30
    assert result.summary['overall']['win_rate'] == 0.25
    assert result.summary['overall']['activations'] == 12_000
    assert len(result.summary['contexts']) == 24
    assert len(calls) == 48  # 4+1 episodi: 2 pool per ogni contesto
    assert len(result.episodes_path.read_text('utf-8').splitlines()) == 120


def test_workers_do_not_change_seed_or_cpu_trajectory(checkpoint_file):
    cfg = replace(conf(checkpoint_file), num_envs=2, episodes_per_context=2)
    one = run_evaluation(replace(cfg, num_workers=1,
                                  output_dir=checkpoint_file.parent / 'single'))
    two = run_evaluation(replace(cfg, num_workers=2,
                                  output_dir=checkpoint_file.parent / 'multiple'))
    assert one.episodes_path.read_text('utf-8') == two.episodes_path.read_text('utf-8')
    assert one.summary['overall'] == two.summary['overall']


def test_performance_A_in_evaluation(checkpoint_file):
    """Report additivo: metriche A1-5,8, checkpoint step e CI bootstrap."""
    cfg = replace(conf(checkpoint_file), contexts=((1,1),(2,1),(3,2),(4,3),(1,5),(2,6)),
                  episodes_per_context=2, bootstrap_replicates=300,
                  output_dir=checkpoint_file.parent / 'a_metrics')
    result = run_evaluation(cfg)
    s = result.summary
    assert s['checkpoint_activations'] == 0
    assert len(s['per_map']) == 4
    assert len(s['per_opponent']) == 5
    assert s['robustness']['worst_six_mean_win_rate'] is not None
    assert s['overall']['stratified_bootstrap_ci'][0] <= s['overall']['win_rate'] <= s['overall']['stratified_bootstrap_ci'][1]
    assert s['robustness']['context_win_rate_std_population'] >= 0
    assert sum(c['wins'] for c in s['per_map']) == s['overall']['wins']
    assert sum(c['wins'] for c in s['per_opponent']) == s['overall']['wins']


def test_bad_bootstrap_replicates(checkpoint_file):
    with pytest.raises(ValueError, match='bootstrap_replicates'):
        conf(checkpoint_file, bootstrap_replicates=99)


def test_behavior_C_episode_and_summary(checkpoint_file):
    """C18/19/20: frequenze legali, catture e combattimento, per episodio e aggregato."""
    cfg = replace(conf(checkpoint_file), contexts=((1,1), (2,2)),
                  num_envs=2, num_workers=1, episodes_per_context=3,
                  output_dir=checkpoint_file.parent / 'behavior_metrics')
    result = run_evaluation(cfg)
    summary = result.summary
    rows = [json.loads(x) for x in result.episodes_path.read_text('utf-8').splitlines()]
    assert len(rows) == 6
    names = {'NORTH', 'SOUTH', 'EAST', 'WEST', 'SHOOT', 'PASS'}
    for row in rows:
        b = row['behavior']
        assert set(b['actions']['counts']) == names
        assert 'DUMMY' not in b['actions']['frequencies']
        assert sum(b['actions']['counts'].values()) == row['agent_decisions']
        assert sum(b['actions']['frequencies'].values()) == pytest.approx(1)
        for name in names:
            assert b['actions']['counts'][name] <= b['actions']['legal_opportunities'][name]
            if b['actions']['legal_opportunities'][name] == 0:
                assert b['actions']['selection_given_legal'][name] is None
        c = b['combat']
        assert c['kills_agent'] == c['shot_kills_agent'] + c['melee_kills_agent']
        assert c['kills_opponent'] == c['shot_kills_opponent'] + c['melee_kills_opponent']
        assert c['deaths_agent'] == c['kills_opponent']
        assert c['deaths_opponent'] == c['kills_agent']
        assert c['melee_encounters'] == c['melee_initiated_agent'] + c['melee_initiated_opponent']
        assert (b['final_vp']['agent'] - b['final_vp']['opponent'] > 0) == row['win']
        assert b['final_vp']['agent'] == b['objective_captures']['agent'] + 2*c['kills_agent'] - 5*c['kills_opponent']
        assert b['final_vp']['opponent'] == b['objective_captures']['opponent'] + 2*c['kills_opponent'] - 5*c['kills_agent']
    for group in [summary['overall'], *summary['contexts']]:
        b = group['behavior']
        relevant = [r for r in rows if ('map_id' not in group or (r['map_id'],r['opponent_id']) == (group['map_id'],group['opponent_id']))]
        assert b['episodes'] == len(relevant)
        assert sum(b['actions']['counts'].values()) == sum(r['agent_decisions'] for r in relevant)
        assert b['objective_captures']['agent_total'] == sum(r['behavior']['objective_captures']['agent'] for r in relevant)
        assert b['combat']['kills_agent_total'] == sum(r['behavior']['combat']['kills_agent'] for r in relevant)
        assert b['combat']['shots_agent_total'] == sum(r['behavior']['combat']['shots_agent'] for r in relevant)
    saved = json.loads(result.summary_path.read_text(encoding='utf-8'))
    assert saved['overall']['behavior'] == summary['overall']['behavior']


def test_behavior_aggregation_handles_zero_denominators():
    from evaluate import _episode_action_stats, _aggregate_behavior
    counts = np.zeros(7, dtype=np.int64)
    legal = np.zeros(7, dtype=np.int64)
    counts[6] = legal[6] = 100
    stats = _episode_action_stats(counts, legal, decisions=0, forced=100)
    assert all(v is None for v in stats['frequencies'].values())
    assert all(v is None for v in stats['selection_given_legal'].values())
    assert _aggregate_behavior([{'agent_decisions': 0}]) is None
    with pytest.raises(RuntimeError, match='Contatori azioni'):
        _episode_action_stats(counts, legal, decisions=1, forced=99)
