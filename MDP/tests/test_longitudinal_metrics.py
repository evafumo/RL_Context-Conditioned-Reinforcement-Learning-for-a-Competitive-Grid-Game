"""Test offline B/C e PNG; nessuna GPU e nessun episodio MDP."""
import csv
import json
from pathlib import Path

import pytest

from analysis.longitudinal_metrics import collect_longitudinal, parse_dqn_updates
from analysis.plots import generate_plots
from analysis.analyze_training import analyze_training


def _log(path, *, legacy=False):
    data = []
    for u, t in ((10, 20000), (20, 20000), (30, 40000)):
        dqn = {'loss': .03 if t == 20000 else .01,
               'mean_q': .2, 'grad_norm': .4, 'mean_target': .3,
               'update_steps': u, 'target_synced': False}
        if not legacy:
            dqn.update(td_mae=.2, td_p90=.3, std_q=.1,
                       mean_max_legal_q=.4, std_max_legal_q=.15)
        data.append({'kind': 'dqn_update', 'update_index': u, 'dqn': dqn,
                     'other': {'environment_activations': t},
                     'epsilon': .5, 'replay_size': t, 'replay_fill_fraction': .5})
    data.append({'kind':'checkpoint', 'environment_activations':40000})
    path.write_text('\n'.join(json.dumps(d) for d in data)+'\n', encoding='utf-8')
    return path


def _behavior(n=10):
    counts = {'NORTH': 2*n, 'SOUTH': 0, 'EAST': 0,
              'WEST': 0, 'SHOOT': 3*n, 'PASS': 5*n}
    return {
        'episodes': n,
        'actions': {'agent_decisions': 10*n, 'counts': counts,
                    'legal_opportunities': {a: 10*n for a in counts}},
        'objective_captures': {'agent_mean_per_episode': 2.,
                               'opponent_mean_per_episode': 1.,
                               'agent_minus_opponent_mean_per_episode': 1.},
        'combat': {
            'shots_agent_total': 3*n, 'shots_opponent_total': n,
            'shot_kills_agent_total': n, 'shot_kills_opponent_total': 0,
            'melee_encounters_total': n, 'melee_initiated_agent_total': n,
            'melee_initiated_opponent_total': 0, 'melee_kills_agent_total': 0,
            'melee_kills_opponent_total': 0, 'kills_agent_mean_per_episode': 1.,
            'kills_opponent_mean_per_episode': 0., 'deaths_agent_mean_per_episode': 0.,
            'deaths_opponent_mean_per_episode': 1., 'shoot_kill_rate_agent': 1/3,
            'shoot_kill_rate_opponent': 0.,
        },
    }


def _summaries():
    results = []
    for t in (20000,40000):
        b = _behavior(10)
        results.append((t, {'overall': {'behavior': _behavior(20)},
                            'contexts': [
                                {'map_id':1,'opponent_id':1,'behavior':b},
                                {'map_id':1,'opponent_id':2,'behavior':_behavior(10)},
                            ]}))
    return results


def test_dqn_parsing_and_interval_means(tmp_path):
    f = _log(tmp_path/'metrics.jsonl')
    parsed = parse_dqn_updates(f)
    assert [r['activations'] for r in parsed] == [20000,20000,40000]
    assert [r['update_steps'] for r in parsed] == [10,20,30]
    result = collect_longitudinal(_summaries(), tmp_path/'out', training_metrics=f)
    assert result['B'] == 'available' and result['C'] == 'available'
    with (tmp_path/'out'/'dqn_checkpoint_summary.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]['logged_updates'] == '2'
    assert float(rows[0]['mean_loss']) == pytest.approx(.03)
    assert rows[1]['logged_updates'] == '1'
    with (tmp_path/'out'/'behavior_learning_curves.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==2
    assert float(rows[0]['action_shoot_frequency']) == pytest.approx(.3)
    assert float(rows[0]['shoot_kill_rate_agent']) == pytest.approx(1/3)
    with (tmp_path/'out'/'behavior_context_curves.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f))) == 4


def test_legacy_logs_and_missing_behavior(tmp_path):
    f = _log(tmp_path/'old.jsonl', legacy=True)
    snapshots = [(80000, {'overall':{}, 'contexts': []})]
    result = collect_longitudinal(snapshots, tmp_path/'legacy', training_metrics=f)
    assert result['C'] == 'unavailable_legacy'
    assert 'td_mae' in result['B_fields_missing']
    assert not (tmp_path/'legacy'/'behavior_learning_curves.csv').exists()
    with pytest.raises(ValueError, match='solo per alcuni'):
        collect_longitudinal([_summaries()[0], snapshots[0]], tmp_path/'partial')


def test_invalid_dqn_updates(tmp_path):
    f = _log(tmp_path/'metrics.jsonl')
    lines = f.read_text().splitlines()
    bad = json.loads(lines[1]); bad['update_index'] = 1
    lines[1] = json.dumps(bad)
    f.write_text('\n'.join(lines))
    with pytest.raises(ValueError, match='monotono'):
        parse_dqn_updates(f)


def test_behavior_invalid_counts(tmp_path):
    snapshots = _summaries()
    snapshots[0][1]['overall']['behavior']['actions']['agent_decisions'] = 10
    with pytest.raises(ValueError, match='Somma counts'):
        collect_longitudinal(snapshots, tmp_path/'bad')


def test_plots_generated_without_GPU(tmp_path):
    pytest.importorskip('matplotlib')
    f = _log(tmp_path/'metrics.jsonl')
    collect_longitudinal(_summaries(),tmp_path/'res', training_metrics=f)
    paths = generate_plots(tmp_path/'res')
    names = {p.name for p in paths}
    assert 'dqn_loss.png' in names
    assert 'behavior_actions.png' in names
    assert all(p.stat().st_size > 1000 for p in paths)


def test_full_analysis_BC_cli_compatible(tmp_path):
    evaluations = tmp_path / 'evaluations'
    for t, wins in ((20000, (5, 6)), (40000, (7, 8))):
        path = evaluations / f'checkpoint_{t}'
        path.mkdir(parents=True)
        contexts = []
        with (path/'episodes.jsonl').open('w', encoding='utf-8') as fp:
            for k, w in enumerate(wins, 1):
                contexts.append({'map_id':1, 'opponent_id':k, 'wins':w, 'episodes':10,
                                 'behavior':_behavior(10)})
                for i in range(10):
                    fp.write(json.dumps({'map_id':1,'opponent_id':k,
                                         'episode_seed':100*k+i, 'win':i<w})+'\n')
        summary = {'format':'Giulio2-dqn-evaluation', 'version':1,
                   'checkpoint':str(path/f'checkpoint_{t}.pt'),
                   'checkpoint_activations':t,
                   'mode':'greedy','epsilon':0,'eval_seed':42,
                   'shaping_lambda':.1,'horizon':100,'episodes_per_context':10,
                   'overall':{'episodes':20, 'wins':sum(wins), 'behavior':_behavior(20)},
                   'contexts':contexts}
        (path/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    log = _log(tmp_path/'metrics.jsonl')
    out = tmp_path/'analysis'
    report = analyze_training(evaluations, out, training_metrics=log,
                              replicates=100, generate_figures=True)
    assert report['learning']['number_of_checkpoints'] == 2
    assert report['longitudinal']['B'] == 'available'
    assert report['longitudinal']['C'] == 'available'
    for filename in ('performance_analysis.json','learning_curves.csv',
                     'dqn_learning_curves.csv','dqn_checkpoint_summary.csv',
                     'behavior_learning_curves.csv','behavior_context_curves.csv'):
        assert (out/filename).is_file()
    assert (out/'plots'/'behavior_kills.png').is_file()
