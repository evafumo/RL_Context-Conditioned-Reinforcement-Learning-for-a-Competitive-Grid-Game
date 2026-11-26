"""Metriche A1-5/8-11: test sintetici deterministici, senza GPU e senza MDP."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.performance_metrics import snapshot_metrics, load_evaluation, paired_comparison
from analysis.learning_metrics import summarize_learning
from analysis.analyze_training import analyze_training, main


def rows(wins=(6,3,2,9), n=10):
    pairs = ((1,1),(1,2),(2,1),(2,2))
    return [dict(map_id=m, opponent_id=k, wins=w, episodes=n)
            for (m,k), w in zip(pairs, wins)]


def create_evaluation(path: Path, wins, step, *, mode='greedy', epsilon=0,
                      eval_seed=42, shuffle=False):
    path.mkdir(parents=True, exist_ok=True)
    contexts = rows(wins)
    if shuffle:
        contexts.reverse()
    summary = {
        'format':'Giulio2-dqn-evaluation', 'version':1,
        'mode':mode,'epsilon':epsilon,'eval_seed':eval_seed,
        'checkpoint':str(path/f'checkpoint_{step}.pt'),
        'shaping_lambda':0.1,'horizon':100,'episodes_per_context':10,
        'overall':{'episodes':40, 'wins':sum(wins)},
        'contexts':contexts,
    }
    (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    with (path/'episodes.jsonl').open('w',encoding='utf-8') as f:
        for ctx in contexts:
            for i in range(10):
                # Seed dipende dal contesto, non dal checkpoint.
                ep = {'map_id':ctx['map_id'],'opponent_id':ctx['opponent_id'],
                      'episode_seed':ctx['map_id']*1000+ctx['opponent_id']*100+i,
                      'win':i < ctx['wins']}
                f.write(json.dumps(ep)+'\n')
    return path


def test_snapshot_aggregations_and_robustness():
    r = rows((6,3,2,9))
    result = snapshot_metrics(r, replicates=400, seed=123)
    assert result['overall_win_rate_equal_context_weight'] == pytest.approx(.5)
    assert result['per_map'][0]['win_rate'] == pytest.approx(.45)
    assert result['per_map'][1]['win_rate'] == pytest.approx(.55)
    assert result['per_opponent'][0]['win_rate'] == pytest.approx(.4)
    assert result['per_opponent'][1]['win_rate'] == pytest.approx(.6)
    assert result['robustness']['worst_six_mean_win_rate'] is None
    assert result['robustness']['context_win_rate_std_population'] == pytest.approx((.075)**.5)
    assert result == snapshot_metrics(r, replicates=400, seed=123)
    assert result['stratified_bootstrap_ci'][0] <= .5 <= result['stratified_bootstrap_ci'][1]


def test_worst_six_and_bad_contexts():
    context = [dict(map_id=m,opponent_id=k,wins=(m+k)%11,episodes=10)
               for m in range(1,5) for k in range(1,7)]
    result = snapshot_metrics(context, replicates=100)
    expected = sum(sorted((c['wins']/10 for c in context))[:6])/6
    assert result['robustness']['worst_six_mean_win_rate'] == pytest.approx(expected)
    with pytest.raises(ValueError):
        snapshot_metrics(rows()+[rows()[0]])
    with pytest.raises(ValueError):
        snapshot_metrics(rows(),replicates=50)


def test_paired_baseline_and_mismatches(tmp_path):
    current = load_evaluation(create_evaluation(tmp_path/'current',(6,3,2,9),20000))
    base = load_evaluation(create_evaluation(tmp_path/'random',(5,2,1,8),0,mode='epsilon_greedy',epsilon=1))
    diff = paired_comparison(current,base,reference_kind='random',replicates=300)
    assert diff['matched_episodes'] == 40
    assert diff['delta_global_win_rate'] == pytest.approx(.1)
    assert diff['per_map'][0]['delta_win_rate'] == pytest.approx(.1)
    assert diff['paired_stratified_bootstrap_ci'][0] <= .1 <= diff['paired_stratified_bootstrap_ci'][1]
    wrong = load_evaluation(create_evaluation(tmp_path/'wrong',(5,2,1,8),0,mode='greedy',eval_seed=99))
    with pytest.raises(ValueError,match='eval_seed'):
        paired_comparison(current,wrong)
    with pytest.raises(ValueError,match='baseline'):
        paired_comparison(current,wrong,reference_kind='random')
    # Vecchie run devono mantenere per forza semi allineati.
    old = load_evaluation(create_evaluation(tmp_path/'old',(5,2,1,8),0,mode='greedy'))
    assert paired_comparison(current,old,replicates=300)['delta_global_win_rate'] == pytest.approx(.1)
    old['episodes'][0]['episode_seed'] = 99999
    with pytest.raises(ValueError,match='Seed'):
        paired_comparison(current,old,replicates=300)


def test_learning_auc_forgetting_threshold():
    a = rows((4,6,2,8))
    b = rows((5,5,3,7))
    c = rows((6,4,4,6))
    snapshots = [{'activations':t,'overall_win_rate':.5,
                  'contexts':[{**r,'win_rate':r['wins']/r['episodes']} for r in rs]}
                 for t, rs in [(20000,a),(40000,b),(60000,c)]]
    out = summarize_learning(snapshots,threshold=.5)
    assert out['aulc_trapezoidal'] == pytest.approx(20000)
    assert out['aulc_normalized'] == pytest.approx(.5)
    assert out['first_observed_at_or_above_threshold'] == 20000
    assert out['checkpoints'][1]['contexts'][1]['forgetting_from_previous_best'] == pytest.approx(.1)
    assert out['checkpoints'][2]['contexts'][1]['forgetting_from_previous_best'] == pytest.approx(.2)
    with pytest.raises(ValueError,match='crescenti'):
        summarize_learning([snapshots[0],snapshots[0]])
    assert summarize_learning([snapshots[0]])['aulc_normalized'] is None


def test_analysis_all_outputs_and_historical(tmp_path):
    eval_root = tmp_path/'evals'
    create_evaluation(eval_root/'checkpoint_20000',(4,6,2,8),20000)
    create_evaluation(eval_root/'checkpoint_40000',(5,5,3,7),40000,shuffle=True)
    random = create_evaluation(tmp_path/'baseline',(3,4,1,5),0,
                               mode='epsilon_greedy',epsilon=1)
    output = tmp_path/'results'
    report = analyze_training(eval_root,output,baseline=random,replicates=300,threshold=.6)
    assert report['learning']['number_of_checkpoints'] == 2
    assert report['learning']['aulc_normalized'] == pytest.approx(.5)
    assert len(report['evaluations']) == 2
    assert report['baseline_status'] == 'matched'
    assert report['evaluations'][0]['improvement_vs_random']['matched_episodes'] == 40
    for filename in ('performance_analysis.json','learning_curves.csv','context_metrics.csv',
                     'random_comparison.csv','checkpoint_deltas.csv',
                     'map_learning_curves.csv','opponent_learning_curves.csv'):
        assert (output/filename).is_file()
    assert main(['--evaluations-root',str(eval_root),'--output-dir',str(tmp_path/'cli'),
                 '--bootstrap-replicates','100']) == 0
    with pytest.raises(ValueError,match='Protocollo diverso'):
        changed = json.loads((eval_root/'checkpoint_40000'/'summary.json').read_text())
        changed['eval_seed'] = 77
        (eval_root/'checkpoint_40000'/'summary.json').write_text(json.dumps(changed))
        analyze_training(eval_root,tmp_path/'bad',replicates=100)


def test_windows_checkpoint_path_legacy(tmp_path):
    import json
    directory = create_evaluation(tmp_path/'old',(5,5,5,5),80000)
    file = directory/'summary.json'
    summary = json.loads(file.read_text())
    summary['checkpoint'] = r'C:\Users\Giulio\Desktop\Giulio2\runs\Training_0\checkpoint_80000.pt'
    file.write_text(json.dumps(summary))
    report = analyze_training(directory,tmp_path/'historical_output',replicates=100)
    assert report['learning']['first_checkpoint'] == 80000
    assert report['learning']['aulc_normalized'] is None
