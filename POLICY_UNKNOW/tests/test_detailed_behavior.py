"""Verifica formule e raggruppamenti C senza eseguire partite o usare la GPU."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import json

import pytest

from analysis.detailed_behavior import collect_detailed_behavior
from analysis.plots import generate_plots


def _context(m: int, k: int, n: int = 100, shift: int = 0) -> dict:
    decisions = 190 + 10*m + 5*k + shift
    north = m*k*2 + shift
    shoot = k + shift
    counts = {"NORTH": north, "SOUTH": 0, "EAST": 0, "WEST": 0,
              "SHOOT": shoot, "PASS": decisions - north - shoot}
    shots_a, shots_o = 5*m+shift, 3*k+shift
    shot_kills_a, shot_kills_o = m, k
    melee_kills_a, melee_kills_o = k // 2, m // 2
    return {
        "map_id": m, "opponent_id": k, "episodes": n,
        "behavior": {
            "episodes": n,
            "actions": {"agent_decisions": decisions, "counts": counts},
            "objective_captures": {"agent_total": 2*m+shift,
                                   "opponent_total": 3*k+shift},
            "combat": {
                "shots_agent_total": shots_a, "shots_opponent_total": shots_o,
                "shot_kills_agent_total": shot_kills_a,
                "shot_kills_opponent_total": shot_kills_o,
                "melee_encounters_total": m+k,
                "melee_initiated_agent_total": m,
                "melee_initiated_opponent_total": k,
                "melee_kills_agent_total": melee_kills_a,
                "melee_kills_opponent_total": melee_kills_o,
                "kills_agent_total": shot_kills_a+melee_kills_a,
                "kills_opponent_total": shot_kills_o+melee_kills_o,
            },
        },
    }


def _snapshot(t: int, *, pairs=None, shift=0) -> tuple[int, dict]:
    if pairs is None:
        pairs = [(m,k) for m in range(1,5) for k in range(1,7)]
    contexts = [_context(m,k, shift=shift) for m,k in pairs]
    actions = {a: sum(ctx['behavior']['actions']['counts'][a] for ctx in contexts)
               for a in ('NORTH','SOUTH','EAST','WEST','SHOOT','PASS')}
    summ = {
        'horizon': 100,
        'contexts': contexts,
        'overall': {'episodes':sum(ctx['episodes'] for ctx in contexts),
                    'behavior': {
                        'actions': {'agent_decisions': sum(
                            c['behavior']['actions']['agent_decisions'] for c in contexts),
                            'counts': actions},
                        'combat': {f'kills_{side}_total': sum(
                            c['behavior']['combat'][f'kills_{side}_total'] for c in contexts)
                            for side in ('agent','opponent')},
                        'objective_captures': {f'{side}_total': sum(
                            c['behavior']['objective_captures'][f'{side}_total']
                            for c in contexts) for side in ('agent','opponent')},
                    }}}
    return t, summ


def _csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def test_full_24_contexts_two_checkpoints_correct_denominators(tmp_path):
    snaps = [_snapshot(20000), _snapshot(40000, shift=2)]
    result = collect_detailed_behavior(snaps, tmp_path)
    assert result['status'] == 'available'
    assert len(result['files']) == 12
    expected = {'global':2, 'map':8, 'opponent':12, 'context':48}
    for kind in ('kill','act','obj'):
        for scope, count in expected.items():
            assert len(_csv(tmp_path/f'{kind}_{scope}_curves.csv')) == count
    kill_global = _csv(tmp_path/'kill_global_curves.csv')[0]
    assert int(kill_global['episodes']) == 2400
    assert int(kill_global['K_agent_total']) == sum(
        c['behavior']['combat']['kills_agent_total'] for c in snaps[0][1]['contexts'])
    assert float(kill_global['K_agent']) == pytest.approx(
        int(kill_global['K_agent_total']) / 2400)
    assert float(kill_global['K_agent']) == pytest.approx(
        float(kill_global['KS_agent']) + float(kill_global['KM_agent']))
    assert float(kill_global['COMB_agent']) == pytest.approx(
        (int(kill_global['SHOTS_agent_total']) + int(kill_global['MELEE_INIT_agent_total'])) / 2400)
    k_map = next(r for r in _csv(tmp_path/'kill_map_curves.csv')
                 if r['activations']=='20000' and r['map_id']=='2')
    assert int(k_map['episodes']) == 600
    k_opponent = next(r for r in _csv(tmp_path/'kill_opponent_curves.csv')
                      if r['activations']=='20000' and r['opponent_id']=='4')
    assert int(k_opponent['episodes']) == 400
    k_context = next(r for r in _csv(tmp_path/'kill_context_curves.csv')
                     if r['activations']=='20000' and r['map_id']=='2' and r['opponent_id']=='4')
    assert int(k_context['episodes']) == 100
    action = _csv(tmp_path/'act_global_curves.csv')[0]
    nd = sum(c['behavior']['actions']['agent_decisions'] for c in snaps[0][1]['contexts'])
    assert int(action['agent_decisions']) == nd
    assert float(action['f_NORTH']) == pytest.approx(int(action['count_NORTH'])/nd)
    assert sum(float(action['f_'+a]) for a in ('NORTH','SOUTH','EAST','WEST','SHOOT','PASS')) == pytest.approx(1)
    assert action['f_DUMMY'] == ''
    assert int(action['dummy_count']) == 2400*100-nd
    obj = _csv(tmp_path/'obj_opponent_curves.csv')[0]
    assert float(obj['OB_agent']) == pytest.approx(int(obj['OB_agent_total'])/400)


def test_grouped_actions_use_sum_of_decisions_not_unweighted_context_means(tmp_path):
    snapshot = _snapshot(20000, pairs=[(1,1),(1,6)])
    collect_detailed_behavior([snapshot], tmp_path)
    glob = _csv(tmp_path/'act_global_curves.csv')[0]
    c1, c2 = _csv(tmp_path/'act_context_curves.csv')
    pooled = (int(c1['count_NORTH']) + int(c2['count_NORTH'])) / (
        int(c1['agent_decisions']) + int(c2['agent_decisions']))
    assert float(glob['f_NORTH']) == pytest.approx(pooled)
    assert pooled != pytest.approx((float(c1['f_NORTH'])+float(c2['f_NORTH']))/2)


def test_missing_legacy_data_and_corrupt_counters(tmp_path):
    old = _snapshot(20000)
    del old[1]['contexts'][0]['behavior']['combat']['kills_agent_total']
    with pytest.raises(ValueError, match='solo in alcuni'):
        collect_detailed_behavior([old], tmp_path)
    older = (20000, {'contexts':[{'episodes':100}], 'overall':{'episodes':100}})
    assert collect_detailed_behavior([older], tmp_path)['status'] == 'unavailable_missing_raw_counts'
    good = _snapshot(20000)
    good[1]['contexts'][0]['behavior']['combat']['kills_agent_total'] += 1
    with pytest.raises(ValueError, match='Kill shooting\\+melee'):
        collect_detailed_behavior([good], tmp_path)


def test_all_plot_folders_and_actual_smoke_data(tmp_path):
    pytest.importorskip('matplotlib')
    # Non sono necessarie tutte le 24 coppie per verificare i percorsi.
    snap1 = _snapshot(20000, pairs=[(1,1),(2,2)])
    snap2 = _snapshot(40000, pairs=[(1,1),(2,2)], shift=1)
    result = collect_detailed_behavior([snap1,snap2], tmp_path)
    assert result['status'] == 'available'
    images = generate_plots(tmp_path)
    assert len(images) == 35  # 2 global + 2 map + 2 opponent + 2 context per tipo
    folders = [f'{s}_KILL_{scope.upper()}' for s in ('Agent','Opponent')
               for scope in ('global','context','map','opponent')]
    folders += [f'{s}_OBJ_{scope.upper()}' for s in ('Agent','Opponent')
                for scope in ('global','context','map','opponent')]
    folders += [f'ACT_{scope.upper()}' for scope in ('global','context','map','opponent')]
    assert len(folders)==20
    for folder in folders:
        assert (tmp_path/'plots'/folder).is_dir(), folder
        assert list((tmp_path/'plots'/folder).glob('*.png'))
    assert all(p.stat().st_size > 2000 for p in images)


def test_original_smoke_evaluation_counts_are_reusable(tmp_path):
    source = Path(__file__).resolve().parents[1]/'evaluations'/'smoke_cuda'/'summary.json'
    if not source.is_file():
        pytest.skip('L\'archivio di test non contiene la valutazione smoke_cuda')
    summary = json.loads(source.read_text(encoding='utf-8'))
    result = collect_detailed_behavior([(summary['checkpoint_activations'],summary)],tmp_path)
    assert result['status']=='available'
    glob = _csv(tmp_path/'kill_global_curves.csv')[0]
    assert int(glob['episodes']) == summary['overall']['episodes']
    assert float(glob['K_agent']) == pytest.approx(
        summary['overall']['behavior']['combat']['kills_agent_total']/int(glob['episodes']))
    assert int(glob['COMB_agent_total']) == (
        summary['overall']['behavior']['combat']['shots_agent_total'] +
        summary['overall']['behavior']['combat']['melee_initiated_agent_total'])
