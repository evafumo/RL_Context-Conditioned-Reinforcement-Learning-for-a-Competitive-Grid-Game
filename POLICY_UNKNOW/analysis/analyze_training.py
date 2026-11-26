"""Analisi offline di tutti i checkpoint DQN; supporta anche Training_0.

    python -m analysis.analyze_training \
      --evaluations-root evaluations/Training_1 \
      --baseline evaluations/baseline_random \
      --output-dir analysis/results/Training_1

Ogni sottocartella checkpoint_*/ deve contenere summary.json + episodes.jsonl.
La baseline si genera UNA volta con evaluate.py --mode epsilon_greedy
--epsilon 1, con gli stessi seed, episodi e parametri dell'evaluation greedy.
"""
from __future__ import annotations

import argparse
import csv
import json
import ntpath
from pathlib import Path
import re
from typing import Any

from analysis.performance_metrics import load_evaluation, paired_comparison, snapshot_metrics
from analysis.learning_metrics import summarize_learning
from analysis.longitudinal_metrics import collect_longitudinal
from analysis.detailed_behavior import collect_detailed_behavior


def _checkpoint_activations(summary: dict[str, Any]) -> int:
    t = summary.get('checkpoint_activations')
    filename = ntpath.basename(str(summary.get('checkpoint', '')))
    match = re.fullmatch(r'checkpoint_(\d+)\.pt', filename)
    if t is None:
        if not match:
            raise ValueError('Checkpoint senza attivazioni: manca checkpoint_activations e nome checkpoint_N.pt')
        t = int(match.group(1))  # compatibilita' con Training_0
    elif type(t) is not int or t < 0:
        raise ValueError('checkpoint_activations non valido')
    if match and int(match.group(1)) != t:
        raise ValueError('Nome e metadati del checkpoint non coincidono')
    return t


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def _csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def analyze_training(
    evaluations_root: str | Path, output_dir: str | Path, *,
    baseline: str | Path | None = None,
    confidence: float = .95, replicates: int = 2000,
    seed: int = 20261015, threshold: float = .5,
    training_metrics: str | Path | None = None, generate_figures: bool = False,
) -> dict[str, Any]:
    root = Path(evaluations_root)
    if not root.exists():
        raise FileNotFoundError(root)
    # Una valutazione singola (Training_0 storico) oppure serie checkpoint_*/.
    directories = ([root] if (root / 'summary.json').is_file() else
                   sorted(p for p in root.iterdir() if p.is_dir() and (p/'summary.json').is_file()))
    if not directories:
        raise ValueError('Nessuna valutazione in checkpoint_*/summary.json')
    baseline_data = load_evaluation(baseline) if baseline is not None else None
    all_data: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    seen_t: set[int] = set()
    for directory in directories:
        evaluation = load_evaluation(directory)
        summary = evaluation['summary']
        if summary.get('mode') != 'greedy' or summary.get('epsilon') != 0:
            raise ValueError(f'Solo checkpoint greedy nell\'analisi: {directory}')
        t = _checkpoint_activations(summary)
        if t in seen_t:
            raise ValueError(f'Checkpoint con stesse attivazioni: {t}')
        seen_t.add(t)
        result = snapshot_metrics(summary['contexts'], confidence=confidence,
                                  replicates=replicates, seed=seed)
        all_data.append((t, evaluation, result))
    all_data.sort(key=lambda item: item[0])
    # Prima di confrontare curve, controllare esplicitamente protocollo e contesti.
    first_summary = all_data[0][1]['summary']
    fixed = ('horizon','shaping_lambda','eval_seed','episodes_per_context')
    original_pairs = set((c['map_id'],c['opponent_id']) for c in all_data[0][1]['contexts'])
    for t, data, _ in all_data[1:]:
        s = data['summary']
        for field in fixed:
            if s.get(field) != first_summary.get(field):
                raise ValueError(f'Protocollo diverso a {t}: {field}')
        if {(c['map_id'],c['opponent_id']) for c in data['contexts']} != original_pairs:
            raise ValueError(f'Contesti differenti a {t}')

    snapshots = [
        {'activations': t, 'overall_win_rate': perf['overall_win_rate_equal_context_weight'],
         'contexts': data['contexts']}
        for t, data, perf in all_data
    ]
    learning = summarize_learning(snapshots, threshold=threshold)
    reports = []
    curve_rows = []
    context_rows = []
    map_rows = []
    opponent_rows = []
    delta_rows = []
    baseline_rows = []
    for index, (t, data, perf) in enumerate(all_data):
        summary = data['summary']
        baseline_cmp = None
        if baseline_data is not None:
            baseline_cmp = paired_comparison(data, baseline_data, reference_kind='random',
                                             confidence=confidence, replicates=replicates, seed=seed)
            for c in baseline_cmp['per_context']:
                baseline_rows.append({'activations': t, **c})
        previous_cmp = None
        if index > 0:
            previous_cmp = paired_comparison(data, all_data[index-1][1],
                                              reference_kind='checkpoint', confidence=confidence,
                                              replicates=replicates, seed=seed)
            delta_rows.append({'activations': t,
                               'previous_activations': all_data[index-1][0],
                               'delta_global_win_rate': previous_cmp['delta_global_win_rate'],
                               'paired_ci_lower': previous_cmp['paired_stratified_bootstrap_ci'][0],
                               'paired_ci_upper': previous_cmp['paired_stratified_bootstrap_ci'][1]})
        reports.append({
            'checkpoint_activations': t,
            'evaluation_path': data['path'],
            'overall': {'episodes': summary['overall']['episodes'],
                        'wins': summary['overall']['wins'],
                        'win_rate': perf['overall_win_rate_equal_context_weight'],
                        'stratified_bootstrap_ci': perf['stratified_bootstrap_ci']},
            'per_map': perf['per_map'], 'per_opponent': perf['per_opponent'],
            'contexts': data['contexts'], 'robustness': perf['robustness'],
            'improvement_vs_random': baseline_cmp,
            'improvement_vs_previous': previous_cmp,
        })
        curve_rows.append({
            'activations': t,
            'win_rate': perf['overall_win_rate_equal_context_weight'],
            'ci_lower': perf['stratified_bootstrap_ci'][0],
            'ci_upper': perf['stratified_bootstrap_ci'][1],
            'context_std': perf['robustness']['context_win_rate_std_population'],
            'worst_six': perf['robustness']['worst_six_mean_win_rate'],
            'delta_vs_previous': learning['checkpoints'][index]['delta_global_from_previous'],
            'mean_forgetting': learning['checkpoints'][index]['mean_context_forgetting'],
            'delta_vs_random': None if baseline_cmp is None else baseline_cmp['delta_global_win_rate'],
        })
        for row in perf['per_map']:
            map_rows.append({'activations': t, **row,
                             'ci_lower': row['stratified_bootstrap_ci'][0],
                             'ci_upper': row['stratified_bootstrap_ci'][1]})
        for row in perf['per_opponent']:
            opponent_rows.append({'activations': t, **row,
                                  'ci_lower': row['stratified_bootstrap_ci'][0],
                                  'ci_upper': row['stratified_bootstrap_ci'][1]})
        # original_ctx = {(r['map_id'], r['opponent_id']): r for r in data['contexts']}
        original_ctx = {(r['map_id'], r['opponent_id']): r for r in summary['contexts']}
        for row in learning['checkpoints'][index]['contexts']:
            original = original_ctx[(row['map_id'], row['opponent_id'])]
            wilson = original.get('wilson_ci')
            context_rows.append({'activations': t, **row,
                                 'ci_lower': wilson[0] if wilson is not None else None,
                                 'ci_upper': wilson[1] if wilson is not None else None})
    longitudinal = collect_longitudinal(
        [(t, data['summary']) for t, data, _ in all_data],
        output_dir, training_metrics=training_metrics,
    )
    detailed_behavior = collect_detailed_behavior(
        [(t, data['summary']) for t, data, _ in all_data], output_dir,
    )
    report = {
        'detailed_behavior': detailed_behavior,
        'longitudinal': longitudinal,
        'format': 'Giulio2-dqn-analysis-A', 'version': 1,
        'confidence': confidence, 'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
        'evaluation_seed': first_summary['eval_seed'],
        'baseline_path': None if baseline is None else str(Path(baseline)),
        'baseline_status': 'matched' if baseline_data is not None else 'not_provided',
        'n_contexts': len(original_pairs),
        'learning': learning,
        'evaluations': reports,
    }
    out = Path(output_dir)
    _atomic_json(out/'performance_analysis.json', report)
    _csv(out/'learning_curves.csv', curve_rows,
         ['activations','win_rate','ci_lower','ci_upper','context_std','worst_six',
          'delta_vs_previous','mean_forgetting','delta_vs_random'])
    _csv(out/'context_metrics.csv', context_rows,
         ['activations','map_id','opponent_id','win_rate','ci_lower','ci_upper',
          'delta_from_previous','forgetting_from_previous_best'])
    _csv(out/'map_learning_curves.csv', map_rows,
         ['activations','map_id','contexts','episodes','wins','win_rate','ci_lower','ci_upper'])
    _csv(out/'opponent_learning_curves.csv', opponent_rows,
         ['activations','opponent_id','contexts','episodes','wins','win_rate','ci_lower','ci_upper'])
    _csv(out/'checkpoint_deltas.csv', delta_rows,
         ['activations','previous_activations','delta_global_win_rate','paired_ci_lower','paired_ci_upper'])
    if baseline is not None:
        _csv(out/'random_comparison.csv', baseline_rows,
             ['activations','map_id','opponent_id','current_win_rate','reference_win_rate','delta_win_rate'])
    if generate_figures:
        from analysis.plots import generate_plots
        report['plots'] = [str(p) for p in generate_plots(out)]
        _atomic_json(out/'performance_analysis.json', report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analisi A dei checkpoint DQN, senza simulare partite')
    parser.add_argument('--evaluations-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=None)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--training-metrics', type=Path, default=None,
                        help='Percorso runs/Training_X/metrics.jsonl per le metriche B')
    parser.add_argument('--plots', action='store_true',
                        help='Genera anche i grafici PNG (richiede matplotlib)')
    args = parser.parse_args(argv)
    results = analyze_training(args.evaluations_root, args.output_dir,
                               baseline=args.baseline, confidence=args.confidence,
                               replicates=args.bootstrap_replicates, seed=args.seed,
                               threshold=args.threshold,
                               training_metrics=args.training_metrics,
                               generate_figures=args.plots)
    print(f"[analysis] checkpoint={results['learning']['number_of_checkpoints']} "
          f"AULC={results['learning']['aulc_normalized']} "
          f"output={args.output_dir / 'performance_analysis.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
