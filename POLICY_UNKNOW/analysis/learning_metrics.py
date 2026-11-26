"""Metriche A10/A11: AULC, soglia osservata, miglioramento e forgetting."""
from __future__ import annotations

from typing import Any, Mapping
from math import isfinite


def summarize_learning(snapshots: list[Mapping[str, Any]], *, threshold: float = .5) -> dict[str, Any]:
    """Input ordinato per step cumulativi: {'activations', 'contexts', 'overall_win_rate'}.

    Non inventa interpolazioni al di fuori dei checkpoint osservati. Il
    superamento di soglia e' 'first observed', non il vero tempo di hitting.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int,float)) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold deve appartenere a [0,1]')
    if not snapshots:
        raise ValueError('Nessun checkpoint disponibile')
    prev_t = -1
    baseline_keys = None
    best_so_far: dict[tuple[int,int], float] = {}
    last: dict[tuple[int,int], float] | None = None
    steps = []
    auc = 0.
    t_prev = None
    p_prev = None
    first_above = None
    for s in snapshots:
        t = s.get('activations')
        p = s.get('overall_win_rate')
        if type(t) is not int or t < 0 or t <= prev_t:
            raise ValueError('Le attivazioni cumulative devono essere interi crescenti')
        if isinstance(p, bool) or not isinstance(p, (int,float)) or not isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Win rate globale non valido')
        by_key = {}
        for c in s['contexts']:
            key = (c['map_id'], c['opponent_id'])
            rate = c['win_rate']
            if key in by_key or not 0 <= rate <= 1:
                raise ValueError('Contesto duplicato o win rate fuori intervallo')
            by_key[key] = rate
        if baseline_keys is None:
            baseline_keys = set(by_key)
        elif set(by_key) != baseline_keys:
            raise ValueError('Contesti diversi tra checkpoint')
        if abs(sum(by_key.values())/len(by_key) - p) > 1e-9:
            raise ValueError('Il globale non coincide con la media dei contesti')
        delta = None if p_prev is None else p - p_prev
        if t_prev is not None:
            auc += (t - t_prev) * (p + p_prev) / 2
        if first_above is None and p >= threshold:
            first_above = t
        details = []
        for (m,k), rate in sorted(by_key.items()):
            previous = None if last is None else last[(m,k)]
            best_before = best_so_far.get((m,k))
            forgetting = 0. if best_before is None else max(0., best_before - rate)
            details.append({'map_id': m, 'opponent_id': k,
                            'win_rate': rate,
                            'delta_from_previous': None if previous is None else rate-previous,
                            'forgetting_from_previous_best': forgetting})
            best_so_far[(m,k)] = max(rate, best_so_far.get((m,k), rate))
        steps.append({'activations': t, 'global_win_rate': p,
                      'delta_global_from_previous': delta,
                      'mean_context_forgetting': sum(x['forgetting_from_previous_best'] for x in details)/len(details),
                      'worst_context_forgetting': max(x['forgetting_from_previous_best'] for x in details),
                      'contexts': details})
        prev_t, t_prev, p_prev, last = t, t, p, by_key
    span = steps[-1]['activations'] - steps[0]['activations']
    return {
        'first_checkpoint': steps[0]['activations'],
        'last_checkpoint': steps[-1]['activations'],
        'number_of_checkpoints': len(steps),
        'aulc_trapezoidal': auc if span > 0 else None,
        'aulc_normalized': auc/span if span > 0 else None,
        'aulc_interval': [steps[0]['activations'],steps[-1]['activations']],
        'threshold': float(threshold),
        'first_observed_at_or_above_threshold': first_above,
        'checkpoints': steps,
        'interpretation': 'Forgetting e variazioni osservate, non test statistici di causalita o generalizzazione.',
    }
