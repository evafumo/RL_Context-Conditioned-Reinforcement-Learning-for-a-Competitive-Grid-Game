"""Metriche A per singola valutazione e confronti appaiati.

Nessun training, caricamento CUDA o interazione con GridGame. L'unita'
statistica e' l'episodio entro uno dei 24 contesti; il globale assegna
peso identico ad ogni contesto e non confonde episodi con contesti.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from pathlib import Path
import json
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np


def _wilson(wins: int, n: int, confidence: float) -> list[float]:
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = wins / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0.,c-h), min(1.,c+h)]


def _validate_confidence(confidence: float, replicates: int, seed: int) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence deve essere in (0,1)')
    if type(replicates) is not int or replicates < 100:
        raise ValueError('replicates deve essere >= 100')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed deve essere uint64')


def _context_rows(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(contexts, list) or not contexts:
        raise ValueError('Contesti assenti')
    pairs = set()
    result = []
    for ctx in contexts:
        m, k, n, w = (ctx.get(s) for s in ('map_id', 'opponent_id', 'episodes', 'wins'))
        if (any(type(v) is not int for v in (m, k, n, w)) or not (1 <= m <= 4)
                or not (1 <= k <= 6) or n < 1 or not 0 <= w <= n or (m,k) in pairs):
            raise ValueError('Contesto, conteggi o coppie non validi')
        pairs.add((m,k))
        result.append({'map_id': m, 'opponent_id': k, 'episodes': n, 'wins': w,
                       'win_rate': w/n})
    # Per il protocollo di questo progetto, ogni contesto ha n episodi uguali.
    if len({x['episodes'] for x in result}) != 1:
        raise ValueError('Il protocollo richiede lo stesso numero di episodi per contesto')
    return sorted(result, key=lambda x: (x['map_id'], x['opponent_id']))


def snapshot_metrics(
    contexts: list[dict[str, Any]], *, confidence: float = .95,
    replicates: int = 2000, seed: int = 20261015,
) -> dict[str, Any]:
    """Metriche 1--5 e 8, IC bootstrap stratificato della media dei contesti.

    Il bootstrap resampling dei Bernoulli osservati e' equivalente a un
    Binomiale(n, wins/n) indipendente per contesto; non cambia il loro peso.
    La deviazione standard e' descrittiva (ddof=0), NON errore standard.
    """
    _validate_confidence(confidence, replicates, seed)
    rows = _context_rows(contexts)
    rates = np.array([r['win_rate'] for r in rows], dtype=np.float64)
    overall = float(rates.mean())
    rng = np.random.default_rng(seed)
    boot = np.zeros(replicates, dtype=np.float64)
    draws_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for row in rows:
        draws = rng.binomial(row['episodes'], row['win_rate'], size=replicates) / row['episodes']
        draws_by_pair[(row['map_id'], row['opponent_id'])] = draws
        boot += draws
    boot /= len(rows)
    alpha = (1-confidence)/2
    ci = [float(v) for v in np.quantile(boot, [alpha, 1-alpha])]

    def aggregate(key: str) -> list[dict[str, Any]]:
        by_key: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_key[r[key]].append(r)
        results = []
        for value, group in sorted(by_key.items()):
            n = sum(c['episodes'] for c in group)
            w = sum(c['wins'] for c in group)
            group_draws = sum((draws_by_pair[(c['map_id'], c['opponent_id'])]
                               for c in group), np.zeros(replicates)) / len(group)
            results.append({key: value, 'contexts': len(group), 'episodes': n,
                            'wins': w, 'win_rate': w/n,
                            'wilson_ci': _wilson(w, n, confidence),
                            'stratified_bootstrap_ci': [float(v) for v in
                                np.quantile(group_draws, [alpha, 1-alpha])]})
        return results

    worst = sorted(rows, key=lambda r: (r['win_rate'], r['map_id'], r['opponent_id']))[:6]
    return {
        'overall_win_rate_equal_context_weight': overall,
        'stratified_bootstrap_ci': ci,
        'bootstrap_replicates': replicates,
        'bootstrap_seed': seed,
        'per_map': aggregate('map_id'),
        'per_opponent': aggregate('opponent_id'),
        'robustness': {
            'context_win_rate_std_population': float(rates.std(ddof=0)),
            'worst_six_mean_win_rate': float(np.mean([r['win_rate'] for r in worst])) if len(rows) >= 6 else None,
            'worst_six_contexts': [dict(map_id=r['map_id'], opponent_id=r['opponent_id'], win_rate=r['win_rate']) for r in worst] if len(rows) >= 6 else [],
            'number_of_contexts': len(rows),
        },
    }


def load_evaluation(path: str | Path) -> dict[str, Any]:
    """Carica summary.json e episodes.jsonl, anche dai vecchi Training_0.

    Verifica la coerenza dei dati: il confronto paired non deve associare
    casualmente episodi che hanno seed diversi.
    """
    p = Path(path)
    directory = p if p.is_dir() else p.parent
    summary_path = directory / 'summary.json' if p.is_dir() else p
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('format') != 'Giulio2-dqn-evaluation' or summary.get('version') != 1:
        raise ValueError(f'Formato valutazione sconosciuto: {summary_path}')
    contexts = _context_rows(summary.get('contexts'))
    if (sum(c['episodes'] for c in contexts) != summary['overall']['episodes']
            or sum(c['wins'] for c in contexts) != summary['overall']['wins']):
        raise ValueError(f'Summary incoerente: {summary_path}')
    episodes_path = directory / 'episodes.jsonl'
    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(episodes) != summary['overall']['episodes']:
        raise ValueError('Lunghezza episodes.jsonl non coincide con summary.json')
    counts: dict[tuple[int,int], list[int]] = defaultdict(list)
    seeds: set[tuple[int,int,int]] = set()
    for ep in episodes:
        key = (ep.get('map_id'), ep.get('opponent_id'))
        if (len(key) != 2 or any(type(v) is not int for v in key)
                or type(ep.get('episode_seed')) is not int or type(ep.get('win')) is not bool):
            raise ValueError('Riga episodio non valida')
        full_key = (*key, ep['episode_seed'])
        if full_key in seeds:
            raise ValueError('Seed duplicato nello stesso contesto')
        seeds.add(full_key)
        counts[key].append(int(ep['win']))
    for c in contexts:
        xs = counts.pop((c['map_id'], c['opponent_id']), [])
        if len(xs) != c['episodes'] or sum(xs) != c['wins']:
            raise ValueError('Contatori episodio/summary non coerenti')
    if counts:
        raise ValueError('Episodi su contesti assenti dal summary')
    return {'summary': summary, 'episodes': episodes, 'path': str(summary_path), 'contexts': contexts}


def _episode_map(evaluation: Mapping[str, Any]) -> dict[tuple[int,int], dict[int,int]]:
    result: dict[tuple[int,int], dict[int,int]] = defaultdict(dict)
    for ep in evaluation['episodes']:
        result[(ep['map_id'], ep['opponent_id'])][ep['episode_seed']] = int(ep['win'])
    return result


def paired_comparison(
    current: Mapping[str, Any], reference: Mapping[str, Any], *,
    reference_kind: str = 'checkpoint', confidence: float = .95,
    replicates: int = 2000, seed: int = 20261015,
) -> dict[str, Any]:
    """Metrica 9 e confronto fra checkpoint (11); bootstrap paired stratificato.

    richiede gli stessi contesti e SEMI di episodio; non accetta confronti
    non omogenei o fra differenti protocolli reward/seed.
    """
    _validate_confidence(confidence, replicates, seed)
    cs, rs = current['summary'], reference['summary']
    if cs.get('mode') != 'greedy' or cs.get('epsilon') != 0:
        raise ValueError('Il checkpoint corrente deve essere valutato greedy')
    if reference_kind == 'random':
        if rs.get('mode') != 'epsilon_greedy' or rs.get('epsilon') != 1:
            raise ValueError('La baseline Random richiede epsilon_greedy con epsilon=1')
    elif reference_kind == 'checkpoint':
        if rs.get('mode') != 'greedy' or rs.get('epsilon') != 0:
            raise ValueError('Il checkpoint di riferimento deve essere greedy')
    else:
        raise ValueError('reference_kind sconosciuto')
    for field in ('eval_seed', 'shaping_lambda', 'horizon', 'episodes_per_context',
                  'observation', 'fog_radius'):
        if cs.get(field) != rs.get(field):
            raise ValueError(f'Valutazioni non confrontabili: {field} diverso')
    now = _episode_map(current)
    old = _episode_map(reference)
    if now.keys() != old.keys():
        raise ValueError('Insieme dei contesti diverso')
    rng = np.random.default_rng(seed)
    boot = np.zeros(replicates)
    per_context = []
    for m, k in sorted(now):
        if now[(m,k)].keys() != old[(m,k)].keys():
            raise ValueError(f'Seed di episodio differenti nel contesto {(m,k)}')
        ids = sorted(now[(m,k)])
        differences = np.array([now[(m,k)][s] - old[(m,k)][s] for s in ids], dtype=np.float64)
        mean = float(differences.mean())
        boot += differences[rng.integers(len(ids), size=(replicates, len(ids)))].mean(axis=1)
        per_context.append({'map_id': m, 'opponent_id': k,
                            'current_win_rate': sum(now[(m,k)].values())/len(ids),
                            'reference_win_rate': sum(old[(m,k)].values())/len(ids),
                            'delta_win_rate': mean})
    boot /= len(now)
    alpha = (1-confidence)/2
    def aggregation(key: str) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        for row in per_context:
            groups[row[key]].append(row['delta_win_rate'])
        return [{key: v, 'delta_win_rate': float(np.mean(deltas))} for v, deltas in sorted(groups.items())]
    return {
        'reference_kind': reference_kind,
        'delta_global_win_rate': float(np.mean([r['delta_win_rate'] for r in per_context])),
        'paired_stratified_bootstrap_ci': [float(v) for v in np.quantile(boot, [alpha, 1-alpha])],
        'per_map': aggregation('map_id'),
        'per_opponent': aggregation('opponent_id'),
        'per_context': per_context,
        'matched_episodes': sum(len(x) for x in now.values()),
        'bootstrap_replicates': replicates,
    }
