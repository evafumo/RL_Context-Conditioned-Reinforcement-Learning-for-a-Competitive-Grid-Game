"""Serie longitudinali DQN (B) e comportamento (C), solo lettura dei log.

Non modifica policy, ambiente, replay o checkpoint. Compatibile con la
mancanza delle nuove statistiche B nei vecchi log di Training_0.
"""
from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

ACTION_NAMES = ("NORTH", "SOUTH", "EAST", "WEST", "SHOOT", "PASS")
DQN_METRICS = (
    "loss", "td_mae", "td_p90", "mean_q", "std_q", "mean_max_legal_q",
    "std_max_legal_q", "mean_target", "grad_norm",
)
DQN_COLUMNS = ("activations", "update_steps", "epsilon", "replay_size",
               "replay_fill_fraction", *DQN_METRICS)
BEHAVIOR_COLUMNS = (
    "activations", "episodes", "agent_decisions",
    *(f"action_{a.lower()}_frequency" for a in ACTION_NAMES),
    *(f"action_{a.lower()}_given_legal" for a in ACTION_NAMES),
    "objectives_agent_mean", "objectives_opponent_mean", "objectives_difference_mean",
    "shots_agent_mean", "shots_opponent_mean", "shot_kills_agent_mean",
    "shot_kills_opponent_mean", "shoot_kill_rate_agent", "shoot_kill_rate_opponent",
    "melee_encounters_mean", "melee_initiated_agent_mean", "melee_initiated_opponent_mean",
    "melee_kills_agent_mean", "melee_kills_opponent_mean", "kills_agent_mean",
    "kills_opponent_mean", "deaths_agent_mean", "deaths_opponent_mean",
)


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"Valore non numerico o non finito: {name}")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"Valore fuori intervallo: {name}")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Intero non valido: {name}")
    return value


def _csv(path: Path, records: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def parse_dqn_updates(path: str | Path) -> list[dict[str, Any]]:
    """Un record per update LOGGATO, senza interpolare update non registrati.

    Step X: environment_activations; asse secondario: update_steps.
    Update consecutivi possono avere lo stesso X, perche' gli update sono
    eseguiti senza raccogliere nuove transizioni.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result: list[dict[str, Any]] = []
    last_t = -1
    last_update = -1
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL non valido alla riga {line_no}: {path}") from exc
            if not isinstance(record, dict) or record.get("kind") != "dqn_update":
                continue
            dqn = record.get("dqn")
            other = record.get("other")
            if not isinstance(dqn, dict) or not isinstance(other, dict):
                raise ValueError(f"dqn/other mancanti nella riga {line_no}")
            t = _integer(other.get("environment_activations"), "environment_activations")
            u = _integer(record.get("update_index"), "update_index", minimum=1)
            if t < last_t or u <= last_update:
                raise ValueError(f"Ordine update/attivazioni non monotono alla riga {line_no}")
            if "update_steps" in dqn and dqn["update_steps"] != u:
                raise ValueError(f"update_steps e update_index non coincidono alla riga {line_no}")
            row = {"activations": t, "update_steps": u}
            for key in ("epsilon", "replay_fill_fraction"):
                row[key] = _number(record.get(key), key, minimum=0, maximum=1,
                                   allow_none=True)
            raw_size = record.get("replay_size")
            row["replay_size"] = None if raw_size is None else _integer(raw_size, "replay_size")
            for key in DQN_METRICS:
                row[key] = _number(dqn.get(key), key, allow_none=True,
                                   minimum=0 if key in ("loss", "td_mae", "td_p90",
                                                    "std_q", "std_max_legal_q", "grad_norm")
                                   else None)
            result.append(row)
            last_t, last_update = t, u
    return result


def _behavior_row(t: int, block: Mapping[str, Any]) -> dict[str, Any]:
    """Legge i campi ESATTI di evaluate._aggregate_behavior, senza risimulare."""
    episodes = _integer(block.get("episodes"), "behavior.episodes", minimum=1)
    actions = block.get("actions")
    obj = block.get("objective_captures")
    combat = block.get("combat")
    if not all(isinstance(item, dict) for item in (actions, obj, combat)):
        raise ValueError("Riepilogo behavior incompleto (actions, objective_captures, combat)")
    decisions = _integer(actions.get("agent_decisions"), "agent_decisions")
    counts = actions.get("counts")
    opps = actions.get("legal_opportunities")
    if not isinstance(counts, dict) or not isinstance(opps, dict):
        raise ValueError("Mancano counts/legal_opportunities")
    if sum(_integer(counts.get(a), f"counts.{a}") for a in ACTION_NAMES) != decisions:
        raise ValueError("Somma counts diversa da agent_decisions")
    row: dict[str, Any] = {"activations": t, "episodes": episodes,
                           "agent_decisions": decisions}
    for a in ACTION_NAMES:
        n = counts[a]
        opportunities = _integer(opps.get(a), f"legal_opportunities.{a}")
        if n > opportunities:
            raise ValueError("Azioni selezionate piu' spesso delle occasioni legali")
        row[f"action_{a.lower()}_frequency"] = n / decisions if decisions else None
        row[f"action_{a.lower()}_given_legal"] = n / opportunities if opportunities else None
    for target, source in (
        ("objectives_agent_mean", "agent_mean_per_episode"),
        ("objectives_opponent_mean", "opponent_mean_per_episode"),
        ("objectives_difference_mean", "agent_minus_opponent_mean_per_episode"),
    ):
        row[target] = _number(obj.get(source), source)
    for target, source in (
        ("shots_agent_mean", "shots_agent_total"),
        ("shots_opponent_mean", "shots_opponent_total"),
        ("shot_kills_agent_mean", "shot_kills_agent_total"),
        ("shot_kills_opponent_mean", "shot_kills_opponent_total"),
        ("melee_encounters_mean", "melee_encounters_total"),
        ("melee_initiated_agent_mean", "melee_initiated_agent_total"),
        ("melee_initiated_opponent_mean", "melee_initiated_opponent_total"),
        ("melee_kills_agent_mean", "melee_kills_agent_total"),
        ("melee_kills_opponent_mean", "melee_kills_opponent_total"),
    ):
        row[target] = _integer(combat.get(source), source) / episodes
    for target, source in (
        ("kills_agent_mean", "kills_agent_mean_per_episode"),
        ("kills_opponent_mean", "kills_opponent_mean_per_episode"),
        ("deaths_agent_mean", "deaths_agent_mean_per_episode"),
        ("deaths_opponent_mean", "deaths_opponent_mean_per_episode"),
    ):
        row[target] = _number(combat.get(source), source, minimum=0)
    for name in ("agent", "opponent"):
        row[f"shoot_kill_rate_{name}"] = _number(
            combat.get(f"shoot_kill_rate_{name}"), f"shoot_kill_rate_{name}",
            minimum=0, maximum=1, allow_none=True)
    return row


def _dqn_interval_summaries(updates: list[dict[str, Any]],
                            checkpoint_steps: list[int]) -> list[dict[str, Any]]:
    result = []
    lower = -1
    for upper in checkpoint_steps:
        batch = [r for r in updates if lower < r["activations"] <= upper]
        row: dict[str, Any] = {"activations": upper, "previous_checkpoint": max(lower, 0),
                               "logged_updates": len(batch)}
        for metric in DQN_METRICS:
            values = [float(r[metric]) for r in batch if r[metric] is not None]
            row[f"mean_{metric}"] = fmean(values) if values else None
            row[f"observations_{metric}"] = len(values)
        result.append(row)
        lower = upper
    return result


def collect_longitudinal(
    snapshots: Sequence[tuple[int, Mapping[str, Any]]],
    output_dir: str | Path, *, training_metrics: str | Path | None = None,
) -> dict[str, Any]:
    """Produce CSV longitudinali separati B/C per i checkpoint valutati.

    snapshots: [(attivazioni_cumulative, summary_di_evaluate), ...].
    Se tutte le valutazioni sono storiche senza behavior, segnala 'unavailable'.
    Se solo alcune mancano di behavior, blocca il confronto (serie incompleta).
    """
    if not snapshots:
        raise ValueError("Nessun checkpoint")
    times = [_integer(t, "checkpoint_activations") for t, _ in snapshots]
    if times != sorted(set(times)):
        raise ValueError("Checkpoint devono avere attivazioni crescenti e uniche")
    output = Path(output_dir)
    result: dict[str, Any] = {"B": "not_requested", "C": "unavailable_legacy",
                              "checkpoints": len(times), "files": []}
    if training_metrics is None:
        for name in ("dqn_learning_curves.csv", "dqn_checkpoint_summary.csv"):
            (output / name).unlink(missing_ok=True)
    if training_metrics is not None:
        updates = parse_dqn_updates(training_metrics)
        if not updates:
            raise ValueError("Non esistono dqn_update in metrics.jsonl")
        _csv(output / "dqn_learning_curves.csv", updates, DQN_COLUMNS)
        intervals = _dqn_interval_summaries(updates, times)
        interval_columns = ["activations", "previous_checkpoint", "logged_updates"] + [
            col for key in DQN_METRICS for col in (f"mean_{key}", f"observations_{key}")
        ]
        _csv(output / "dqn_checkpoint_summary.csv", intervals, interval_columns)
        result["B"] = "available"
        result["dqn_logged_updates"] = len(updates)
        result["B_fields_missing"] = [k for k in DQN_METRICS
                                       if all(r[k] is None for r in updates)]
        result["files"].extend(["dqn_learning_curves.csv", "dqn_checkpoint_summary.csv"])
    availability = ["behavior" in summary.get("overall", {}) for _, summary in snapshots]
    if any(availability) and not all(availability):
        raise ValueError("Metriche C presenti solo per alcuni checkpoint: rivalutare i mancanti")
    if not any(availability):
        for name in ("behavior_learning_curves.csv", "behavior_context_curves.csv"):
            (output / name).unlink(missing_ok=True)
    if all(availability):
        rows = []
        context_rows = []
        contexts0 = None
        for t, summary in snapshots:
            rows.append(_behavior_row(t, summary["overall"]["behavior"]))
            original_contexts = summary.get("contexts", [])
            context_keys = set()
            for ctx in original_contexts:
                m = _integer(ctx.get("map_id"), "map_id", minimum=1)
                k = _integer(ctx.get("opponent_id"), "opponent_id", minimum=1)
                if (m, k) in context_keys:
                    raise ValueError("Contesto duplicato")
                context_keys.add((m, k))
                if "behavior" not in ctx:
                    raise ValueError(f"Manca behavior nel contesto {(m, k)} al checkpoint {t}")
                row = _behavior_row(t, ctx["behavior"])
                context_rows.append({"map_id": m, "opponent_id": k, **row})
            if contexts0 is None:
                contexts0 = context_keys
            elif context_keys != contexts0:
                raise ValueError("Contesti diversi tra checkpoint")
        # Verifica: il numero di partite globali deve coincidere con la somma per contesto.
        for t, summary in snapshots:
            if summary["overall"]["behavior"]["episodes"] != sum(
                    c["behavior"]["episodes"] for c in summary["contexts"]):
                raise ValueError(f"Numero episodi behavior incoerente al checkpoint {t}")
        _csv(output / "behavior_learning_curves.csv", rows, BEHAVIOR_COLUMNS)
        _csv(output / "behavior_context_curves.csv", context_rows,
             ["map_id", "opponent_id", *BEHAVIOR_COLUMNS])
        result["C"] = "available"
        result["files"].extend(["behavior_learning_curves.csv", "behavior_context_curves.csv"])
    return result


__all__ = ["ACTION_NAMES", "DQN_METRICS", "DQN_COLUMNS", "BEHAVIOR_COLUMNS",
           "parse_dqn_updates", "collect_longitudinal"]
