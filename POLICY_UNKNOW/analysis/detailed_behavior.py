"""Metriche C dettagliate per checkpoint: kill, azioni e obiettivi.

Usa esclusivamente i contatori preesistenti di summary.json (uno per contesto).
Non esegue episodi, non carica checkpoint e non modifica l'MDP.

Definizioni:
- K, KS, KM e OB: eventi / partite del gruppo.
- COMB: (attacchi SHOOT + incontri melee iniziati dal giocatore) / partite.
  Ogni SHOOT conta un attacco anche se non uccide; un melee e' contato
  una sola volta per il giocatore indicato come melee_attacker.
- f(a): selezioni Agent dell'azione a / decisioni libere Agent.
  DUMMY non e' una decisione libera; il suo conteggio viene conservato
  soltanto come diagnostica e non entra nella distribuzione delle sei azioni.

I gruppi per mappa/avversario sono sommati sui conteggi grezzi, NON ottenuti
mediando frequenze per contesto che hanno denominatori differenti.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

ACTION_NAMES = ("NORTH", "SOUTH", "EAST", "WEST", "SHOOT", "PASS")
SIDES = ("agent", "opponent")
SCOPES = ("global", "context", "map", "opponent")
COMBAT_COUNTERS = (
    "shots_agent", "shots_opponent", "shot_kills_agent", "shot_kills_opponent",
    "melee_encounters", "melee_initiated_agent", "melee_initiated_opponent",
    "melee_kills_agent", "melee_kills_opponent", "kills_agent", "kills_opponent",
)
RAW_COMBAT = tuple(f"{name}_total" for name in COMBAT_COUNTERS)
RAW_OBJECTIVES = ("agent_total", "opponent_total")


def _whole(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Contatore non valido: {label}")
    return value


def _raw_counts(ctx: Mapping[str, Any], horizon: int) -> dict[str, int]:
    """Estrae e verifica i contatori esatti di UN contesto."""
    m, k = (_whole(ctx.get(key), key) for key in ("map_id", "opponent_id"))
    if not (1 <= m <= 4 and 1 <= k <= 6):
        raise ValueError(f"Contesto fuori intervallo {(m, k)}")
    n = _whole(ctx.get("episodes"), "episodes")
    if not n:
        raise ValueError("Contesto senza episodi")
    behavior = ctx["behavior"]
    if _whole(behavior.get("episodes"), "behavior.episodes") != n:
        raise ValueError(f"Numero partite behavior incoerente in {(m, k)}")
    actions, objects, combat = (behavior[key] for key in
                                 ("actions", "objective_captures", "combat"))
    result = {"episodes": n, "agent_decisions": _whole(actions.get("agent_decisions"), "agent_decisions")}
    counts = actions["counts"]
    for a in ACTION_NAMES:
        result[f"action_{a.lower()}_count"] = _whole(counts.get(a), f"counts.{a}")
    if sum(result[f"action_{a.lower()}_count"] for a in ACTION_NAMES) != result["agent_decisions"]:
        raise ValueError(f"Azioni e agent_decisions discordano in {(m, k)}")
    result["dummy_count"] = n*horizon - result["agent_decisions"]
    if result["dummy_count"] < 0:
        raise ValueError(f"Decisioni Agent oltre l'orizzonte in {(m, k)}")
    for s in SIDES:
        result[f"objectives_{s}_total"] = _whole(objects.get(f"{s}_total"), f"objectives.{s}_total")
    for name in COMBAT_COUNTERS:
        result[f"{name}_total"] = _whole(combat.get(f"{name}_total"), f"combat.{name}_total")
    _verify_combat(result)
    return result


def _verify_combat(raw: Mapping[str, int]) -> None:
    for side in SIDES:
        if (raw[f"shot_kills_{side}_total"] + raw[f"melee_kills_{side}_total"]
                != raw[f"kills_{side}_total"]):
            raise ValueError(f"Kill shooting+melee incoerenti per {side}")
        if raw[f"shot_kills_{side}_total"] > raw[f"shots_{side}_total"]:
            raise ValueError(f"Shot kill oltre numero di tiri per {side}")
    if (raw["melee_initiated_agent_total"] + raw["melee_initiated_opponent_total"]
            != raw["melee_encounters_total"]):
        raise ValueError("Melee iniziati incoerenti")


def _sum_raw(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not items:
        raise ValueError("Gruppo senza contesti")
    keys = tuple(items[0].keys())
    return {name: sum(item[name] for item in items) for name in keys}


def _rates(t: int, raw: Mapping[str, int], *, map_id: int | None,
           opponent_id: int | None) -> dict[str, Any]:
    n = raw["episodes"]
    nd = raw["agent_decisions"]
    r: dict[str, Any] = {
        "activations": t, "map_id": map_id, "opponent_id": opponent_id,
        "episodes": n, "agent_decisions": nd, "dummy_count": raw["dummy_count"],
    }
    for s in SIDES:
        for dest, source in (("K", "kills"), ("KS", "shot_kills"),
                             ("KM", "melee_kills")):
            r[f"{dest}_{s}"] = raw[f"{source}_{s}_total"] / n
            r[f"{dest}_{s}_total"] = raw[f"{source}_{s}_total"]
        combat = raw[f"shots_{s}_total"] + raw[f"melee_initiated_{s}_total"]
        r[f"COMB_{s}"] = combat / n
        r[f"COMB_{s}_total"] = combat
        r[f"SHOTS_{s}_total"] = raw[f"shots_{s}_total"]
        r[f"MELEE_INIT_{s}_total"] = raw[f"melee_initiated_{s}_total"]
        r[f"OB_{s}"] = raw[f"objectives_{s}_total"] / n
        r[f"OB_{s}_total"] = raw[f"objectives_{s}_total"]
    for a in ACTION_NAMES:
        count = raw[f"action_{a.lower()}_count"]
        r[f"f_{a}"] = count / nd if nd else None
        r[f"count_{a}"] = count
    # DUMMY non fa parte delle azioni con cui si calcola f(a).
    r["f_DUMMY"] = None
    return r


_BASE = ("activations", "map_id", "opponent_id", "episodes", "agent_decisions", "dummy_count")
_KILL_FIELDS = tuple(x for s in SIDES for metric in ("K", "KS", "KM", "COMB")
                     for x in (f"{metric}_{s}", f"{metric}_{s}_total")) + tuple(
                         x for s in SIDES for x in (f"SHOTS_{s}_total", f"MELEE_INIT_{s}_total"))
_OBJECTIVE_FIELDS = tuple(x for s in SIDES for x in (f"OB_{s}", f"OB_{s}_total"))
_ACTION_FIELDS = tuple(x for a in ACTION_NAMES for x in (f"f_{a}", f"count_{a}")) + ("f_DUMMY",)
_FILE_FIELDS = {"kill": _BASE + _KILL_FIELDS,
                "act": _BASE + _ACTION_FIELDS,
                "obj": _BASE + _OBJECTIVE_FIELDS}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _has_raw_counts(ctx: Mapping[str, Any]) -> bool:
    behavior = ctx.get("behavior")
    if not isinstance(behavior, dict):
        return False
    obj = behavior.get("objective_captures", {})
    combat = behavior.get("combat", {})
    return (isinstance(obj, dict) and isinstance(combat, dict)
            and all(key in obj for key in RAW_OBJECTIVES)
            and all(key in combat for key in RAW_COMBAT))


def collect_detailed_behavior(
    snapshots: Sequence[tuple[int, Mapping[str, Any]]], output_dir: str | Path,
) -> dict[str, Any]:
    """Crea 12 CSV: kill/act/obj x global/context/map/opponent.

    Restituisce uno status esplicito per dati legacy senza conteggi grezzi.
    Non inventa zeri per i dati assenti e non combina checkpoint eterogenei.
    """
    output = Path(output_dir)
    filenames = [f"{kind}_{scope}_curves.csv" for kind in _FILE_FIELDS for scope in SCOPES]
    readiness = [_has_raw_counts(ctx)
                 for _, summary in snapshots for ctx in summary.get("contexts", [])]
    if not readiness or not any(readiness):
        for fn in filenames:
            (output / fn).unlink(missing_ok=True)
        return {"status": "unavailable_missing_raw_counts", "files": []}
    if not all(readiness):
        raise ValueError("Contatori C dettagliati presenti solo in alcuni contesti/checkpoint")
    by_scope: dict[str, list[dict[str, Any]]] = {scope: [] for scope in SCOPES}
    for t, summary in snapshots:
        if type(t) is not int or t < 0:
            raise ValueError("Attivazioni checkpoint non valide")
        horizon = _whole(summary.get("horizon"), "horizon")
        if not horizon:
            raise ValueError("Horizon assente")
        per_context: dict[tuple[int, int], dict[str, int]] = {}
        for ctx in summary["contexts"]:
            key = (ctx["map_id"], ctx["opponent_id"])
            if key in per_context:
                raise ValueError(f"Contesto duplicato {key}")
            per_context[key] = _raw_counts(ctx, horizon)
        if not per_context:
            raise ValueError("Nessun contesto")
        all_raw = _sum_raw(list(per_context.values()))
        if all_raw["episodes"] != summary["overall"]["episodes"]:
            raise ValueError("Episodi globali incoerenti")
        overall_behavior = summary["overall"].get("behavior")
        if isinstance(overall_behavior, dict):
            if all_raw["agent_decisions"] != overall_behavior["actions"]["agent_decisions"]:
                raise ValueError("N_DA globale incoerente con il summary")
            for side in SIDES:
                if all_raw[f"kills_{side}_total"] != overall_behavior["combat"][f"kills_{side}_total"]:
                    raise ValueError("Kill globali incoerenti con il summary")
                if all_raw[f"objectives_{side}_total"] != overall_behavior["objective_captures"][f"{side}_total"]:
                    raise ValueError("Obiettivi globali incoerenti con il summary")
        by_scope["global"].append(_rates(t, all_raw, map_id=None, opponent_id=None))
        for (m, k), raw in sorted(per_context.items()):
            by_scope["context"].append(_rates(t, raw, map_id=m, opponent_id=k))
        for field, scope in (("map_id", "map"), ("opponent_id", "opponent")):
            grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
            for (m, k), raw in per_context.items():
                grouped[m if field == "map_id" else k].append(raw)
            for val, parts in sorted(grouped.items()):
                by_scope[scope].append(_rates(t, _sum_raw(parts),
                                              map_id=val if scope == "map" else None,
                                              opponent_id=val if scope == "opponent" else None))
    generated = []
    for kind, columns in _FILE_FIELDS.items():
        for scope in SCOPES:
            name = f"{kind}_{scope}_curves.csv"
            _write_csv(output / name, by_scope[scope], columns)
            generated.append(name)
    return {"status": "available", "files": generated,
            "definitions": {"COMB": "(shots + melee_initiated) / episodes",
                            "DUMMY": "excluded from free-agent action distributions"}}


__all__ = ["collect_detailed_behavior", "ACTION_NAMES", "SIDES", "SCOPES"]
