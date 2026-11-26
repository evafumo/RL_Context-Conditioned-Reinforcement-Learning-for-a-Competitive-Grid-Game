"""Grafici PNG offline delle serie A/B/C; nessuna operazione sull'MDP o GPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _numeric(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def _plot_detailed_behavior(root: Path, directory: Path, plt: Any) -> list[Path]:
    """Grafici longitudinali C suddivisi nelle 20 cartelle richieste.

    Ogni grafico riguarda UNO specifico gruppo (globale/mappa/opponent/contesto)
    e mostra l'evoluzione ai checkpoint, non le medie dei checkpoint.
    """
    from analysis.detailed_behavior import ACTION_NAMES, SCOPES, SIDES

    plots: list[Path] = []
    kinds = ("kill", "act", "obj")
    metric_labels = {"K": "Kill totali", "KS": "Kill SHOOT",
                     "KM": "Kill melee", "COMB": "Combattimenti iniziati"}
    for scope in SCOPES:
        for kind in kinds:
            rows = _read(root / f"{kind}_{scope}_curves.csv")
            if not rows:
                continue
            groups: dict[tuple[int | None, int | None], list[dict[str, Any]]] = {}
            for row in rows:
                m = int(row["map_id"]) if row["map_id"] else None
                k = int(row["opponent_id"]) if row["opponent_id"] else None
                groups.setdefault((m, k), []).append(row)
            for (m, k), group in sorted(groups.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0)):
                group.sort(key=lambda row: int(row["activations"]))
                if scope == "global":
                    caption, filetag = "globale", "global"
                elif scope == "map":
                    caption, filetag = f"mappa {m}", f"m{m}"
                elif scope == "opponent":
                    caption, filetag = f"avversario {k}", f"k{k}"
                else:
                    caption, filetag = f"mappa {m}, avversario {k}", f"m{m}_k{k}"
                sides = SIDES if kind != "act" else (None,)
                for side in sides:
                    if kind == "kill":
                        folder = directory / f"{side.capitalize()}_KILL_{scope.upper()}"
                        series = [(f"{metric}_{side}", label)
                                  for metric, label in metric_labels.items()]
                        title = f"Combattimento {side.capitalize()} - {caption}"
                        ylabel = "Eventi medi per partita"
                        scale = 1.
                    elif kind == "obj":
                        folder = directory / f"{side.capitalize()}_OBJ_{scope.upper()}"
                        series = [(f"OB_{side}", "Obiettivi catturati")]
                        title = f"Obiettivi {side.capitalize()} - {caption}"
                        ylabel = "Obiettivi medi per partita"
                        scale = 1.
                    else:
                        folder = directory / f"ACT_{scope.upper()}"
                        series = [(f"f_{action}", action) for action in ACTION_NAMES]
                        title = f"Distribuzione azioni Agent - {caption} (DUMMY esclusa)"
                        ylabel = "Frequenza sulle decisioni Agent (%)"
                        scale = 100.
                    figure, ax = plt.subplots(figsize=(9.5, 5.2))
                    shown = False
                    for field, legend in series:
                        xs, ys = [], []
                        for r in group:
                            v = _numeric(r, field)
                            if v is not None:
                                xs.append(int(r["activations"]))
                                ys.append(v * scale)
                        if xs:
                            ax.plot(xs, ys, marker="o", markersize=2.5,
                                    linewidth=1.5, label=legend)
                            shown = True
                    if not shown:
                        plt.close(figure)
                        continue
                    ax.set_title(title)
                    ax.set_xlabel("Attivazioni cumulative di training (checkpoint)")
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=.25)
                    if kind == "act":
                        ax.set_ylim(0, 100)
                    ax.legend(loc="best", fontsize=8)
                    figure.tight_layout()
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{kind}_{side + '_' if side else ''}{filetag}.png"
                    figure.savefig(path, dpi=150)
                    plt.close(figure)
                    plots.append(path)
    return plots


def generate_plots(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        import matplotlib
        matplotlib.use("Agg")  # Windows e server senza display
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Installare matplotlib con: python -m pip install matplotlib") from exc
    directory = root / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    def line_plot(source: str, image: str, series: list[tuple[str, str]],
                  title: str, ylabel: str, *, percentage: bool = False,
                  x_key: str = "activations") -> None:
        rows = _read(root / source)
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(9, 4.8))
        plotted = False
        for key, legend in series:
            pairs = [(float(r[x_key]), _numeric(r, key)) for r in rows
                     if _numeric(r, key) is not None]
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            if percentage:
                ys = tuple(v*100 for v in ys)
            ax.plot(xs, ys, label=legend, linewidth=1.6)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Attivazioni cumulative di training")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        if len(series) > 1:
            ax.legend(fontsize=9)
        fig.tight_layout()
        path = directory / image
        fig.savefig(path, dpi=150)
        plt.close(fig)
        produced.append(path)

    # Metriche A, CSV gia' prodotti da analyze_training.py.
    line_plot("learning_curves.csv", "win_rate_global.png",
              [("win_rate", "Win rate globale"), ("worst_six", "Worst-6")],
              "Prestazioni ai checkpoint", "Win rate (%)", percentage=True)
    line_plot("learning_curves.csv", "robustezza_contesti.png",
              [("context_std", "Deviazione standard tra contesti")],
              "Eterogeneita' nei 24 contesti", "Deviazione standard", percentage=False)

    # Metriche B, ogni update LOGGATO (possono esserci piu' punti con stesso t).
    line_plot("dqn_learning_curves.csv", "dqn_loss.png",
              [("loss", "Huber loss")], "Loss DQN", "Loss")
    line_plot("dqn_learning_curves.csv", "dqn_td_error.png",
              [("td_mae", "MAE TD"), ("td_p90", "P90 |TD|")],
              "Temporal Difference Error", "Errore TD")
    line_plot("dqn_learning_curves.csv", "dqn_q_values.png",
              [("mean_q", "Media Q(s,a)"), ("mean_max_legal_q", "Media max Q legale")],
              "Valori Q della online network", "Q-value")
    line_plot("dqn_learning_curves.csv", "dqn_q_std.png",
              [("std_q", "Std Q(s,a)"), ("std_max_legal_q", "Std max Q legale")],
              "Dispersione dei Q-values", "Deviazione standard")
    line_plot("dqn_learning_curves.csv", "dqn_epsilon.png",
              [("epsilon", "Epsilon")], "Esplorazione epsilon-greedy", "Epsilon")
    line_plot("dqn_checkpoint_summary.csv", "dqn_loss_per_checkpoint.png",
              [("mean_loss", "Huber media nell'intervallo")],
              "Loss media tra checkpoint", "Loss")

    # Metriche C, una osservazione per checkpoint valutato.
    line_plot("behavior_learning_curves.csv", "behavior_actions.png",
              [(f"action_{a}_frequency", a.upper()) for a in
               ("north", "south", "east", "west", "shoot", "pass")],
              "Distribuzione delle azioni Agent (DUMMY esclusa)", "Frequenza (%)",
              percentage=True)
    line_plot("behavior_learning_curves.csv", "behavior_objectives.png",
              [("objectives_agent_mean", "Agent"),
               ("objectives_opponent_mean", "Opponent")],
              "Obiettivi conquistati per episodio", "Obiettivi medi")
    line_plot("behavior_learning_curves.csv", "behavior_kills.png",
              [("kills_agent_mean", "Agent"),
               ("kills_opponent_mean", "Opponent")],
              "Eliminazioni per episodio", "Eliminazioni medie")
    line_plot("behavior_learning_curves.csv", "behavior_shooting.png",
              [("shoot_kill_rate_agent", "Agent"),
               ("shoot_kill_rate_opponent", "Opponent")],
              "Successo dei tiri (kill / tentativi)", "Kill per tiro (%)",
              percentage=True)

    # Heatmap ultimo checkpoint, i contesti sono 4x6; lasciare NaN nei mancanti.
    contexts = _read(root / "context_metrics.csv")
    if contexts:
        import numpy as np
        last = max(int(row["activations"]) for row in contexts)
        values = np.full((4, 6), np.nan)
        for row in contexts:
            if int(row["activations"]) == last:
                m = int(row["map_id"])
                k = int(row["opponent_id"])
                if 1 <= m <= 4 and 1 <= k <= 6:
                    values[m-1, k-1] = float(row["win_rate"])
        if np.isfinite(values).any():
            fig, ax = plt.subplots(figsize=(9, 4.5))
            im = ax.imshow(values * 100, vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(6), [f"Avversario {k}" for k in range(1, 7)])
            ax.set_yticks(range(4), [f"Mappa {m}" for m in range(1, 5)])
            ax.set_title(f"Win rate per contesto al checkpoint {last:,}")
            for i in range(4):
                for j in range(6):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i,j]*100:.1f}%", ha="center", va="center")
            fig.colorbar(im, ax=ax, label="Win rate (%)")
            fig.tight_layout()
            path = directory / "heatmap_ultimo_checkpoint.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            produced.append(path)
    produced.extend(_plot_detailed_behavior(root, directory, plt))
    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crea PNG dai CSV dell'analisi Giulio2")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = generate_plots(args.results_dir)
    print(f"[plots] immagini={len(paths)} directory={args.results_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
