"""Render the report figures from results/ into results/figures/*.png.

    python -m lahja.eval.figures

Palette: validated categorical slots 1-3 from the data-viz reference (blue #2a78d6,
orange #eb6834, aqua #1baf7a) on the light chart surface; validator reports worst adjacent
CVD dE 9.2 and normal-vision dE 27.6, with aqua below 3:1 contrast - relieved by the visible
direct labels here and the tables in docs/results.md. Nominal categories share slot 1: bar
length already encodes magnitude, so hue is not spent re-encoding it.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lahja import RESULTS
from lahja.data.filters import EGY_MARKERS
from lahja.data.normalize import normalize
from lahja.data.schema import load_examples, read_jsonl
from lahja.eval.report import paired_delta

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
FIGDIR = RESULTS / "figures"

LABELS = {
    "E-D-camelbert-da": "E-D  encoder 110M +EGY",
    "E-C-camelbert-da": "E-C  encoder 110M",
    "D-qwen3-17b": "D  Qwen3-1.7B +EGY",
    "C-qwen3-17b": "C  Qwen3-1.7B",
    "D-qwen3-06b": "D  Qwen3-0.6B +EGY",
    "C-qwen3-06b": "C  Qwen3-0.6B",
    "A_sonnet5_5shot": "A  Sonnet 5, 5-shot",
    "A_sonnet5_zeroshot": "A  Sonnet 5, zero-shot",
    "B-qwen3-06b-5shot": "B  Qwen3-0.6B untrained",
}
SHORT = {
    "E-D-camelbert-da": "encoder 110M",
    "D-qwen3-06b": "Qwen3-0.6B",
    "D-qwen3-17b": "Qwen3-1.7B",
}
PARAMS = {  # millions
    "E-D-camelbert-da": 110,
    "E-C-camelbert-da": 110,
    "D-qwen3-06b": 596,
    "C-qwen3-06b": 596,
    "D-qwen3-17b": 1720,
    "C-qwen3-17b": 1720,
}


def style(ax, xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=1, solid_capstyle="butt")
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)


def legend_above(ax, ncols: int) -> None:
    """Legend outside the plot area - never on top of the marks."""
    ax.legend(
        frameon=False,
        fontsize=9.5,
        ncols=ncols,
        labelcolor=INK2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.2,
        columnspacing=1.8,
    )


def scores(run: str, eval_set: str) -> dict | None:
    path = RESULTS / run / eval_set / "scores.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_headline(eval_set: str, title: str, fname: str) -> None:
    rows = [(r, s) for r in LABELS if (s := scores(r, eval_set))]
    rows.sort(key=lambda t: t[1]["exact_match"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ys = list(range(len(rows)))
    values = [s["exact_match"] for _, s in rows]
    lo = [v - s["exact_match_ci95"][0] for v, (_, s) in zip(values, rows, strict=True)]
    hi = [s["exact_match_ci95"][1] - v for v, (_, s) in zip(values, rows, strict=True)]
    ax.barh(ys, values, height=0.52, color=S1, zorder=3)
    ax.errorbar(
        values, ys, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4
    )
    for y, (_, s) in zip(ys, rows, strict=True):
        ax.text(
            s["exact_match_ci95"][1] + 0.012,
            y,
            f"{s['exact_match']:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(ys, [LABELS[r] for r, _ in rows], fontsize=9.5, color=INK2)
    ax.set_xlim(0, 0.80)
    style(ax, "Exact match (intent + all slots correct), 95% CI")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, dpi=200)
    plt.close(fig)


def fig_egyptian_effect() -> None:
    """Paired D-C differences: does the synthetic Egyptian data help?"""
    pairs = [
        ("encoder 110M", "E-D-camelbert-da", "E-C-camelbert-da"),
        ("Qwen3-0.6B", "D-qwen3-06b", "C-qwen3-06b"),
        ("Qwen3-1.7B", "D-qwen3-17b", "C-qwen3-17b"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for si, (eval_set, label, color) in enumerate(
        [("egy_test", "Egyptian test", S1), ("massive_ar_test", "MASSIVE test", S2)]
    ):
        ys, xs, los, his = [], [], [], []
        for pi, (_, a, b) in enumerate(pairs):
            d = paired_delta(a, b, eval_set)
            if not d:
                continue
            y = pi + (0.16 if si == 0 else -0.16)
            ys.append(y)
            xs.append(d["diff"])
            los.append(d["diff"] - d["ci"][0])
            his.append(d["ci"][1] - d["diff"])
            if d["significant"]:
                ax.text(d["ci"][1] + 0.005, y, "significant", va="center", fontsize=8, color=INK2)
        ax.errorbar(
            xs,
            ys,
            xerr=[los, his],
            fmt="o",
            markersize=8,
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=0,
            label=label,
            zorder=3,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    ax.axvline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=10, color=INK2)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlim(-0.08, 0.16)
    style(ax, "Change in exact match from adding synthetic Egyptian data (D − C), 95% CI")
    ax.set_title(
        "Egyptian data helps the encoder; not measurably the LLMs",
        loc="left",
        fontsize=12,
        color=INK,
        pad=30,
    )
    legend_above(ax, 2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "egyptian_effect.png", dpi=200)
    plt.close(fig)


def fig_marker_split() -> None:
    test = {e.id: e for e in load_examples("data/processed/egy_test.jsonl")}
    marked = {i for i, e in test.items() if EGY_MARKERS & set(normalize(e.utt).split())}
    runs = [
        "C-qwen3-06b",
        "D-qwen3-06b",
        "C-qwen3-17b",
        "D-qwen3-17b",
        "E-C-camelbert-da",
        "E-D-camelbert-da",
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for si, (ids, label, color) in enumerate(
        [
            (marked, f"dialect-marked ({len(marked)})", S1),
            (set(test) - marked, f"unmarked ({len(test) - len(marked)})", S2),
        ]
    ):
        vals = []
        for run in runs:
            s = {x["id"]: x for x in read_jsonl(RESULTS / run / "egy_test" / "scored.jsonl")}
            vals.append(sum(s[i]["em"] for i in ids) / len(ids))
        ys = [i + (0.19 if si == 0 else -0.19) for i in range(len(runs))]
        ax.barh(ys, vals, height=0.34, color=color, label=label, zorder=3)
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + 0.008, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(runs)), [LABELS[r] for r in runs], fontsize=9.5, color=INK2)
    ax.set_xlim(0, 0.80)
    style(ax, "Exact match on the Egyptian test set")
    ax.set_title(
        "Where the Egyptian data pays off: sentences with dialect markers",
        loc="left",
        fontsize=12,
        color=INK,
        pad=30,
    )
    legend_above(ax, 2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "marker_split.png", dpi=200)
    plt.close(fig)


def fig_error_composition() -> None:
    runs = [r for r in LABELS if (RESULTS / r / "egy_test" / "scored.jsonl").exists()]
    runs.sort(key=lambda r: scores(r, "egy_test")["exact_match"])
    parts = {"correct": [], "wrong intent": [], "intent right, slots wrong": []}
    for run in runs:
        items = list(read_jsonl(RESULTS / run / "egy_test" / "scored.jsonl"))
        parts["correct"].append(sum(i["em"] for i in items))
        parts["wrong intent"].append(sum(1 for i in items if not i["intent_ok"]))
        parts["intent right, slots wrong"].append(
            sum(1 for i in items if i["intent_ok"] and not i["em"])
        )
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    left = [0] * len(runs)
    # Segments are separated by a surface-coloured edge, not by padding the values:
    # inflating the stack would push a 200-item row past the 200 mark.
    for (label, vals), color in zip(parts.items(), (S1, S2, S3), strict=True):
        ax.barh(
            range(len(runs)),
            vals,
            left=left,
            height=0.52,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_yticks(range(len(runs)), [LABELS[r] for r in runs], fontsize=9.5, color=INK2)
    ax.set_xlim(0, 200)
    style(ax, "Egyptian test items (200)")
    ax.set_title("What goes wrong: intents vs. slots", loc="left", fontsize=12, color=INK, pad=30)
    legend_above(ax, 3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "error_composition.png", dpi=200)
    plt.close(fig)


def fig_size_accuracy() -> None:
    """Discrete architectures, so points - not a line, which would imply interpolation."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for runs, label, color in [
        (["E-D-camelbert-da", "D-qwen3-06b", "D-qwen3-17b"], "+ Egyptian data (D)", S1),
        (["E-C-camelbert-da", "C-qwen3-06b", "C-qwen3-17b"], "MSA only (C)", S2),
    ]:
        xs = [PARAMS[r] for r in runs]
        ys = [scores(r, "egy_test")["exact_match"] for r in runs]
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker="o",
            color=color,
            markersize=9,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
    for run, short in SHORT.items():
        ax.annotate(
            short,
            (PARAMS[run], scores(run, "egy_test")["exact_match"] + 0.016),
            ha="center",
            fontsize=8.5,
            color=INK2,
        )
    ax.set_xscale("log")
    ax.set_xlim(70, 2800)
    ax.set_xticks([110, 596, 1720], ["110M", "596M", "1.72B"])
    ax.set_ylim(0.44, 0.69)
    style(ax, "Parameters (log scale)")
    ax.set_ylabel("Exact match, Egyptian test", color=INK2, fontsize=9.5)
    ax.set_title(
        "A 110M encoder matches a 1.7B LLM on this task", loc="left", fontsize=12, color=INK, pad=30
    )
    legend_above(ax, 2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "size_accuracy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig_headline("egy_test", "Egyptian test set (200 native-written commands)", "headline_egy.png")
    fig_headline("massive_ar_test", "MASSIVE test set (Saudi/MSA)", "headline_massive.png")
    fig_egyptian_effect()
    fig_marker_split()
    fig_error_composition()
    fig_size_accuracy()
    print(f"wrote {len(list(FIGDIR.glob('*.png')))} figures to {FIGDIR}")


if __name__ == "__main__":
    main()
