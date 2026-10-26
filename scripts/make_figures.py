"""
Generate the README figures from the finished runs (runs/*) and the dataset, into docs/images/.

    python scripts/make_figures.py
    python scripts/make_figures.py --out docs/images

Needs the run folders listed in RUNS (results.json, history_fold*.csv) and runs/analysis/analysis.json
(python scripts/analyze_oof.py). Missing inputs only skip the figures that need them.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# run folder -> (display name, color)
RUNS = {
    "runs/ml_tfidf_logreg": ("TF-IDF + LR (baseline)", "#8C8C8C"),
    "runs/baseline_roberta_large": ("RoBERTa-large (baseline)", "#E0782F"),
    "runs/roberta_large_rtx5060": ("RoBERTa-large (ours)", "#2C6BAA"),
    "runs/unixcoder-base": ("UniXcoder-base", "#59A14F"),
    "runs/deberta-v3-large": ("DeBERTa-v3-large", "#8E6BBF"),
    "runs/modernbert-large": ("ModernBERT-large", "#B8860B"),
}
MAIN, BASELINE, ML = "runs/roberta_large_rtx5060", "runs/baseline_roberta_large", "runs/ml_tfidf_logreg"
LANG_COLORS = {"java": "#E0782F", "python": "#2C6BAA", "pharo": "#8E6BBF"}
INK, MUTED, GRID = "#14213D", "#5A6175", "#DEDAD0"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def load_results():
    out = {}
    for run in RUNS:
        p = Path(run) / "results.json"
        if p.exists():
            out[run] = json.loads(p.read_text())
    return out


def save(fig, out, name):
    path = out / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  ✔ {path}")


def fig_label_distribution(out):
    df = pd.read_csv("data/raw/sentences.csv")
    rows = []
    for lang, labels in zip(df["lang"], df["labels"]):
        lang = {"JAVA": "java", "PY": "python", "PHARO": "pharo"}[lang]
        for lab in str(labels).split(";"):
            name = f"{lab}_{lang}" if lab in ("summary", "usage", "expand") else lab
            rows.append((name, lang))
    counts = pd.DataFrame(rows, columns=["label", "lang"]).value_counts().reset_index(name="n").sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh(counts["label"], counts["n"], color=[LANG_COLORS[l] for l in counts["lang"]])
    for y, n in enumerate(counts["n"]):
        ax.text(n + 8, y, str(n), va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Sentences carrying the label")
    ax.set_title(f"Label distribution · {len(df):,} sentences · 19 labels")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in LANG_COLORS.values()],
              labels=["Java", "Python", "Pharo"], loc="lower right", frameon=False)
    save(fig, out, "label_distribution.png")


def fig_results_overview(res, analysis, out):
    names, means, stds, colors = [], [], [], []
    for run, (name, color) in RUNS.items():
        if run in res:
            vals = [m["f1_samples"] for m in res[run]["fold_metrics"]]
            names.append(name); means.append(np.mean(vals)); stds.append(np.std(vals)); colors.append(color)
    if analysis and "ensemble" in analysis:
        e = analysis["ensemble"]["metrics"]["f1_samples"]
        names.append("Ensemble of 4 (post-hoc)"); means.append(e["mean"]); stds.append(e["std"]); colors.append(INK)
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(order))
    ax.barh(y, [means[i] for i in order], xerr=[stds[i] for i in order], color=[colors[i] for i in order],
            error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_yticks(y, [names[i] for i in order])
    for yy, i in zip(y, order):
        ax.text(means[i] + stds[i] + 0.01, yy, f"{means[i]:.4f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (samples), mean ± std over 5 folds")
    ax.set_title("5-fold cross-validation results")
    ax.grid(axis="y", visible=False)
    save(fig, out, "results_overview.png")


def fig_per_fold(res, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for run, (name, color) in RUNS.items():
        if run not in res:
            continue
        folds = res[run].get("folds", list(range(len(res[run]["fold_metrics"]))))
        vals = [m["f1_samples"] for m in res[run]["fold_metrics"]]
        ax.plot([f + 1 for f in folds], vals, marker="o", color=color, label=name,
                linewidth=2.4 if run == MAIN else 1.4)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Fold (identical sentences for every method)")
    ax.set_ylabel("F1 (samples)")
    ax.set_title("Every method on the same 5 folds")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="center right")
    save(fig, out, "per_fold.png")


def fig_gain(res, analysis, out):
    if not analysis or MAIN not in res:
        return
    steps = [("RoBERTa baseline\n(threshold 0.5)", analysis["ablation"]["threshold 0.5"]["mean"], RUNS[BASELINE][1]),
             ("+ at least one\nlabel", analysis["ablation"]["+ at least one label"]["mean"], RUNS[BASELINE][1]),
             ("+ per-label\nthresholds", analysis["ablation"]["+ thresholds + at least one"]["mean"], RUNS[BASELINE][1]),
             ("Ours: + prompt,\nASL, sampler", res[MAIN]["avg_metrics"]["f1_samples"], RUNS[MAIN][1])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(steps))
    ax.bar(x, [s[1] for s in steps], color=[s[2] for s in steps], width=0.6)
    for i, (_, v, _) in enumerate(steps):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, color=INK, fontweight="bold")
        if i:
            ax.text(i, v / 2, f"+{v - steps[i - 1][1]:.3f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(x, [s[0] for s in steps])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (samples)")
    ax.set_title("Where the gain over the RoBERTa baseline comes from")
    ax.grid(axis="x", visible=False)
    save(fig, out, "gain_decomposition.png")


def mean_history(run, column):
    hs = [pd.read_csv(p).set_index("epoch")[column] for p in sorted(Path(run).glob("history_fold*.csv"))]
    return pd.concat(hs, axis=1).mean(axis=1) if hs else None


def fig_training_curves(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for run, (name, color) in RUNS.items():
        f1 = mean_history(run, "f1_samples") if run != ML else None
        if f1 is None:
            continue
        ax1.plot(f1.index, f1.values, color=color, label=name, linewidth=2.4 if run == MAIN else 1.4)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation F1 (samples)")
    ax1.set_title("Validation F1 per epoch (mean over folds)")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    # losses of different runs are not comparable (BCE vs ASL): show train vs validation loss of the main model
    train_loss, val_loss = mean_history(MAIN, "train_loss"), mean_history(MAIN, "val_loss")
    if train_loss is not None:
        ax2.plot(train_loss.index, train_loss.values, color=RUNS[MAIN][1], linestyle="--", label="train loss")
        ax2.plot(val_loss.index, val_loss.values, color=RUNS[MAIN][1], label="validation loss", linewidth=2)
        best = int(val_loss.idxmin())
        ax2.axvline(best, color=MUTED, linestyle=":", linewidth=1)
        ax2.text(best + 0.4, ax2.get_ylim()[1] * 0.9, f"lowest mean validation loss\n(epoch {best})", fontsize=8, color=MUTED)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("Asymmetric Loss")
    ax2.set_title("RoBERTa-large (ours): train vs validation loss")
    ax2.legend(frameon=False, fontsize=8)
    save(fig, out, "training_curves.png")


def lang_of(res, run):
    return {lab: v.get("lang") for lab, v in res[run]["per_label_f1"].items()}


def fig_per_language(res, out):
    runs = [r for r in RUNS if r in res and res[r].get("per_label_f1")]
    langs = ["java", "python", "pharo"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(runs)
    for k, run in enumerate(runs):
        pl, lg = res[run]["per_label_f1"], lang_of(res, run)
        vals = [np.mean([v["mean"] for lab, v in pl.items() if lg[lab] == lang]) for lang in langs]
        xs = np.arange(3) + (k - (len(runs) - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=RUNS[run][1], label=RUNS[run][0])
    ax.set_xticks(np.arange(3), ["Java (7 labels)", "Python (5 labels)", "Pharo (7 labels)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro F1 over the language’s labels")
    ax.set_title("Results per language")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, out, "per_language.png")


def fig_per_label(res, out):
    runs = [r for r in (ML, BASELINE, MAIN) if r in res and res[r].get("per_label_f1")]
    if MAIN not in runs:
        return
    order = sorted(res[MAIN]["per_label_f1"], key=lambda lab: res[MAIN]["per_label_f1"][lab]["mean"])
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(order))
    h = 0.8 / len(runs)
    for k, run in enumerate(runs):
        vals = [res[run]["per_label_f1"][lab]["mean"] for lab in order]
        ax.barh(y + (k - (len(runs) - 1) / 2) * h, vals, height=h, color=RUNS[run][1], label=RUNS[run][0])
    lg = lang_of(res, MAIN)
    ax.set_yticks(y, [f"{lab} ({lg[lab]})" for lab in order], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 per label (mean over 5 folds)")
    ax.set_title("Results per label")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, out, "per_label.png")


def params_of(run):
    log = Path(run) / "train.log"
    if log.exists():
        m = re.search(r"trainable [\d.]+M / ([\d.]+)M", log.read_text(errors="ignore"))
        if m:
            return float(m.group(1))
    return None


# label placement (points) to keep the bubbles' labels apart
LABEL_OFFSETS = {MAIN: (14, 8, "left"), "runs/modernbert-large": (-14, -26, "right")}


def fig_backbone_cost(res, out):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for run, (name, color) in RUNS.items():
        if run not in res or run == ML or not res[run].get("fold_times_s"):
            continue
        hours = sum(res[run]["fold_times_s"]) / 3600
        f1 = res[run]["avg_metrics"]["f1_samples"]
        params = params_of(run)
        ax.scatter(hours, f1, s=(params or 150) * 1.6, color=color, alpha=0.85, edgecolor="white", linewidth=1)
        label = f"{name}\n{params:.0f}M params" if params else name
        dx, dy, ha = LABEL_OFFSETS.get(run, (12, -6, "left"))
        ax.annotate(label, (hours, f1), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha)
    ax.set_xlabel("5-fold training time on one RTX 5060 (hours)")
    ax.set_ylabel("F1 (samples)")
    ax.set_title("Quality vs cost (bubble size = parameters)")
    ax.set_xlim(0, 5); ax.set_ylim(0.5, 0.9)
    save(fig, out, "backbone_cost.png")


def fig_errors(analysis, out):
    if not analysis or "errors" not in analysis:
        return
    err = analysis["errors"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.8})
    groups = err["exact_match"]
    names = [g.replace("lang ", "").replace("PY", "Python").replace("JAVA", "Java").replace("PHARO", "Pharo")
             for g in groups]
    vals = [v["pct"] for v in groups.values()]
    colors = [RUNS[MAIN][1] if "label" in g else "#8E6BBF" for g in groups]
    ax1.bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax1.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax1.set_xticks(range(len(vals)), names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 100); ax1.set_ylabel("exact match (%)")
    ax1.set_title("Exact match by subgroup (main model)")
    ax1.grid(axis="x", visible=False)
    top = err["top_confusions"][:8][::-1]
    ax2.barh(range(len(top)), [t["count"] for t in top], color="#C0392B", alpha=0.8)
    ax2.set_yticks(range(len(top)), [f"{t['missed']} → {t['predicted']}" for t in top], fontsize=8)
    ax2.set_xlabel("sentences")
    ax2.set_title("Most frequent confusions (missed → predicted)")
    ax2.grid(axis="y", visible=False)
    save(fig, out, "error_analysis.png")


def render_text(lines, out, name, title):
    """Render a log excerpt like a terminal window."""
    fig = plt.figure(figsize=(12, 0.2 * len(lines) + 0.6))
    fig.patch.set_facecolor("#1E1F29")
    fig.text(0.01, 0.995, title, color="#8FA3BF", family="DejaVu Sans Mono", fontsize=9, va="top")
    fig.text(0.01, 0.995 - 0.35 / (0.2 * len(lines) + 0.6), "\n".join(lines), color="#E6E6E6",
             family="DejaVu Sans Mono", fontsize=8.5, va="top", linespacing=1.45)
    fig.savefig(out / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ {out / name}")


def fig_logs(out, log_run="runs/deberta-v3-large"):
    log = Path(log_run) / "train.log"
    if log.exists():
        lines = log.read_text(errors="ignore").splitlines()
        start = next(i for i, l in enumerate(lines) if "TRAINING RUN" in l) - 1
        first = next(i for i, l in enumerate(lines) if l.startswith("Epoch   1/"))
        epoch = next(i for i, l in enumerate(lines) if l.startswith("Epoch  27/30"))
        excerpt = lines[start:first - 1] + ["", "  ...", ""] + lines[epoch - 1:epoch + 9]
        render_text(excerpt, out, "training_log.png", f"$ python scripts/run_experiments.py   ·   {log}")
        summary = next(i for i, l in enumerate(lines) if "CROSS-VALIDATION SUMMARY" in l) - 1
        end = next(i for i, l in enumerate(lines[summary:], summary) if "Total CV time" in l) + 1
        render_text(lines[summary:end], out, "cv_summary_log.png", f"End of cross-validation   ·   {log}")
    exp = Path("runs/experiments.log")
    if exp.exists():
        lines = exp.read_text(errors="ignore").splitlines()
        starts = [i for i, l in enumerate(lines) if "RUN EXPERIMENTS" in l]
        done = [i for i, l in enumerate(lines) if l.startswith("SUMMARY ·")]
        if starts and done:
            s = max(i for i in starts if i < done[-1]) - 1
            e = next(i for i, l in enumerate(lines[done[-1]:], done[-1]) if l.startswith("▶ Comparing")) + 1
            keep = [l for l in lines[s:e] if l.strip()]
            render_text(keep, out, "run_experiments_log.png", "$ python scripts/run_experiments.py   ·   runs/experiments.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = load_results()
    a_path = Path("runs/analysis/analysis.json")
    analysis = json.loads(a_path.read_text()) if a_path.exists() else None
    print(f"Runs found: {len(res)} / {len(RUNS)} · analysis.json: {'yes' if analysis else 'no'}")

    fig_label_distribution(out)
    fig_results_overview(res, analysis, out)
    fig_per_fold(res, out)
    fig_gain(res, analysis, out)
    fig_training_curves(out)
    fig_per_language(res, out)
    fig_per_label(res, out)
    fig_backbone_cost(res, out)
    fig_errors(analysis, out)
    fig_logs(out)


if __name__ == "__main__":
    main()
