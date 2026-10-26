"""
Compare the cross-validation results of several training runs (each runs/<model> folder has a results.json).
The first run is the reference (baseline).

    python scripts/compare_runs.py runs/roberta_large_rtx5060 runs/deberta-v3-large
    python scripts/compare_runs.py runs/a runs/b runs/c --out runs/comparison

Output: tables printed to the terminal + summary.csv, per_label_f1.csv, per_label_f1.png saved to --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [("f1_samples", "F1-samples"), ("f1_micro", "F1-micro"), ("f1_macro", "F1-macro"),
           ("pr_auc_macro", "PR-AUC"), ("roc_auc_macro", "ROC-AUC"),
           ("jaccard_samples", "Jaccard"), ("subset_accuracy", "Exact-match"), ("hamming_loss", "Hamming↓")]
LANG_ORDER = ["java", "python", "pharo", "other"]


def fmt_time(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def load_run(run_dir: Path):
    res_path = run_dir / "results.json"
    if not res_path.exists():
        raise SystemExit(f"✘ {res_path} not found")
    with open(res_path) as f:
        res = json.load(f)
    if res.get("split") != "5fold":
        raise SystemExit(f"✘ {res_path} is not a cross-validation result")

    model, seed, n_splits = run_dir.name, None, len(res["fold_metrics"])
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = cfg.get("model_name", model)
        seed = cfg.get("train_params", {}).get("seed")
    return {
        "name": run_dir.name,
        "model": model,
        "seed": seed,
        "n_splits": n_splits,
        "folds": res.get("folds", list(range(n_splits))),
        "fold_metrics": res["fold_metrics"],
        "best_epochs": res.get("fold_best_epochs"),
        "times": res.get("fold_times_s"),
        "per_label": res.get("per_label_f1"),
        "label_list": res["label_list"],
    }


def summary_table(runs):
    rows = []
    for r in runs:
        row = {"run": r["name"], "model": r["model"]}
        for key, name in METRICS:
            vals = [m.get(key, np.nan) for m in r["fold_metrics"]]  # older results may lack the new metrics
            row[name] = "n/a" if np.all(np.isnan(vals)) else f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}"
            row[f"{name}_mean"] = float(np.mean(vals))
        row["best epoch (mean)"] = f"{np.mean(r['best_epochs']):.1f}" if r["best_epochs"] else "n/a"
        row["CV time"] = fmt_time(sum(r["times"])) if r["times"] else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_folds(base, other, key="f1_samples"):
    """Same seed + same number of folds -> same split, so folds can be compared one by one."""
    a = {f: m[key] for f, m in zip(base["folds"], base["fold_metrics"])}
    b = {f: m[key] for f, m in zip(other["folds"], other["fold_metrics"])}
    common = sorted(set(a) & set(b))
    diffs = np.array([b[f] - a[f] for f in common])
    return common, diffs


def best_run(runs):
    """The run with the highest mean F1-samples, excluding the reference."""
    return max(runs[1:], key=lambda r: np.mean([m["f1_samples"] for m in r["fold_metrics"]]))


def delta_col(runs):
    return f"Δ ({best_run(runs)['name']} − reference)"


def per_label_frame(runs):
    labels = runs[0]["label_list"]
    df = pd.DataFrame({"label": labels})
    df["lang"] = [(runs[0]["per_label"] or {}).get(l, {}).get("lang", "") for l in labels]
    for r in runs:
        df[r["name"]] = [(r["per_label"] or {}).get(l, {}).get("mean", np.nan) for l in labels]
    if len(runs) > 1:
        df[delta_col(runs)] = df[best_run(runs)["name"]] - df[runs[0]["name"]]
    return df


def save_chart(df, runs, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    order = df.sort_values(runs[0]["name"]).reset_index(drop=True)
    y = np.arange(len(order))
    h = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(order) + 1.5))
    for i, r in enumerate(runs):
        ax.barh(y + (i - (len(runs) - 1) / 2) * h, order[r["name"]], height=h, label=r["model"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l} ({g})" for l, g in zip(order["label"], order["lang"])])
    ax.set_xlabel("F1 (mean over folds)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("F1 per label")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Result folders; the first one is the reference")
    parser.add_argument("--out", default="runs/comparison", help="Folder for the tables and the chart")
    args = parser.parse_args()

    runs = [load_run(Path(p)) for p in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    # 1. Overview
    summ = summary_table(runs)
    print("\n=== 1. Overview (mean ± std over folds) ===")
    print(summ[["run", "model"] + [n for _, n in METRICS] + ["best epoch (mean)", "CV time"]].to_string(index=False))
    summ.to_csv(out / "summary.csv", index=False)

    # 2. Fold-by-fold comparison with the reference
    base = runs[0]
    if len(runs) > 1:
        print(f"\n=== 2. Fold-by-fold vs reference '{base['name']}' (F1-samples) ===")
        for r in runs[1:]:
            if r["n_splits"] != base["n_splits"] or (r["seed"] and base["seed"] and r["seed"] != base["seed"]):
                print(f"  {r['name']}: different number of folds/seed → folds do not share data, skipping fold-by-fold comparison")
                continue
            common, diffs = paired_folds(base, r)
            wins = int((diffs > 0).sum())
            print(f"  {r['name']}: Δ per fold {' '.join(f'{d:+.4f}' for d in diffs)}"
                  f" · mean {diffs.mean():+.4f} ± {diffs.std():.4f} · wins {wins}/{len(common)} folds")
        print("  (Same seed + same number of folds ⇒ same data split. With 5 folds, a gap smaller than"
              " the standard deviation that does not win on every fold is not enough to call one model better.)")

    # 3. Per language + per label
    if all(r["per_label"] for r in runs):
        pl = per_label_frame(runs)
        print("\n=== 3. Macro F1 per language ===")
        lang_tbl = pl.groupby("lang")[[r["name"] for r in runs]].mean()
        lang_tbl = lang_tbl.reindex([l for l in LANG_ORDER if l in lang_tbl.index])
        print(lang_tbl.round(4).to_string())

        print("\n=== 4. F1 per label (mean over folds)" + (", sorted by Δ" if len(runs) > 1 else "") + " ===")
        sort_col = delta_col(runs) if len(runs) > 1 else runs[0]["name"]
        print(pl.sort_values(sort_col).round(4).to_string(index=False))
        pl.to_csv(out / "per_label_f1.csv", index=False)
        chart = save_chart(pl, runs, out / "per_label_f1.png")
    else:
        chart = None
        print("\n(Skipping the per-label part: older results.json has no per_label_f1)")

    print(f"\nSaved: {out / 'summary.csv'}" + (f", {out / 'per_label_f1.csv'}, {chart}" if chart else ""))


if __name__ == "__main__":
    main()
