"""
Deeper analysis of finished cross-validation runs, from their out-of-fold predictions (oof_fold*.npz):

  1. prediction behaviour      (empty predictions, labels per sentence)
  2. decision-rule ablation    (threshold 0.5 / at least one label / per-label thresholds) on one run
  3. paired t-tests            (every run vs the reference, over the shared folds)
  4. ensemble                  (mean probability of several runs, thresholds re-tuned per fold)
  5. error analysis            (confusions, exact match by subgroup) of the reference run

    python scripts/analyze_oof.py
    python scripts/analyze_oof.py --reference runs/unixcoder-base --out runs/analysis_unixcoder

Writes <out>/analysis.json. Runs trained before oof files existed: python scripts/rebuild_oof.py <run_dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # train.py lives in src/
import train as T  # noqa: E402

DEFAULT_RUNS = ["runs/baseline_roberta_large", "runs/ml_tfidf_logreg", "runs/roberta_large_rtx5060",
                "runs/unixcoder-base", "runs/deberta-v3-large", "runs/modernbert-large"]
DEFAULT_ENSEMBLE = ["runs/roberta_large_rtx5060", "runs/unixcoder-base", "runs/deberta-v3-large",
                    "runs/modernbert-large"]


def load_oof(run):
    files = sorted(Path(run).glob("oof_fold*.npz"), key=lambda p: int(p.stem.replace("oof_fold", "")))
    if not files:
        raise SystemExit(f"✘ {run} has no oof_fold*.npz (run scripts/rebuild_oof.py {run})")
    out = []
    for f in files:
        z = np.load(f)
        out.append({"fold": int(f.stem.replace("oof_fold", "")), "ids": z["ids"], "probs": z["probs"],
                    "labels": z["labels"].astype(int), "thr": z["thresholds"]})
    return out


def tune(y, p):
    return T.optimize_thresholds_samples(y, p, base=T.optimize_thresholds_per_label(y, p), iters=2)


def decide(p, thr, at_least_one=False):
    pred = (p >= thr).astype(int)
    if at_least_one:
        empty = pred.sum(1) == 0
        pred[empty, p[empty].argmax(1)] = 1
    return pred


def f1_samples(folds, rule):
    """rule: 'fixed' (0.5) | 'saved' (thresholds of the run) | 'tuned' (re-tuned per fold); + at-least-one flag."""
    mode, alo = rule
    vals = []
    for f in folds:
        thr = {"fixed": np.full(f["probs"].shape[1], 0.5), "saved": f["thr"]}.get(mode)
        if thr is None:
            thr = tune(f["labels"], f["probs"])
        vals.append(T.compute_metrics(f["labels"], decide(f["probs"], thr, alo), f["probs"])["f1_samples"])
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Runs to describe and test")
    parser.add_argument("--reference", default="runs/roberta_large_rtx5060", help="Main model (t-tests, errors)")
    parser.add_argument("--ablation-run", default="runs/baseline_roberta_large", help="Run for the decision-rule ablation")
    parser.add_argument("--ensemble", nargs="+", default=DEFAULT_ENSEMBLE, help="Runs averaged in the ensemble")
    parser.add_argument("--out", default="runs/analysis")
    args = parser.parse_args()

    runs = [r for r in args.runs if (Path(r) / "results.json").exists()]
    labels = json.load(open(Path(args.reference) / "results.json"))["label_list"]
    data = pd.read_csv("data/raw/sentences.csv").set_index("id")
    oof = {r: load_oof(r) for r in set(runs) | {args.reference, args.ablation_run} | set(args.ensemble)}
    report = {}

    print("=== 1. Prediction behaviour (all folds pooled) ===")
    report["behaviour"] = {}
    for r in runs:
        y = np.vstack([f["labels"] for f in oof[r]])
        pred = np.vstack([decide(f["probs"], f["thr"]) for f in oof[r]])
        b = {"empty_pct": 100 * float((pred.sum(1) == 0).mean()), "mean_pred": float(pred.sum(1).mean()),
             "mean_true": float(y.sum(1).mean()), "over_predict_pct": 100 * float((pred.sum(1) > y.sum(1)).mean())}
        report["behaviour"][r] = b
        print(f"  {r:32s} empty {b['empty_pct']:5.1f}% · labels/sentence {b['mean_pred']:.2f} "
              f"(true {b['mean_true']:.2f}) · over-predict {b['over_predict_pct']:4.1f}%")

    print(f"\n=== 2. Decision-rule ablation on {args.ablation_run} (F1 samples, mean ± std) ===")
    report["ablation"] = {}
    for name, rule in [("threshold 0.5", ("fixed", False)), ("+ at least one label", ("fixed", True)),
                       ("+ per-label thresholds", ("tuned", False)), ("+ thresholds + at least one", ("tuned", True))]:
        m, s = f1_samples(oof[args.ablation_run], rule)
        report["ablation"][name] = {"mean": m, "std": s}
        print(f"  {name:32s} {m:.4f} ± {s:.4f}")

    print(f"\n=== 3. Paired t-tests vs {args.reference} (F1 samples per fold) ===")
    f1 = {r: {f: m["f1_samples"] for f, m in zip(json.load(open(Path(r) / "results.json")).get("folds", range(5)),
                                                    json.load(open(Path(r) / "results.json"))["fold_metrics"])}
          for r in runs + [args.reference]}
    report["ttests"] = {}
    for r in runs:
        if r == args.reference:
            continue
        common = sorted(set(f1[r]) & set(f1[args.reference]))
        a = np.array([f1[args.reference][k] for k in common]); b = np.array([f1[r][k] for k in common])
        t, p = stats.ttest_rel(a, b)
        report["ttests"][r] = {"mean_diff_ref_minus_run": float((a - b).mean()), "ref_wins": int((a > b).sum()),
                               "folds": len(common), "t": float(t), "p": float(p)}
        print(f"  ref − {r:32s} Δ {(a - b).mean():+.4f} · ref wins {(a > b).sum()}/{len(common)} · t={t:6.2f} · p={p:.3g}")

    print(f"\n=== 4. Ensemble of {len(args.ensemble)} runs (mean probability, thresholds re-tuned per fold) ===")
    vals = []
    for k, fold in enumerate(oof[args.ensemble[0]]):
        for r in args.ensemble:
            assert (oof[r][k]["ids"] == fold["ids"]).all(), f"{r}: fold {k} has different sentences"
        p = np.mean([oof[r][k]["probs"] for r in args.ensemble], axis=0)
        vals.append(T.compute_metrics(fold["labels"], decide(p, tune(fold["labels"], p)), p))
    ens = {key: {"mean": float(np.mean([v[key] for v in vals])), "std": float(np.std([v[key] for v in vals]))}
           for key in vals[0]}
    ref_folds = [f1[args.reference][k] for k in sorted(f1[args.reference])]
    diffs = np.array([v["f1_samples"] for v in vals]) - np.array(ref_folds)
    t, p = stats.ttest_1samp(diffs, 0)
    report["ensemble"] = {"runs": args.ensemble, "metrics": ens, "diff_vs_reference": diffs.tolist(), "t": float(t), "p": float(p)}
    for key in ["f1_samples", "f1_micro", "f1_macro", "pr_auc_macro", "subset_accuracy", "hamming_loss"]:
        print(f"  {key:16s} {ens[key]['mean']:.4f} ± {ens[key]['std']:.4f}")
    print(f"  vs reference per fold: {' '.join(f'{d:+.4f}' for d in diffs)} · t={t:.2f} · p={p:.3g}")

    print(f"\n=== 5. Error analysis of {args.reference} ===")
    o = oof[args.reference]
    ids = np.concatenate([f["ids"] for f in o]); y = np.vstack([f["labels"] for f in o])
    pred = np.vstack([decide(f["probs"], f["thr"]) for f in o]); lab = np.array(labels)
    pairs = {}
    for i in range(len(y)):
        for a in lab[(y[i] == 1) & (pred[i] == 0)]:
            for c in lab[(y[i] == 0) & (pred[i] == 1)]:
                pairs[(a, c)] = pairs.get((a, c), 0) + 1
    top = sorted(pairs.items(), key=lambda x: -x[1])[:10]
    exact = (pred == y).all(1)
    lang = data.loc[ids, "lang"].values
    groups = {f"{k} true label(s)": exact[y.sum(1) == k] for k in (1, 2)}
    groups["3+ true labels"] = exact[y.sum(1) >= 3]
    groups.update({f"lang {l}": exact[lang == l] for l in ["JAVA", "PY", "PHARO"]})
    report["errors"] = {"top_confusions": [{"missed": a, "predicted": c, "count": n} for (a, c), n in top],
                        "exact_match": {g: {"pct": 100 * float(v.mean()), "n": int(len(v))} for g, v in groups.items()}}
    print("  Most frequent confusions (true label missed → wrong label predicted):")
    for (a, c), n in top:
        print(f"    {a:24s} → {c:24s} {n}")
    print("  Exact match: " + " · ".join(f"{g} {100 * v.mean():.1f}% (n={len(v)})" for g, v in groups.items()))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out / 'analysis.json'}")


if __name__ == "__main__":
    main()
