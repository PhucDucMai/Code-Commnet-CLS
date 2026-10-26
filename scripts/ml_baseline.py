"""
Machine Learning baseline: TF-IDF + Logistic Regression One-vs-Rest, multi-label with 19 classes.

Uses EXACTLY the 5 folds of train.py (same MultilabelStratifiedKFold, same seed from the config), the same
threshold selection and the same metrics, and writes results.json in the same format -> sits in the same
table as the transformers via compare_runs.py. Runs on CPU in a few minutes.

    python scripts/ml_baseline.py                          # LogReg -> runs/ml_tfidf_logreg
    python scripts/ml_baseline.py --model rf               # Random Forest (the kind of baseline NLBSE'23 used)
    python scripts/ml_baseline.py --no-threshold-search    # threshold 0.5 for every label
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # train.py lives in src/
import train as T  # noqa: E402

MODELS = {
    "logreg": ("TF-IDF + LogisticRegression (One-vs-Rest)",
               # n_jobs=1: liblinear fails with "WRITEBACKIFCOPY base is read-only" on read-only data shared by joblib
               lambda seed: OneVsRestClassifier(
                   LogisticRegression(C=4.0, class_weight="balanced", solver="liblinear", max_iter=2000),
                   n_jobs=1)),
    "rf": ("TF-IDF + RandomForest",
           lambda seed: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=seed)),
}


def make_features(train_text, val_text):
    """Word TF-IDF (1-2 grams) + character TF-IDF (2-5 grams); fitted on the training fold only to avoid leakage."""
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    x_val = hstack([word.transform(val_text), char.transform(val_text)]).tocsr()
    return x_train, x_val


def predict_probs(clf, x):
    probs = clf.predict_proba(x)
    if isinstance(probs, list):  # multi-output RandomForest: a list of (n, 2) arrays, one per label
        probs = np.stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(p)) for p in probs], axis=1)
    return probs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="configs/train_rtx5060.yaml",
                        help="Take the data, label list, number of folds and seed from this config")
    parser.add_argument("--model", choices=list(MODELS), default="logreg")
    parser.add_argument("--no-threshold-search", action="store_true", help="Threshold 0.5 for every label")
    parser.add_argument("--output-dir", help="Default: runs/ml_tfidf_<model>")
    args = parser.parse_args()

    config = T.read_yaml(Path(args.config))
    model_desc, factory = MODELS[args.model]
    results_dir = Path(args.output_dir or f"runs/ml_tfidf_{args.model}")
    results_dir.mkdir(parents=True, exist_ok=True)
    T.set_log_file(results_dir / "train.log")
    t_run = time.time()

    # Data + label matrix + fold split: identical to main() in train.py
    df = pd.read_csv(config["data"]["raw_path"])
    df["__lang_norm__"] = df["lang"].apply(T.norm_lang)
    label_cols = config["data"]["label_list"]
    labels_mat = T.build_label_matrix(df, label_cols, expand_to_19=bool(config["data"].get("expand_to_19_classes", False)))
    label_langs = T.infer_label_langs(df["__lang_norm__"].values, labels_mat, label_cols)
    n_splits = int(config.get("cv", {}).get("n_splits", 5))
    seed = int(config["train_params"]["seed"])
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    texts = df["sentence"].astype(str).values
    do_thr = not args.no_threshold_search

    T.log("\n" + T.DOUBLE_RULE)
    T.log(f"ML BASELINE · {model_desc}")
    T.log(T.DOUBLE_RULE)
    T.log(f"  Data      {len(df):,} sentences · {len(label_cols)} labels · input: comment sentence only")
    T.log(f"  Split     {n_splits}-fold multilabel stratified · seed {seed} (same as train.py)")
    T.log(f"  Threshold {'searched per label as in train.py' if do_thr else '0.5 for every label'}")
    T.log(f"  Output    {results_dir}/")

    fold_states = []
    for fold, (tr_idx, va_idx) in enumerate(mskf.split(df["sentence"].values, labels_mat)):
        t0 = time.time()
        x_train, x_val = make_features(texts[tr_idx], texts[va_idx])
        clf = factory(seed)
        clf.fit(x_train, labels_mat[tr_idx])
        probs = predict_probs(clf, x_val)
        y_val = labels_mat[va_idx]

        if do_thr:
            base_thr = T.optimize_thresholds_per_label(y_val, probs)
            thr = T.optimize_thresholds_samples(y_val, probs, base=base_thr, iters=2)
        else:
            thr = np.full(len(label_cols), 0.5)
        preds = (probs >= thr).astype(int)

        state = {
            "fold": fold,
            "thresholds": [float(t) for t in thr],
            "metrics": T.compute_metrics(y_val, preds, probs),
            "per_label": T.per_label_metrics(y_val, preds, probs, label_cols),
            "epoch": "-",
            "epochs_run": "-",
            "time": time.time() - t0,
        }
        fold_states.append(state)
        with open(results_dir / f"fold{fold}_state.json", "w") as f:
            json.dump(state, f, indent=2)
        T.save_oof(results_dir / f"oof_fold{fold}.npz", df.iloc[va_idx], probs, y_val, thr)
        m = state["metrics"]
        T.log(f"  Fold {fold+1}/{n_splits} · train {len(tr_idx):,} · val {len(va_idx):,} · {x_train.shape[1]:,} features"
              f" · F1 samples {m['f1_samples']:.4f} · micro {m['f1_micro']:.4f} · macro {m['f1_macro']:.4f}"
              f" · {T.fmt_time(state['time'])}")

    avg, std = T.aggregate_fold_metrics(fold_states)
    thr_median = np.median(np.vstack([st["thresholds"] for st in fold_states]), axis=0)
    f1s = np.array([[r["f1"] for r in st["per_label"]] for st in fold_states])
    summary = {
        "split": "5fold",
        "folds": [st["fold"] for st in fold_states],
        "fold_metrics": [st["metrics"] for st in fold_states],
        "fold_times_s": [st["time"] for st in fold_states],
        "avg_metrics": avg,
        "std_metrics": std,
        "per_label_f1": {lab: {"mean": float(f1s[:, j].mean()), "std": float(f1s[:, j].std()),
                               "lang": label_langs.get(lab)} for j, lab in enumerate(label_cols)},
        "median_thresholds": thr_median.tolist(),
        "label_list": label_cols,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    # so that compare_runs.py shows the right method name
    used = {"model_name": model_desc, "ml_baseline": {"model": args.model, "threshold_search": do_thr},
            "train_params": {"seed": seed}, "cv": {"n_splits": n_splits}, "data": config["data"],
            "logging": {"output_dir": str(results_dir)}}
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(used, f, allow_unicode=True, sort_keys=False)

    T.log_cv_summary(fold_states, label_cols, label_langs, thr_median, time.time() - t_run)
    T.log(f"\n✅ Done · results in {results_dir}/")


if __name__ == "__main__":
    main()
