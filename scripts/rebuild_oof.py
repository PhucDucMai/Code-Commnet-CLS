"""
Rebuild out-of-fold predictions + the new multi-label metrics (Hamming, exact-match, Jaccard) for an EXISTING
training run, from its best_model_fold*.pt checkpoints and best_thresholds_fold*.json — no retraining needed.

Writes oof_fold*.npz and updates results.json (the old version is kept as results.before_rebuild.json).
Meant for runs trained before train.py saved out-of-fold predictions itself.

    python scripts/rebuild_oof.py runs/roberta_large_rtx5060
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # train.py lives in src/
import train as T  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = T.read_yaml(run_dir / "config_used.yaml")
    res_path = run_dir / "results.json"
    with open(res_path) as f:
        res = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data + fold split: identical to main() in train.py
    df = pd.read_csv(config["data"]["raw_path"])
    df["__lang_norm__"] = df["lang"].apply(T.norm_lang)
    label_cols = config["data"]["label_list"]
    to_19 = bool(config["data"].get("expand_to_19_classes", False))
    labels_mat = T.build_label_matrix(df, label_cols, expand_to_19=to_19)
    label_langs = T.infer_label_langs(df["__lang_norm__"].values, labels_mat, label_cols)
    n_splits = int(config.get("cv", {}).get("n_splits", 5))
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config["train_params"]["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    old = dict(zip(res.get("folds", range(n_splits)), res["fold_metrics"]))
    precision = config.get("precision", "fp32")

    print(f"Rebuild OOF · {run_dir} · {config['model_name']} · {device}")
    fold_states = []
    for fold, (_, va_idx) in enumerate(mskf.split(df["sentence"].values, labels_mat)):
        ckpt, thr_path = run_dir / f"best_model_fold{fold}.pt", run_dir / f"best_thresholds_fold{fold}.json"
        if not (ckpt.exists() and thr_path.exists()):
            print(f"  Fold {fold}: missing checkpoint/thresholds → skipped")
            continue
        val_df = df.iloc[va_idx].reset_index(drop=True)
        ds = T.CodeCommentDataset(val_df, label_cols, tokenizer, max_len=config["max_len"],
                                  text_cfg=config.get("text_features", {}), to_19=to_19)
        loader = DataLoader(ds, batch_size=int(config["train_params"].get("eval_batch_size", 32)), shuffle=False,
                            num_workers=config["train_params"].get("num_workers", 4), pin_memory=True)

        model = T.build_model(config, len(label_cols))
        state = torch.load(ckpt, map_location="cpu")
        T.load_classifier_state(model, state["model"] if isinstance(state, dict) and "model" in state else state)
        model.to(device)
        probs, labels, _ = T.evaluate(model, loader, device, precision=precision, desc=f"fold {fold}")
        thr = np.array(json.load(open(thr_path)), dtype=np.float32)
        preds = (probs >= thr).astype(int)

        metrics = T.compute_metrics(labels, preds, probs)
        T.save_oof(run_dir / f"oof_fold{fold}.npz", val_df, probs, labels, thr)
        prev = old.get(fold, {}).get("f1_samples")
        print(f"  Fold {fold}: F1 samples {metrics['f1_samples']:.4f}"
              + (f" (at training time: {prev:.4f}, diff {metrics['f1_samples'] - prev:+.4f})" if prev is not None else "")
              + f" · Hamming {metrics['hamming_loss']:.4f} · exact {metrics['subset_accuracy']:.4f}"
              + f" · Jaccard {metrics['jaccard_samples']:.4f}")
        fold_states.append({"fold": fold, "metrics": metrics, "thresholds": thr.tolist(),
                            "per_label": T.per_label_metrics(labels, preds, probs, label_cols)})
        del model, state
        T.free_cuda()

    if len(fold_states) != len(res["fold_metrics"]):
        raise SystemExit("✘ Number of rebuilt folds differs from results.json → not updating")

    backup = run_dir / "results.before_rebuild.json"
    if not backup.exists():
        shutil.copy(res_path, backup)
    avg, std = T.aggregate_fold_metrics(fold_states)
    f1s = np.array([[r["f1"] for r in st["per_label"]] for st in fold_states])
    res.update({
        "folds": [st["fold"] for st in fold_states],
        "fold_metrics": [st["metrics"] for st in fold_states],
        "avg_metrics": avg,
        "std_metrics": std,
        "per_label_f1": {lab: {"mean": float(f1s[:, j].mean()), "std": float(f1s[:, j].std()),
                               "lang": label_langs.get(lab)} for j, lab in enumerate(label_cols)},
        "oof_rebuilt": True,
    })
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n✅ F1 samples {avg['f1_samples']:.4f} ± {std['f1_samples']:.4f} · Hamming {avg['hamming_loss']:.4f}"
          f" · exact {avg['subset_accuracy']:.4f} · Jaccard {avg['jaccard_samples']:.4f}")
    print(f"   Updated {res_path} (old version: {backup}) and wrote oof_fold*.npz")


if __name__ == "__main__":
    main()
