"""
Train every model in configs/experiments.yaml one after another with ONE command, then compare the results.

    python scripts/run_experiments.py                         # uses configs/experiments.yaml
    python scripts/run_experiments.py --dry-run               # only print the plan, no training
    python scripts/run_experiments.py --only deberta-v3-large # run only one (or a few) models from the list

For each model (each one runs train.py in its own process so that all VRAM is released):
  - results.json with all folds + final_model.pt matching the config -> skipped
  - CV done but the final model is missing/outdated                  -> train.py --final-only
  - not finished                                                    -> train.py (CV + final model), resuming from finished folds
  - CUDA out of memory -> halve batch_size, double grad_accum, retry
  - any other error -> recorded, then move on to the next model
Finally runs compare_runs.py on the models that have results. Log: runs/experiments.log
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # train.py lives in src/
from train import EXIT_OOM, apply_cli_overrides, fmt_time, read_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # repo root: every path in the configs is relative to it
LOG_PATH = ROOT / "runs" / "experiments.log"
MAX_ATTEMPTS = 8


def log(msg=""):
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def effective_config(base_cfg, exp, batch_size, grad_accum):
    """Config train.py will use: the experiment's own config (if any) + the matching CLI arguments."""
    args = Namespace(model=exp.get("model"), output_dir=exp.get("output_dir"),
                     batch_size=batch_size, grad_accum=grad_accum)
    return apply_cli_overrides(read_yaml(exp_config_path(base_cfg, exp)), args)


def exp_config_path(base_cfg, exp):
    return Path(exp.get("config", base_cfg))


def final_paths(cfg):
    out = Path(cfg["logging"]["output_dir"])
    save_path = Path(cfg.get("final_training", {}).get("save_path", out / "final_model.pt"))
    return save_path, save_path.parent / "final_meta.json"


def run_state(cfg):
    """('done' | 'final' | 'train', number of finished folds)."""
    out = ROOT / cfg["logging"]["output_dir"]
    n_splits = int(cfg.get("cv", {}).get("n_splits", 5))
    cv_done = False
    res_path = out / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
        cv_done = res.get("split") == "5fold" and len(res.get("fold_metrics", [])) >= n_splits
    folds_done = n_splits if cv_done else len(list(out.glob("fold*_state.json")))

    ft = cfg.get("final_training", {})
    if not ft.get("enabled"):
        return ("done" if cv_done else "train"), folds_done
    save_path, meta_path = final_paths(cfg)
    final_done = False
    if (ROOT / save_path).exists() and (ROOT / meta_path).exists():
        with open(ROOT / meta_path) as f:
            meta = json.load(f)
        epochs = int(ft.get("epochs", cfg["train_params"]["epochs"]))
        sched = max(epochs, int(ft.get("schedule_epochs", epochs)))
        final_done = (meta.get("model_name") == cfg["model_name"]
                      and meta.get("epochs") == epochs and meta.get("schedule_epochs") == sched)
    if cv_done:
        return ("done" if final_done else "final"), folds_done
    return "train", folds_done


def describe(state, cfg):
    if state == "done":
        return "done → skipped"
    if state == "final":
        return "CV done → retrain the final model only"
    return "train CV + final model" if cfg.get("final_training", {}).get("enabled") else "train CV (no final model needed)"


def preflight(cfg):
    """Load the config + tokenizer (cheap) to catch a wrong model name / missing library early, before hours of training."""
    from transformers import AutoConfig, AutoTokenizer
    try:
        AutoConfig.from_pretrained(cfg["model_name"])
        AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}"


def run_train(base_cfg, cfg, exp, batch, accum, final_only):
    cmd = [sys.executable, "src/train.py", str(exp_config_path(base_cfg, exp)),
           "--output-dir", cfg["logging"]["output_dir"],
           "--batch-size", str(batch), "--grad-accum", str(accum), "--resume"]
    if exp.get("model"):
        cmd += ["--model", exp["model"]]
    if final_only:
        cmd.append("--final-only")
    # reduce GPU memory fragmentation, helpful with little VRAM
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def f1_samples_of(cfg):
    res_path = ROOT / cfg["logging"]["output_dir"] / "results.json"
    if not res_path.exists():
        return "n/a"
    with open(res_path) as f:
        res = json.load(f)
    m = res.get("avg_metrics", {})
    s = res.get("std_metrics", {})
    return f"{m['f1_samples']:.4f} ± {s['f1_samples']:.4f}" if "f1_samples" in m else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="?", default="configs/experiments.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Only run the experiments with these names")
    args = parser.parse_args()

    with open(args.experiments) as f:
        spec = yaml.safe_load(f)
    base_cfg = Path(spec["base_config"])
    exps = [e for e in spec["experiments"] if e.get("enabled", True)]
    if args.only:
        unknown = set(args.only) - {e["name"] for e in exps}
        if unknown:
            sys.exit(f"✘ No (enabled) experiment named: {', '.join(sorted(unknown))}")
        exps = [e for e in exps if e["name"] in args.only]
    for e in exps:
        tp = read_yaml(exp_config_path(base_cfg, e))["train_params"]
        e.setdefault("batch_size", int(tp["batch_size"]))
        e.setdefault("grad_accum", int(tp["grad_accum"]))
    outs = [effective_config(base_cfg, e, e["batch_size"], e["grad_accum"])["logging"]["output_dir"] for e in exps]
    if len(set(outs)) != len(outs):
        sys.exit("✘ Two experiments share the same output_dir")

    # ---- Plan ----
    log("\n" + "═" * 96)
    log(f"RUN EXPERIMENTS · {datetime.now():%Y-%m-%d %H:%M:%S} · {args.experiments} · base {base_cfg}")
    log("═" * 96)
    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    log(f"  Free disk space: {free_gb:.1f} GB" + ("  ⚠ low" if free_gb < 10 else ""))
    plans = []
    for i, exp in enumerate(exps, 1):
        cfg = effective_config(base_cfg, exp, exp["batch_size"], exp["grad_accum"])
        state, folds = run_state(cfg)
        error = preflight(cfg) if state != "done" else None
        plans.append((exp, state, error))
        extra = f" ({folds} fold(s) done, resuming)" if state == "train" and folds else ""
        log(f"  {i}. {exp['name']:<24} {cfg['model_name']:<30} batch {exp['batch_size']}×{exp['grad_accum']:<3} "
            f"→ {describe(state, cfg)}{extra}" + (f"\n     ✘ CANNOT RUN: {error}" if error else ""))
    if args.dry_run:
        log("\n(--dry-run: no training)")
        return

    # ---- Run ----
    summary = []
    t_all = time.time()
    try:
        for i, (exp, state, error) in enumerate(plans, 1):
            if error:
                summary.append((exp, "skipped (failed to load model/tokenizer)", 0.0))
                continue
            batch, accum = exp["batch_size"], exp["grad_accum"]
            t0, status, prev_state, last_rc = time.time(), None, None, None
            for _ in range(MAX_ATTEMPTS):
                cfg = effective_config(base_cfg, exp, batch, accum)
                state, folds = run_state(cfg)
                if state == "done":
                    status = "already done" if prev_state is None else "done"
                    break
                if state == prev_state and last_rc == 0:  # succeeded but the state did not change -> do not loop
                    status = f"train.py finished but results are incomplete, see {cfg['logging']['output_dir']}/train.log"
                    break
                log(f"\n▶ [{i}/{len(plans)}] {exp['name']} · {describe(state, cfg)}"
                    f"{f' (from fold {folds})' if state == 'train' and folds else ''} · batch {batch}×{accum}"
                    f" · {datetime.now():%H:%M:%S}")
                rc = run_train(base_cfg, cfg, exp, batch, accum, final_only=(state == "final"))
                prev_state, last_rc = state, rc
                if rc == 0:
                    continue  # the next iteration re-checks the state
                if rc == EXIT_OOM and spec.get("oom_retry", True) and batch // 2 >= int(spec.get("min_batch_size", 2)):
                    batch, accum = batch // 2, accum * 2
                    log(f"  ⚠ {exp['name']}: out of GPU memory → retrying with batch {batch}×{accum} (same effective batch)")
                    continue
                status = "out of GPU memory" if rc == EXIT_OOM else f"error (exit code {rc}), see {cfg['logging']['output_dir']}/train.log"
                break
            else:
                status = "stopped after too many attempts"
            elapsed = time.time() - t0
            summary.append((exp, status, elapsed))
            log(f"■ {exp['name']}: {status} · {fmt_time(elapsed)}")
    except KeyboardInterrupt:
        log("\n⏸ Stopped (Ctrl+C). Run the same command again to resume where it left off.")
        sys.exit(130)

    # ---- Summary + comparison ----
    log("\n" + "═" * 96)
    log(f"SUMMARY · {fmt_time(time.time() - t_all)}")
    log("═" * 96)
    done_dirs = []
    for exp, status, elapsed in summary:
        cfg = effective_config(base_cfg, exp, exp["batch_size"], exp["grad_accum"])
        f1 = f1_samples_of(cfg)
        log(f"  {exp['name']:<20} {status:<45} {fmt_time(elapsed):>8}   F1-samples {f1}")
        if f1 != "n/a":
            done_dirs.append(cfg["logging"]["output_dir"])

    # compare_runs in the yaml: column order + reference (the first folder), may include runs/ml_tfidf_*;
    # only folders that already have results.json are used
    if spec.get("compare_runs"):
        done_dirs = [d for d in spec["compare_runs"] if (ROOT / d / "results.json").exists()]
    if len(done_dirs) >= 1:
        out = spec.get("compare_out", "runs/comparison")
        log(f"\n▶ Comparing {len(done_dirs)} results → {out}/  (reference: {done_dirs[0]})")
        subprocess.run([sys.executable, "scripts/compare_runs.py", *done_dirs, "--out", out], cwd=ROOT)


if __name__ == "__main__":
    main()
