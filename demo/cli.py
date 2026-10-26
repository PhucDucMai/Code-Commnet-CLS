"""
Command-line demo (backup for the Streamlit app): classify the class comments of code files.

    python demo/cli.py demo/examples/DownloadManager.java          # one file
    python demo/cli.py --all                                         # every sample + test file, accuracy vs expected labels
    python demo/cli.py --tests                                       # only demo/test_files/
    python demo/cli.py MyFile.java --run runs/unixcoder-base        # another trained model
"""
import argparse
import sys
from pathlib import Path

from inference import (DEFAULT_RUN, EXAMPLES_DIR, LABEL_INFO, ROOT, TEST_DIR, classify, load_bundle, match_status,
                       sample_files)

sys.path.insert(0, str(ROOT / "src"))
from train import style  # noqa: E402

MARK = {"exact": style("✔ correct", "green"), "partial": style("~ partial", "yellow"), "wrong": style("✘ wrong", "red")}


def show_file(bundle, path, args):
    print("\n" + style(f"■ {path.name}", "bold", "cyan"))
    try:
        rows, probs, _ = classify(bundle, path.read_text(encoding="utf-8", errors="replace"), path.name,
                                  lang_mask=not args.no_lang_mask, at_least_one=not args.no_at_least_one)
    except SyntaxError as e:
        print(style(f"  Python syntax error, line {e.lineno}: {e.msg} -> file skipped", "red"))
        return []
    if not rows:
        print("  No class comment found (Javadoc before a class, Python class docstring, Pharo class comment).")
        return []
    label_list = bundle["label_list"]
    current = None
    for i, r in enumerate(rows):
        if r["class"] != current:
            current = r["class"]
            print(style(f"  class {current} (line {r['line']})", "bold"))
        labels = ", ".join(f"{lab} {probs[i, label_list.index(lab)]:.2f}" for lab in r["pred"])
        status = match_status(r["pred"], r["expected"])
        tail = f"   {MARK[status]}" + (f" (expected: {', '.join(r['expected'])})" if status != "exact" else "") if status else ""
        print(f"    {r['raw'][:70]:<72} → {style(labels, 'cyan')}{tail}")
    return [match_status(r["pred"], r["expected"]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=".java / .py / .st files")
    parser.add_argument("--all", action="store_true", help="Run every file of demo/examples/ and demo/test_files/")
    parser.add_argument("--examples", action="store_true", help=f"Run every file of {EXAMPLES_DIR.relative_to(ROOT)}/")
    parser.add_argument("--tests", action="store_true", help=f"Run every file of {TEST_DIR.relative_to(ROOT)}/")
    parser.add_argument("--run", default=DEFAULT_RUN, help="Trained run folder (with config_used.yaml + final_model.pt)")
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--no-lang-mask", action="store_true", help="Also allow labels of other languages")
    parser.add_argument("--no-at-least-one", action="store_true", help="Allow sentences without any label")
    parser.add_argument("--labels", action="store_true", help="Print the meaning of every label and exit")
    args = parser.parse_args()

    if args.labels:
        for lab, desc in LABEL_INFO.items():
            print(f"  {lab:<25} {desc}")
        return
    files = [Path(f) for f in args.files]
    for flag, folder in [(args.examples or args.all, EXAMPLES_DIR), (args.tests or args.all, TEST_DIR)]:
        if flag:
            files += [p for p in sample_files() if p.parent == folder]
    if not files:
        parser.error("give one or more files, or --all / --examples / --tests")

    print(f"Loading {args.run} …")
    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    bundle = load_bundle(run_dir / "config_used.yaml", device=args.device)
    print(f"Model {bundle['config']['model_name']} · device {bundle['device']} · thresholds: {bundle['thr_source']}")

    statuses = []
    for path in files:
        statuses += show_file(bundle, path, args)
    checked = [s for s in statuses if s]
    if checked:
        n = len(checked)
        print("\n" + style(f"Against the hand-written expected labels: {checked.count('exact')}/{n} correct "
                           f"({100 * checked.count('exact') / n:.0f}%), {checked.count('partial')} partial, "
                           f"{checked.count('wrong')} wrong", "bold"))


if __name__ == "__main__":
    main()
