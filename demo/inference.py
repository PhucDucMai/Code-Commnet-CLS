"""
Shared inference code for the demo (used by demo/app.py and demo/cli.py, no Streamlit dependency):
load a trained run, classify the class comments of a code file, compare with expected labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_comments import DEFAULT_TEMPLATE, detect_lang, extract_comments, to_model_input  # noqa: E402
from train import build_model, load_classifier_state, read_yaml  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"      # showcase files for the presentation
TEST_DIR = Path(__file__).resolve().parent / "test_files"        # test files: varied styles + edge cases
SAMPLE_DIRS = [EXAMPLES_DIR, TEST_DIR]
EXPECTED_PATH = EXAMPLES_DIR / "expected_labels.json"            # reference labels for both folders
DEFAULT_RUN = "runs/roberta_large_rtx5060"

LANG_NAMES = {"java": "Java", "python": "Python", "pharo": "Pharo"}
LANG_LABELS = {
    "java": ["summary_java", "usage_java", "expand_java", "pointer", "deprecation", "rational", "ownership"],
    "python": ["summary_python", "usage_python", "expand_python", "parameters", "developmentnotes"],
    "pharo": ["intent", "responsibilities", "collaborators", "keymessages", "example",
              "keyimplementationpoints", "classreferences"],
}
# Category definitions of the NLBSE'23 code comment taxonomy
LABEL_INFO = {
    "summary_java": "What the class is / does, in one or two sentences",
    "usage_java": "How to use the class: calls, expected inputs, usage examples",
    "expand_java": "Further details on behaviour or implementation beyond the summary",
    "pointer": "References to other classes, methods or resources (@see, @link, URLs)",
    "deprecation": "The class or API is deprecated, what replaces it, when it will be removed",
    "rational": "Why a design or implementation choice was made",
    "ownership": "Authors, owners, versions (@author, @since, @version)",
    "summary_python": "What the class is / does, in one or two sentences",
    "usage_python": "How to use the class: calls, expected inputs, usage examples",
    "expand_python": "Further details on behaviour or implementation beyond the summary",
    "parameters": "Arguments, attributes and their meaning (Args:, Attributes:)",
    "developmentnotes": "Notes for developers: TODOs, known limitations, future work",
    "intent": "The purpose of the class, usually \"I am / I represent ...\"",
    "responsibilities": "What the class knows and does",
    "collaborators": "Other classes it works with and how",
    "keymessages": "The main messages (methods) of its public API",
    "example": "Example code showing how to use the class",
    "keyimplementationpoints": "Internal representation, instance variables, implementation details",
    "classreferences": "Mentions of other related classes",
}


def sample_files():
    """Code files of every sample folder, as paths."""
    return [p for d in SAMPLE_DIRS if d.exists() for p in sorted(d.iterdir()) if detect_lang(p.name)]


def trained_runs():
    """Run folders that have both config_used.yaml and final_model.pt, the default run first."""
    runs = [p for p in sorted((ROOT / "runs").glob("*")) if (p / "config_used.yaml").exists() and (p / "final_model.pt").exists()]
    return sorted(runs, key=lambda p: p.relative_to(ROOT).as_posix() != DEFAULT_RUN)


def resolve_thresholds(ckpt_obj, thr_path, results_dir: Path, num_labels: int):
    """Priority: thresholds stored in the checkpoint -> json file -> results.json -> 0.5."""
    if isinstance(ckpt_obj, dict) and "thresholds" in ckpt_obj:
        return np.array(ckpt_obj["thresholds"], dtype=np.float32), "checkpoint (median over CV folds)"
    if thr_path and Path(thr_path).exists():
        with open(thr_path) as f:
            return np.array(json.load(f), dtype=np.float32), str(thr_path)
    results = results_dir / "results.json"
    if results.exists():
        with open(results) as f:
            r = json.load(f)
        thr = r.get("median_thresholds") or r.get("thresholds")
        if thr:
            return np.array(thr, dtype=np.float32), str(results)
    return np.full(num_labels, 0.5, dtype=np.float32), "default 0.5 (no threshold file found)"


def load_bundle(config_path, ckpt_path=None, thr_path=None, device=None):
    """Load a trained model. ckpt_path defaults to final_training.save_path of the config."""
    config = read_yaml(Path(config_path))
    label_list = config["data"]["label_list"]
    ckpt_path = ckpt_path or config.get("final_training", {}).get(
        "save_path", str(Path(config["logging"]["output_dir"]) / "final_model.pt"))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model = build_model(config, len(label_list))  # built exactly as in training so the state_dict keys match
    obj = torch.load(ROOT / ckpt_path, map_location="cpu")
    # final_model.pt = {"model", "thresholds"}; best_model_fold*.pt = bare state_dict
    load_classifier_state(model, obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
    model.to(device).eval()

    thresholds, thr_source = resolve_thresholds(obj, thr_path, ROOT / config["logging"]["output_dir"], len(label_list))
    return {"model": model, "tokenizer": tokenizer, "device": device, "config": config,
            "max_len": int(config["max_len"]), "label_list": label_list,
            "thresholds": thresholds, "thr_source": thr_source, "checkpoint": str(ckpt_path)}


@torch.no_grad()
def predict_probs(bundle, texts, batch_size=16):
    tok, model, device = bundle["tokenizer"], bundle["model"], bundle["device"]
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], max_length=bundle["max_len"], padding="max_length",
                  truncation=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.vstack(probs)


def decide_labels(probs, thresholds, label_list, allowed, at_least_one):
    """probs: (N, L). Only labels in `allowed` are considered; returns the list of labels per sentence."""
    idx = [j for j, lab in enumerate(label_list) if lab in allowed]
    preds = []
    for row in probs:
        chosen = [label_list[j] for j in idx if row[j] >= thresholds[j]]
        if not chosen and at_least_one:  # every sentence in the training data has at least one label
            chosen = [label_list[max(idx, key=lambda j: row[j])]]
        preds.append(chosen)
    return preds


def model_inputs(items, config):
    text_cfg = config.get("text_features", {})
    return [to_model_input(it, template=text_cfg.get("template", DEFAULT_TEMPLATE),
                           include_lang=text_cfg.get("include_lang", False),
                           include_class_id=text_cfg.get("include_class_id", False)) for it in items]


def load_expected(filename):
    """Hand-written reference labels of a sample file: {normalized sentence: [labels]}, or {}."""
    if not EXPECTED_PATH.exists():
        return {}
    with open(EXPECTED_PATH) as f:
        return json.load(f).get("files", {}).get(Path(filename).name, {})


def classify(bundle, source, filename, lang_mask=True, at_least_one=True):
    """Extract the class comments of a code file and classify every sentence.

    Returns (rows, probs, allowed_labels); each row has class/line/raw/sentence/input/pred/expected."""
    lang = detect_lang(filename)
    if lang is None:
        raise ValueError("The file name must end with .java, .py or .st")
    items = extract_comments(source, filename, lang)
    if not items:
        return [], None, []
    label_list = bundle["label_list"]
    allowed = [lab for lab in label_list if lab in LANG_LABELS[lang]] if lang_mask else list(label_list)
    allowed = allowed or list(label_list)
    inputs = model_inputs(items, bundle["config"])
    probs = predict_probs(bundle, inputs)
    preds = decide_labels(probs, bundle["thresholds"], label_list, set(allowed), at_least_one)
    expected = load_expected(filename)
    rows = [{"class": it.class_name, "line": it.line, "raw": it.raw, "sentence": it.sentence, "input": x,
             "pred": p, "expected": expected.get(it.sentence)} for it, x, p in zip(items, inputs, preds)]
    return rows, probs, allowed


def match_status(pred, expected):
    """'exact' | 'partial' | 'wrong' | None (no reference)."""
    if expected is None:
        return None
    p, e = set(pred), set(expected)
    if p == e:
        return "exact"
    return "partial" if p & e else "wrong"
