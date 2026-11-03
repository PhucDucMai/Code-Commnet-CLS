import os
import re
import gc
import sys
import json
import math
import time
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    accuracy_score,
    jaccard_score,
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers import logging as hf_logging

# PEFT / LoRA (optional)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()  # hide the "Some weights ... not initialized" warning repeated every fold


# ===================== Logging =====================
# log() prints to the terminal (colored on a TTY) and appends an uncolored copy to <output_dir>/train.log

_USE_COLOR = sys.stdout.isatty()
_SHOW_BARS = sys.stderr.isatty()  # disable tqdm bars when piped (e.g. | tee) to keep logs clean
_ANSI = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOG_FILE = None

LANG_ORDER = ["java", "python", "pharo", "other"]
RULE = "─" * 96
DOUBLE_RULE = "═" * 96


def style(text, *styles):
    if not _USE_COLOR or not styles:
        return str(text)
    return f"\x1b[{';'.join(_ANSI[s] for s in styles)}m{text}\x1b[0m"


def set_log_file(path: Path):
    global _LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(path, "a", encoding="utf-8")


def log(msg=""):
    print(msg, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(_ANSI_RE.sub("", str(msg)) + "\n")
        _LOG_FILE.flush()


def fmt_time(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_num(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_delta(cur, prev, higher_is_better=True, digits=4):
    """'▲ +0.0123' (green when better) / '▼ -0.0040' (red when worse)."""
    if cur is None or prev is None or np.isnan(cur) or np.isnan(prev):
        return ""
    d = cur - prev
    if abs(d) < 0.5 * 10 ** -digits:
        return style(f"= {0:+.{digits}f}", "dim")
    arrow = "▲" if d > 0 else "▼"
    good = (d > 0) == higher_is_better
    return style(f"{arrow} {d:+.{digits}f}", "green" if good else "red")


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def gpu_peak_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 ** 3


# ===================== Utils =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_yaml(path: Path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def norm_lang(v):
    s = str(v).strip().lower()
    if "java" in s:
        return "java"
    if ("python" in s) or (s in ("py", "py3", "python3")):
        return "python"
    if ("pharo" in s) or ("smalltalk" in s):
        return "pharo"
    return "other"


def map_labels_for_row(labels_str, lang_norm, expand_to_19):
    labs = str(labels_str or "").split(";")
    labs = [x.strip().lower() for x in labs if x.strip()]
    if not expand_to_19:
        return labs
    mapped = []
    for lab in labs:
        if lab in ("summary", "usage", "expand"):
            if lang_norm in ("java", "python"):
                mapped.append(f"{lab}_{lang_norm}")
            else:
                # rare for pharo; keep as is if present
                mapped.append(lab)
        else:
            mapped.append(lab)
    return mapped


def build_label_matrix(df, label_list, expand_to_19=True):
    name_to_idx = {name: i for i, name in enumerate(label_list)}
    L = len(label_list)
    mat = np.zeros((len(df), L), dtype=np.float32)
    for i, row in df.iterrows():
        lang = norm_lang(row.get("lang", ""))
        labs = map_labels_for_row(row.get("labels", ""), lang, expand_to_19)
        for lab in labs:
            if lab in name_to_idx:
                mat[i, name_to_idx[lab]] = 1.0
    return mat


def infer_label_langs(langs, label_matrix, label_cols):
    """Language of a label = the most frequent language among the sentences carrying that label."""
    langs = np.asarray(langs)
    out = {}
    for j, lab in enumerate(label_cols):
        mask = label_matrix[:, j] > 0
        out[lab] = pd.Series(langs[mask]).mode().iat[0] if mask.any() else "other"
    return out


EXIT_OOM = 3  # exit code on CUDA OOM, so run_experiments.py can halve the batch and retry


def trainable_state_dict(model):
    """Keep only trained parameters (LoRA + classifier head); the base encoder is reloaded from HuggingFace."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def checkpoint_state(model, config):
    full = not config["logging"].get("save_trainable_only", False)
    return move_state_to_cpu(model.state_dict() if full else trainable_state_dict(model))


def load_classifier_state(model, state):
    """Load a full checkpoint or one holding only the trained parameters (logging.save_trainable_only)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint keys do not match the model (wrong model/config?): {unexpected[:5]}")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing_trained = [k for k in missing if k in trainable]
    if missing_trained:
        raise RuntimeError(f"Checkpoint is missing trained parameters: {missing_trained[:5]}")


def move_state_to_cpu(state_dict):
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================== Losses =====================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        loss_pos = loss_pos * (1 - probs_pos) ** self.gamma_pos
        loss_neg = loss_neg * probs_pos ** self.gamma_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * bce_loss
        return loss.mean()


# ===================== Dataset =====================

class CodeCommentDataset(Dataset):
    def __init__(self, df, label_cols, tokenizer, max_len=128, text_cfg=None, to_19=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = label_cols
        self.text_cfg = text_cfg or {}
        self.to_19 = to_19

        self.name_to_idx = {name: i for i, name in enumerate(self.label_cols)}
        self.label_matrix = np.zeros((len(self.df), len(self.label_cols)), dtype=np.float32)

        for i, row in self.df.iterrows():
            lang = norm_lang(row.get("lang", ""))
            labs = map_labels_for_row(row.get("labels", ""), lang, self.to_19)
            for lab in labs:
                if lab in self.name_to_idx:
                    self.label_matrix[i, self.name_to_idx[lab]] = 1.0

    def __len__(self):
        return len(self.df)

    def _build_text(self, row):
        sentence = str(row["sentence"])
        lang = norm_lang(row.get("lang", ""))
        class_id = str(row.get("class_id", "UNKNOWN"))
        include_lang = self.text_cfg.get("include_lang", False)
        include_class_id = self.text_cfg.get("include_class_id", False)
        template = self.text_cfg.get("template", "[{lang}] {class_id} | {sentence}")
        if include_lang or include_class_id:
            return template.format(lang=lang.upper(), class_id=class_id, sentence=sentence)
        return sentence

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._build_text(row)
        labels = self.label_matrix[idx]

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


# ===================== Model =====================

# Names of the Linear layers that get LoRA adapters, per architecture (encoder.config.model_type)
LORA_TARGETS = {
    "roberta": ["query", "key", "value", "dense"],      # roberta, codebert, unixcoder, graphcodebert
    "xlm-roberta": ["query", "key", "value", "dense"],
    "bert": ["query", "key", "value", "dense"],
    "electra": ["query", "key", "value", "dense"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],  # deberta-v3-*
    "modernbert": ["Wqkv", "Wo", "Wi"],
}


def resolve_lora_targets(encoder, requested):
    """'auto'/None -> per architecture; a list that does not match the architecture -> warn and use 'auto'."""
    model_type = getattr(encoder.config, "model_type", "")
    auto = LORA_TARGETS.get(model_type, "all-linear")
    if requested in (None, "auto"):
        return auto
    names = {n.split(".")[-1] for n, _ in encoder.named_modules()}
    missing = [t for t in requested if t not in names]
    if missing:
        log(style(f"⚠ LoRA targets {missing} not found in architecture '{model_type}' → using {auto}", "yellow"))
        return auto
    return requested


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules=None,
        gradient_checkpointing: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.num_labels = num_labels
        self.pooling = pooling
        self.use_lora = use_lora and HAS_PEFT

        self.encoder = AutoModel.from_pretrained(model_name)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        if self.use_lora:
            target_modules = resolve_lora_targets(self.encoder, lora_target_modules)
            self.lora_targets = target_modules
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # encoder
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
        elif use_lora and not HAS_PEFT:
            log(style("⚠ peft is not installed → running full fine-tuning instead of LoRA", "yellow"))

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "cls":
            # First token: <s> for RoBERTa, [CLS] for BERT-style tokenizers
            pooled = last_hidden[:, 0]
        else:
            # Mean pooling over non-padding tokens (attention mask)
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom

        logits = self.classifier(pooled)
        return logits


def build_model(config, num_labels):
    peft_cfg = config["peft"]
    return TransformerClassifier(
        model_name=config["model_name"],
        num_labels=num_labels,
        dropout=peft_cfg.get("dropout", 0.1),
        use_lora=peft_cfg.get("enabled", False),
        lora_r=peft_cfg.get("r", 16),
        lora_alpha=peft_cfg.get("alpha", 32),
        lora_dropout=peft_cfg.get("dropout", 0.1),
        lora_target_modules=peft_cfg.get("target_modules", None),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        pooling=config.get("pooling", "mean"),  # configs without the key keep mean pooling
    )


# ===================== Metrics & Thresholds =====================

def safe_roc_auc(y_true, y_probs, average="macro"):
    try:
        valid = []
        for j in range(y_true.shape[1]):
            col = y_true[:, j]
            if col.min() != col.max():
                valid.append(j)
        if not valid:
            return np.nan
        return roc_auc_score(y_true[:, valid], y_probs[:, valid], average=average)
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred, y_probs):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )

    try:
        roc_auc_micro = safe_roc_auc(y_true, y_probs, average="micro")
        roc_auc_macro = safe_roc_auc(y_true, y_probs, average="macro")
        pr_auc_micro = average_precision_score(y_true, y_probs, average="micro")
        pr_auc_macro = average_precision_score(y_true, y_probs, average="macro")
    except Exception:
        roc_auc_micro = roc_auc_macro = pr_auc_micro = pr_auc_macro = np.nan

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_samples": precision_samples,
        "recall_samples": recall_samples,
        "f1_samples": f1_samples,
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_micro": pr_auc_micro,
        "pr_auc_macro": pr_auc_macro,
        # standard multi-label metrics
        "hamming_loss": hamming_loss(y_true, y_pred),                  # fraction of wrong (sentence, label) cells, lower is better
        "subset_accuracy": accuracy_score(y_true, y_pred),             # fraction of sentences whose WHOLE label set is correct
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
    }


def per_label_metrics(y_true, y_pred, y_probs, label_cols):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    rows = []
    for j, lab in enumerate(label_cols):
        ap = average_precision_score(y_true[:, j], y_probs[:, j]) if y_true[:, j].any() else float("nan")
        rows.append({
            "label": lab, "support": int(support[j]),
            "precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j]), "ap": float(ap),
        })
    return rows


def lang_macro_f1(per_label, label_langs):
    if not label_langs:
        return {}
    out = {}
    for lang in LANG_ORDER:
        f = [r["f1"] for r in per_label if label_langs.get(r["label"]) == lang]
        if f:
            out[lang] = float(np.mean(f))
    return out


def format_per_label_table(rows, thresholds, label_langs=None, prev_rows=None, indent="  "):
    """P/R/F1/AP table per label, grouped by language; ΔF1 against prev_rows (the previous best)."""
    label_langs = label_langs or {}
    prev = {r["label"]: r["f1"] for r in (prev_rows or [])}
    head = f"{'Lang':<8}{'Label':<25}{'n':>5}{'Thr':>6}{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}  {'ΔF1':<10}"
    lines = [indent + style(head, "bold"), indent + "─" * len(head)]
    for lang in LANG_ORDER:
        idx = [j for j, r in enumerate(rows) if label_langs.get(r["label"], "other") == lang]
        if not idx:
            continue
        for k, j in enumerate(idx):
            r = rows[j]
            f1_txt = f"{r['f1']:>8.4f}"
            if r["f1"] < 0.5:
                f1_txt = style(f1_txt, "red")
            flag = "  ⚠ weak" if r["f1"] < 0.5 else ""
            delta = fmt_delta(r["f1"], prev.get(r["label"])) if r["label"] in prev else ""
            lines.append(
                indent
                + f"{(lang if k == 0 else ''):<8}{r['label']:<25}{r['support']:>5}{thresholds[j]:>6.2f}"
                + f"{r['precision']:>8.4f}{r['recall']:>8.4f}" + f1_txt + f"{r['ap']:>8.4f}  "
                + delta + flag
            )
        macro = np.mean([rows[j]["f1"] for j in idx])
        n = sum(rows[j]["support"] for j in idx)
        lines.append(indent + style(f"{'':<8}{'↳ macro ' + lang:<25}{n:>5}{'':>6}{'':>8}{'':>8}{macro:>8.4f}", "dim"))
    return "\n".join(lines)


def save_curves(history, path: Path, title: str):
    """Plot loss + F1 per epoch to a PNG (skipped if matplotlib is unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ep, [h["train_loss"] for h in history], marker="o", label="train loss")
    if all(h["val_loss"] is not None for h in history):
        ax1.plot(ep, [h["val_loss"] for h in history], marker="o", label="val loss")
    ax1.set_title("Loss")
    for key, lab in [("f1_samples", "F1 samples"), ("f1_micro", "F1 micro"), ("f1_macro", "F1 macro")]:
        ax2.plot(ep, [h[key] for h in history], marker="o", label=lab)
    best = max(history, key=lambda h: h["f1_samples"])["epoch"]
    ax2.axvline(best, ls="--", c="gray", lw=1, label=f"best epoch {best}")
    ax2.set_title("Validation F1")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_oof(path: Path, val_df, probs, labels, thresholds):
    """Out-of-fold predictions (probabilities for each validation sentence) for error analysis / extra metrics."""
    ids = val_df["id"].values if "id" in val_df.columns else np.arange(len(val_df))
    np.savez_compressed(path, ids=ids, probs=probs.astype(np.float32), labels=labels.astype(np.int8),
                        thresholds=np.asarray(thresholds, dtype=np.float32))


def optimize_thresholds_per_label(y_true, y_probs, num_thresholds=100):
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    L = y_true.shape[1]
    best_t = np.ones(L) * 0.5
    for j in range(L):
        yt = y_true[:, j]
        yp = y_probs[:, j]
        best_f, bt = -1.0, 0.5
        for t in thresholds:
            pred = (yp >= t).astype(int)
            f = f1_score(yt, pred, zero_division=0)
            if f > best_f:
                best_f, bt = f, t
        best_t[j] = bt
    return best_t


def optimize_thresholds_samples(y_true, y_probs, base=None, iters=2, grid=50):
    """
    Coordinate descent maximizing the global F1(samples).
    - base: initial per-label thresholds (recommended).
    """
    L = y_true.shape[1]
    thr = (base.copy() if base is not None else np.ones(L) * 0.5).astype(np.float32)

    def f1s(th):
        pred = (y_probs >= th).astype(int)
        return f1_score(y_true, pred, average="samples", zero_division=0)

    if base is None:
        ts = np.linspace(0.05, 0.95, grid)
        for j in range(L):
            best_f, bt = -1.0, 0.5
            for t in ts:
                pred = (y_probs[:, j] >= t).astype(int)
                f = f1_score(y_true[:, j], pred, zero_division=0)
                if f > best_f:
                    best_f, bt = f, t
            thr[j] = bt

    ts = np.linspace(0.05, 0.95, grid)
    for _ in range(iters):
        improved = False
        for j in range(L):
            cur_best = f1s(thr)
            cur_t = thr[j]
            best_f, bt = cur_best, cur_t
            for t in ts:
                tmp = thr.copy()
                tmp[j] = t
                f = f1s(tmp)
                if f > best_f:
                    best_f, bt = f, t
            if bt != cur_t:
                thr[j] = bt
                improved = True
        if not improved:
            break
    return thr


# ===================== Train/Eval loops =====================

def make_scheduler(optimizer, total_updates, warmup_ratio, name="cosine"):
    warmup_steps = int(total_updates * float(warmup_ratio))
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    else:
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)


def get_amp_setup(precision):
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    return use_amp, amp_dtype, scaler


def make_sampler_if_needed(train_labels, cfg):
    samp_cfg = cfg.get("sampler", {})
    if not samp_cfg.get("enabled", False):
        return None
    typ = samp_cfg.get("type", "weighted")
    if typ != "weighted":
        return None

    eps = float(samp_cfg.get("smooth_eps", 1e-3))
    freq = train_labels.sum(0) + eps
    inv = 1.0 / freq  # rare label -> large weight
    sample_w = (train_labels * inv).sum(1)
    # sample without any label (rare): give it the minimum weight
    min_w = float(inv.min()) * 0.5
    sample_w = np.where(sample_w > 0, sample_w, min_w)
    sample_w = torch.tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, precision="fp32",
                    grad_accum=1, max_grad_norm=1.0, desc="train"):
    model.train()
    running = 0.0
    step_in_accum = 0
    grad_norms = []
    n_samples = 0
    use_amp, amp_dtype, scaler = get_amp_setup(precision)
    t0 = time.time()

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not _SHOW_BARS)
    for step, batch in enumerate(bar, 1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets) / grad_accum
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, targets) / grad_accum

        if precision == "fp16":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step_in_accum += 1
        if step_in_accum % grad_accum == 0:
            if precision == "fp16":
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            grad_norms.append(float(gn))  # norm BEFORE clipping

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            step_in_accum = 0

        running += loss.item() * grad_accum
        n_samples += input_ids.size(0)
        if step % 10 == 0 or step == len(loader):
            bar.set_postfix(loss=f"{running / step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    elapsed = time.time() - t0
    finite = [g for g in grad_norms if np.isfinite(g)]
    return {
        "loss": running / len(loader),
        "grad_norm": float(np.mean(finite)) if finite else float("nan"),
        "lr": optimizer.param_groups[0]["lr"],
        "samples_per_sec": n_samples / max(elapsed, 1e-9),
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device, precision="fp32", criterion=None, desc="eval"):
    model.eval()
    probs_all, labels_all = [], []
    loss_sum, n_seen = 0.0, 0
    use_amp, amp_dtype, _ = get_amp_setup(precision)

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not _SHOW_BARS):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = logits.float()  # bf16 autocast -> numpy does not support bfloat16
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * input_ids.size(0)
        n_seen += input_ids.size(0)

        probs = torch.sigmoid(logits)
        probs_all.append(probs.cpu().numpy())
        labels_all.append(targets.cpu().numpy())

    probs_all = np.vstack(probs_all)
    labels_all = np.vstack(labels_all)
    val_loss = loss_sum / max(n_seen, 1) if criterion is not None else None
    return probs_all, labels_all, val_loss


def build_dataloaders(train_df, val_df, label_cols, tokenizer, config):
    text_cfg = config.get("text_features", {})
    to_19 = bool(config["data"].get("expand_to_19_classes", False))

    train_bs = int(config["train_params"]["batch_size"])
    eval_bs = int(config["train_params"].get("eval_batch_size", train_bs))  # NEW: separate validation batch size

    train_ds = CodeCommentDataset(train_df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg, to_19=to_19)
    val_ds = CodeCommentDataset(val_df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg, to_19=to_19)

    sampler = make_sampler_if_needed(train_ds.label_matrix, config)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=train_bs,
            sampler=sampler, num_workers=config["train_params"].get("num_workers", 4),
            pin_memory=True, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=train_bs,
            shuffle=True, num_workers=config["train_params"].get("num_workers", 4),
            pin_memory=True, drop_last=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=eval_bs,          # NEW: no longer *2
        shuffle=False, num_workers=config["train_params"].get("num_workers", 4),
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_ds


def make_loss(config, device, train_label_matrix=None):
    lt = config.get("loss_type", "asl").lower()
    if lt == "asl":
        p = config.get("loss_params", {})
        return AsymmetricLoss(gamma_pos=p.get("gamma_pos", 0), gamma_neg=p.get("gamma_neg", 4), clip=p.get("clip", 0.05))
    elif lt == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    else:
        if not config.get("loss_params", {}).get("pos_weight", True):
            return nn.BCEWithLogitsLoss()  # plain BCE (baseline)
        # BCE with pos_weight
        assert train_label_matrix is not None, "Need train_label_matrix for BCE pos_weight."
        pos = train_label_matrix.sum(0)
        neg = len(train_label_matrix) - pos
        pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32), device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_fold(
    fold_idx,
    train_df,
    val_df,
    label_cols,
    config,
    device,
    n_folds=1,
    label_langs=None,
):
    t_fold = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_loader, val_loader, train_ds = build_dataloaders(train_df, val_df, label_cols, tokenizer, config)

    model = build_model(config, len(label_cols)).to(device)

    criterion = make_loss(config, device, train_label_matrix=train_ds.label_matrix)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train_params"]["lr"],
        weight_decay=config["train_params"]["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, int(config["train_params"]["grad_accum"])))
    total_updates = steps_per_epoch * int(config["train_params"]["epochs"])
    scheduler = make_scheduler(
        optimizer, total_updates=total_updates,
        warmup_ratio=config["train_params"]["warmup"],
        name=config["train_params"]["scheduler"],
    )

    precision = config.get("precision", "fp32")
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    n_epochs = int(config["train_params"]["epochs"])
    grad_accum = int(config["train_params"]["grad_accum"])
    early_stop = int(config["logging"].get("early_stop", 6))
    # no early stopping before min_epochs: with BCE + threshold 0.5 the first epochs often predict all-negative (F1 = 0)
    min_epochs = int(config["logging"].get("min_epochs", 0))
    do_thr_search = bool(config["eval"].get("threshold_search", True))
    per_label_mode = str(config["logging"].get("per_label_table", "best"))  # best | every | none
    results_dir = Path(config["logging"]["output_dir"])

    # ---- Fold header ----
    trainable, total = count_params(model)
    peft_cfg = config["peft"]
    model_desc = config["model_name"] + (
        f" + LoRA r={peft_cfg.get('r')} α={peft_cfg.get('alpha')} on {model.lora_targets}"
        if model.use_lora else " (full fine-tune)"
    ) + f" · {model.pooling} pooling"
    log("\n" + style(DOUBLE_RULE, "cyan"))
    log(style(f"FOLD {fold_idx+1}/{n_folds}", "bold", "cyan")
        + f"   train {len(train_df):,} sentences · val {len(val_df):,} sentences")
    log(f"  Model    {model_desc} · trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M ({trainable/total*100:.2f}%)")
    log(f"  Steps    {len(train_loader)} batch/epoch · {steps_per_epoch} update/epoch · "
        f"up to {total_updates:,} updates · warmup {int(total_updates * float(config['train_params']['warmup'])):,}")
    log(style(DOUBLE_RULE, "cyan"))

    history, epoch_times = [], []
    for epoch in range(n_epochs):
        t_epoch = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tr = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            precision=precision, grad_accum=grad_accum,
            desc=f"fold {fold_idx+1} · epoch {epoch+1}/{n_epochs}",
        )

        t_eval = time.time()
        val_probs, val_labels, val_loss = evaluate(
            model, val_loader, device, precision=precision, criterion=criterion, desc="  eval"
        )
        eval_time = time.time() - t_eval

        t_thr = time.time()
        if do_thr_search:
            base_thr = optimize_thresholds_per_label(val_labels, val_probs)
            thr = optimize_thresholds_samples(val_labels, val_probs, base=base_thr, iters=2)
        else:
            thr = np.ones(len(label_cols)) * 0.5
        thr_time = time.time() - t_thr

        val_preds = (val_probs >= thr).astype(int)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        per_label = per_label_metrics(val_labels, val_preds, val_probs, label_cols)
        by_lang = lang_macro_f1(per_label, label_langs)

        epoch_time = time.time() - t_epoch
        epoch_times.append(epoch_time)
        peak = gpu_peak_gb()
        prev = history[-1] if history else None

        is_best = metrics["f1_samples"] > best_f1
        prev_best_f1, prev_best_epoch = best_f1, best_epoch
        prev_best_rows = best_state["per_label"] if best_state else None
        if is_best:
            best_f1, best_epoch = metrics["f1_samples"], epoch + 1
            best_oof = (val_probs.copy(), val_labels.copy())
            # Keep the state_dict on CPU so holding best_state in RAM does not use VRAM
            best_state = {
                "fold": fold_idx,
                "model": checkpoint_state(model, config),
                "thresholds": thr.tolist(),
                "metrics": metrics,
                "per_label": per_label,
                "epoch": epoch + 1,
            }
            patience = 0
        else:
            patience += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": tr["loss"],
            "val_loss": val_loss,
            "lr": tr["lr"],
            "grad_norm": tr["grad_norm"],
            **{k: float(v) for k, v in metrics.items()},
            **{f"f1_macro_{lang}": v for lang, v in by_lang.items()},
            "epoch_time_s": epoch_time,
            "gpu_peak_gb": peak,
            "is_best": is_best,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(results_dir / f"history_fold{fold_idx}.csv", index=False)

        # ---- Epoch report ----
        g = lambda key: prev[key] if prev else None  # noqa: E731
        log(style(RULE, "dim"))
        log(style(f"Epoch {epoch+1:>3}/{n_epochs}", "bold")
            + f"  ·  {fmt_time(epoch_time)} (train {fmt_time(tr['time'])} · eval {fmt_time(eval_time)}"
            + f" · thr {fmt_time(thr_time)})  ·  {tr['samples_per_sec']:.0f} sentences/s")
        loss_line = f"  Loss      train {fmt_num(tr['loss'])} {fmt_delta(tr['loss'], g('train_loss'), False)}"
        if val_loss is not None:
            loss_line += (f"   val {fmt_num(val_loss)} {fmt_delta(val_loss, g('val_loss'), False)}"
                          f"   gap(val-train) {val_loss - tr['loss']:+.4f}")
        log(loss_line)
        mem = f"   GPU peak {peak:.2f} GB" if peak is not None else ""
        log(f"  Optim     lr {tr['lr']:.2e}   grad-norm {fmt_num(tr['grad_norm'], 3)}{mem}")
        log(f"  Val F1    samples {style(fmt_num(metrics['f1_samples']), 'bold')} "
            f"{fmt_delta(metrics['f1_samples'], g('f1_samples'))}"
            f"   micro {fmt_num(metrics['f1_micro'])} {fmt_delta(metrics['f1_micro'], g('f1_micro'))}"
            f"   macro {fmt_num(metrics['f1_macro'])} {fmt_delta(metrics['f1_macro'], g('f1_macro'))}")
        log(f"  Val P/R   samples P {fmt_num(metrics['precision_samples'])} R {fmt_num(metrics['recall_samples'])}"
            f"   ·   macro P {fmt_num(metrics['precision_macro'])} R {fmt_num(metrics['recall_macro'])}")
        log(f"  Ranking   PR-AUC(macro) {fmt_num(metrics['pr_auc_macro'])}   ROC-AUC(macro) {fmt_num(metrics['roc_auc_macro'])}")
        log(f"  Multi     Jaccard {fmt_num(metrics['jaccard_samples'])}   exact-match {fmt_num(metrics['subset_accuracy'])}"
            f"   Hamming loss {fmt_num(metrics['hamming_loss'])} {fmt_delta(metrics['hamming_loss'], g('hamming_loss'), False)}")
        if by_lang:
            weakest = min(per_label, key=lambda r: r["f1"])
            log("  By lang   F1 macro  " + " · ".join(f"{lang.capitalize()} {fmt_num(v)}" for lang, v in by_lang.items())
                + f"   │ weakest: {weakest['label']} {fmt_num(weakest['f1'])}")

        remaining = n_epochs - (epoch + 1)
        avg_epoch = float(np.mean(epoch_times))
        eta = (f"ETA fold ≤ {fmt_time(avg_epoch * remaining)} · early stop after {early_stop - patience} more epoch(s) "
               f"without improvement") if remaining and patience < early_stop else ""
        if is_best:
            prev_txt = f" (previous: {fmt_num(prev_best_f1)} @ epoch {prev_best_epoch})" if prev_best_epoch else ""
            log("  Status    " + style(f"★ NEW BEST F1(samples) = {fmt_num(best_f1)}", "green", "bold")
                + prev_txt + f" · patience 0/{early_stop}" + (f" · {eta}" if eta else ""))
        else:
            log(f"  Status    no improvement · best {fmt_num(best_f1)} @ epoch {best_epoch} · "
                + style(f"patience {patience}/{early_stop}", "yellow") + (f" · {eta}" if eta else ""))

        if per_label_mode == "every" or (per_label_mode == "best" and is_best):
            note = "vs previous best" if per_label_mode == "best" else "vs current best"
            log(f"\n  Per-label details (ΔF1 {note}):")
            log(format_per_label_table(per_label, thr, label_langs, prev_best_rows))

        if patience >= early_stop and epoch + 1 >= min_epochs:
            log(style(f"\n⏹ Early stopping at epoch {epoch+1}: no F1(samples) improvement for {early_stop} epochs in a row", "yellow"))
            break

    # ---- Fold summary ----
    fold_time = time.time() - t_fold
    best_state["time"] = fold_time
    best_state["epochs_run"] = len(history)
    m = best_state["metrics"]
    log("\n" + style(f"✔ Fold {fold_idx+1}/{n_folds} done in {fmt_time(fold_time)}", "green", "bold")
        + f" · best epoch {best_epoch}/{len(history)} · F1 samples {fmt_num(m['f1_samples'])}"
        + f" · micro {fmt_num(m['f1_micro'])} · macro {fmt_num(m['f1_macro'])}")
    curves = save_curves(history, results_dir / f"curves_fold{fold_idx}.png",
                         f"Fold {fold_idx+1} · {config['model_name']}")
    save_oof(results_dir / f"oof_fold{fold_idx}.npz", val_df, *best_oof, best_state["thresholds"])
    log(f"  Epoch history → {results_dir / f'history_fold{fold_idx}.csv'}"
        + (f" · curves → {curves}" if curves else ""))

    # Free RAM/GPU memory used by this fold
    del train_loader, val_loader, train_ds, tokenizer, optimizer, scheduler, criterion, model
    free_cuda()

    return best_state


def train_full(df, label_cols, config, device, save_path: Path, med_thresholds: np.ndarray):
    t_full = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    # Full-data training has no holdout → training only; thresholds are the median thresholds from CV
    text_cfg = config.get("text_features", {})
    full_ds = CodeCommentDataset(
        df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg,
        to_19=bool(config["data"].get("expand_to_19_classes", False))
    )
    sampler = make_sampler_if_needed(full_ds.label_matrix, config)
    full_loader = DataLoader(
        full_ds,
        batch_size=int(config["train_params"]["batch_size"]),
        sampler=sampler if sampler is not None else None,
        shuffle=(sampler is None),
        num_workers=config["train_params"].get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(config, len(label_cols)).to(device)

    criterion = make_loss(config, device, train_label_matrix=full_ds.label_matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train_params"]["lr"], weight_decay=config["train_params"]["weight_decay"])

    ft = config.get("final_training", {})
    epochs = int(ft.get("epochs", config["train_params"]["epochs"]))
    # The LR schedule may be longer than the epochs actually run, so the final model ends in the same state as the CV models at their best epoch
    # (e.g. CV used a 30-epoch schedule with best epoch ~25 -> epochs: 25, schedule_epochs: 30)
    schedule_epochs = max(epochs, int(ft.get("schedule_epochs", epochs)))

    steps_per_epoch = math.ceil(len(full_loader) / max(1, int(config["train_params"]["grad_accum"])))
    total_updates = steps_per_epoch * schedule_epochs
    scheduler = make_scheduler(
        optimizer, total_updates=total_updates,
        warmup_ratio=ft.get("warmup", config["train_params"]["warmup"]),
        name=config["train_params"]["scheduler"],
    )

    precision = config.get("precision", "fp32")

    log("\n" + style(DOUBLE_RULE, "cyan"))
    log(style("FULL TRAIN", "bold", "cyan") + f"   {len(full_ds):,} sentences (all data) · {epochs} epochs"
        f" on a {schedule_epochs}-epoch LR schedule · no validation, uses the median thresholds from CV")
    log(style(DOUBLE_RULE, "cyan"))

    prev_loss, epoch_times = None, []
    for epoch in range(epochs):
        tr = train_one_epoch(
            model, full_loader, optimizer, scheduler, criterion, device,
            precision=precision, grad_accum=int(config["train_params"]["grad_accum"]),
            desc=f"full · epoch {epoch+1}/{epochs}",
        )
        epoch_times.append(tr["time"])
        eta = fmt_time(np.mean(epoch_times) * (epochs - epoch - 1))
        log(f"[Full] Epoch {epoch+1:>3}/{epochs} · loss {fmt_num(tr['loss'])} {fmt_delta(tr['loss'], prev_loss, False)}"
            f" · lr {tr['lr']:.2e} · grad-norm {fmt_num(tr['grad_norm'], 3)} · {fmt_time(tr['time'])}"
            f" · {tr['samples_per_sec']:.0f} sentences/s · ETA {eta}")
        prev_loss = tr["loss"]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint_state(model, config), "thresholds": med_thresholds.tolist()}, save_path)
    # run_experiments.py reads this file to know whether the final model matches the current config
    with open(save_path.parent / "final_meta.json", "w") as f:
        json.dump({"model_name": config["model_name"], "epochs": epochs, "schedule_epochs": schedule_epochs,
                   "time_s": time.time() - t_full, "finished_at": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    log(style(f"✔ Full train done in {fmt_time(time.time() - t_full)}", "green", "bold") + f" · model → {save_path}")

    # cleanup
    del full_loader, full_ds, tokenizer, optimizer, scheduler, criterion, model
    free_cuda()


def log_run_header(cfg_path, config, device, df, labels_mat, label_cols, results_dir):
    tp, peft_cfg = config["train_params"], config["peft"]
    log("\n" + style(DOUBLE_RULE, "bold"))
    log(style(f"CODE COMMENT CLASSIFICATION · TRAINING RUN · {datetime.now():%Y-%m-%d %H:%M:%S}", "bold"))
    log(style(DOUBLE_RULE, "bold"))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        dev = f"cuda · {props.name} ({props.total_memory / 1024**3:.1f} GB)"
    else:
        dev = style("cpu (no GPU — will be very slow)", "yellow")
    lora = (f"LoRA r={peft_cfg.get('r')} α={peft_cfg.get('alpha')} dropout={peft_cfg.get('dropout')}"
            if peft_cfg.get("enabled") else "full fine-tune")
    loss_p = config.get("loss_params", {})
    sampler = config.get("sampler", {}).get("type", "weighted") if config.get("sampler", {}).get("enabled") else "off"
    split = ("single 80/20 multilabel stratified" if config.get("use_single_split")
             else f"{config.get('cv', {}).get('n_splits', 5)}-fold multilabel stratified")
    langs = df["__lang_norm__"].value_counts()
    multi = (labels_mat.sum(1) > 1).mean() * 100
    rows = [
        ("Config", str(cfg_path)),
        ("Output", f"{results_dir}/  (log: {results_dir / 'train.log'})"),
        ("Device", f"{dev} · precision {config.get('precision', 'fp32')}"),
        ("Model", f"{config['model_name']} · {lora} · {config.get('pooling', 'mean')} pooling · max_len {config['max_len']}"),
        ("Optim", f"AdamW lr {tp['lr']} · wd {tp['weight_decay']} · {tp['scheduler']} warmup {tp['warmup']}"
                  f" · batch {tp['batch_size']}×{tp['grad_accum']}={int(tp['batch_size']) * int(tp['grad_accum'])}"),
        ("Schedule", f"up to {tp['epochs']} epochs · early stop {config['logging'].get('early_stop', 6)}"
                     f" · seed {tp['seed']}"),
        ("Loss", f"{config.get('loss_type', 'asl')} {loss_p if loss_p else ''} · sampler {sampler}"),
        ("Data", f"{len(df):,} sentences · " + " · ".join(f"{k} {v:,}" for k, v in langs.items())
                 + f" · {len(label_cols)} labels · {multi:.1f}% multi-label"),
        ("Split", f"{split} · threshold search {'on' if config['eval'].get('threshold_search', True) else 'off'}"),
    ]
    for k, v in rows:
        log(f"  {k:<9} {v}")


def aggregate_fold_metrics(fold_states):
    """mean/std of each metric over folds (ignores folds missing a metric, e.g. folds saved before it existed)."""
    keys = list(dict.fromkeys(k for st in fold_states for k in st["metrics"]))
    vals = {k: [st["metrics"].get(k, np.nan) for st in fold_states] for k in keys}
    avg = {k: float(np.nanmean(v)) for k, v in vals.items()}
    std = {k: float(np.nanstd(v)) for k, v in vals.items()}
    return avg, std


def log_cv_summary(fold_states, label_cols, label_langs, thr_median, total_time):
    log("\n" + style(DOUBLE_RULE, "bold"))
    log(style(f"CROSS-VALIDATION SUMMARY · {len(fold_states)} folds", "bold"))
    log(style(DOUBLE_RULE, "bold"))
    cols = [("f1_samples", "F1-samp"), ("f1_micro", "F1-micro"), ("f1_macro", "F1-macro"),
            ("pr_auc_macro", "PR-AUC"), ("roc_auc_macro", "ROC-AUC"),
            ("jaccard_samples", "Jaccard"), ("subset_accuracy", "Exact"), ("hamming_loss", "Hamming↓")]
    head = f"  {'Fold':>4} {'Best ep':>9}" + "".join(f"{name:>10}" for _, name in cols) + f"{'Time':>10}"
    log(style(head, "bold"))
    log("  " + "─" * (len(head) - 2))
    for st in fold_states:
        ep = f"{st['epoch']}/{st['epochs_run']}"
        log(f"  {st['fold']:>4} {ep:>9}" + "".join(f"{fmt_num(st['metrics'].get(k, np.nan)):>10}" for k, _ in cols)
            + f"{fmt_time(st['time']):>10}")
    log("  " + "─" * (len(head) - 2))
    for agg, fn in [("mean", np.nanmean), ("std", np.nanstd)]:
        log(style(f"  {agg:>4} {'':>9}" + "".join(
            f"{fmt_num(float(fn([st['metrics'].get(k, np.nan) for st in fold_states]))):>10}" for k, _ in cols), "bold"))

    # Per-label F1 averaged over folds, weakest label first
    f1s = np.array([[r["f1"] for r in st["per_label"]] for st in fold_states])
    order = np.argsort(f1s.mean(0))
    log("\n  Per-label F1 (mean ± std over folds, weakest first) · threshold = median over folds:")
    log(style(f"  {'Label':<25}{'Lang':<8}{'F1':>8}{'± std':>8}{'Thr':>7}", "bold"))
    for j in order:
        mean, std = f1s[:, j].mean(), f1s[:, j].std()
        txt = f"  {label_cols[j]:<25}{(label_langs or {}).get(label_cols[j], ''):<8}{mean:>8.4f}{std:>8.4f}{thr_median[j]:>7.2f}"
        log(style(txt + "  ⚠ weak", "red") if mean < 0.5 else txt)
    log(f"\n  Total CV time: {fmt_time(total_time)}")


def apply_cli_overrides(config, args):
    """--model switches the backbone (LoRA targets picked per architecture) and uses a separate results folder."""
    if args.model:
        config["model_name"] = config["tokenizer_name"] = args.model
        config["peft"]["target_modules"] = "auto"
    if args.batch_size:
        config["train_params"]["batch_size"] = args.batch_size
        # eval keeps no gradients so it can use twice the batch; it still shrinks when the train batch shrinks after an OOM
        eval_bs = config["train_params"].get("eval_batch_size", args.batch_size)
        config["train_params"]["eval_batch_size"] = min(int(eval_bs), 2 * args.batch_size)
    if args.grad_accum:
        config["train_params"]["grad_accum"] = args.grad_accum
    if getattr(args, "resume", False):
        config.setdefault("cv", {})["skip_if_checkpoint_exists"] = True
    if getattr(args, "pooling", None):
        config["pooling"] = args.pooling
    out = args.output_dir or (f"runs/{args.model.rstrip('/').split('/')[-1].lower()}" if args.model else None)
    if out:
        config["logging"]["output_dir"] = out
        config.setdefault("final_training", {})["save_path"] = str(Path(out) / "final_model.pt")
    return config


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/train_rtx5060.yaml",
                        help="YAML config (default: configs/train_rtx5060.yaml, fits an 8 GB GPU)")
    parser.add_argument("--final-only", action="store_true",
                        help="Skip CV and only retrain the final model on all data "
                             "using median_thresholds from <output_dir>/results.json")
    parser.add_argument("--model", help="Override model_name/tokenizer_name, e.g. microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", help="Results folder (default with --model: runs/<model name>)")
    parser.add_argument("--batch-size", type=int, help="Override train_params.batch_size (keep batch × accum = 64)")
    parser.add_argument("--grad-accum", type=int, help="Override train_params.grad_accum")
    parser.add_argument("--resume", action="store_true",
                        help="Skip folds that already have a checkpoint (resume after an interruption)")
    parser.add_argument("--pooling", choices=["mean", "cls"],
                        help="Override the sentence representation: mean of the tokens, or the first token (<s> / [CLS])")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = apply_cli_overrides(read_yaml(cfg_path), args)
    t_run = time.time()

    set_seed(int(config["train_params"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = Path(config["data"]["raw_path"])
    if not data_path.exists():
        alt = Path("sentences.csv")
        if alt.exists():
            data_path = alt
    df = pd.read_csv(data_path)

    assert "sentence" in df.columns and "labels" in df.columns, "The CSV needs 'sentence' and 'labels' columns"

    # Label list (19 labels) + checks
    label_cols = config["data"].get("label_list", None)
    assert label_cols and isinstance(label_cols, list), "The config needs 'data.label_list'."
    assert len(label_cols) == int(config["num_labels"]), "num_labels does not match the number of labels in label_list."

    # Normalize lang so folds are split correctly
    df = df.copy()
    df["__lang_norm__"] = df.get("lang", "").apply(norm_lang)

    # Label matrix used for splitting (already mapped to 19 labels)
    labels_mat = build_label_matrix(df, label_cols, expand_to_19=bool(config["data"].get("expand_to_19_classes", False)))

    results_dir = Path(config["logging"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    set_log_file(results_dir / "train.log")
    # Effective config (after CLI overrides), reused by demo/app.py, compare_runs.py and --final-only
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    label_langs = infer_label_langs(df["__lang_norm__"].values, labels_mat, label_cols)
    log_run_header(cfg_path, config, device, df, labels_mat, label_cols, results_dir)

    if args.final_only:
        res_path = results_dir / "results.json"
        if not res_path.exists():
            log(style(f"✘ --final-only needs {res_path} from a previous CV run", "red"))
            sys.exit(1)
        with open(res_path) as f:
            res = json.load(f)
        if res.get("label_list") != label_cols:
            log(style(f"✘ label_list in {res_path} differs from the current config", "red"))
            sys.exit(1)
        thr = np.array(res.get("median_thresholds") or res["thresholds"], dtype=np.float32)
        if res.get("fold_best_epochs"):
            log(f"\n↻ --final-only: using median thresholds from {res_path} · best epoch per fold "
                f"{res['fold_best_epochs']} (mean {np.mean(res['fold_best_epochs']):.1f})")
        save_path = Path(config.get("final_training", {}).get("save_path", results_dir / "final_model.pt"))
        train_full(df, label_cols, config, device, save_path, med_thresholds=thr)
        log("\n" + style(f"✅ Done in {fmt_time(time.time() - t_run)} · model → {save_path}", "green", "bold"))
        return

    fold_states = []
    if bool(config.get("use_single_split", False)):
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(config["train_params"]["seed"]))
        X = df["sentence"].values
        (tr_idx, va_idx) = next(msss.split(X, labels_mat))

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        best_state = train_fold(0, train_df, val_df, label_cols, config, device, n_folds=1, label_langs=label_langs)
        fold_states.append(best_state)

        if config["logging"].get("save_best", True):
            torch.save(best_state["model"], results_dir / f"best_model_fold0.pt")
            with open(results_dir / f"best_thresholds_fold0.json", "w") as f:
                json.dump(best_state["thresholds"], f, indent=2)

        with open(results_dir / "results.json", "w") as f:
            json.dump({"split": "single", "metrics": best_state["metrics"], "thresholds": best_state["thresholds"],
                       "per_label": best_state["per_label"], "best_epoch": best_state["epoch"],
                       "label_list": label_cols}, f, indent=2)

        log("\n" + style("Per-label details at the best epoch (20% validation split):", "bold"))
        log(format_per_label_table(best_state["per_label"], best_state["thresholds"], label_langs))
    else:
        n_splits = int(config.get("cv", {}).get("n_splits", 5))
        start_fold = int(config.get("cv", {}).get("start_fold", 0))  # NEW: resume from any fold (0-based)
        skip_if_exists = bool(config.get("cv", {}).get("skip_if_checkpoint_exists", False))  # NEW

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config["train_params"]["seed"]))
        X = df["sentence"].values

        for fold_idx, (tr_idx, va_idx) in enumerate(mskf.split(X, labels_mat)):
            if fold_idx < start_fold:
                log(style(f"⏩ Skipping fold {fold_idx} (start_fold={start_fold})", "yellow"))
                continue

            ckpt_path = results_dir / f"best_model_fold{fold_idx}.pt"
            thr_path  = results_dir / f"best_thresholds_fold{fold_idx}.json"
            state_path = results_dir / f"fold{fold_idx}_state.json"
            if skip_if_exists and ckpt_path.exists() and thr_path.exists():
                if state_path.exists():
                    # resuming after an interruption: reload the old fold result so the summary covers every fold
                    with open(state_path) as f:
                        fold_states.append(json.load(f))
                    log(style(f"⏩ Skipping fold {fold_idx} (checkpoint exists, result reloaded from {state_path.name})", "yellow"))
                else:
                    log(style(f"⏩ Skipping fold {fold_idx} (checkpoint exists but {state_path.name} is missing "
                              "→ this fold is not in the summary)", "yellow"))
                continue

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[va_idx].reset_index(drop=True)

            best_state = train_fold(fold_idx, train_df, val_df, label_cols, config, device,
                                    n_folds=n_splits, label_langs=label_langs)
            fold_states.append(best_state)

            if config["logging"].get("save_best", True):
                torch.save(best_state["model"], ckpt_path)
                with open(thr_path, "w") as f:
                    json.dump(best_state["thresholds"], f, indent=2)
                with open(state_path, "w") as f:
                    json.dump({k: v for k, v in best_state.items() if k != "model"}, f, indent=2)
                log(f"  Checkpoint → {ckpt_path} · thresholds → {thr_path}")
            best_state.pop("model", None)  # already on disk; do not keep it in RAM for the next folds

            # freeing memory here after each fold is also fine (best_state only holds CPU tensors)
            free_cuda()

        if not fold_states:
            log("\nℹ️ No fold was trained in this run (possibly because of start_fold/skip_if_checkpoint_exists).")
            return

        avg, std = aggregate_fold_metrics(fold_states)

        # Median thresholds across folds (recommended for deployment)
        thr_stack = np.vstack([np.array(st["thresholds"]) for st in fold_states])
        thr_median = np.median(thr_stack, axis=0)

        f1s = np.array([[r["f1"] for r in st["per_label"]] for st in fold_states])
        summary = {
            "split": "5fold",
            "folds": [st["fold"] for st in fold_states],
            "fold_metrics": [st["metrics"] for st in fold_states],
            "fold_best_epochs": [st["epoch"] for st in fold_states],
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

        log_cv_summary(fold_states, label_cols, label_langs, thr_median, time.time() - t_run)
        log(f"  Suggested final_training: epochs ≈ mean best epoch "
            f"{np.mean([st['epoch'] for st in fold_states]):.1f} · schedule_epochs = {config['train_params']['epochs']}")

        # (Optional) retrain on the full data with the median thresholds
        if bool(config.get("final_training", {}).get("enabled", False)):
            save_path = Path(config["final_training"]["save_path"])
            train_full(df, label_cols, config, device, save_path, med_thresholds=thr_median)

    log("\n" + style(f"✅ Done in {fmt_time(time.time() - t_run)} · results in {results_dir}/", "green", "bold"))


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        log(style(f"\n✘ Out of GPU memory (CUDA out of memory): {str(e).splitlines()[0]}", "red", "bold"))
        log("  On an 8 GB GPU use configs/train_rtx5060.yaml (batch 16 × grad_accum 4, bf16), or lower --batch-size\n"
            "  and raise --grad-accum accordingly (scripts/run_experiments.py does this automatically).")
        sys.exit(EXIT_OOM)