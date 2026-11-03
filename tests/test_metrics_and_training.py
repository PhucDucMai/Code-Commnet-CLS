"""Metrics, threshold search, loss, sampler and config helpers of src/train.py. CPU only, a few seconds."""
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from inference import decide_labels, match_status
from train import (
    AsymmetricLoss,
    aggregate_fold_metrics,
    apply_cli_overrides,
    compute_metrics,
    make_sampler_if_needed,
    optimize_thresholds_per_label,
    optimize_thresholds_samples,
)


# ---------------------------------------------------------------- metrics

def test_perfect_prediction_scores():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    m = compute_metrics(y, y, y.astype(float))
    for key in ("f1_micro", "f1_macro", "f1_samples", "subset_accuracy", "jaccard_samples",
                "roc_auc_macro", "pr_auc_macro"):
        assert m[key] == pytest.approx(1.0), key
    assert m["hamming_loss"] == 0.0


@pytest.mark.filterwarnings("ignore:No positive class found")  # label 3 has no positive: expected here
def test_metrics_on_a_hand_computed_example():
    y_true = np.array([[1, 0, 0], [1, 1, 0]])
    y_pred = np.array([[1, 1, 0], [1, 1, 0]])   # sentence 1: one extra label; sentence 2: exact
    m = compute_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["subset_accuracy"] == pytest.approx(0.5)            # 1 of 2 label sets fully right
    assert m["hamming_loss"] == pytest.approx(1 / 6)             # 1 wrong cell of 6
    assert m["f1_samples"] == pytest.approx((2 / 3 + 1.0) / 2)   # F1 per sentence, averaged
    assert m["jaccard_samples"] == pytest.approx((1 / 2 + 1.0) / 2)
    assert m["f1_micro"] == pytest.approx(2 * 3 / (2 * 3 + 1))   # TP = 3, FP = 1, FN = 0


def test_aggregate_fold_metrics_ignores_missing_values():
    folds = [{"metrics": {"f1_samples": 0.8, "hamming_loss": 0.02}},
             {"metrics": {"f1_samples": 0.6}}]                   # an older fold without hamming_loss
    avg, std = aggregate_fold_metrics(folds)
    assert avg["f1_samples"] == pytest.approx(0.7)
    assert std["f1_samples"] == pytest.approx(0.1)
    assert avg["hamming_loss"] == pytest.approx(0.02)


# ---------------------------------------------------------------- thresholds

def _separable_probs(seed=0):
    """Label 0 is separable at 0.3, label 1 at 0.7."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(200, 2))
    probs = np.where(y == 1, rng.uniform(0.32, 0.5, y.shape), rng.uniform(0.0, 0.28, y.shape))
    probs[:, 1] = np.where(y[:, 1] == 1, rng.uniform(0.72, 1.0, 200), rng.uniform(0.5, 0.68, 200))
    return y, probs


def test_per_label_thresholds_find_the_separating_value():
    y, probs = _separable_probs()
    thr = optimize_thresholds_per_label(y, probs)
    for j in range(2):
        highest_negative = probs[y[:, j] == 0, j].max()
        lowest_positive = probs[y[:, j] == 1, j].min()
        assert highest_negative < thr[j] <= lowest_positive
    pred = (probs >= thr).astype(int)
    assert (pred == y).all()


def test_samples_threshold_search_never_lowers_f1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(150, 4))
    probs = np.clip(y * 0.4 + rng.uniform(0, 0.6, y.shape), 0, 1)
    base = optimize_thresholds_per_label(y, probs)
    tuned = optimize_thresholds_samples(y, probs, base=base)

    def f1s(t):
        return compute_metrics(y, (probs >= t).astype(int), probs)["f1_samples"]

    assert f1s(tuned) >= f1s(base) - 1e-12
    assert f1s(tuned) >= f1s(np.full(4, 0.5)) - 1e-12


# ---------------------------------------------------------------- loss

def test_asymmetric_loss_without_focusing_equals_bce():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()
    asl = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert asl.item() == pytest.approx(bce.item(), rel=1e-5)


def test_asymmetric_loss_down_weights_easy_negatives():
    easy_negative = torch.tensor([[-4.0]])           # p ≈ 0.018, target 0
    target = torch.zeros(1, 1)
    asl = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)(easy_negative, target)
    bce = F.binary_cross_entropy_with_logits(easy_negative, target)
    assert asl.item() < bce.item() * 1e-3


def test_asymmetric_loss_keeps_positive_loss():
    logits = torch.tensor([[0.0]])                    # p = 0.5, target 1: gamma_pos = 0 -> plain log loss
    loss = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)(logits, torch.ones(1, 1))
    assert loss.item() == pytest.approx(np.log(2), rel=1e-5)


# ---------------------------------------------------------------- sampler

def test_weighted_sampler_favours_rare_labels():
    labels = np.array([[1, 0]] * 9 + [[0, 1]], dtype=np.float32)   # label 1 is 9x rarer
    sampler = make_sampler_if_needed(labels, {"sampler": {"enabled": True, "type": "weighted"}})
    w = sampler.weights.numpy()
    assert w[-1] / w[0] == pytest.approx(9.0, rel=1e-2)
    assert len(sampler) == len(labels)


def test_sampler_is_off_when_disabled():
    labels = np.ones((4, 2), dtype=np.float32)
    assert make_sampler_if_needed(labels, {"sampler": {"enabled": False}}) is None
    assert make_sampler_if_needed(labels, {}) is None


# ---------------------------------------------------------------- command-line overrides

def _args(**kw):
    base = dict(model=None, batch_size=None, grad_accum=None, resume=False, pooling=None, output_dir=None)
    base.update(kw)
    return Namespace(**base)


def test_cli_overrides(main_config):
    import copy

    cfg = apply_cli_overrides(copy.deepcopy(main_config),
                              _args(model="microsoft/unixcoder-base", batch_size=8, pooling="cls"))
    assert cfg["model_name"] == cfg["tokenizer_name"] == "microsoft/unixcoder-base"
    assert cfg["peft"]["target_modules"] == "auto"
    assert cfg["train_params"]["batch_size"] == 8
    assert cfg["train_params"]["eval_batch_size"] <= 16
    assert cfg["pooling"] == "cls"
    assert cfg["logging"]["output_dir"] == "runs/unixcoder-base"
    assert cfg["final_training"]["save_path"] == "runs/unixcoder-base/final_model.pt"


def test_cli_without_overrides_keeps_config(main_config):
    import copy

    cfg = apply_cli_overrides(copy.deepcopy(main_config), _args())
    assert cfg == main_config


# ---------------------------------------------------------------- demo decision rules

LABELS = ["summary_java", "pointer", "summary_python"]


def test_decide_labels_uses_thresholds_and_language_mask():
    probs = np.array([[0.9, 0.8, 0.95]])
    preds = decide_labels(probs, np.full(3, 0.5), LABELS, allowed={"summary_java", "pointer"}, at_least_one=True)
    assert preds == [["summary_java", "pointer"]]   # summary_python is outside the file's language


def test_decide_labels_at_least_one():
    probs = np.array([[0.2, 0.4, 0.9]])
    allowed = {"summary_java", "pointer"}
    assert decide_labels(probs, np.full(3, 0.5), LABELS, allowed, at_least_one=False) == [[]]
    assert decide_labels(probs, np.full(3, 0.5), LABELS, allowed, at_least_one=True) == [["pointer"]]


@pytest.mark.parametrize(
    "pred, expected, status",
    [(["a", "b"], ["b", "a"], "exact"), (["a"], ["a", "b"], "partial"), (["c"], ["a"], "wrong"), (["a"], None, None)],
)
def test_match_status(pred, expected, status):
    assert match_status(pred, expected) == status
