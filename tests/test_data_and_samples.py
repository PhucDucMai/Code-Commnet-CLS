"""Consistency of the dataset, the configs and the demo sample files. Needs no GPU or model weights."""
import json

import numpy as np
import pytest
from conftest import EXPECTED_LABELS, SAMPLE_DIRS

from inference import LANG_LABELS
from preprocess_comments import detect_lang, extract_comments, normalize_sentence
from train import build_label_matrix, map_labels_for_row


# ---------------------------------------------------------------- dataset

def test_dataset_size_and_languages(dataset):
    assert len(dataset) == 6738
    assert dataset["lang"].value_counts().to_dict() == {"PY": 2555, "JAVA": 2418, "PHARO": 1765}


def test_every_sentence_has_a_label_and_912_are_multi_label(dataset, label_list):
    matrix = build_label_matrix(dataset, label_list, expand_to_19=True)
    per_sentence = matrix.sum(axis=1)
    assert (per_sentence >= 1).all()
    assert int((per_sentence > 1).sum()) == 912


def test_label_frequencies(dataset, label_list):
    counts = dict(zip(label_list, build_label_matrix(dataset, label_list).sum(axis=0).astype(int)))
    assert counts["usage_java"] == 912          # most frequent label
    assert counts["classreferences"] == 77      # rarest label
    assert min(counts.values()) == 77 and max(counts.values()) == 912


def test_normalization_reproduces_dataset_sentences(dataset):
    """Our normalization maps dataset sentences onto themselves.

    Known exceptions: chained dotted names such as "java^net.bindexception", where the dataset replaced only
    some of the in-word dots, and 8 cells that a spreadsheet converted to values ("TRUE", "FALSE", "12:00 AM")."""
    spreadsheet_values = {"TRUE", "FALSE", "12:00 AM", "2:00 AM"}
    sentences = dataset["sentence"].tolist()
    changed = [(s, normalize_sentence(s)) for s in sentences if normalize_sentence(s) != s]
    assert len(changed) / len(sentences) < 0.03

    def only_dots_differ(a, b):
        return len(a) == len(b) and all(x == y or (x == "." and y == "^") for x, y in zip(a, b))

    unexplained = [(s, n) for s, n in changed if not (only_dots_differ(s, n) or s in spreadsheet_values)]
    assert unexplained == []


# ---------------------------------------------------------------- labels and configs

def test_config_labels_match_language_groups(label_list):
    assert len(label_list) == 19 == len(set(label_list))
    assert set(label_list) == {lab for labs in LANG_LABELS.values() for lab in labs}
    assert [len(LANG_LABELS[lang]) for lang in ("java", "python", "pharo")] == [7, 5, 7]


@pytest.mark.parametrize(
    "labels, lang, expected",
    [
        ("summary;pointer", "java", ["summary_java", "pointer"]),
        ("usage; Expand ", "python", ["usage_python", "expand_python"]),
        ("intent", "pharo", ["intent"]),
        ("", "java", []),
    ],
)
def test_shared_label_names_are_split_by_language(labels, lang, expected):
    assert map_labels_for_row(labels, lang, expand_to_19=True) == expected


def test_ablation_config_differs_from_main_config_only_where_intended(main_config):
    import yaml
    from conftest import ROOT

    with open(ROOT / "configs" / "ablation_cls_pooling.yaml") as f:
        ablation = yaml.safe_load(f)
    differing = {k for k in set(main_config) | set(ablation) if main_config.get(k) != ablation.get(k)}
    assert differing == {"pooling", "logging", "final_training"}
    assert (main_config["pooling"], ablation["pooling"]) == ("mean", "cls")
    other_logging = {k for k in main_config["logging"] if main_config["logging"][k] != ablation["logging"].get(k)}
    assert other_logging == {"output_dir"}


# ---------------------------------------------------------------- demo sample files

SAMPLE_FILES = sorted(p for d in SAMPLE_DIRS for p in d.iterdir() if detect_lang(p.name))


@pytest.fixture(scope="module")
def expected_labels():
    with open(EXPECTED_LABELS) as f:
        return json.load(f)["files"]


def test_there_are_18_sample_files():
    assert len(SAMPLE_FILES) == 18


@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_file_is_parsed(path):
    items = extract_comments(path.read_text(encoding="utf-8"), path.name)
    if path.name == "NoClassComment.java":
        assert items == []
    else:
        assert items, f"no class comment extracted from {path.name}"
        assert all(it.sentence and it.lang == detect_lang(path.name) for it in items)


@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_expected_labels_refer_to_extracted_sentences(path, expected_labels):
    """Every annotated sentence must still be produced by the extractor, with labels of the file's language."""
    expected = expected_labels.get(path.name, {})
    sentences = {it.sentence for it in extract_comments(path.read_text(encoding="utf-8"), path.name)}
    assert set(expected) <= sentences
    allowed = set(LANG_LABELS[detect_lang(path.name)])
    for sentence, labels in expected.items():
        assert labels and set(labels) <= allowed, sentence


def test_expected_labels_cover_all_19_labels(expected_labels):
    used = {lab for per_file in expected_labels.values() for labs in per_file.values() for lab in labs}
    assert len(used) == 19
    assert sum(len(per_file) for per_file in expected_labels.values()) == 180


def test_label_matrix_shape(dataset, label_list):
    matrix = build_label_matrix(dataset.head(10), label_list)
    assert matrix.shape == (10, 19) and matrix.dtype == np.float32
