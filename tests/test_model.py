"""TransformerClassifier with a tiny randomly initialised RoBERTa saved locally: no download, no GPU."""
import pytest
import torch

from train import TransformerClassifier, load_classifier_state, trainable_state_dict

NUM_LABELS = 19


@pytest.fixture(scope="module")
def tiny_roberta(tmp_path_factory):
    from transformers import RobertaConfig, RobertaModel

    config = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, pad_token_id=1)
    path = tmp_path_factory.mktemp("tiny_roberta")
    torch.manual_seed(0)
    RobertaModel(config).save_pretrained(path)
    return str(path)


def _model(path, **kw):
    torch.manual_seed(0)
    return TransformerClassifier(path, NUM_LABELS, **kw).eval()


def _batch():
    input_ids = torch.tensor([[0, 5, 6, 7, 2, 1, 1, 1],
                              [0, 8, 9, 2, 1, 1, 1, 1]])
    attention_mask = (input_ids != 1).long()
    return input_ids, attention_mask


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_output_shape(tiny_roberta, pooling):
    model = _model(tiny_roberta, pooling=pooling)
    with torch.no_grad():
        logits = model(*_batch())
    assert logits.shape == (2, NUM_LABELS)
    assert torch.isfinite(logits).all()


def test_invalid_pooling_is_rejected(tiny_roberta):
    with pytest.raises(ValueError):
        TransformerClassifier(tiny_roberta, NUM_LABELS, pooling="max")


def test_mean_pooling_ignores_padding(tiny_roberta):
    """Changing the token ids at padded positions must not change the output."""
    model = _model(tiny_roberta, pooling="mean")
    input_ids, mask = _batch()
    other = input_ids.clone()
    other[mask == 0] = 42
    with torch.no_grad():
        assert torch.allclose(model(input_ids, mask), model(other, mask), atol=1e-5)


def test_mean_and_cls_pooling_differ(tiny_roberta):
    mean_model = _model(tiny_roberta, pooling="mean")
    cls_model = _model(tiny_roberta, pooling="cls")
    cls_model.load_state_dict(mean_model.state_dict())   # same weights, only the pooling differs
    with torch.no_grad():
        assert not torch.allclose(mean_model(*_batch()), cls_model(*_batch()))


def test_lora_trains_only_adapters_and_head(tiny_roberta):
    model = _model(tiny_roberta, use_lora=True, lora_r=4, lora_alpha=8,
                   lora_target_modules=["query", "key", "value", "dense"])
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in n or n.startswith("classifier.") for n in trainable)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_train < n_total


def test_unknown_lora_targets_fall_back_to_auto(tiny_roberta):
    model = _model(tiny_roberta, use_lora=True, lora_r=4, lora_target_modules=["does_not_exist"])
    assert model.lora_targets == ["query", "key", "value", "dense"]


def test_trainable_checkpoint_round_trip(tiny_roberta):
    """save_trainable_only checkpoints (LoRA + head) restore the same predictions."""
    kw = dict(use_lora=True, lora_r=4, lora_alpha=8, lora_target_modules=["query", "key", "value", "dense"])
    trained = _model(tiny_roberta, **kw)
    with torch.no_grad():
        for p in trained.parameters():
            if p.requires_grad:
                p.add_(0.05)                               # pretend training changed the weights
        expected = trained(*_batch())
    state = trainable_state_dict(trained)

    restored = _model(tiny_roberta, **kw)
    load_classifier_state(restored, state)
    with torch.no_grad():
        assert torch.allclose(restored(*_batch()), expected, atol=1e-6)


def test_checkpoint_from_another_model_is_rejected(tiny_roberta):
    model = _model(tiny_roberta, pooling="mean")
    with pytest.raises(RuntimeError):
        load_classifier_state(model, {"not_a_real.weight": torch.zeros(1)})
