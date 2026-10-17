import torch
from unittest.mock import patch

from src.runtime.generator import Generator
from src.runtime.reranker import CrossEncoderReranker


@patch("src.runtime.generator.AutoModelForCausalLM.from_pretrained")
def test_generator_loads_in_bfloat16(mock_from_pretrained):
    """The generator must load the base model with bf16 weights."""
    Generator(
        model_id="test/model",
        max_input_tokens=100,
        max_new_tokens=10,
        truncation=True,
        return_tensors="pt",
        tokenizer=object(),
    )
    assert mock_from_pretrained.call_args.kwargs["torch_dtype"] is torch.bfloat16


@patch("src.runtime.reranker.CrossEncoder")
def test_reranker_loads_in_bfloat16(mock_cross_encoder):
    """The cross-encoder must receive bf16 via model_kwargs."""
    reranker = CrossEncoderReranker(model_id="test/reranker", max_length=128)
    reranker._ensure_model()
    kwargs = mock_cross_encoder.call_args.kwargs
    assert kwargs["model_kwargs"]["torch_dtype"] is torch.bfloat16
