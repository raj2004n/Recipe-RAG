import logging
from unittest.mock import patch

import pytest

from src.runtime.generator import Generator


def _generator(adapter_path, tokenizer=None):
    return Generator(
        model_id="test/model",
        max_input_tokens=100,
        max_new_tokens=10,
        truncation=True,
        return_tensors="pt",
        adapter_path=adapter_path,
        tokenizer=tokenizer or object(),
    )


@patch("src.runtime.generator.PeftModel.from_pretrained")
@patch("src.runtime.generator.AutoModelForCausalLM.from_pretrained")
def test_adapter_path_reaches_peft(mock_from_pretrained, mock_peft, tmp_path):
    """The configured adapter directory must be the one peft loads."""
    adapter_dir = tmp_path / "epoch_02"
    adapter_dir.mkdir()

    _generator(str(adapter_dir))

    base_model = mock_from_pretrained.return_value
    args, _ = mock_peft.call_args
    assert args[0] is base_model
    assert args[1] == str(adapter_dir)


@patch("src.runtime.generator.PeftModel.from_pretrained")
@patch("src.runtime.generator.AutoModelForCausalLM.from_pretrained")
def test_missing_adapter_path_raises_before_peft(
    mock_from_pretrained, mock_peft, tmp_path
):
    """A path that is not a directory fails naming the path, not inside peft."""
    missing = tmp_path / "not_there"

    with pytest.raises(FileNotFoundError, match="not_there"):
        _generator(str(missing))

    mock_peft.assert_not_called()


@patch("src.runtime.generator.PeftModel.from_pretrained")
@patch("src.runtime.generator.AutoModelForCausalLM.from_pretrained")
def test_no_adapter_leaves_the_base_model_alone(
    mock_from_pretrained, mock_peft
):
    """Without an adapter configured, peft is never involved."""
    _generator(None)

    mock_peft.assert_not_called()


@patch("src.runtime.generator.PeftModel.from_pretrained")
@patch("src.runtime.generator.AutoModelForCausalLM.from_pretrained")
def test_load_logs_base_and_adapter(
    mock_from_pretrained, mock_peft, tmp_path, caplog
):
    """The load line names both weights, so a served answer is traceable."""
    adapter_dir = tmp_path / "epoch_02"
    adapter_dir.mkdir()

    with caplog.at_level(logging.INFO, logger="src.runtime.generator"):
        _generator(str(adapter_dir))

    assert "test/model" in caplog.text
    assert str(adapter_dir) in caplog.text
