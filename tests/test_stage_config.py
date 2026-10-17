import pytest

from src.core.config import (
    BASE_CONFIG,
    BASE_CONFIG_FILE,
    ENCODER_CONFIG,
    PATHS_CONFIG_FILE,
    STAGE_CONFIG_KEYS,
    UndeclaredConfigKey,
    stage_config,
)


def test_declared_read_returns_the_configured_value():
    config = stage_config("transform")
    assert config.get(BASE_CONFIG_FILE, "encoder.model_id") == ENCODER_CONFIG["model_id"]
    assert config.get(BASE_CONFIG_FILE, "encoder.batch_size") == ENCODER_CONFIG["batch_size"]
    assert (
        config.get(BASE_CONFIG_FILE, "encoder.normalize_embeddings")
        == ENCODER_CONFIG["normalize_embeddings"]
    )


def test_a_key_outside_the_row_is_refused():
    with pytest.raises(UndeclaredConfigKey, match="retriever.top_k"):
        stage_config("transform").get(BASE_CONFIG_FILE, "retriever.top_k")


def test_a_file_outside_the_row_is_refused():
    with pytest.raises(UndeclaredConfigKey, match="prompts_config.yaml"):
        stage_config("transform").get("prompts_config.yaml", "generate")


def test_chunk_reads_the_base_config_pointer_but_nothing_in_the_file():
    """`chunk` carries the `configs` block only, and refuses the base config itself.

    The row names no file of that name, so the refusal is the file's rather than
    one key's, and the message says which file the stage never declared.
    """
    config = stage_config("chunk")
    assert config.get(PATHS_CONFIG_FILE, "configs")
    with pytest.raises(UndeclaredConfigKey, match="base_config.yaml"):
        config.get(BASE_CONFIG_FILE, "rag.strategy")


@pytest.mark.parametrize(
    ("stage", "file", "key"),
    [
        (stage, file, key)
        for stage, row in STAGE_CONFIG_KEYS.items()
        for file, keys in row.items()
        for key in keys
    ],
)
def test_every_declared_key_exists_in_its_config(stage, file, key):
    """A row naming a key the config does not carry would fail at stage start."""
    assert stage_config(stage).get(file, key) is not None


def test_set_strategy_declares_the_three_keys_it_reads():
    config = stage_config("set_strategy")
    assert config.get(BASE_CONFIG_FILE, "rag.strategy") == BASE_CONFIG["rag"]["strategy"]
    assert config.get(PATHS_CONFIG_FILE, "data.cleaned_recipes")
    assert config.get(PATHS_CONFIG_FILE, "configs")


def test_the_base_config_pointer_is_declared_by_every_stage():
    """Repointing `configs.base_config` has to invalidate every stage, not just two."""
    for stage in STAGE_CONFIG_KEYS:
        assert stage_config(stage).get(PATHS_CONFIG_FILE, "configs")
