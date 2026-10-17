from src.build.tokenizer import Tokenizer


def test_tokenizer_is_deterministic():
    t = Tokenizer()
    text = "Chicken tikka masala with basmati rice."
    assert t.tokenize(text) == t.tokenize(text)


def test_tokenizer_lowercases_and_strips_punctuation():
    t = Tokenizer()
    tokens = t.tokenize("Chicken, Tikka-Masala! (with basmati)")
    assert tokens == ["chicken", "tikka", "masala", "with", "basmati"]


def test_tokenizer_returns_empty_on_empty_input():
    assert Tokenizer().tokenize("") == []
    assert Tokenizer().tokenize("!!! --- ...") == []


def test_tokenizer_keeps_digits_as_tokens():
    t = Tokenizer()
    tokens = t.tokenize("Bake at 180C for 30 minutes")
    assert "180c" in tokens
    assert "30" in tokens
