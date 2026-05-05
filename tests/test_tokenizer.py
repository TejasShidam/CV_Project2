from anpr.tokenizer import PlateTokenizer


def test_tokenizer_roundtrip() -> None:
    tokenizer = PlateTokenizer("ABC123")
    tokens = tokenizer.encode("AB12", max_length=8)
    decoded = tokenizer.decode(tokens)
    assert decoded == "AB12"
