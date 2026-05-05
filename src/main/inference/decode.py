from __future__ import annotations

from anpr.tokenizer import PlateTokenizer


def decode_batch(tokens, tokenizer: PlateTokenizer) -> list[str]:
    return [tokenizer.decode(row) for row in tokens]
