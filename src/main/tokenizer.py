from __future__ import annotations

from dataclasses import dataclass

import torch


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


@dataclass
class PlateTokenizer:
    charset: str

    def __post_init__(self) -> None:
        unique_chars = sorted(set(self.charset))
        self.idx_to_token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + unique_chars
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}
        self.pad_id = self.token_to_idx[PAD_TOKEN]
        self.sos_id = self.token_to_idx[SOS_TOKEN]
        self.eos_id = self.token_to_idx[EOS_TOKEN]
        self.unk_id = self.token_to_idx[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        tokens = [self.sos_id]
        for ch in text.upper().strip():
            tokens.append(self.token_to_idx.get(ch, self.unk_id))
        tokens.append(self.eos_id)

        if len(tokens) > max_length:
            tokens = tokens[: max_length - 1] + [self.eos_id]
        if len(tokens) < max_length:
            tokens.extend([self.pad_id] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()

        chars: list[str] = []
        for idx in indices:
            token = self.idx_to_token[idx]
            if token in {PAD_TOKEN, SOS_TOKEN}:
                continue
            if token == EOS_TOKEN:
                break
            chars.append(token)
        return "".join(chars)
