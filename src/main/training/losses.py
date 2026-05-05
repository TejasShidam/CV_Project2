from __future__ import annotations

import torch
from torch import nn


def cross_entropy_sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    # logits: [B, T-1, V], targets: [B, T]
    target_out = targets[:, 1:]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))


def ctc_sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    # logits expected [T, B, V] for CTC.
    log_probs = logits.log_softmax(dim=-1)
    input_lengths = torch.full(
        size=(log_probs.size(1),),
        fill_value=log_probs.size(0),
        dtype=torch.long,
        device=log_probs.device,
    )
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    return criterion(log_probs, targets, input_lengths, target_lengths)
