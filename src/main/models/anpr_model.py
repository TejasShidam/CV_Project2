from __future__ import annotations

import torch
from torch import nn

from anpr.models.cnn_backbone import CNNBackbone
from anpr.models.transformer_decoder import TransformerTextDecoder


class ANPRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        max_label_length: int = 16,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(out_dim=d_model)
        self.decoder = TransformerTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=8,
            num_layers=3,
            max_len=max_label_length,
        )
        self.max_label_length = max_label_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        batch, channels, height, width = feat.shape
        memory = feat.view(batch, channels, height * width).permute(0, 2, 1)
        return memory

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self.encode_image(image)
        decoder_input = target[:, :-1]
        return self.decoder(decoder_input, memory)

    @torch.no_grad()
    def greedy_decode(self, image: torch.Tensor) -> torch.Tensor:
        memory = self.encode_image(image)
        batch_size = image.size(0)
        tokens = torch.full(
            (batch_size, 1),
            fill_value=self.sos_id,
            dtype=torch.long,
            device=image.device,
        )

        for _ in range(self.max_label_length - 1):
            logits = self.decoder(tokens, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == self.eos_id).all():
                break

        return tokens
