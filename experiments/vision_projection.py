import torch
import torch.nn as nn


class VisionProjection(nn.Module):

    def __init__(
        self,
        siglip_dim=768,
        gemma_dim=2304
    ):
        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                siglip_dim,
                gemma_dim
            ),

            nn.GELU(),

            nn.Linear(
                gemma_dim,
                gemma_dim
            )
        )

    def forward(self, visual_tokens):

        return self.projection(
            visual_tokens
        )