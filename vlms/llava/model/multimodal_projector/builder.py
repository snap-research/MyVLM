import re

import torch.nn as nn

from myvlm.myvlm_layer import MyVLMLayer


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class VisionProjector(nn.Module):

    def __init__(self, projector: nn.Sequential):
        super().__init__()
        self.linear1 = projector[0]
        self.act = projector[1]
        self.linear2 = projector[2]

    def forward(self, x, concept_signals=None):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out, concept_signals) if type(self.linear2) == MyVLMLayer else self.linear2(out)
        return out

    @property
    def config(self):
        return self.projector.config


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # return VisionProjector(mlp_depth=mlp_depth, config=config)
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
