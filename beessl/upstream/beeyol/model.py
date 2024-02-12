import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BeeYOLConfig:
    feature_extractor: object
    conv_extractor: nn.Module
    encoder: nn.Module


class BeeYOL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = BeeYOLConfig(**config)
        self.feature_extractor = self.config.feature_extractor
        self.conv_extractor = self.config.conv_extractor
        self.encoder = self.config.encoder

    def forward(self, wavs):
        wavs = wavs.unsqueeze(-1)
        feats = self.conv_extractor(wavs)
        if self.encoder:
            feats = self.encoder(feats)
        else:
            feats = feats.transpose(1, 2)

        return [feats]
