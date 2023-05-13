import torch
import torch.nn as nn

class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict):
        super().__init__()
        self.extractor = model_config["extractor"]
        self.delta = model_config.get("delta")

    def forward(self, wavs, lens=None):
        feats = self.extractor(wavs)
        if self.delta:
            d = self.delta(feats)
            dd = self.delta(d)
            feats = torch.cat([feats, d, dd], dim=1)

        return {
            "hidden_states": feats,
        }
