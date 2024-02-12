import torch
import torch.nn as nn
from beessl.upstream.beeyol.model import BeeYOL

class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict, ckpt:str=None):
        super().__init__()
        self.extractor = BeeYOL(model_config)
        if ckpt:
            self.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def forward(self, wavs, lens=None):
        feats = self.extractor(wavs)
        return {
            "hidden_states": feats,
        }
