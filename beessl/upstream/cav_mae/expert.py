import os
import torch
import torch.nn as nn
import speechbrain as sb
from beessl.upstream.cav_mae.model import CAVMAE


class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict, ckpt:str=None):
        super().__init__()
        # all CAV_MAE models are trained with 10s audio
        # all models trained with 11 modality-specific layers and 1 shared layer
        self.model = CAVMAE(**model_config)

        weights = torch.load(ckpt, map_location="cpu")
        audio_model = torch.nn.DataParallel(self.model)
        audio_model.load_state_dict(weights, strict=False)

    def forward(self, wavs, lens=None):
        feats = self.model.forward_audio(wavs)
        return {"hidden_states": feats}
