import torch
import torch.nn as nn
from beessl.upstream.beats.beats.BEATs import BEATs, BEATsConfig


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt:str=None):
        super().__init__()
        # load the pre-trained checkpoints
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location="cpu")
        else:
            raise ValueError("UpstreamExpert::Specify the path to the ckpt")

        self.cfg = BEATsConfig(ckpt['cfg'])
        self.model = BEATs(self.cfg)
        self.model.load_state_dict(ckpt['model'])

    def forward(self, wavs, lens=None):
        padding_mask = torch.zeros_like(wavs).bool()
        _, _, hidden_states = self.model.extract_features(wavs, padding_mask=padding_mask) # L x [BxTxF]
        hidden_states = [h.transpose(1, 2) for h in hidden_states] # L x [BxFxT]

        return {"hidden_states": hidden_states}
