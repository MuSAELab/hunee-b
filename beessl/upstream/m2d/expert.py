import torch
import torch.nn as nn
from beessl.upstream.m2d.m2d.runtime_audio import RuntimeM2D


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt:str=None):
        super().__init__()
        # The expected ckpt is in the format "80x512p16x16_0425/checkpoint-100.pth"
        self.model = RuntimeM2D(weight_file=ckpt)

    def forward(self, wavs, lens=None):
        batch_lms = self.model.to_feature(wavs)
        batch_lms = (batch_lms - batch_lms.mean()) / (batch_lms.std() + torch.finfo().eps)
        frame_level = self.model.encode_lms(batch_lms, return_layers=True)
        frame_level = [f.transpose(1, 2) for f in frame_level]

        return {"hidden_states": frame_level}
