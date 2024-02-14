import os
import torch
import torch.nn as nn
import speechbrain as sb
from beessl.upstream.beeyol.model import BeeYOL

class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict, ckpt:str=None):
        super().__init__()
        self.extractor = BeeYOL(model_config)
        if ckpt is not None:
            print(f"[UpstreamExpert] Loading checkpoint from {ckpt}")
            if os.path.isfile(ckpt):
                self.load_state_dict(torch.load(ckpt, map_location="cpu"))

            elif os.path.isdir(ckpt):
                checkpointer = sb.utils.checkpoints.Checkpointer(
                    checkpoints_dir=ckpt,
                    recoverables={'student': self}
                )
                checkpointer.recover_if_possible(min_key="byol_loss")

    def forward(self, wavs, lens=None):
        feats = self.extractor(wavs)
        return {
            "hidden_states": feats,
        }
