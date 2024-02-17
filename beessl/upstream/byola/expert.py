import os
import torch
import torchaudio
import torch.nn as nn
from beessl.upstream.byola.augmentations import PrecomputedNorm
from beessl.upstream.byola.models import AudioNTT2020Task6X

class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict, ckpt:str=None):
        super().__init__()
        self.cfg = model_config
        self.stats = [-5.4919195,  5.0389895]
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.f_min,
            f_max=self.cfg.f_max,
        )
        self.normalizer = PrecomputedNorm(self.stats)

        # Load pretrained weights.
        self.feature_extractor = AudioNTT2020Task6X(n_mels=self.cfg.n_mels, d=self.cfg.feature_d)

        if ckpt is not None:
            if os.path.isfile(ckpt):
                self.feature_extractor.load_weight(ckpt, "cpu")

    def forward(self, wavs, lens=None):
        x = self.normalizer((self.to_melspec(wavs) + torch.finfo(torch.float).eps).log())
        feats = self.feature_extractor(x.unsqueeze(1), layered=False)
        # feats = self.feature_extractor.by_layers(feats)
        return {
            "hidden_states": [feats.transpose(1, 2)],
        }
