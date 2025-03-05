import os
import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model


class UpstreamExpert(nn.Module):
    def __init__(self, model_config: dict, ckpt: str = None):
        super().__init__()
        self.cfg = model_config
        self.feature_extractor = wav2vec2_model(**self.cfg, aux_num_out=None)

        if ckpt is not None:
            if os.path.isfile(ckpt):
                self.feature_extractor.load_state_dict(torch.load(ckpt))
                self.feature_extractor.requires_grad_(False)

    def forward(self, wavs, lens=None):
        feats = self.feature_extractor.extract_features(wavs)[0]
        return {
            "hidden_states": feats,
        }


if __name__ == "__main__":
    from hyperpyyaml import load_hyperpyyaml

    model_config = "hparams/birdaves.yaml"
    with open(model_config, "r") as f:
        model_config = load_hyperpyyaml(f)

    # ckpt = "/media/heitor/Research/raw_files/aves-base-all.torchaudio.pt"
    ckpt = "/media/heitor/Research/raw_files/birdaves-biox-large.torchaudio.pt"
    expert = UpstreamExpert(model_config, ckpt)
    wavs = torch.randn(1, 16000)
    print(expert(wavs))
