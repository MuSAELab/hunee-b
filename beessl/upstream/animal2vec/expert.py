import os
import torch
import torch.nn as tnn
import torchaudio
from torchaudio.models import wav2vec2_model
from fairseq import checkpoint_utils

import beessl.upstream.animal2vec.nn as nn


def torch_resample(wavs, sr, target_sr):
    return torchaudio.functional.resample(
        wavs,
        sr,
        target_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


class UpstreamExpert(tnn.Module):
    def __init__(self, ckpt: str = None):
        super().__init__()

        # Import inside the method where it's needed
        import beessl.upstream.animal2vec.nn as nn

        if ckpt is None or not os.path.isfile(ckpt):
            raise ValueError("ckpt file is required")

        self.feature_extractor, self.feature_extractor_args = (
            checkpoint_utils.load_model_ensemble(filenames=[ckpt])
        )
        self.feature_extractor = self.feature_extractor[0]

    def forward(self, wavs, lens=None):
        wavs = torch_resample(wavs, 16000, 8000)

        # Normalize to zero mean and unit variance
        wavs = wavs / wavs.std(dim=-1, keepdim=True)

        feats = self.feature_extractor(source=wavs)["layer_results"]
        feats = [f.transpose(1, 2) for f in feats]

        return {
            "hidden_states": feats,
        }


if __name__ == "__main__":
    expert = UpstreamExpert(
        "/media/heitor/Research/raw_files/animal2vec_large_finetuned_MeerKAT_240507.pt"
    )
    wavs = torch.randn(2, 32000)
    out = expert(wavs)
    print(out["hidden_states"][-1].shape)
    print([hs.shape for hs in out["hidden_states"]])
