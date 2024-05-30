import os
import torch
import torch.nn as nn
import speechbrain as sb
from beessl.upstream.ssast.model import ASTModel
from beessl.upstream.ssast.audio import FeatureExtractor

FBANK_SAMPLE_STRIDE = 160
SAMPLE_RATE = 16000

class UpstreamExpert(nn.Module):
    def __init__(self, model_config:dict, ckpt:str=None):
        super().__init__()
        self.window_secs = model_config['window_secs']
        self.stride_secs = model_config['window_secs']
        target_length = int(self.window_secs * SAMPLE_RATE / FBANK_SAMPLE_STRIDE)
        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )

        self.tstride = 10
        self.model = ASTModel(
            tstride=self.tstride,
            input_tdim=target_length,
            load_pretrained_mdl_path=ckpt,
            **model_config
        )
        self.vertical_num_patches = (128 - 16) // 10 + 1  # 12
        self.model = self.model.cpu()

    def get_downsample_rates(self, key: str = None) -> int:
        return int(FBANK_SAMPLE_STRIDE * self.tstride)

    def forward(self, wavs, lens=None):
        max_wav_len = wavs.size(-1)
        start_points = list(range(0, max_wav_len, int(self.stride_secs * SAMPLE_RATE)))
        if max_wav_len - start_points[-1] < (int(self.window_secs * SAMPLE_RATE) // 2):
            start_points.pop()

        all_features = []
        for start in start_points:
            subwavs = [
                wav[start : start + int(self.window_secs * SAMPLE_RATE)]
                for wav in wavs
            ]
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in subwavs]
            features = torch.stack(
                features, dim=0
            )  # (batch_size, segment_seq_len, hidden_size)
            all_features.append(features)

        all_features = torch.stack(all_features, dim=0)
        num_segment, batch_size, segment_seq_len, hidden_size = all_features.shape

        flatten_features = all_features.reshape(-1, segment_seq_len, hidden_size)
        hidden_states, _ = self.model(flatten_features)

        reshaped_hidden_states = [
            (
                h.reshape(num_segment, batch_size, -1, h.size(-1))
                .transpose(
                    0, 1
                )  # (batch_size, num_segment, num_horizon_patch, num_vertical_patch * hidden_size)
                .flatten(
                    1, 2
                )  # (batch_size, num_segment * num_horizon_patch, num_vertical_patch * hidden_size)
                .transpose(
                    1, 2
                ) # B x F x T
                .float()
            )
            for h in hidden_states
        ]

        return {"hidden_states": reshaped_hidden_states}
