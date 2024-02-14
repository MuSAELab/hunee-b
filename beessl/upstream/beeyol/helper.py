# Note that most of the speechbrain models (lobes) return
# a linear projection from the encoder. This helper function
# is used to extract the sequence of features from the model.

import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import TDNNBlock
from speechbrain.lobes.models.ECAPA_TDNN import SERes2NetBlock

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        input_size,
        activation=torch.nn.ReLU,
        channels=[256, 256, 256, 256, 768],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        res2net_scale=8,
        se_channels=128,
        groups=[1, 1, 1, 1, 1],
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

    def forward(self, x, lengths=None):
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        return self.mfa(x).transpose(1,2)