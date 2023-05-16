import torch
import torch.nn as nn
import torch.nn.functional as F

class Featurizer(nn.Module):
    def __init__(
            self,
            num_hidden_layers:int=12,
            *args, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.weights_stack = nn.Parameter(
            torch.ones(self.num_hidden_layers)
        )

    def forward(self, x):
        # Perform the weighted sum
        layer_num = len(x)
        stacked_feature = torch.stack(x, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
