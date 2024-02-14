import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

class EMA:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + new * (1 - self.alpha)


class MLP(nn.Module):
    def __init__(
            self,
            in_dim:int,
            hidden_dim:int,
            out_dim:int,
            attentive_pooling:bool=False,
            attention_channels:int=64
        ):
        super().__init__()
        self.attentive_pooling = attentive_pooling
        if self.attentive_pooling:
            self.pooling = AttentiveStatisticsPooling(
                channels=in_dim,
                attention_channels=attention_channels,
                global_context=True
            )
            in_dim = 2 * in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        

    def forward(self, x):
        if self.attentive_pooling:
            x = x.transpose(1, 2)
            x = self.pooling(x).squeeze(-1)
        
        x = self.fc1(x)
        x = self.bn(x)
        x = self.activation(x)
        return self.fc2(x)


def byol_loss(z, z_prime):
    z = F.normalize(z, dim=-1, p=2)
    z_prime = F.normalize(z_prime, dim=-1, p=2)
    loss = 2 - 2 * (z * z_prime).sum(dim=-1)
    
    return loss

