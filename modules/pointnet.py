import torch
from torch import nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim