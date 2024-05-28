import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class Inception(nn.Module):
    def __init__(
        self,
        device
    ):
        super(Inception, self).__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        self.model.fc = nn.Identity()
        self.feature_size = 2048
    
    def forward(self, x):
        return self.model(x).logits