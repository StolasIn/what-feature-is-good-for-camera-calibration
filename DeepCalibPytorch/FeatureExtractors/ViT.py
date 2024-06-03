import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViT(nn.Module):
    def __init__(
        self,
        device
    ):
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(device)
        self.model.heads = nn.Identity()
        self.feature_size = 768
    
    def forward(self, x):
        return self.model(x)