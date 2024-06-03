import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, InceptionOutputs

class ResNet(nn.Module):
    def __init__(
        self,
        device
    ):
        super(ResNet, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        self.model.fc = nn.Identity()
        self.feature_size = 2048
    
    def forward(self, x):
        return self.model(x)