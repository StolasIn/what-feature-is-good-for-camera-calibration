import torch
import torch.nn as nn
from FeatureExtractors.Inception import Inception

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        extractor_name, # [Inception, ResNet, Swin, VGG, ViT]
        device
    ):
        super(FeatureExtractor, self).__init__()
        
        if extractor_name == "Inception":
            self.model = Inception(device)
        else:
            raise NotImplementedError
        
    def get_feature_size(self):
        return self.model.feature_size
    
    def forward(self, x):
        return self.model(x)