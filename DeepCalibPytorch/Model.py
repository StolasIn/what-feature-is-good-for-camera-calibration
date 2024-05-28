import torch.nn as nn
from FeatureExtractor import FeatureExtractor

class Model(nn.Module):
    def __init__(
        self,
        extractor_name,
        n_focal,
        n_distortion,
        device
    ):
        super().__init__()
        self.model = FeatureExtractor(extractor_name, device)
        self.focal_layer = nn.Linear(self.model.get_feature_size(), n_focal).to(device)
        self.distortion_layer = nn.Linear(self.model.get_feature_size(), n_distortion).to(device)

    def forward(self, x):
        image_feature = self.model(x)
        focal = self.focal_layer(image_feature)
        distortion = self.distortion_layer(image_feature)
        return focal, distortion