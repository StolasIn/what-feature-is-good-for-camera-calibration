import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class Model(nn.Module):
    def __init__(
        self,
        n_focal,
        n_distortion
    ):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()
        in_features = 2048
        self.focal_layer = nn.Linear(in_features, n_focal)
        self.distortion_layer = nn.Linear(in_features, n_distortion)

    def forward(self, x):
        image_feature = self.model(x)
        focal = self.focal_layer(image_feature)
        distortion = self.distortion_layer(image_feature)
        return focal, distortion