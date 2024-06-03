import torch.nn as nn
from FeatureExtractor import FeatureExtractor
from torch.nn import functional as F

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
        self.focal_layer1 = nn.Linear(2048, n_focal).to(device)
        self.focal_layer2 = nn.Linear(1024, n_focal).to(device)
        self.distortion_layer1 = nn.Linear(2048, n_distortion).to(device)
        self.distortion_layer2 = nn.Linear(1024, n_distortion).to(device)
        self.bn1 = nn.BatchNorm1d(n_focal).to(device)
        self.bn2 = nn.BatchNorm1d(n_distortion).to(device)
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()


    def forward(self, x):
        image_feature = self.model(x)
        # focal = self.focal_layer(image_feature)
        # distortion = self.distortion_layer(image_feature)
        focal_1 = F.relu(self.bn1(self.focal_layer1(image_feature)))
        focal_2 = F.softmax(self.focal_layer2(focal_1))
        distortion_1 = F.relu(self.bn2(self.distortion_layer1(image_feature)))
        distortion_2 = F.softmax(self.distortion_layer2(distortion_1))
        return focal_2, distortion_2
