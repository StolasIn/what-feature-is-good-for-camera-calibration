import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class Swin(nn.Module):
    def __init__(
        self,
        device
    ):
        super(Swin, self).__init__()
        self.model = swin_v2_t(weights = Swin_V2_T_Weights.DEFAULT).eval().to(device)
        print(self.model)
        self.model.head = nn.Identity()
        self.feature_size = 768
    
    def forward(self, x):
        return self.model(x)