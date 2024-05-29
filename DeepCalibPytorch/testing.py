import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import ImageDataset
from Model import Model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 90 + 1, 4) / 100.)
    device = 'cuda:1'
    model_path = "model_80.pth"
    
    dataset = ImageDataset("../dataset/train500k/", 50000, classes_focal, classes_distortion)
    model = torch.load(model_path).to(device)
    test_data = DataLoader(dataset, batch_size=64, shuffle=True)

    focal_acc = 0
    distortion_acc = 0
    total_acc = 0
    with torch.no_grad():
        for x, y_focal, y_distortion in tqdm(test_data):
            x = x.to(device)
            y_focal = y_focal.to(device)
            y_distortion = y_distortion.to(device)

            focal, distortion = model(x)
            focal_class = torch.argmax(focal, dim = 1)
            distortion_class = torch.argmax(distortion, dim = 1)
            focal_acc += (y_focal == focal_class).sum()
            distortion_acc += (y_distortion == distortion_class).sum()
            print(focal_class, distortion_class)
            print(y_focal, y_distortion)

            temp_acc = torch.ones_like(focal_class)
            temp_acc[y_focal != focal_class] = 0
            temp_acc[y_distortion != distortion_class] = 0
            total_acc += temp_acc.sum()
        
        print(focal_acc / len(dataset), distortion_acc / len(dataset), total_acc / len(dataset))
