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

    device = 'cuda:0'
    model_path = "checkpoints/ResNet/model_best.pth"
    classes_focal_torch = torch.Tensor(classes_focal).to(device)
    classes_distortion_torch = torch.Tensor(classes_distortion).to(device)
    
    dataset = ImageDataset("../dataset/test66k/", 25000, classes_focal, classes_distortion)
    model = torch.load(model_path).to(device)
    test_data = DataLoader(dataset, batch_size = 512, shuffle=True)

    # error = 0
    # max_error = 0
    focal_all = 0
    distortion_all = 0
    focal_error_all = 0
    distortion_error_all = 0
    with torch.no_grad():
        for x, y_focal, y_distortion in tqdm(test_data):
            x = x.to(device)
            y_focal = y_focal.to(device)
            y_distortion = y_distortion.to(device)

            focal, distortion = model(x)
            focal_class = torch.argmax(focal, dim = 1)
            distortion_class = torch.argmax(distortion, dim = 1)
            # error = (torch.abs(focal_class - y_focal) + torch.abs(distortion_class - y_distortion)).sum()
            # max_error = torch.max(y_focal, len(classes_focal) - y_focal).sum() + torch.max(distortion_class, len(classes_distortion) - y_focal).sum()
            
            focal_error_all += (torch.abs(focal_class - y_focal) / torch.max(y_focal, len(classes_focal) - y_focal)).sum()
            distortion_error_all += (torch.abs(distortion_class - y_distortion) / torch.max(y_distortion, len(classes_distortion) - y_distortion)).sum()
            focal_all += torch.abs(classes_focal_torch[focal_class] - classes_focal_torch[y_focal]).sum()
            distortion_all += torch.abs(classes_distortion_torch[distortion_class] - classes_distortion_torch[y_distortion]).sum()
    
    print(f"focal error is: {(focal_error_all / len(dataset)):.4f}")
    print(f"distortion error is: {(distortion_error_all / len(dataset)):.4f}")
    print(f"mean abs focal diff is: {(focal_all / len(dataset)):.4f}")
    print(f"mean abs distortion diff is: {(distortion_all / len(dataset)):.4f}")
    # print(focal_acc / len(dataset), distortion_acc / len(dataset), total_acc / len(dataset))
