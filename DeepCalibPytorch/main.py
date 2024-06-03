import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import ImageDataset
from Model import Model
import matplotlib.pyplot as plt
import random
import os
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def new_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

if __name__ == '__main__':
    set_seed(2024)
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 90 + 1, 4) / 100.)
    device = "cuda:0"
    folder = "checkpoints/ResNet_freeze/"
    new_folder(folder)
    extractor_name = "ResNet"

    n_epochs = 100
    model = Model(extractor_name, len(classes_focal), len(classes_distortion), device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    train_dataset = ImageDataset("../dataset/train500k/", 50000, classes_focal, classes_distortion, data_augmentation=True)
    val_dataset = ImageDataset("../dataset/valid10k/", 10000, classes_focal, classes_distortion, data_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    total_losses = []
    total_errors = []

    par = tqdm(range(n_epochs))
    for epoch in par:
        model.train()
        losses = []
        for x, y_focal, y_distortion in train_loader:
            x = x.to(device)
            y_focal = y_focal.to(device)
            y_distortion = y_distortion.to(device)

            focal, distortion = model(x)
            loss = criterion(focal, y_focal) + criterion(distortion, y_distortion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        scheduler.step()
        par.set_description(f"epoch: {epoch+1}, lr: {scheduler.get_last_lr()}, loss: {sum(losses) / len(losses)}")
        total_losses.append(sum(losses) / len(losses))
        
        if (epoch + 1) % 5 == 0:
            print("\nevaluation...")
            model.eval()
            error = 0
            total_error = 0
            min_error = 1e9
            for x, y_focal, y_distortion in val_loader:
                x = x.to(device)
                y_focal = y_focal.to(device)
                y_distortion = y_distortion.to(device)

                focal, distortion = model(x)
                focal_class = torch.argmax(focal, dim = 1)
                distortion_class = torch.argmax(distortion, dim = 1)
                error += (torch.abs(focal_class - y_focal) + torch.abs(distortion_class - y_distortion)).sum()
                total_error += torch.max(y_focal, len(classes_focal) - y_focal).sum()
                total_error += torch.max(distortion_class, len(classes_distortion) - y_focal).sum()

            print(f"validation error: {error / total_error}")
            total_errors.append((error / total_error).item())
            if min_error > (error / total_error):
                print("model saved...")
                min_error = (error / total_error)
                torch.save(model, f"{folder}/model_best.pth")            

        plt.clf()
        plt.plot(range(len(total_losses)), total_losses)
        plt.savefig(f"{folder}/loss_curve.png")
        plt.clf()
        plt.plot(np.arange(len(total_errors)) * 5, total_errors)
        plt.savefig(f"{folder}/error_curve.png")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model, f"{folder}/model_{epoch + 1}.pth")

    torch.save(model, f"{folder}/model_final.pth")