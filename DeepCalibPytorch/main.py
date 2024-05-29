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

if __name__ == '__main__':
    set_seed(2024)
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 90 + 1, 4) / 100.)
    device = 'cuda:0'
    extractor_name = "Inception"

    n_epochs = 200
    model = Model(extractor_name, len(classes_focal), len(classes_distortion), device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    dataset = ImageDataset("../dataset/train500k/", 50, classes_focal, classes_distortion)
    train_data = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    total_losses = []

    par = tqdm(range(n_epochs))
    for epoch in par:
        losses = []
        for x, y_focal, y_distortion in train_data:
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
        plt.plot(range(len(total_losses)), total_losses)
        plt.savefig("loss_curve.png")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model, f"model_{epoch + 1}.pth")
    torch.save(model, f"model_final.pth")