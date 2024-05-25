import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import ImageDataset
from Model import Model

if __name__ == '__main__':
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 120 + 1, 2) / 100.)
    device = 'cuda:0'

    n_epochs = 100
    model = Model(len(classes_focal), len(classes_distortion))
    model.to(device)
    dataset = ImageDataset("test/", classes_focal, classes_distortion)
    train_data = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

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
        par.set_description(f"loss: {sum(losses) / len(losses)}")
    torch.save(model, "model.pth")