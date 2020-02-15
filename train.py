import torch
import torch.optim as optim
import numpy as np
import tqdm
import random

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from celluloid import Camera

from model import Model
from dataset import get_dataloader
from loss import SPLLoss


fig = plt.figure()
camera = Camera(fig)


def setup():
    seed = 1445
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    setup()

    model = Model(2, 2)
    dataloader = get_dataloader()
    criterion = SPLLoss(n_samples=len(dataloader.dataset))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):
        for index, data, target in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, index)
            loss.backward()
            optimizer.step()
        criterion.increase_threshold()
        plot(dataloader.dataset, model, criterion)

    animation = camera.animate()
    animation.save("plot.gif")


def find_anomaly():
    setup()

    model = Model(2, 2)
    dataloader = get_dataloader()
    criterion = SPLLoss(n_samples=len(dataloader.dataset))
    optimizer = optim.Adam(model.parameters())

    # for epoch in range(10):
    epoch = 0
    while (criterion.v == 0).sum() > 5:
        for index, data, target in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, index)
            loss.backward()
            optimizer.step()
        criterion.increase_threshold()
        plot(dataloader.dataset, model, criterion)
        epoch += 1

    anomalies = np.where(criterion.v == 0)[0]
    print(anomalies)
    plot_anomalies(dataloader.dataset, model, criterion, anomalies)

    animation = camera.animate()
    animation.save("anomalies.gif")


def plot(dataset, model, criterion):
    x = dataset.X[criterion.v == 1]
    y = dataset.y[criterion.v == 1]

    plt.scatter(dataset.X[:, 0], dataset.X[:, 1], alpha=0)
    plot_decision_regions(x.detach().numpy(), y.detach().numpy(), clf=model, legend=None)
    camera.snap()


def plot_anomalies(dataset, model, criterion, anomalies):
    x = dataset.X[criterion.v == 1]
    y = dataset.y[criterion.v == 1]

    plot_decision_regions(x.detach().numpy(), y.detach().numpy(), clf=model, legend=None)
    plt.scatter(dataset.X[anomalies, 0], dataset.X[anomalies, 1], c="red", marker="o")
    camera.snap()


if __name__ == "__main__":
    train()
