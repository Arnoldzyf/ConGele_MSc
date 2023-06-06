from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import argparse
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def get_args():
    """
    Defines training-specific hyper-parameters.
    """
    parser = argparse.ArgumentParser('contrastive VAE Model')
    # add sth later
    # ...
    args = parser.parse_args()
    return args

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == '__main__':
    print(f"Using {device} device")
    args = get_args()

    test_target = np.random.random((10, 1, 160, 192, 160))  # range:[0,1), but actual data may be not in
    test_background = np.random.random((10, 1, 160, 192, 160))
    
