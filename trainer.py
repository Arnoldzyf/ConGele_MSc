from torch.utils.data import DataLoader#, ConcatDataset
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

    parser.add_argument('--batch_size', default=2, help='specify the batch size')
    # add sth later
    # ...
    args = parser.parse_args()
    return args

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

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
    print(args)

    # The size of train_loader depends on the smaller dataset
    ##  ! so we better make the size of target and background dataset the same
    ##  Another way: use circuite-loop ranther than cvae class, train on target and background seperately
    '''training data and dataloader'''
    train_target = np.random.random((10, 1, 160, 192, 160))  ## range:[0,1), but actual data may be not in
    train_background = np.random.random((10, 1, 160, 192, 160))
    # shuffle through first index
    np.random.shuffle(train_target)
    np.random.shuffle(train_background)
    train_loader = DataLoader(
            ConcatDataset(train_target, train_background),
            batch_size=args.batch_size, shuffle=True, num_workers=1)  # parallelized shuffle

    '''validation data and dataloader'''
    val_target = np.random.random((5, 1, 160, 192, 160)) 
    val_background = np.random.random((5, 1, 160, 192, 160))
    # shuffle through first index
    np.random.shuffle(val_target)
    np.random.shuffle(val_background)
    train_loader = DataLoader(
        ConcatDataset(val_target, val_background),
        batch_size=args.batch_size, shuffle=True, num_workers=1)





