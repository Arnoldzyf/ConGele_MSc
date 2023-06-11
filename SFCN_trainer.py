import sys
import os
SFCN_path = "../UKBiobank_deep_pretrain-master"
sys.path.append(SFCN_path)

from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from utils import ConcatDataset, save_checkpoint, load_checkpoint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.linalg import lstsq
import argparse


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

    # Add data arguments
    parser.add_argument('--batch_size', default=1, type=int,  ## ! change to larger numbers later
                        help='maximum number of target or background samples in a batch')

    # Add optimization arguments
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--max-epoch', default=50, type=int,
                        help='force stop training at specified epoch')  ## ! change to 100 later
    parser.add_argument('--patience', default=3, type=int,  ## ! change to 10 later
                        help='number of epochs without improvement on validation set before early stopping')

    # Add logging/checkpoint arguments
    parser.add_argument('--logging-interval', default=5, type=int,
                        help='the batch interval to print training loss')  ## ! change to 100 later
    parser.add_argument('--save-root-dir', default="./trained_SFCN", type=str,
                        help='the directory to save all the models '
                             'and their training info')
    parser.add_argument('--trial-name', default="test0", type=str, help='the name of the current model, will be used '
                                                                        'to store model and training info')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')

    args = parser.parse_args()
    return args


def SFCN_train(dataloader, model, optimizer, args):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        n_sample = X.shape[0]
        X, y = X.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

        # Compute prediction error
        output = model(X)
        pred = output[0].reshape([n_sample, -1])  # shape[n_samples, 40]

        loss = dpl.my_KLDivLoss(pred, y)  # avg

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % args.logging_interval == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def SFCN_validate(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            n_sample = X.shape[0]
            X, y = X.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

            output = model(X)
            pred = output[0].reshape([n_sample, -1])  # shape[n_samples, 40]

            loss = dpl.my_KLDivLoss(pred, y).item()  # avg
            val_loss += loss
    val_loss /= num_batches
    print(f"Validation avg loss: {val_loss:>8f}")
    val_info = {"loss": val_loss}
    return val_info


if __name__ == '__main__':
    # get args from bash command
    args = get_args()
    print(args)

    # get the device on which the model will be trained on
    print(f"Using {device} device")

    ''' model saving path '''
    # parent dir of the saved model
    if not os.path.exists(args.save_root_dir):
        os.makedirs(args.save_root_dir)
    # dir to save the current model info
    model_info_dir = os.path.join(args.save_root_dir, args.trial_name)
    if not os.path.exists(model_info_dir):
        os.makedirs(model_info_dir)
    # path of a possible last checkpoint
    last_model_path = os.path.join(model_info_dir, args.trial_name + "_last.pth")

    ''' get the small dataset (20 samples)'''
    with open('./arr_img_data_small.npy', 'rb') as f:
        data = np.load(f)
    label = pd.read_pickle("./df_info_path_data_small.pkl")["age"].to_numpy()

    bin_range = [14, 94]
    bin_step = 2

    ''' Training data and dataloader '''
    n_sample = 15
    data_train = data[:n_sample]
    label_train = label[:n_sample]

    # a normal distribution centered at the label (true chronological age)
    y_train, bin_center = dpu.num2vect(x=label_train, bin_range=bin_range, bin_step=bin_step, sigma=1)
    # print(y_train.shape)  # (n_sample, 40)
    # print(bin_center.shape)  # (40,)

    # create training data loader
    train_loader = DataLoader(
        ConcatDataset(data_train, y_train),
        batch_size=args.batch_size, shuffle=True, num_workers=1)  # parallelized shuffle
    print(f"Load training data of size {len(train_loader.dataset)}")

    ''' Validation data and dataloader '''
    data_val = data[n_sample:]
    label_val = label[n_sample:]

    # a normal distribution centered at the label (true chronological age)
    y_val, _ = dpu.num2vect(x=label_val, bin_range=bin_range, bin_step=bin_step, sigma=1)

    # create validation data loader
    val_loader = DataLoader(
        ConcatDataset(data_val, y_val),
        batch_size=args.batch_size, shuffle=True, num_workers=1)  # parallelized shuffle
    print(f"Load validation data of size {len(val_loader.dataset)}")

    ''' using the pre-trained model param to initialize the model '''
    model = SFCN()
    model = torch.nn.DataParallel(model)
    fp_ = os.path.join(SFCN_path, './brain_age/run_20190719_00_epoch_best_mae.p')
    model.load_state_dict(torch.load(fp_))
    model.to(device)

    ''' Instantiate optimizer and learning rate scheduler '''
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    ''' Load last checkpoint if one exists '''
    state_dict = load_checkpoint(last_model_path, model, optimizer)
    if state_dict is not None:
        last_epoch = state_dict["epoch"]
        best_loss, bad_epochs = state_dict["best_loss"], state_dict["bad_epochs"]
        val_info_history = state_dict["val_info_history"]
    else:  # Initialize training params if its a new trail
        last_epoch = -1
        best_loss, bad_epochs = float('inf'), 0
        val_info_history = {"loss": []}

    ''' Start training '''
    for epoch in range(last_epoch + 1, args.max_epoch):
        print(f"Epoch {epoch + 1} -------------------------------")
        SFCN_train(dataloader=train_loader, model=model, optimizer=optimizer, args=args)
        val_info = SFCN_validate(dataloader=val_loader, model=model)

        # record validation loss
        for key in set(val_info_history):
            new = val_info.get(key, None)
            val_info_history.get(key, []).append(new)

        # check whether to use early stop
        early_stop = False
        val_loss = val_info["loss"]
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            # save the best model so far
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=model, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name="best")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            early_stop = True

        # Save checkpoints
        if epoch % args.save_interval == 0:
            name = "last"
            if early_stop:
                name = "final"
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=model, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name=name)

        # early stop
        if early_stop:
            print('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break
        # force stop
        if epoch == args.max_epoch - 1:
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=model, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name="final")
            print("! Reaching the maximum epoch number")

    print("Done!")





