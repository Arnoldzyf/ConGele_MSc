import sys
import os
SFCN_path = "../UKBiobank_deep_pretrain-master"
sys.path.append(SFCN_path)

from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from utils import ConcatDataset, save_checkpoint, load_checkpoint, create_dataset_from_nii_path_list

import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split


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
    parser = argparse.ArgumentParser('SFCN Model')

    # Add data arguments
    parser.add_argument('--dataset', default="./data_info/T1_MNI_20252_2_0/HC_info.csv", type=str,
                        help='csv file storing nii path and chronological ages of subjects')
    parser.add_argument('--col_name_path', default="path", type=str,
                        help='name of the column that stores nii paths')
    parser.add_argument('--col_name_age', default="f.21003.2.0", type=str,
                        help='name of the column that stores chronological ages')

    parser.add_argument('--val_rate', default=0.2, type=float,
                        help='rate of validation set as to all the data')
    parser.add_argument('--random_state', default=42, type=int,
                        help='random state for splitting training and validation set')

    parser.add_argument('--batch_size', default=20, type=int,
                        help='maximum number of target or background samples in a batch')

    # Add optimization arguments
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--max_epoch', default=100, type=int,
                        help='force stop training at specified epoch')  ## ! change to 100 later
    parser.add_argument('--patience', default=3, type=int,  ## ! change to 10 later
                        help='number of epochs without improvement on validation set before early stopping')

    # Add logging/checkpoint arguments
    parser.add_argument('--logging_interval', default=25, type=int,
                        help='the batch interval to print training loss')
    parser.add_argument('--save_root_dir', default="./trained_SFCN", type=str,
                        help='the directory to save all the models '
                             'and their training info')
    parser.add_argument('--trial_name', default="test1", type=str, help='the name of the current model, will be used '
                                                                        'to store model and training info')
    parser.add_argument('--save_interval', type=int, default=1, help='save a checkpoint every N epochs')

    args = parser.parse_args()
    return args


def SFCN_train(dataloader, model, optimizer, args):
    logging_interval = args.logging_interval

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # path --> np array --> tensor
        X = torch.from_numpy(create_dataset_from_nii_path_list(X))

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

        if batch % logging_interval == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def SFCN_validate(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # path --> np array --> tensor
            X = torch.from_numpy(create_dataset_from_nii_path_list(X))

            n_sample = X.shape[0]
            X, y = X.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

            output = model(X)
            pred = output[0].reshape([n_sample, -1])  # shape[n_samples, 40]

            loss = dpl.my_KLDivLoss(pred, y).item()  # avg
            val_loss += loss
    val_loss /= num_batches
    logging.info(f"Validation avg loss: {val_loss:>8f}")
    val_info = {"loss": val_loss}
    return val_info


if __name__ == '__main__':
    # get args from bash command
    args = get_args()

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

    '''initialize logging'''
    # log
    log_file = os.path.join(model_info_dir, "logging.txt")
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode='a')]
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logging.info("! start logging -------")
    logging.info(f"! Model: {model_info_dir}")

    logging.info(args)
    # get the device on which the model will be trained on
    logging.info(f"Using {device} device")


    ''' train and val datasets: nii path + chronological age'''
    dataset = pd.read_csv(args.dataset)
    nii_paths = dataset[args.col_name_path].tolist()
    ages = dataset[args.col_name_age].tolist()

    # split into train and validate set
    paths_train, paths_val, ages_train, ages_val  \
        = train_test_split(nii_paths, ages,  test_size=args.val_rate, shuffle=True, random_state=args.random_state)

    ''' Training dataloader '''
    # specify bin range and step
    bin_range = [42, 82]
    bin_step = 1

    # a normal distribution centered at the label (true chronological age)
    y_train, bin_center = dpu.num2vect(x=ages_train, bin_range=bin_range, bin_step=bin_step, sigma=1)
    # print(y_train.shape)  # (n_sample, 40)
    # print(bin_center.shape)  # (40,)

    # create training data loader
    train_loader = DataLoader(
        ConcatDataset(paths_train, y_train),
        batch_size=args.batch_size, shuffle=True, num_workers=3)  # parallelized shuffle
    logging.info(f"Load training data of size {len(train_loader.dataset)}")

    ''' Validation dataloader '''
    # a normal distribution centered at the label (true chronological age)
    y_val, _ = dpu.num2vect(x=ages_val, bin_range=bin_range, bin_step=bin_step, sigma=1)

    # create validation data loader
    val_loader = DataLoader(
        ConcatDataset(paths_val, y_val),
        batch_size=args.batch_size, shuffle=True, num_workers=3)  # parallelized shuffle
    logging.info(f"Load validation data of size {len(val_loader.dataset)}")

    ''' using the pre-trained model param to initialize the model '''
    model = SFCN()
    logging.info("Built a model with {:d} parameters".format(sum(p.numel() for p in model.parameters())))
    model = torch.nn.DataParallel(model)
    fp_ = os.path.join(SFCN_path, './brain_age/run_20190719_00_epoch_best_mae.p')
    logging.info("Initialize the model with pre-trained parameter from the SFCN github repo")
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
    logging.info(f"Start training with batch size of {args.batch_size}")
    for epoch in range(last_epoch + 1, args.max_epoch):
        logging.info(f"Epoch {epoch} -------------------------------")
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
            logging.info('No validation loss improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break
        # force stop
        if epoch == args.max_epoch - 1:
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=model, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name="final")
            logging.info("! Reaching the maximum epoch number")

    logging.info("Done!")





