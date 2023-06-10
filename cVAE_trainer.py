"""
search `?` for sth unsure
search `!!` for unfinished work
"""

from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score
import os

from cVAE_utils import ContrastiveVAE
from utils import plot_latent_features_2D, save_checkpoint, load_checkpoint

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

    # Add testing option while building the model
    parser.add_argument('--build_test', default=True, type=bool,  ## ! change to false later
                        help='use smaller data (-1,1,6,9,6) while building the model')

    # Add data arguments
    parser.add_argument('--batch_size', default=20, type=int,
                        help='maximum number of target or background samples in a batch')

    # Add model arguments
    parser.add_argument('--s_dim', default=2, type=int, help='the dimension of a salient feature')
    parser.add_argument('--z_dim', default=6, type=int, help='the dimension of an irrelevant feature')
    parser.add_argument('--disentangle', default=True, type=bool,
                        help='whether to force independence between salient and irrelevant features of the target set '
                             'while training')

    # Add optimization arguments
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--beta', default=1, type=float, help='scaling coefficient of KL loss')
    parser.add_argument('--gamma', default=1, type=float, help='scaling coefficient of TC loss')
    parser.add_argument('--max-epoch', default=100, type=int,
                        help='force stop training at specified epoch')  ## ! change to 100 later
    parser.add_argument('--patience', default=3, type=int,  ## ! change to 10 later
                        help='number of epochs without improvement on validation set before early stopping')

    # Add logging/checkpoint arguments
    parser.add_argument('--logging_interval', default=10, type=int,
                        help='the batch interval to print training loss')  ## ! change to 100 later
    parser.add_argument('--plot-interval', default=1, type=int, help='the epoch interval to plot salient features '
                                                                     'while validation')
    parser.add_argument('--save-root-dir', default="./trained_models", type=str,
                        help='the directory to save all the models '
                             'and their training info')
    parser.add_argument('--trial-name', default="test0", type=str, help='the name of the current model, will be used '
                                                                        'to store model and training info')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')

    # add sth later ...

    args = parser.parse_args()
    return args


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def compute_loss(X_tg, X_bg, pred, disentangle, beta, gamma, batch_size):
    """ compute the avg loss of a sample within a batch """
    # square error:
    reconst_loss_tg = ((pred["reconst_tg"] - X_tg) ** 2).sum() / batch_size
    reconst_loss_bg = ((pred["reconst_bg"] - X_bg) ** 2).sum() / batch_size
    reconst_loss = reconst_loss_tg + reconst_loss_bg

    # KL( p || N(0,1) )
    KL_s_tg = (- 0.5 * (
            1 + pred["s_lv_tg"] - torch.exp(pred["s_lv_tg"]) - torch.square(pred["s_mu_tg"]))).sum() / batch_size
    KL_z_tg = (- 0.5 * (
            1 + pred["z_lv_tg"] - torch.exp(pred["z_lv_tg"]) - torch.square(pred["z_mu_tg"]))).sum() / batch_size
    KL_z_bg = (- 0.5 * (
            1 + pred["z_lv_bg"] - torch.exp(pred["z_lv_bg"]) - torch.square(pred["z_mu_bg"]))).sum() / batch_size
    KL_loss = KL_s_tg + KL_z_tg + KL_z_bg

    # forcing independence
    TC_loss = 0
    discriminator_loss = 0
    if disentangle and batch_size != 1:
        if pred["v_score"] is None or pred["v_bar_score"] is None:
            print("Oops! None value returned by the discriminator ------------")
        else:
            # min v_score --> 0
            TC_loss = (torch.log(pred["v_score"]) - torch.log(1 - pred["v_score"])).sum() / batch_size
            # max v_score --> 1, min v_bar_score --> 0
            discriminator_loss = (- torch.log(pred["v_score"]) - torch.log(1 - pred["v_bar_score"])).sum() / batch_size

    # average total loss of a sample
    loss = reconst_loss + beta * KL_loss + gamma * TC_loss + discriminator_loss

    loss_dict = {"reconst_loss_tg": reconst_loss_tg, "reconst_loss_bg": reconst_loss_bg, "reconst_loss": reconst_loss,
                 "KL_s_tg": KL_s_tg, "KL_z_tg": KL_z_tg, "KL_z_bg": KL_z_bg, "KL_loss": KL_loss,
                 "TC_loss": TC_loss, "discriminator_loss": discriminator_loss,
                 "loss": loss}

    return loss_dict


def train(train_loader, cVAE, disentangle, optimizer, beta, gamma, batch_size, logging_interval):
    size = len(train_loader.dataset)
    cVAE.train()
    for batch, (X_tg, X_bg) in enumerate(train_loader):
        current_batch_size = X_tg.shape[0]  # current batch size

        # convert data to float type and move it to cuda
        X_tg, X_bg = X_tg.type(torch.FloatTensor).to(device), X_bg.type(torch.FloatTensor).to(device)

        # Compute prediction error
        pred = cVAE(X_tg, X_bg)  # dict
        loss = compute_loss(X_tg=X_bg, X_bg=X_bg, pred=pred, disentangle=disentangle, beta=beta, gamma=gamma,
                            batch_size=current_batch_size)  # dict per batch

        # Backpropagation
        loss["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        if batch % logging_interval == 0:
            current = (batch + 1) * batch_size
            print(f"loss: {loss['loss']:>7f}  [{current:>5d}/{size:>5d}] || "
                  f"reconst_loss: {loss['reconst_loss']:>7f}, KL_loss: {loss['KL_loss']:>7f}, "
                  f"TC_loss: {loss['TC_loss']:>7f}, discriminator_loss: {loss['discriminator_loss']:>7f}")


def validate(val_loader, cVAE, disentangle, beta, gamma, val_label, plot_interval=1, epoch=-1, plot_dir="./fig"):
    """
    validation function of a epoch in the training loop
    can also plot the salient features
    return mean-batch-loss dict and ss score
    """
    num_batches = len(val_loader)

    cVAE.eval()
    loss_dict = {"reconst_loss_tg": 0, "reconst_loss_bg": 0, "reconst_loss": 0,
                 "KL_s_tg": 0, "KL_z_tg": 0, "KL_z_bg": 0, "KL_loss": 0,
                 "TC_loss": 0, "discriminator_loss": 0,
                 "loss": 0}
    s_mu_list = []  # for compute ss score and plotting for target dataset

    with torch.no_grad():  # no need back-prop when validating
        for X_tg, X_bg in val_loader:
            current_batch_size = X_tg.shape[0]  # current batch size

            # convert data to float type and move it to cuda
            X_tg, X_bg = X_tg.type(torch.FloatTensor).to(device), X_bg.type(torch.FloatTensor).to(device)

            # Compute prediction error
            pred = cVAE(X_tg, X_bg)  # dict
            loss = compute_loss(X_tg=X_tg, X_bg=X_bg, pred=pred, disentangle=disentangle, beta=beta, gamma=gamma,
                                batch_size=current_batch_size)  # dict per batch

            # accumulate loss value and means of salient features
            loss_dict = {i: loss_dict.get(i, 0) + loss.get(i, 0)
                         for i in set(loss_dict)}
            s_mu_list.append(pred["s_mu_tg"])

    # avg batch loss for each epoch
    loss_dict = {i: loss_dict.get(i) / num_batches
                 for i in set(loss_dict)}
    # overall ss score for all validation target samples
    s_mu = torch.vstack(s_mu_list).cpu()
    ss = silhouette_score(s_mu, val_label)

    loss_dict['ss'] = ss
    info_dict = loss_dict

    print(f"Validation --- loss: {loss['loss']:>7f}  ss:{ss:>7f} || "
          f"reconst_loss: {loss['reconst_loss']:>7f}, KL_loss: {loss['KL_loss']:>7f}, "
          f"TC_loss: {loss['TC_loss']:>7f}, discriminator_loss: {loss['discriminator_loss']:>7f}")

    # plot salient features
    if epoch % plot_interval == 0:
        plot_path = os.path.join(plot_dir, f"epoch-{epoch}.png")
        plot_latent_features_2D(mu=s_mu, label=val_label, ss=round(ss, 3), name='salient', run=False, path=plot_path)

    return info_dict


def save_model(model, path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), path)


if __name__ == '__main__':

    # get args from bash command
    args = get_args()
    print(args)

    # get the device on which the model will be trained on
    print(f"Using {device} device")

    '''
    initialize the path to save training info:
        - args.save_root_dir
            - args.trial_name
                - model_best.pth
                - model_last.pth
                - val_plot_salient
                    - .png files
                    - ...
                - 
            - model_save_path
            - ...
    '''
    # parent dir of the saved model
    if not os.path.exists(args.save_root_dir):
        os.makedirs(args.save_root_dir)
    # dir to save the current model info
    model_info_dir = os.path.join(args.save_root_dir, args.trial_name)
    if not os.path.exists(model_info_dir):
        os.makedirs(model_info_dir)
    # dir to save the plotted salient features when validating
    plot_val_dir = os.path.join(model_info_dir, "val_plot_salient")
    # path of a possible last checkpoint
    last_model_path = os.path.join(model_info_dir, args.trial_name + "_last.pth")

    # The size of ConcatDataset loader depends on the smaller dataset
    ##  ! so we better make the size of target and background dataset the same
    ##  Another way: use circuite-loop ranther than cvae class, train on target and background seperately
    '''training data and dataloader'''
    train_target = np.random.random((10, 1, 160, 192, 160))  ## range:[0,1), but actual data may be not in
    train_background = np.random.random((10, 1, 160, 192, 160))
    if args.build_test:
        train_target = np.random.random((1000, 1, 6, 9, 6))
        train_background = np.random.random((1000, 1, 6, 9, 6))
    # shuffle through the first axis (idx in a batch)
    np.random.shuffle(train_target)
    np.random.shuffle(train_background)
    # create training data loader
    train_loader = DataLoader(
        ConcatDataset(train_target, train_background),
        batch_size=args.batch_size, shuffle=True, num_workers=1)  # parallelized shuffle
    print(f"Load training data of size {len(train_loader.dataset)}")

    '''validation data and dataloader'''
    val_target = np.random.random((5, 1, 160, 192, 160))
    val_label = np.random.randint(low=1, high=4, size=5, dtype=int)
    val_background = np.random.random((5, 1, 160, 192, 160))
    if args.build_test:
        val_target = np.random.random((500, 1, 6, 9, 6))
        val_label = np.random.randint(low=1, high=4, size=500, dtype=int)
        val_background = np.random.random((500, 1, 6, 9, 6))
    # shuffle through the first index
    val_target, val_label = shuffle(val_target, val_label)
    np.random.shuffle(val_background)
    # create validation dataloader
    val_loader = DataLoader(
        ConcatDataset(val_target, val_background),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(f"Load validation data of size {len(val_loader.dataset)}")

    ''' Initialize the model '''
    cVAE = ContrastiveVAE(salient_dim=args.s_dim, irrelevant_dim=args.z_dim, disentangle=args.disentangle,
                          build_test=args.build_test)
    cVAE = cVAE.to(device)
    print("Built a model with {:d} parameters".format(sum(p.numel() for p in cVAE.parameters())))

    ''' Instantiate optimizer and learning rate scheduler '''
    optimizer = torch.optim.Adam(cVAE.parameters(), args.lr)  ## weight_decay=1e-5 ?
    ## may add learning rate scheduler later ...

    ''' Load last checkpoint if one exists '''
    state_dict = load_checkpoint(last_model_path, cVAE, optimizer)
    if state_dict is not None:
        last_epoch = state_dict["epoch"]
        best_loss, bad_epochs = state_dict["best_loss"], state_dict["bad_epochs"]
        val_info_history = state_dict["val_info_history"]
    else:  # Initialize training params if its a new trail
        last_epoch = -1
        best_loss, bad_epochs = float('inf'), 0
        val_info_history = {"reconst_loss_tg": [], "reconst_loss_bg": [], "reconst_loss": [],
                            "KL_s_tg": [], "KL_z_tg": [], "KL_z_bg": [], "KL_loss": [],
                            "TC_loss": [], "discriminator_loss": [],
                            "loss": [],
                            "ss": []}

    ''' Start training'''
    for epoch in range(last_epoch + 1, args.max_epoch):
        print(f"Epoch {epoch + 1} -------------------------------:")

        train(train_loader=train_loader, cVAE=cVAE, disentangle=args.disentangle, optimizer=optimizer,
              beta=args.beta, gamma=args.gamma, batch_size=args.batch_size, logging_interval=args.logging_interval)

        val_info = validate(val_loader=val_loader, cVAE=cVAE, disentangle=args.disentangle, beta=args.beta,
                            gamma=args.gamma, val_label=val_label, plot_interval=args.plot_interval, epoch=epoch,
                            plot_dir=plot_val_dir)

        # record validation loss
        for key in set(val_info_history):
            if key != 'ss':
                new = val_info.get(key, None).cpu().item()
            else:
                new = val_info.get(key, None)
            val_info_history.get(key, []).append(new)

        # check whether to use early stop
        early_stop = False
        val_loss = val_info["loss"]
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            # save the best model so far
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=cVAE, optimizer=optimizer,
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
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=cVAE, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name=name)

        # early stop
        if early_stop:
            print('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break
        # force stop
        if epoch == args.max_epoch-1:
            save_checkpoint(model_info_dir=model_info_dir, trial_name=args.trial_name, model=cVAE, optimizer=optimizer,
                            epoch=epoch, best_loss=best_loss, bad_epochs=bad_epochs, val_info_history=val_info_history,
                            name="final")
            print("! Reaching the maximum epoch number")


    print("Done!")
    ## !! test func haven't implement
