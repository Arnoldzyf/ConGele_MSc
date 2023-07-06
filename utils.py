import sys
import os

# the path that contains SFCN codes
SFCN_path = "../UKBiobank_deep_pretrain-master"
sys.path.append(SFCN_path)

import torch
from torch import nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import nibabel as nib

from dp_model import dp_utils as dpu


def create_dataset_from_nii_path_list(path_list):
    """
    extract array of nii image from path_list,
    crop,
    return np array of shap (n_samples, 1, 160, 192, 160)
    """
    arr_list = []

    for path in path_list:
        # get image array data
        img_arr = obtain_arr_from_nii(path)  # (160, 192, 160)

        # reshape -- add batch size, channel size
        sp = (1, 1) + img_arr.shape
        img_arr_reshape = img_arr.reshape(sp)  # (1, 1, 160, 192, 160)

        arr_list.append(img_arr_reshape)

    data_arr = np.vstack(arr_list)
    return data_arr


def obtain_arr_from_nii(path):
    """
    extract + crop
    """
    # extract nii object
    img = nib.load(path)
    # extract raw arr data
    img_raw_arr = img.get_fdata()
    # preprocess
    img_arr_0 = img_raw_arr/img_raw_arr.mean()
    img_arr = dpu.crop_center(img_arr_0, (160, 192, 160)) # crop
    return img_arr

def load_checkpoint(last_model_path, model, optimizer):
    if os.path.isfile(last_model_path):
        state_dict = torch.load(last_model_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print('Loaded from a previous saved model {}'.format(last_model_path))
        return state_dict


def save_checkpoint(model_info_dir, trial_name, model, optimizer, epoch, best_loss, bad_epochs, val_info_history, name):
    " name: best/last/final "
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'bad_epochs': bad_epochs,
        'val_info_history': val_info_history
    }
    os.makedirs(model_info_dir, exist_ok=True)
    path = os.path.join(model_info_dir, trial_name + "_" + name + ".pth")
    torch.save(state_dict, path)
    print(f"Finish saving the {name} model.")


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def plot_latent_features_2D(mu=[0], label=[0], ss=-999, name='salient', path=None,
                            run=False, encoder=None, sample=None):
    """
    use mean value inferred by encoder and sample labels to plot
    if run = True, take in encoder and samples and infer mean; otherwise take in mean value directly
    """
    if run:
        mu, _, _ = encoder(sample)
    # plot:
    plt.figure()
    plt.scatter(mu[:, 0], mu[:, 1], c=label, cmap='Accent')
    plt.title(name + ', Silhouette score: ' + str(ss))
    if path is None:
        plt.show()
    else:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(path)
        plt.close()


class GaussianSampleLayer(nn.Module):
    """
    Sampling once from a Gaussian distribution
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu, lv):
        """
        :param mu: mean of the feature
        :param lv: log variance of the feature
        :return: sample: mu + std * N(0,1)
        """
        std = torch.sqrt(torch.exp(lv))
        eps = torch.randn_like(std)
        sample = eps.mul(std).add(mu)
        return sample

def bias_correction(y: np.ndarray, y_pred: np.ndarray):
    """
    :param y: chronological age (ground truth)
    :param y_pred: brain age before bias correction (predictions)
    :return: bias-corrected brain age
    """
    # Smith2019 bias correction
    linear_fit = LinearRegression(fit_intercept=True).fit(
        X=y, y=y_pred
    )
    intercept, slope = linear_fit.intercept_, linear_fit.coef_[0]
    y_pred_unbiased = (y_pred - intercept) / (slope + np.finfo(np.float32).eps) # avoid division by 0
    return y_pred_unbiased, intercept, slope


""" 
set args.building_test to True if using test classes:
"""
input_dim = (-1, 1, 6, 9, 6)
input_dim_flatten = np.prod(input_dim) * -1


class test_encoder(nn.Module):
    """ input size (-1, 1, 6, 9, 6) """

    def __init__(self, intermediate_dim=128, latent_dim=2, use_bias=True):
        super().__init__()
        self.to_intermediate = nn.Linear(input_dim_flatten, intermediate_dim, bias=use_bias)
        self.to_mean = nn.Linear(intermediate_dim, latent_dim, bias=use_bias)
        self.to_log_var = nn.Linear(intermediate_dim, latent_dim, bias=use_bias)
        self.sample = GaussianSampleLayer()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = self.to_intermediate(x)
        mean = self.to_mean(h)
        log_var = self.to_log_var(h)
        f = self.sample(mean, log_var)
        return mean, log_var, f


class test_decoder(nn.Module):
    """ output size (-1, 1, 6, 9, 6) """

    def __init__(self, intermediate_dim=128, latent_dim=8, use_bias=True):
        super().__init__()
        self.back_to_intermediate = nn.Linear(latent_dim, intermediate_dim, bias=use_bias)
        self.back_to_original_dim = nn.Linear(intermediate_dim, input_dim_flatten, bias=use_bias)

    def forward(self, salient, irrelevant):
        f = torch.hstack((salient, irrelevant))
        h = self.back_to_intermediate(f)
        h = self.back_to_original_dim(h)
        reconstructed_x = h.view(input_dim)
        return reconstructed_x
