import torch
from torch import nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_latent_features_2D(mu=[0], label=[0], ss=-999, name='salient', run=False, encoder=None, sample=None,
                            path=None):
    """
    use mean value inferred by encoder and sample labels to plot
    if run = True, take in encoder and samples and infer mean; otherwise take in mean value directly
    """
    if run:
        mu, _, _ = encoder(sample)
    # plot:
    plt.figure()
    plt.scatter(mu[:, 0],mu[:, 1], c=label, cmap='Accent')
    plt.title(name + ', Silhouette score: ' + str(ss))
    if path is None:
        plt.show()
    else:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(path)


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


""" set args.building_test to True"""
input_dim = (-1, 1, 6,9,6)
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

