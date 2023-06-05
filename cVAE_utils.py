import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchinfo import summary
import numpy as  np

conv_shape = [-1, 512, 3, 4, 3]
flattened_dim = np.prod(conv_shape) * -1

class GaussianSampleLayer(nn.Module):
    """
    Return a sample of a Gaussian distribution
    :param mu: mean of the feature
    :param lv: log variance of the feature
    :return: mu + std * N(0,1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, lv):
        std = torch.sqrt(torch.exp(lv))
        eps = torch.randn_like(std)
        sample = eps.mul(std).add(mu)
        return sample

class  cVAE_encoder(nn.Module):
    """
    Encoders for inferring both salient and irrelevant features
    input size: [batch_size, n_channels=1, 160, 192, 160]
    Output dim: [batch_size, latent_dim]
    maybe set salient dim to 2 and irrlevant dim to 6
    """
    def __init__(self, intermediate_dim=128, latent_dim=2, use_bias=True):
        """
        the model structure is just a draft, not sure makes sense
        adapted from https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py
        can add residual later
        """
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, padding=5, bias=use_bias),  # padding = "same"
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 5, 5), stride=(3, 3, 3)))
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 5, 5), stride=(3, 3, 3)))
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)))
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)))

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_dim, intermediate_dim, bias=use_bias),
            nn.ReLU(), ## ? choose of activation func
            nn.Dropout(0.2))

        self.fc_mean_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))

        self.fc_log_var_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))

        self.sample_block = GaussianSampleLayer()


    def forward(self, x):
        # conv layers to shrink the size down
        h = self.conv_block_1(x)
        h = self.conv_block_2(h)
        h = self.conv_block_3(h)
        h = self.conv_block_4(h)

        # flatten the hidden feature and convert to intermediate vec
        h = h.view(h.size(0), -1)
        h = self.fc_block(h)

        # Infer the mean
        mean = self.fc_mean_block(h)

        # Infer the log variance
        log_var = self.fc_log_var_block(h)

        # Sample the feature vector
        f = self.sample_block(mean, log_var)
        return mean, log_var, f


class  cVAE_decoder(nn.Module):
    """
    cVAE decoder,
    take in [batch_size, salient_dim + irrelevant_dim],
    reconstruct input of size [batch_size, 1, 160, 192, 160]
    """
    def __init__(self, intermediate_dim=128, latent_dim=8, use_bias=True):
        """
        draft of decoder structure
        adapted from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
        :param latent_dim: sum of the dim of the salient and the irrelevant features
        """
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim, bias=use_bias),
            nn.Linear(intermediate_dim, flattened_dim, bias=use_bias),  # in this setting intermediate_dim not needed
            nn.ReLU())

        self.trans_conv_block1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU())
        self.trans_conv_block2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU())
        self.trans_conv_block3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=[6,9,6], stride=3, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.Tanh())
        self.trans_conv_block4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=[10,12,10], stride=5, padding=0),
            nn.BatchNorm3d(num_features=1),
            nn.Tanh())  ## ? what's the range of input MRI

    def forward(self, f):
        h = self.fc_block(f)
        h = h.view(conv_shape)

        h = self.trans_conv_block1(h)
        h = self.trans_conv_block2(h)
        h = self.trans_conv_block3(h)
        x = self.trans_conv_block4(h)
        return x





if __name__ == '__main__':
    # enc = cVAE_encoder()
    # print(enc)
    # summary(enc, input_size=(1, 1, 160, 192, 160))  ## large batch_size will lead to cuda out of memory

    latent_dim_sum = 2 + 6
    dec = cVAE_decoder()
    print(dec)
    summary(dec, input_size=(1, 1, latent_dim_sum))



