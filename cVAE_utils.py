import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchinfo import summary


def GaussianSampleLayer(mu, lv):
    """
    Return a sample of a Gaussian distribution
    :param mu: mean of the feature
    :param lv: log variance of the feature
    :return: mu + std * N(0,1)
    """
    std = torch.sqrt(torch.exp(lv))
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class  cVAE_encoder(nn.Module):
    """
    Encoders for inferring both salient and irrelevant features
    input size: [batch_size, n_channels=1, 160, 192, 160]
    Output dim: [latent_dim,]
    maybe set salient dim to 2 and irrlevant dim to 6
    """
    def __init__(self, intermediate_dim=128, latent_dim=2, use_bias=True):
        """
        the model structure is just a draft, not sure makes sense
        adapted from https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py
        """
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, padding=5),
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
            nn.Linear(512 * 3 * 4 * 3, intermediate_dim, bias=use_bias),
            nn.Tanh(),
            nn.Dropout(0.2))

        self.fc_mean_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))

        self.fc_log_var_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))


    def forward(self, x):
        ## conv layers to shrink the size down
        h = self.conv_block_1(x)
        h = self.conv_block_2(h)
        h = self.conv_block_3(h)
        h = self.conv_block_4(h)

        ## flatten the hidden feature and convert to intermediate vec
        h = h.view(h.size(0), -1)
        h = self.fc_block(h)

        ## Infer the mean
        mean = self.fc_mean_block(h)

        ## Infer the log variance
        log_var = self.fc_log_var_block(h)

        ## Sample the feature vector
        f = 0
        return mean, log_var, f


class  cVAE_decoder(nn.Module):
    def __init__(self, intermediate_dim=128, latent_dim_sum=8):
        super().__init__()



if __name__ == '__main__':
    enc = cVAE_encoder()
    print(enc)
    summary(enc, input_size=(1, 1, 160, 192, 160))  ## large batch_size will lead to cuda out of memory
