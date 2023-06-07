import torch
from torch import nn
from torchinfo import summary
import numpy as np

from test_utils import test_decoder, test_encoder

conv_shape = [-1, 512, 3, 4, 3]
flattened_dim = np.prod(conv_shape) * -1


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


class cVAE_encoder(nn.Module):
    """
    Encoders for inferring both salient and irrelevant features
    input size: [batch_size, n_channels=1, 160, 192, 160]
    Output dim: [batch_size, latent_dim]
    maybe set salient dim to 2 and irrlevant dim to 6
    """

    def __init__(self, intermediate_dim=128, latent_dim=2, use_bias=True):
        """
        ! the model structure is just a draft, not sure makes sense
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
            nn.Conv3d(64, 128, kernel_size=7, padding=3, bias=use_bias),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(5, 5, 5), stride=(3, 3, 3)))
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=5, padding=2, bias=use_bias),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)))
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2)))

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_dim, intermediate_dim, bias=use_bias),
            nn.ReLU(),  ## ? choose of activation func
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


class cVAE_decoder(nn.Module):
    """
    cVAE decoder,
    output reconstructed input of size [batch_size, 1, 160, 192, 160]
    """

    def __init__(self, intermediate_dim=128, latent_dim=8, use_bias=True):
        """
        ! draft of decoder structure, not sure makes sense
        adapted from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
        :param latent_dim: sum of the dim of the salient and the irrelevant features
        """
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim, bias=use_bias),
            nn.Linear(intermediate_dim, flattened_dim, bias=use_bias),  # in this setting intermediate_dim not needed
            nn.ReLU())

        self.trans_conv_block1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU())
        self.trans_conv_block2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU())
        self.trans_conv_block3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=[6, 9, 6], stride=3, padding=1,
                               bias=use_bias),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU())
        self.trans_conv_block4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=[10, 12, 10], stride=5, padding=0,
                               bias=use_bias),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU())  ## ?? need to pre-process the input range to >=0

    def forward(self, salient, irrelevant):
        f = torch.hstack((salient, irrelevant))
        # print(f.shape)

        h = self.fc_block(f)
        h = h.view(conv_shape)

        h = self.trans_conv_block1(h)
        h = self.trans_conv_block2(h)
        h = self.trans_conv_block3(h)
        reconstructed_x = self.trans_conv_block4(h)
        return reconstructed_x


class cVAE_discriminator(nn.Module):
    """
    no use when batch_size=1
    """

    def __init__(self, latent_dim=8):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid())

    def forward(self, salient, irrelevant):
        # split the batch of latent feature (row-wise)
        batch_size = salient.shape[0]
        s1 = salient[:int(batch_size / 2), :]  # first half batch of the salient features
        s2 = salient[int(batch_size / 2):, :]  # second half batch of the salient features
        z1 = irrelevant[:int(batch_size / 2), :]  # first half batch of the irrelevant features
        z2 = irrelevant[int(batch_size / 2):, :]  # second half batch of the irrelevant features

        # sample from q joint
        v = torch.hstack((salient, irrelevant))
        # should --> 0
        v_score = self.fc_block(v)  ## ? +1)*0.85 in CVAE-ASD repo

        # sample from q prod
        v_bar = torch.vstack((torch.hstack((s1, z2)), torch.hstack((s2, z1))))
        # should --> 1
        v_bar_score = self.fc_block(v_bar)  ## ? +1)*0.85 in CVAE-ASD repo
        return v_score, v_bar_score


class ContrastiveVAE(nn.Module):
    def __init__(self, intermediate_dim=128, salient_dim=2, irrelevant_dim=6, disentangle=True, use_bias=True,
                 building_test=False):
        super().__init__()

        self.salient_encoder = cVAE_encoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim,
                                            use_bias=use_bias)
        self.irrelevant_encoder = cVAE_encoder(intermediate_dim=intermediate_dim, latent_dim=irrelevant_dim,
                                               use_bias=use_bias)

        self.decoder = cVAE_decoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim + irrelevant_dim,
                                    use_bias=True)

        self.discriminator = None
        if disentangle:
            self.discriminator = cVAE_discriminator(latent_dim=salient_dim + irrelevant_dim)

        self.disentangle = disentangle

        ''' use smaller input while building the model'''
        if building_test:
            self.salient_encoder = test_encoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim,
                                                use_bias=use_bias)
            self.irrelevant_encoder = test_encoder(intermediate_dim=intermediate_dim, latent_dim=irrelevant_dim,
                                                   use_bias=use_bias)
            self.decoder = test_decoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim + irrelevant_dim,
                                        use_bias=True)


    def forward(self, x_tg, x_bg):
        """
        ? In the original Keras code: s --> irrelevant, z --> salient
        **         HERE:              s --> salient, z --> irrelevant      **
        :param x:
        :return:
        """

        ''' 
        For the target dataset: 
        '''
        # get salient features
        s_mu_tg, s_lv_tg, s_tg = self.salient_encoder(x_tg)
        # get irrelevant features
        z_mu_tg, z_lv_tg, z_tg = self.irrelevant_encoder(x_tg)

        # get reconstructed input
        reconst_tg = self.decoder(s_tg, z_tg)

        # disentangle: force the independence between the salient and the irrelevant features.
        v_score_tg, v_bar_score_tg = None, None
        batch_size = s_tg.shape[0]
        # no disentangle if there's only one sample in a batch
        if batch_size != 1 and self.disentangle:  # no need self.training
            v_score_tg, v_bar_score_tg = self.discriminator(s_tg, z_tg)

        ''' 
        For the background dataset: 
        '''
        # get salient features
        s_zero = torch.zeros_like(s_tg)
        # get irrelevant features
        z_mu_bg, z_lv_bg, z_bg = self.irrelevant_encoder(x_bg)

        # get reconstructed input
        reconst_bg = self.decoder(s_zero, z_bg)

        ''' 
        Output Dict: 
          '''
        output_dict = {"s_mu_tg": s_mu_tg, "s_lv_tg": s_lv_tg, "s_tg": s_tg,
                       "z_mu_tg": z_mu_tg, "z_lv_tg": z_lv_tg, "z_tg": z_tg,
                       "reconst_tg": reconst_tg,
                       "v_score": v_score_tg, "v_bar_score": v_bar_score_tg,  # background have no v_(bar_)score
                       "z_mu_bg": z_mu_bg, "z_lv_bg": z_lv_bg, "z_bg": z_bg,
                       "reconst_bg": reconst_bg}
        return output_dict


if __name__ == '__main__':
    batch_size = 1  ##! batch_size==2 will lead to cuda out of memory...
    s_dim = 2
    z_dim = 6  # increasing the size of irrelevant features can help with the result

    # enc = cVAE_encoder()
    # print(enc)
    # summary(enc, input_size=(batch_size, 1, 160, 192, 160))

    # dec = cVAE_decoder()
    # print(dec)
    # summary(dec, input_size=((batch_size, s_dim), (batch_size, z_dim)))

    # disc = cVAE_discriminator()
    # print(disc)
    # summary(disc, input_size=((batch_size, s_dim), (batch_size, z_dim)))

    cVAE = ContrastiveVAE(salient_dim=s_dim, irrelevant_dim=z_dim)
    print(cVAE)
    summary(cVAE, input_size=((batch_size, 1, 160, 192, 160), (batch_size, 1, 160, 192, 160)))
