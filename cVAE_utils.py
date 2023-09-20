import torch
from torch import nn
from torchinfo import summary
import numpy as np
import math
import logging

from utils import test_decoder, test_encoder, GaussianSampleLayer


conv_shape_asd = [-1, 256, 10, 12, 10]
flattened_dim_asd = np.prod(conv_shape_asd) * -1

class cVAE_encoder_asd(nn.Module):
    """
    https://github.com/sccnlab/pub-CVAE-MRI-ASD/blob/main/Notebooks/make_models2.py
    """
    def __init__(self, intermediate_dim=128, latent_dim=2, use_bias=True):
        super().__init__()

        filters = 16
        kernel_size = 3

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(1, filters*2, kernel_size=kernel_size, stride=2, padding=(1,1,1), bias=use_bias),
            nn.BatchNorm3d(filters*2),
            nn.ReLU())
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(filters*2, filters * 4, kernel_size=kernel_size, stride=2, padding=(1,1,1), bias=use_bias),
            nn.BatchNorm3d(filters * 4),
            nn.ReLU())
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(filters * 4, filters * 8, kernel_size=kernel_size, stride=2, padding=(1, 1, 1),
                      bias=use_bias),
            nn.BatchNorm3d(filters * 8),
            nn.ReLU())
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(filters * 8, filters * 16, kernel_size=kernel_size, stride=2, padding=(1, 1, 1),
                      bias=use_bias),
            nn.BatchNorm3d(filters * 16),
            nn.ReLU())

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_dim_asd, intermediate_dim, bias=use_bias),
            nn.ReLU())

        self.fc_mean_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))

        self.fc_log_var_block = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim, bias=use_bias))

        self.sample_block = GaussianSampleLayer()

    def forward(self, x):
        h = self.conv_block_1(x)  # [1, 64, 80, 96, 80]
        h = self.conv_block_2(h)  # [1, 128, 40, 48, 40]
        h = self.conv_block_3(h)  # [1, 256, 20, 24, 20]
        h = self.conv_block_4(h)  # [1, 512, 10, 12, 10]

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


class cVAE_decoder_asd(nn.Module):
    """
    https://github.com/sccnlab/pub-CVAE-MRI-ASD/blob/main/Notebooks/make_models2.py
    """

    def __init__(self, intermediate_dim=128, latent_dim=8, use_bias=True):
        super().__init__()

        filters = 16
        kernel_size = 3

        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(intermediate_dim, flattened_dim_asd, bias=use_bias),
            nn.ReLU())

        self.trans_conv_block1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=filters*16, out_channels=filters*8, kernel_size=kernel_size, stride=2, padding=0, output_padding=1, bias=use_bias),
            nn.BatchNorm3d(num_features=filters*8),
            nn.ReLU())
        self.trans_conv_block2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=filters*8, out_channels=filters*4, kernel_size=kernel_size, stride=2, padding=3, output_padding=1, bias=use_bias),
            nn.BatchNorm3d(num_features=filters*4),
            nn.ReLU())
        self.trans_conv_block3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=filters*4, out_channels=filters*2, kernel_size=kernel_size, stride=2, padding=0, output_padding=0,
                               bias=use_bias),
            nn.BatchNorm3d(num_features=filters*2),
            nn.ReLU())
        self.trans_conv_block4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=filters*2, out_channels=1, kernel_size=kernel_size, stride=2, padding=2, output_padding=1,
                               bias=use_bias),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid())

    def forward(self, salient, irrelevant):
        f = torch.hstack((salient, irrelevant))
        # print(f.shape)

        h = self.fc_block(f)
        h = h.view(conv_shape_asd)

        h = self.trans_conv_block1(h)
        h = self.trans_conv_block2(h)
        h = self.trans_conv_block3(h)
        reconstructed_x = self.trans_conv_block4(h)
        return reconstructed_x


conv_shape = [-1, 512, 3, 4, 3]
flattened_dim = np.prod(conv_shape) * -1

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
            nn.Sigmoid())
        self.trans_conv_block4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=[10, 12, 10], stride=5, padding=0,
                               bias=use_bias),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid())

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

    def __init__(self, latent_dim=8, batch_norm=True):
        super().__init__()

        self.batch_norm = batch_norm

        if batch_norm:
            self.fc = nn.Linear(latent_dim, 1)
            self.bn = nn.BatchNorm1d(num_features=1)
            self.prob = nn.Sigmoid()
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(latent_dim, 1),
                # nn.BatchNorm1d(num_features=1), # cannot handle batch size = 1 # to prevent Function 'SigmoidBackward0' returned nan values in its 0th output.
                nn.Sigmoid())

    def forward(self, salient, irrelevant):
        batch_size = salient.shape[0]

        # build v: sample from q joint
        v = torch.hstack((salient, irrelevant))

        # build v_bar: sample from q prod
        ## I think shuffle z can be more convenient here
        if batch_size==1:
            v_bar = torch.hstack((salient, irrelevant))
        else:
            # split the batch of latent feature (row-wise)
            edge = int(math.ceil(batch_size / 2))
            s1 = salient[:edge, :]  # first half batch of the salient features
            s2 = salient[edge:, :]  # second half batch of the salient features
            z1 = irrelevant[:edge, :]  # first half batch of the irrelevant features
            z2 = irrelevant[edge:, :]  # second half batch of the irrelevant features

            if batch_size % 2 == 1:  # handle the case when batch_size is not even
                s0 = salient[0]
                z0 = irrelevant[0]
                s1 = s1[1:, :]
                z1 = z1[1:, :]

            v_bar = torch.vstack((torch.hstack((s1, z2)), torch.hstack((s2, z1))))
            if batch_size % 2 == 1:
                v_bar = torch.vstack((v_bar, torch.hstack((s0, z0))))
                #print(v_bar.shape)

        # calculate scores
        if self.batch_norm:
            v_score = self.fc(v)
            v_bar_score = self.fc(v_bar)
            if batch_size != 1:
                v_score = self.bn(v_score)
                v_bar_score = self.bn(v_bar_score)
            v_score = self.prob(v_score)
            v_bar_score = self.prob(v_bar_score)
        else:
            v_score = self.fc_block(v)  # should --> 0
            v_bar_score = self.fc_block(v_bar)  # should --> 1
        return v_score, v_bar_score

class ContrastiveVAE(nn.Module):
    def __init__(self, intermediate_dim=128, salient_dim=2, irrelevant_dim=6, disentangle=True, use_bias=True,
                 build_test=False, model_type="asd", disc_BN=True):
        super().__init__()

        if model_type == "asd":
            self.salient_encoder = cVAE_encoder_asd(intermediate_dim=intermediate_dim, latent_dim=salient_dim,
                                                use_bias=use_bias)
            self.irrelevant_encoder = cVAE_encoder_asd(intermediate_dim=intermediate_dim, latent_dim=irrelevant_dim,
                                                   use_bias=use_bias)
            self.decoder = cVAE_decoder_asd(intermediate_dim=intermediate_dim, latent_dim=salient_dim + irrelevant_dim,
                                        use_bias=True)
            logging.info("Creating cVAE of type 'asd'")
        elif model_type == "default":
            self.salient_encoder = cVAE_encoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim,
                                                use_bias=use_bias)
            self.irrelevant_encoder = cVAE_encoder(intermediate_dim=intermediate_dim, latent_dim=irrelevant_dim,
                                                   use_bias=use_bias)
            self.decoder = cVAE_decoder(intermediate_dim=intermediate_dim, latent_dim=salient_dim + irrelevant_dim,
                                        use_bias=True)
            logging.info("Creating cVAE of type 'default'")
        else:
            logging.info("!! ERROR -- Using undefined model type!")
            exit(-777)

        self.discriminator = None
        if disentangle:
            logging.info("Creating discriminator")
            self.discriminator = cVAE_discriminator(latent_dim=salient_dim + irrelevant_dim, batch_norm=disc_BN)

        self.disentangle = disentangle

        ''' use smaller input while building the model'''
        if build_test:
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
        if self.disentangle:  # no need self.training or batch_size!=1
            v_score_tg, v_bar_score_tg = self.discriminator(s_tg, z_tg)

        ''' 
        For the background dataset: 
        '''
        # get salient features
        s_zero = torch.zeros_like(s_tg)
        s_mu_bg, s_lv_bg, s_bg = self.salient_encoder(x_bg)  # won't be counted in loss, just for record
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
                       "s_mu_bg": s_mu_bg, "s_lv_bg": s_lv_bg, "s_bg": s_bg,
                       "z_mu_bg": z_mu_bg, "z_lv_bg": z_lv_bg, "z_bg": z_bg,
                       "reconst_bg": reconst_bg}
        return output_dict

    def get_salient_encoder(self):
        return self.salient_encoder

    def get_decoder(self):
        return self.decoder


if __name__ == '__main__':
    batch_size = 2  ##! batch_size==2 will lead to cuda out of memory...
    s_dim = 32
    z_dim = 32  # increasing the size of irrelevant features can help with the result

    # enc = cVAE_encoder_asd(latent_dim=s_dim)
    #print(enc)
    # summary(enc, input_size=(batch_size, 1, 160, 192, 160))

    dec = cVAE_decoder_asd(latent_dim=s_dim+z_dim)
    # print(dec)
    summary(dec, input_size=((batch_size, s_dim), (batch_size, z_dim)))

    # disc = cVAE_discriminator()
    # print(disc)
    # summary(disc, input_size=((batch_size, s_dim), (batch_size, z_dim)))

    # cVAE = ContrastiveVAE(salient_dim=s_dim, irrelevant_dim=z_dim, model_type="asd")
    # print(cVAE)
    # summary(cVAE, input_size=((batch_size, 1, 160, 192, 160), (batch_size, 1, 160, 192, 160)))
