import torch
from torch import nn
from torch.nn import functional as F

from vqvae import Encoder, Decoder, Quantize


class UnifModel(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):

    super().__init__()

    self.mnist_enc_b = Encoder(1, channel, n_res_block, n_res_channel, stride=4)
    self.mnist_enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
    self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
    self.quantize_t = Quantize(embed_dim, n_embed)
    self.mnist_dec_t = Decoder(
        embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
    )
    self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
    self.quantize_b = Quantize(embed_dim, n_embed)
    self.upsample_t = nn.ConvTranspose2d(
        embed_dim, embed_dim, 4, stride=2, padding=1
    )
    self.mni_dec = Decoder(
        embed_dim + embed_dim,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
        stride=4,
    )
    self.mnist_discrim = 


    self.svhn_enc_b = Encoder(1, channel, n_res_block, n_res_channel, stride=4)
    self.svhn_enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
    self.svquantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
    self.svquantize_t = Quantize(embed_dim, n_embed)
    self.mnist_dec_t = Decoder(
        embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
    )
    self.svquantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
    self.svquantize_b = Quantize(embed_dim, n_embed)
    self.svupsample_t = nn.ConvTranspose2d(
        embed_dim, embed_dim, 4, stride=2, padding=1
    )
    self.svdec = Decoder(
        embed_dim + embed_dim,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
        stride=4,
    )
    self.svhn_discrim = 