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
        decay=0.99):

        super().__init__()

        self.enc_mnist = Encoder(1, channel, n_res_block, n_res_channel, stride=4)
        self.enc_svhn = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(2*channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)

        self.quantize_conv_b_mnist = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b_mnist = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        
        self.quantize_conv_b_svhn = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b_svhn = Quantize(embed_dim, n_embed)

        

        self.dec_mnist = Decoder(
            embed_dim + embed_dim,
            1,
            channel,
            n_res_block,
            n_res_channel,
            stride=4)
        self.dec_svhn = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4)

    def forward(self, mnist, svhn):
        enc_m = self.enc_mnist(mnist)
        enc_s = self.enc_svhn(svhn)
        quant_t, quant_b_mnist, quant_b_svhn, diff, _, _, _ = self.encode(enc_m, enc_s)
        dec_m, dec_s = self.decode(quant_t, quant_b_mnist, quant_b_svhn)

        return dec_m, dec_s, diff

    def encode(self, enc_m, enc_s):
        enc_b = torch.cat([enc_m, enc_s], 1)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)

        #===========MNIST ============
        enc_m = torch.cat([dec_t, enc_m], 1)
        quant_b_mnist = self.quantize_conv_b_mnist(enc_m).permute(0, 2, 3, 1)
        quant_b_mnist, diff_b_mnist, id_b_mnist = self.quantize_b_mnist(quant_b_mnist)
        quant_b_mnist = quant_b_mnist.permute(0, 3, 1, 2)
        diff_b_mnist = diff_b_mnist.unsqueeze(0)
        #===========svhn================
        enc_s = torch.cat([dec_t, enc_s], 1)
        quant_b_svhn = self.quantize_conv_b_svhn(enc_s).permute(0, 2, 3, 1)
        quant_b_svhn, diff_b_svhn, id_b_svhn = self.quantize_b_svhn(quant_b_svhn)
        quant_b_svhn = quant_b_svhn.permute(0, 3, 1, 2)
        diff_b_svhn = diff_b_svhn.unsqueeze(0)

        diff = diff_t + diff_b_mnist + diff_b_svhn

        return quant_t, quant_b_mnist, quant_b_svhn,  diff, id_t, id_b_mnist, id_b_svhn
    
    def decode(self, quant_t, quant_b_mnist, quant_b_svhn):
        upsample_t = self.upsample_t(quant_t)
        quant_m  = torch.cat([upsample_t, quant_b_mnist], 1)
        quant_s  = torch.cat([upsample_t, quant_b_svhn], 1)
        dec_m = self.dec_mnist(quant_m)
        dec_s = self.dec_svhn(quant_s)
        return dec_m, dec_s

    #def decode_code(self, code_t, code_b):
    #    quant_t = self.quantize_t.embed_code(code_t)
    #    quant_t = quant_t.permute(0, 3, 1, 2)
    #    quant_b = self.quantize_b.embed_code(code_b)
    #    quant_b = quant_b.permute(0, 3, 1, 2)

    #    dec = self.decode(quant_t, quant_b)

    #    return dec
