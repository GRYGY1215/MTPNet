import torch
from torch import nn
from einops import rearrange

# Implementation from other source code.
from models.MTPNet_EncDec import tsformer_Encoder, tsformer_Decoder
from models.Attentions.REVIN import RevIN

from models.build_model_util import series_decomp_multi


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mode = configs.mode
        assert self.mode in ["pretrain", 'finetune', "forecasting"], "Error mode."
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channel = configs.data_dim
        self.embed_dim = configs.embed_dim
        # self.mask_ratio = configs.mask_ratio
        self.encoder_depth = configs.encoder_depth
        self.decoder_depth = configs.decoder_depth
        self.decoder_embed_dim = configs.decoder_embed_dim
        self.dropout = configs.dropout
        self.epoch = 0
        configs.activation = 'gelu'

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)
        self.revin_layer_dec = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)

        # Seasonal encoder and decoder
        self.encoder_seasonal = tsformer_Encoder(configs, mode='Seasonal')
        self.decoder_seasonal = tsformer_Decoder(configs, mode='Seasonal')
        self.output_layer = nn.Conv2d(in_channels=self.embed_dim * self.decoder_seasonal.H_depth,
                                      out_channels=1,
                                      kernel_size=(1, 1))

        # Trend Encoder and Decoder
        self.encoder_trend = tsformer_Encoder(configs, mode='Trend')
        self.decoder_trend = tsformer_Decoder(configs, mode='Trend')
        self.output_layer_trend = nn.Conv2d(in_channels=self.embed_dim * self.decoder_trend.H_depth,
                                            out_channels=1,
                                            kernel_size=(1, 1))

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:
        x_enc = self.revin_layer(x_enc, 'norm')

        x_enc, trend_enc = self.decomp_multi(x_enc)
        x_dec, trend_dec = self.decomp_multi(x_dec)

        # Seasonal
        encoder_output = self.encoder_seasonal(x_enc)
        decoder_output = self.decoder_seasonal(x_dec, encoder_output)

        final_predict = self.decoder_seasonal.decoder_segments[0].concat(decoder_output[0])
        for i in range(1, self.decoder_seasonal.H_depth):
            final_predict = torch.cat((final_predict, self.decoder_seasonal.decoder_segments[i].concat(decoder_output[i])), dim=1)
        final_predict = self.output_layer(final_predict)

        # Trend
        encoder_trend_output = self.encoder_trend(trend_enc)
        decoder_trend_output = self.decoder_trend(trend_dec, encoder_trend_output)
        # Trend output
        trend_predict = self.decoder_trend.decoder_segments[0].concat(decoder_trend_output[0])
        for i in range(1, self.decoder_trend.H_depth):
            trend_predict = torch.cat((trend_predict, self.decoder_trend.decoder_segments[i].concat(decoder_trend_output[i])), dim=1)
        trend_predict = self.output_layer_trend(trend_predict)

        # Concate Trend and Seasonal
        final_predict += trend_predict
        final_predict = rearrange(final_predict, 'b 1 seq_len ts_d -> b seq_len ts_d')[:, -self.pred_len:, :]

        final_predict = self.revin_layer(final_predict, 'denorm')

        return final_predict

