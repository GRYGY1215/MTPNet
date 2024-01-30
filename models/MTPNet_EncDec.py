import math
import torch
from torch import nn
from einops import rearrange
from models.Attentions.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.Attentions.SelfAttention_Family import FullAttention, AttentionLayer
from models.build_model_util import DI_embedding, TS_Segment


class tsformer_Encoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim

        self.H_depth = len(self.patch_size)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_val_embeddings = nn.ModuleList()
        self.encoder_segments = nn.ModuleList()
        self.encoder_pos_embeds = nn.ParameterList()
        self.encoder_pre_norms = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_feat_fuse = nn.ModuleList()
        for i in range(self.H_depth):
            H_patch_size = self.patch_size[i]
            d_model_lvl = configs.embed_dim * H_patch_size
            dff_lvl = configs.d_ff * H_patch_size

            self.encoder_blocks.append(
                Encoder(
                    [EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                            d_model_lvl,
                            configs.n_heads),
                        d_model=d_model_lvl,
                        d_ff=dff_lvl,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.encoder_depth)
                    ],
                    norm_layer=None
                ))

            self.encoder_val_embeddings.append(DI_embedding(H_patch_size, configs.embed_dim, configs.dropout))
            self.encoder_segments.append(TS_Segment(configs.seq_len, H_patch_size))
            self.encoder_pos_embeds.append(nn.Parameter(torch.randn(1,
                                                                    configs.embed_dim,
                                                                    self.encoder_segments[i].seg_num,
                                                                    H_patch_size,
                                                                    self.in_channel
                                                                    )))
            self.encoder_pre_norms.append(nn.LayerNorm(d_model_lvl))
            self.encoder_norms.append(nn.LayerNorm(d_model_lvl))
            if i > 0:
                self.encoder_feat_fuse.append(nn.Conv2d(in_channels=2*configs.embed_dim,
                                                out_channels=configs.embed_dim,
                                                kernel_size=(1, 1)))

    def forward(self, x_enc):
        encoder_outputs = []
        for i in range(self.H_depth):
            # value embedding
            patches = self.encoder_val_embeddings[i](rearrange(x_enc, 'b seq_len ts_d -> b 1 seq_len ts_d'))
            identity = patches

            if i > 0:
                encoder_output_pre_layer = self.encoder_segments[i - 1].concat(encoder_outputs[i - 1])
                patches = self.encoder_feat_fuse[i - 1](torch.cat((patches, encoder_output_pre_layer), 1))
            # Segment
            patches = self.encoder_segments[i](patches)
            # Add pos
            patches = patches + self.encoder_pos_embeds[i]

            patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
            # PreNorm
            patches = self.encoder_pre_norms[i](patches)

            encoder_output, attns = self.encoder_blocks[i](patches)
            # PostNorm
            encoder_output = self.encoder_norms[i](encoder_output)

            # skip connection
            encoder_output = rearrange(encoder_output,
                                       '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                       seg_len=self.patch_size[i], ts_d=self.in_channel)
            encoder_output = self.encoder_segments[i].concat(encoder_output)
            encoder_output = encoder_output + identity
            encoder_output = self.encoder_segments[i](encoder_output)

            encoder_outputs.append(encoder_output)
        return encoder_outputs


class tsformer_Decoder(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        self.patch_size = configs.patch_size if mode == 'Seasonal' else configs.trend_patch_size
        self.in_channel = configs.data_dim
        self.H_depth = len(self.patch_size)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_val_embeddings = nn.ModuleList()
        self.decoder_pos_embeds = nn.ParameterList()
        self.decoder_segments = nn.ModuleList()
        self.decoder_cross_segments = nn.ModuleList()
        self.decoder_pre_norms = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_feat_fuse = nn.ModuleList()
        for i in range(self.H_depth):
            H_patch_size = self.patch_size[i]
            d_model_lvl = configs.decoder_embed_dim * H_patch_size
            dff_lvl = configs.d_ff * H_patch_size
            self.decoder_blocks.append(
                Decoder(
                    [
                        DecoderLayer(
                            AttentionLayer(
                                FullAttention(False, attention_dropout=configs.dropout,
                                              output_attention=False),
                                d_model_lvl,
                                configs.n_heads),
                            AttentionLayer(
                                FullAttention(False, attention_dropout=configs.dropout,
                                              output_attention=False),
                                d_model_lvl,
                                configs.n_heads),
                            d_model=d_model_lvl,
                            d_ff=dff_lvl,
                            dropout=configs.dropout,
                            activation=configs.activation,
                        )
                        for l in range(configs.decoder_depth)
                    ],
                    norm_layer=None,
                    projection=None
                ))
            self.decoder_val_embeddings.append(DI_embedding(H_patch_size, configs.decoder_embed_dim, configs.dropout))
            self.decoder_cross_segments.append(TS_Segment(configs.seq_len, H_patch_size))
            self.decoder_segments.append(TS_Segment(configs.label_len + configs.pred_len, H_patch_size))
            self.decoder_pos_embeds.append(nn.Parameter(torch.randn(1,
                                                                    configs.decoder_embed_dim,
                                                                    self.decoder_segments[i].seg_num,
                                                                    H_patch_size,
                                                                    self.in_channel
                                                                    )))
            self.decoder_pre_norms.append(nn.LayerNorm(d_model_lvl))
            self.decoder_norms.append(nn.LayerNorm(d_model_lvl))
            if i > 0:
                self.decoder_feat_fuse.append(nn.Conv2d(in_channels=2 * configs.decoder_embed_dim,
                                                        out_channels=configs.decoder_embed_dim,
                                                        kernel_size=(1, 1)))


    def forward(self, x_dec, cross):
        '''
                x: the output of last decoder layer
                cross: the output of the corresponding encoder layer
                '''
        decoder_outputs = []
        for i in range(self.H_depth - 1, -1, -1):
            # value Embedding and add pos info
            patches = self.decoder_val_embeddings[i](rearrange(x_dec, 'b seq_len ts_d -> b 1 seq_len ts_d'))
            identity = patches

            if i != self.H_depth - 1:
                decoder_output_pre_layer = self.decoder_segments[i + 1].concat(decoder_outputs[self.H_depth - 1 - i - 1])
                patches = self.decoder_feat_fuse[i - 1](torch.cat((patches, decoder_output_pre_layer), 1))
            # Segment
            patches = self.decoder_segments[i](patches)
            # Add pos
            patches = patches + self.decoder_pos_embeds[i]
            cross_tmp = cross[i]

            cross_tmp = rearrange(cross_tmp, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
            patches = rearrange(patches, 'b d_model seg_num seg_len ts_d -> (b ts_d) seg_num (seg_len d_model)')
            # decoder
            patches = self.decoder_pre_norms[i](patches)
            decoder_output = self.decoder_blocks[i](patches, cross_tmp)
            decoder_output = self.decoder_norms[i](decoder_output)

            # skip connection
            decoder_output = rearrange(decoder_output,
                                       '(b ts_d) seg_num (seg_len d_model) -> b d_model seg_num seg_len ts_d',
                                       seg_len=self.patch_size[i], ts_d=self.in_channel)

            decoder_output = self.decoder_segments[i].concat(decoder_output)

            decoder_output = decoder_output + identity

            decoder_output = self.decoder_segments[i](decoder_output)
            decoder_outputs.append(decoder_output)
        decoder_outputs.reverse()
        return decoder_outputs

