import torch
from torch import nn
from math import ceil
from einops import rearrange


# Dimension invariant embedding
class DI_embedding(nn.Module):
    def __init__(self, seg_len, embed_dim, dropout):
        super(DI_embedding, self).__init__()
        self.convs = nn.ModuleList()
        self.seg_len = seg_len
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=embed_dim,
                                kernel_size=(3, 1),
                                padding='same')
        # for i in range(seg_len//6):
        #     self.convs.append(BasicBlock(inplanes=embed_dim, planes=embed_dim))
            # self.convs.append(nn.Conv2d(in_channels=embed_dim,
            #                     out_channels=embed_dim,
            #                     kernel_size=(3, 1),
            #                     padding='same'))
        self.norm = nn.LayerNorm(embed_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        return x

class TS_Segment(nn.Module):
    def __init__(self, seq_len, seg_len):
        super(TS_Segment, self).__init__()
        self.seg_len = seg_len
        self.seg_num = ceil(seq_len / self.seg_len)
        self.pad_len = self.seg_num * self.seg_len - seq_len

    def concat(self, x):
        # The shape of x:[batch_size, d_model, seg_num, seg_len, feature dims]
        batch_size, emb_d, seg_num, seg_len, ts_d = x.shape
        x = rearrange(x, 'b d_model seg_num seg_len ts_d -> b d_model (seg_num seg_len) ts_d')
        if self.pad_len != 0:
            x = x[:, :, :(seg_num*seg_len-self.pad_len), :]
        return x

    def forward(self, x):
        # The shape of x:[batch_size, d_model, seq_len, feature dims]
        batch_size, emb_d, ts_len, ts_d = x.shape
        if self.pad_len != 0:
            x = torch.cat([x, torch.zeros(batch_size, emb_d, self.pad_len, ts_d, device=x.device)], dim=2)
        # conduct segmentation to time series data to the time step dimension.
        x_segment = rearrange(x,
                              'b d_model (seg_num seg_len) ts_d -> b d_model seg_num seg_len ts_d',
                              seg_len=self.seg_len)
        return x_segment

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=0)
    # find period by amplitudes
    frequency_list = abs(xf).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[0] // top_list
    period = period[period < 200]
    return period, abs(xf).mean(-1)[top_list]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean