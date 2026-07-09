import torch
import torch.nn as nn

class ConvMCDO(nn.Module):
    def __init__(self, ch_out, kernel_size, stride, dropout_rate=0.7):
        super(ConvMCDO, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_out, out_channels=ch_out,
                              kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
