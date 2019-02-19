import time
import torch
import torch.nn as nn
from torch_deform_conv.layers import ConvOffset2D


class TestDCN(nn.Module):
    def __init__(self, in_channels=128):
        super(TestDCN, self).__init__()
        self.deformable_conv = ConvOffset2D(in_channels)

    def forward(self, x):
        x = self.deformable_conv(x)
        return x


class TestConv(nn.Module):
    def __init__(self, in_channels=128):
        super(TestConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def unit_test_DCN():
    in_channels = 512
    net_dcn = TestDCN(in_channels)
    net_conv = TestConv(in_channels)
    inputs = torch.randn(4, in_channels, 120, 80)
    if torch.cuda.is_available():
        net_dcn = net_dcn.cuda()
        net_conv = net_conv.cuda()
        inputs = inputs.cuda()
    old = time.time()
    output = net_dcn(inputs)
    new = time.time()
    print('net_dcn time:{}'.format(new - old))
    print('net_dcn output shape:{}'.format(output.shape))
    old = time.time()
    output = net_conv(inputs)
    new = time.time()
    print('net_conv time:{}'.format(new - old))
    print('net_conv output shape:{}'.format(output.shape))


if __name__ == "__main__":
    unit_test_DCN()
