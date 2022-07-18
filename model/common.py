import torch
import torch.nn as nn
import MinkowskiEngine as ME

from collections import OrderedDict


class BasicConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dimension, norm_fn=None):
        super().__init__()
        self.downsample = None
        if norm_fn is None:
            norm_fn = ME.MinkowskiBatchNorm
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=False, dimension=dimension)
            )

        self.conv_branch = nn.Sequential(
            norm_fn(in_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, bias=False, dimension=dimension),
            norm_fn(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, bias=False, dimension=dimension)
        )

    def forward(self, x):
        identity = x
        x = self.conv_branch(x)

        if self.downsample is not None:
          identity = self.downsample(identity)
        
        x += identity
        
        return x


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dimension, norm_fn=None):
        super().__init__()
        if norm_fn is None:
            norm_fn = ME.MinkowskiBatchNorm

        self.conv_layers = nn.Sequential(
            norm_fn(in_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, bias=False, dimension=dimension)
        )

    def forward(self, x):
        return self.conv_layers(x)


class UBlock(nn.Module):
    
    def __init__(self, nPlanes, norm_fn, block_reps, block):

        super().__init__()

        self.nPlanes = nPlanes
        self.D = 3

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], self.D, norm_fn) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if len(nPlanes) > 1:
            self.conv = nn.Sequential(
                norm_fn(nPlanes[0]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, dimension=self.D)
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block)

            self.deconv = nn.Sequential(
                norm_fn(nPlanes[1]),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(nPlanes[1], nPlanes[0], kernel_size=2, stride=2, bias=False, dimension=self.D)
            )

            blocks_tail = {'block{}'.format(i): block(nPlanes[0] * (2 - i), nPlanes[0], self.D, norm_fn) for i in range(block_reps)}
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, x):
        out = self.blocks(x)
        identity = out

        if len(self.nPlanes) > 1:
            out = self.conv(out)
            out = self.u(out)
            out = self.deconv(out)

            out = ME.cat(identity, out)

            out = self.blocks_tail(out)

        return out


class SparseConvEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConvolutionBlock(input_dim, 32, 3)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64, 3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(128, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, 3),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x