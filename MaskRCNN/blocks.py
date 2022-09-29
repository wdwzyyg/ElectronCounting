from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Creates block of layers each consisting of convolution operation,
    leaky relu and (optionally) dropout and batch normalization
    Args:
        ndim:
            Data dimensionality (1D or 2D)
        nb_layers:
            Number of layers in the block
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        kernel_size:
            Size of convolutional filter (in pixels)
        stride:
            Stride of convolutional filter
        padding:
            Value for edge padding
        batch_norm:
            Add batch normalization to each layer in the block
        lrelu_a:
            Value of alpha parameter in leaky ReLU activation
            for each layer in the block
        dropout_:
            Dropout value for each layer in the block
    """
    def __init__(self,
                 ndim: int, nb_layers: int,
                 input_channels: int, output_channels: int,
                 kernel_size: Union[Tuple[int], int] = 1,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 0,  # 1
                 batch_norm: bool = False, lrelu_a: float = 0.01,
                 dropout_: float = 0) -> None:
        """
        Initializes module parameters
        """
        super(ConvBlock, self).__init__()
        if not 0 < ndim < 3:
            raise AssertionError("ndim must be equal to 1 or 2")
        conv = nn.Conv2d if ndim == 2 else nn.Conv1d
        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(conv(input_channels,
                         output_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         padding_mode='circular'))
            if dropout_ > 0:
                block.append(nn.Dropout(dropout_))
            block.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if batch_norm:
                if ndim == 2:
                    block.append(nn.BatchNorm2d(output_channels))
                else:
                    block.append(nn.BatchNorm1d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        output = self.block(x)
        return output


class ResModule(nn.Module):
    """
    Stitches multiple convolutional blocks with residual connections together
    Args:
            ndim: Data dimensionality (1D or 2D)
            res_depth: Number of residual blocks in a residual module
            input_channels: Number of filters in the input layer
            output_channels: Number of channels in the output layer
            batch_norm: Batch normalization for non-unity layers in the block
            lrelu_a: value of negative slope for LeakyReLU activation
    """
    def __init__(self,
                 ndim: int,
                 res_depth: int,
                 input_channels: int,
                 output_channels: int,
                 batch_norm: bool = True,
                 lrelu_a: float = 0.01
                 ) -> None:
        """
        Initializes module parameters
        """
        super(ResModule, self).__init__()
        res_module = []
        for i in range(res_depth):
            input_channels = output_channels if i > 0 else input_channels
            res_module.append(
                ResBlock(ndim, input_channels, output_channels,
                         lrelu_a=lrelu_a, batch_norm=batch_norm))
        self.res_module = nn.Sequential(*res_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        x = self.res_module(x)
        return x


class ResBlock(nn.Module):
    """
    Builds a residual block
    Args:
        ndim:
            Data dimensionality (1D or 2D)
        nb_layers:
            Number of layers in the block
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        kernel_size:
            Size of convolutional filter (in pixels)
        stride:
            Stride of convolutional filter
        padding:
            Value for edge padding
        batch_norm:
            Add batch normalization to each layer in the block
        lrelu_a:
            Value of alpha parameter in leaky ReLU activation
            for each layer in the block
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[Tuple[int], int] = 3,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 1,
                 batch_norm: bool = True,
                 lrelu_a: float = 0.01
                 ) -> None:
        """
        Initializes block's parameters
        """
        super(ResBlock, self).__init__()
        if not 0 < ndim < 3:
            raise AssertionError("ndim must be equal to 1 or 2")
        conv = nn.Conv2d if ndim == 2 else nn.Conv1d
        self.lrelu_a = lrelu_a
        self.batch_norm = batch_norm
        self.c0 = conv(input_channels,
                       output_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       padding_mode='circular')
        self.c1 = conv(output_channels,
                       output_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,  # 1
                       padding_mode='circular')
        self.c2 = conv(output_channels,
                       output_channels,
                       kernel_size=1,
                       stride=1,
                       padding=0,  # 1
                       padding_mode='circular')
        if batch_norm:
            bn = nn.BatchNorm2d if ndim == 2 else nn.BatchNorm1d
            self.bn1 = bn(output_channels)
            self.bn2 = bn(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        x = self.c0(x)
        residual = x
        out = self.c1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        out = self.c2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += residual
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        return out


class UpsampleBlock(nn.Module):
    """
    Defines upsampling block performed using bilinear
    or nearest-neigbor interpolation followed by 1-by-1 convolution
    (the latter can be used to reduce a number of feature channels)
    Args:
        ndim:
            Data dimensionality (1D or 2D)
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        scale_factor:
            Scale factor for upsampling
        mode:
            Upsampling mode. Select between "bilinear" and "nearest"
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int,
                 output_channels: int,
                 scale_factor: int = 2,
                 mode: str = "bilinear") -> None:
        """
        Initializes module parameters
        """
        super(UpsampleBlock, self).__init__()
        if not any([mode == 'bilinear', mode == 'nearest']):
            raise NotImplementedError(
                "use 'bilinear' or 'nearest' for upsampling mode")
        if not 0 < ndim < 3:
            raise AssertionError("ndim must be equal to 1 or 2")
        conv = nn.Conv2d if ndim == 2 else nn.Conv1d
        self.scale_factor = scale_factor
        self.mode = mode if ndim == 2 else "nearest"
        self.conv = conv(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0, padding_mode='circular')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)
