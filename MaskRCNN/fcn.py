from typing import List

from MaskRCNN.blocks import *


class TinySegResNet(nn.Module):
    """
    Builds a fully convolutional neural network based on residual blocks
    for semantic segmentation
    Args:
        nb_classes:
            Number of classes in the ground truth
        nb_filters:
            Number of filters in 1st residual block
            (gets multiplied by 2 in each next block)
        batch_norm:
            Use batch normalization after each convolutional layer
            (Default: True)
        upsampling_mode:
            Select between "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate,but adds additional (small)
            randomness. For full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
        **layers (list):
            3-element list with a number of residual blocks
            in each residual segment (Default: [2, 2])
    """
    def __init__(self,
                 nb_classes: int = 1,
                 nb_filters: int = 32,
                 batch_norm: bool = True,
                 upsampling_mode: str = "bilinear",
                 **kwargs: List[int]
                 ) -> None:
        """
        Initializes module parameters
        """
        super(TinySegResNet, self).__init__()
        nbl = kwargs.get("layers", [1, 1, 1])
        self.c1 = ConvBlock(
            2, 1, 1, nb_filters, batch_norm=batch_norm # ndim, nbl, inputchannel, outputchannel
        )

        self.bn = ResModule(
            2, nbl[1], nb_filters, nb_filters*2, batch_norm=batch_norm
        )
        self.upsample_block1 = UpsampleBlock(
            2, nb_filters*2, nb_filters, 2, upsampling_mode
        )

        self.c2 = ConvBlock(
            2, 1, nb_filters*2, nb_filters, batch_norm=batch_norm
        )
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines a forward pass"""
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        # Bottleneck
        bn = self.bn(d1)
        # Expanding path
        u1 = self.upsample_block1(bn)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c2(u1)
        # pixel-wise classification
        px = self.px(u1)
        return px
