from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign

from MaskRCNN.anchor_custom import AnchorGenerator
from MaskRCNN.faster_rcnn_custom import FasterRCNN, TwoMLPHead
from MaskRCNN.fcn import TinySegResNet, TinySegResNet_ori


def map01(raw):
    raw -= raw.min()
    raw /= raw.max()
    return raw


class channellayer(nn.Module):
    """ Custom layer to delete two channels in input """

    def __init__(self):
        super().__init__()
        self.out_channels = 1

    def forward(self, x):
        return x[:, :1, :, :]


class CovBackbone(nn.Sequential):
    def __init__(self, nb_filters: int = 64):
        super(CovBackbone, self).__init__()
        self.channel2one = channellayer()
        self.depth_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, groups=1, padding='same')
        self.point_conv = nn.Conv2d(in_channels=1, out_channels=nb_filters, kernel_size=1)
        self.depth_conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=3, groups=1,
                                     padding='same')
        self.point_conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=1)
        self.out_channels = nb_filters
        features: List[nn.Module] = [self.channel2one, self.depth_conv, self.point_conv, self.depth_conv2,
                                     self.point_conv2]

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.channel2one(x)
        x = F.relu(self.point_conv(self.depth_conv(x)))
        # x = F.relu(self.point_conv(self.depth_conv(x)))

        x = F.relu(self.point_conv2(self.depth_conv2(x)))
        return x


class FCNBackbone(nn.Sequential):
    def __init__(self,
                 kernel: int = 3):
        super(FCNBackbone, self).__init__()
        self.channel2one = channellayer()
        self.kernel = kernel
        if self.kernel == 3:
            self.fcov = TinySegResNet_ori()
        elif self.kernel == 1:
            self.fcov = TinySegResNet()
        else:
            raise ValueError("use kernel = 3 or 1")

        features: List[nn.Module] = [self.channel2one] + list(self.fcov.c1.block.children())[:1] + list(
            self.fcov.bn.res_module[0].children())[:3]

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.channel2one(x)
        x = self.fcov(x)
        return x


class Mathlayers(nn.Module):
    """
  input: tensor of shape [batch, c, h, w]
  output: tensor of shape [batch, 4*c, h, w]
  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.clip(map01(torch.log(x + 0.01)), 0.15)
        b = torch.clip(map01(torch.sqrt(x)), 0.1)
        c = torch.clip(map01(torch.cat(list(torch.gradient(x, dim=[-1, -2])), dim=1)), 0.1)
        res = torch.cat((a, b, c), dim=1)
        return res


class MathBackbone(nn.Module):
    def __init__(self):
        super(MathBackbone, self).__init__()
        self.channel2one = channellayer()
        self.math = Mathlayers()
        self.math2 = Mathlayers()
        self.out_channels = 16

    def forward(self, x):
        x = self.channel2one(x)
        x = self.math(x)
        x = self.math2(x)
        return x


def faster_rcnn_math(num_classes, setting_dict):
    """
    Constructs a Mask R-CNN model with a math backbone.

    Arguments:
        num_classes (int): number of classes (including the background).
        setting_dict: dict of all the model parameters
    """

    backbone = MathBackbone()

    anchor_generator = AnchorGenerator(sizes=((1, 2,),), aspect_ratios=((0.25, 1, 2),), stride_multiplier=1)
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

    resolution = box_roi_pool.output_size[0]  # 7
    representation_size = 1024
    box_head = TwoMLPHead(backbone.out_channels * resolution ** 2, representation_size)  # (2*7^2, 1024)
    # box_predictor = FastRCNNPredictor(representation_size, num_classes=None)

    # load an instance segmentation model pre-trained on COCO
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator, rpn_head=rpn_head,
                       box_head=box_head, box_roi_pool=box_roi_pool,
                       **setting_dict)
    return model


def faster_rcnn_fcn(kernel, pretrained, num_classes, weights_path, setting_dict):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        kernel(int): 3 or 1 for the cov layers in TinySegNet.
        pretrained (bool): If True, returns a model with pre-trained feature extraction layer.
        num_classes (int): number of classes (including the background).
        weights_path(directory str): the source model path for the feature extraction layers
        setting_dict: dict of all the model parameters

    """

    backbone = FCNBackbone(kernel=kernel)

    if pretrained:
        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        backbone.fcov.load_state_dict(model_state_dict['state_dict'])

    backbone = backbone.features

    stage_indices = [1, 2, 4]  # the three conv layers with original resolution
    num_stages = len(stage_indices)

    backbone.out_channels = 64
    returned_layers = [0, 1, 2]
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
    return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]

    backboneFPN = BackboneWithFPN(backbone, return_layers, in_channels_list, backbone.out_channels, extra_blocks=None)

    # change the padding model of cov layers in backboneFPN
    for layer in backboneFPN.fpn.inner_blocks:
        list(layer.children())[0].padding_mode = 'circular'
    for layer in backboneFPN.fpn.layer_blocks:
        list(layer.children())[0].padding_mode = 'circular'

    anchor_generator = AnchorGenerator(sizes=((1, 2, 4),) * 4, aspect_ratios=((0.5, 1, 2),) * 4, stride_multiplier=1)
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "pool"], output_size=7, sampling_ratio=2)
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

    resolution = box_roi_pool.output_size[0]  # 7
    representation_size = 1024
    box_head = TwoMLPHead(backbone.out_channels * resolution ** 2, representation_size)  # (64*7^2, 1024)
    # box_predictor = FastRCNNPredictor(representation_size, num_classes=None)

    # load an instance segmentation model pre-trained on COCO
    model = FasterRCNN(backbone=backboneFPN, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator, rpn_head=rpn_head,
                       box_head=box_head, box_roi_pool=box_roi_pool,
                       **setting_dict)

    return model


def faster_rcnn_2conv(pretrained, num_classes, weights_path, setting_dict):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model with pre-trained feature extraction layer.
        num_classes (int): number of classes (including the background).
        weights_path(directory str): the source model path for the feature extraction layers
        setting_dict: dict of all the model parameters
    """

    backbone = CovBackbone()

    if pretrained:
        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(8, 16)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = backbone.state_dict()
        for i, name in enumerate(msd):
            msd[name].copy_(pretrained_msd[i])
            if i >= (len(pretrained_msd) - 1):
                break
        backbone.load_state_dict(msd)

    backbone = backbone.features
    backbone.out_channels = 64
    stage_indices = [0] + [
        i for i, b in enumerate(backbone)
        if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)
    # trainable_layers = 0
    #
    # if trainable_layers == 0:
    #     freeze_before = num_stages
    # else:
    #     freeze_before = stage_indices[num_stages - trainable_layers]

    # for b in backbone[:freeze_before]:
    #     for parameter in b.parameters():
    #         parameter.requires_grad_(False)

    out_channels = 64

    returned_layers = [num_stages - 2, num_stages - 1]
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
    return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    backboneFPN = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=None)
    print(backboneFPN)
    # transform = GeneralizedRCNNTransform(min_size=256, max_size=256, image_mean=None,
    # image_std=None, fixed_size=(256,256), _skip_resize=True)
    # _skip_resize=True does not take effect
    anchor_generator = AnchorGenerator(sizes=(1, 2, 4), aspect_ratios=(0.5, 1, 2))
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    # anchor_generator = AnchorGenerator(sizes=(1, 2, 3, 4), aspect_ratios=(0.25, 0.5, 1, 2))

    resolution = box_roi_pool.output_size[0]  # 7
    representation_size = 1024
    out_channels = backbone.out_channels
    box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)  # (2*7^2, 1024)
    # box_predictor = FastRCNNPredictor(representation_size, num_classes=None)

    # load an instance segmentation model pre-trained on COCO
    model = FasterRCNN(backbone=backboneFPN, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_head=box_head,
                       **setting_dict)

    return model
