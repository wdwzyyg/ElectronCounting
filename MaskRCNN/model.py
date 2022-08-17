from typing import List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from MaskRCNN.transform import CustomTransform # cannot use custom transform.
from torchvision.ops import MultiScaleRoIAlign


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


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


def faster_rcnn_2conv(pretrained, num_classes, weights_path, setting_dict):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model with pre-trained feature extraction layer.
        num_classes (int): number of classes (including the background).
        weights_path(directory str): the source model path for the feature extraction layers
        seting_dict: dict of all the model parameters
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
    trainable_layers = 0

    if trainable_layers == 0:
        freeze_before = num_stages
    else:
        freeze_before = stage_indices[num_stages - trainable_layers]

    # for b in backbone[:freeze_before]:
    #     for parameter in b.parameters():
    #         parameter.requires_grad_(False)

    out_channels = 64

    returned_layers = [num_stages - 2, num_stages - 1]
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
    return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    backboneFPN = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=None)
    # transform = GeneralizedRCNNTransform(min_size=256, max_size=256, image_mean=None, image_std=None, fixed_size=(256,256), _skip_resize=True)
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
    model = torchvision.models.detection.FasterRCNN(backbone=backboneFPN, num_classes=num_classes,
                                                    rpn_anchor_generator=anchor_generator,
                                                    box_head=box_head,
                                                    **setting_dict)

    return model
