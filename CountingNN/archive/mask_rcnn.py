from collections import OrderedDict

import torch
import torch.nn.functional as F
from CountingNN.pooler import RoIAlign
from CountingNN.roi_heads import RoIHeads
from CountingNN.rpn import RPNHead, RegionProposalNetwork
from CountingNN.transform import Transformer
from CountingNN.utils import AnchorGenerator
from torch import nn


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.
    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
    The model returns a Dict[Tensor], containing the classification and regression losses
    for both the RPN and the R-CNN, and the mask loss.
    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format,
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).

        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes, for the four parameters.
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.

    """

    def __init__(self, backbone, num_classes, setting):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        # ------------- RPN --------------------------
        anchor_sizes = (1, 2, 4)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=setting.rpn_pre_nms_top_n_train,
                                 testing=setting.rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=setting.rpn_post_nms_top_n_train,
                                  testing=setting.rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head, setting.forcecpu,
            setting.rpn_fg_iou_thresh, setting.rpn_bg_iou_thresh,
            setting.rpn_num_samples, setting.rpn_positive_fraction,
            setting.rpn_reg_weights,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, setting.rpn_nms_thresh)

        # ------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = RoIHeads(
            box_roi_pool, box_predictor,
            setting.box_fg_iou_thresh, setting.box_bg_iou_thresh,
            setting.box_num_samples, setting.box_positive_fraction,
            setting.box_reg_weights,
            setting.box_score_thresh, setting.box_nms_thresh, setting.box_num_detections)

        # self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)

        # layers = (256, 256, 256, 256)
        # dim_reduced = 256
        # self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        # ------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=256, max_size=256,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            size_divisible=32)

    def forward(self, images, targets=None):
        # reference: torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # ori_image_shape = image.shape[-2:]
        images, targets= self.transformer(images, targets)
        image_shape = images[0].shape[-2:]
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposal, rpn_losses = self.rpn(features, image_shape, targets)
        result, roi_losses = self.head(features, proposal, image_shape, targets)
        # auto detect if training
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            # result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result, dict(**rpn_losses, **roi_losses)


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """

        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class CovBackbone(nn.Module):
    def __init__(self, nb_filters: int = 64):
        super(CovBackbone, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, groups=1, padding='same')
        self.point_conv = nn.Conv2d(in_channels=1, out_channels=nb_filters, kernel_size=1)
        self.depth_conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=3, groups=1,
                                     padding='same')
        self.point_conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=1)
        self.out_channels = nb_filters

    def forward(self, x):
        x = F.relu(self.point_conv(self.depth_conv(x)))
        x = F.relu(self.point_conv2(self.depth_conv2(x)))
        return x


def maskrcnn_2conv(pretrained, num_classes, weights_path, setting):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model with pre-trained feature extraction layer.
        num_classes (int): number of classes (including the background).
        weights_path(directory str): the source model path for the feature extraction layers
        setting: class Maskrcnn_param of all the model parameters
    """

    backbone = CovBackbone()

    model = MaskRCNN(backbone, num_classes, setting=setting)

    if pretrained:

        model_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(8, 16)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        for i, name in enumerate(msd):
            msd[name].copy_(pretrained_msd[i])
            if i >= (len(pretrained_msd) - 1):
                break
        model.load_state_dict(msd)

    return model
