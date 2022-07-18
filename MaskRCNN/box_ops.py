import math

import torch
import torchvision


class BoxCoder:
    """
    Transforming between raw format (xmin, ymin, xmax, ymax) and regression format
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes (gt)
        Arguments:
            reference_boxes (Tensor[N, 4]): reference boxes
            proposals (Tensor[N, 4]): boxes to be encoded
        """

        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_x = reference_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = reference_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log(gt_width / width)
        dh = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """

        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width + ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = torch.exp(dw) * width
        pred_h = torch.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target


@torch.jit.script
def box_iou(box_a, box_b):
    """
    use torch.jit to save GPU memory
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    print("Will compute boxes: ", box_a.size(dim=0),box_b.size(dim=0))
    #box_a = box_a.type(torch.int16)
    #box_b = box_a.type(torch.int16)
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    print("max min done")

    # wh = (rb - lt).clamp(min=0, max=math.inf)
    # inter = wh[:, :, 0] * wh[:, :, 1]
    # area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    #
    # return inter / (area_a[:, None] + area_b - inter)
    N = int(len(box_a))
    M = int(len(box_b))
    iou = torch.zeros([N, M]).to(box_a.device)

    for i in range(0, N, 20):

        area_a = torch.prod(box_a[i:min(i+20, N), 2:] - box_a[i:min(i+20, N), :2], 1)
        wh = (rb[i:min(i+20, N), :] - lt[i:min(i+20, N), :]).clamp(min=0)  # [<=20,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [<=20,M]

        # handle empty boxes
        iou[i:min(i+20, N), :] = torch.where(
            inter > 0,
            inter / (area_a[:, None] + area_b - inter),
            torch.zeros(1, dtype=inter.dtype, device=inter.device),
        )
    return iou


def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small.
    """

    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1])
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0])

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score


def nms(box, score, threshold):
    """
    Arguments:
        box (Tensor[N, 4])
        score (Tensor[N]): scores of the boxes.
        threshold (float): iou threshold.
    Returns:
        keep (Tensor): indices of boxes filtered by NMS.
    """

    return torchvision.ops.nms(box, score, threshold)
