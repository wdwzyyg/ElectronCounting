import itertools
import math
from typing import List

import numpy as np
import torch


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


class Locator:
    """
    Implements Faster R-CNN on a single image (1 px events excluded)
    and followed with a FCN trained on Faster R-CNN predicted boxes
    to assign the entry positions


    Args:
        Inputs image patches with fixed width. e.g.12

    Example::

        >>>
    """

    def __init__(self, fastrcnn_model, device, process_stride=64, **kwargs):
        super().__init__()
        self.fastrcnn_model = fastrcnn_model
        self.device = device
        self.prelimit = False
        self.process_stride = process_stride

    def model_tune(self, arr):
        """
        Change the detection limits of Fast R-CNN model by estimating the image sparsity
        ? Best image sparsity estimation function?
        :param arr:
        :return:
        """

    @torch.no_grad()
    def grid_predict(self, inputs: List[torch.tensor]):
        """
        apply model on image one patch after another. The patches size equals the image size in training data, so that
        no need to tune the model detection limits for different limit sizes.
        """
        boxes_list = []
        for whole_img in inputs:
            max_size = list(whole_img.shape)
            max_size[1] = int(math.ceil(float(max_size[1]) / self.process_stride) * self.process_stride)
            max_size[2] = int(math.ceil(float(max_size[2]) / self.process_stride) * self.process_stride)

            divisible_img = whole_img.new_full(max_size, 0)
            divisible_img[: whole_img.shape[0], : whole_img.shape[1], : whole_img.shape[2]].copy_(whole_img)
            boxes = []
            for i, j in list(itertools.product(range(max_size[1]/self.process_stride), range(max_size[2]/self.process_stride) )):
                image_cell = divisible_img[:, i*self.process_stride:(i+1)*self.process_stride,
                             j*self.process_stride:(j+1)*self.process_stride]
                output = self.fastrcnn_model([image_cell])[0]['boxes']
                increment = torch.zeros_like(output)
                increment[:, 0] = i * self.process_stride
                increment[:, 2] = i * self.process_stride
                increment[:, 1] = j * self.process_stride
                increment[:, 3] = j * self.process_stride
                boxes.append(output + increment)

                del image_cell, increment, output

            boxes_list.append(torch.stack(boxes, dim=0))

            del divisible_img

        return boxes_list

    def forward(self, x: np.ndarray) -> List[torch.tensor]:
        """
        input:x: [n, w, h]
        output: list of [N,4]
        """
        self.fastrcnn_model.transform.crop_max = x.shape[1]
        # make size_divisible equals process_stride here to avoid inconsistent padding issue.
        self.fastrcnn_model.transform.size_divisible = self.process_stride
        self.fastrcnn_model.eval()
        images = []
        out_list = []
        for i, im in enumerate(x):
            image = map01(im)
            image = torch.tensor(image, dtype=torch.float32)
            image = image[None, None, ...]
            image = torch.nn.Upsample(scale_factor=2, mode='bilinear')(image)
            images.append(image[0])  # return dimension [C, H, W]

        inputs = list(im_.to(self.device) for im_ in images)
        boxes_list = self.grid_predict(inputs)
        for boxes in boxes_list:
            select = []
            for row, value in enumerate(boxes):
                if 1 < (value[2] - value[0]) < 30 and 1 < (value[3] - value[1]) < 30:
                    select.append(row)
            select = torch.as_tensor(select, dtype=torch.int, device=self.device)
            filtered_boxes = torch.index_select(boxes, 0, select)
            filtered_boxes = filtered_boxes / 2.0
            out_list.append(filtered_boxes)

        return out_list
