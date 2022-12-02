import itertools
import math
from typing import List

import numpy as np
import torch


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


class Locator:
    """
    Implements Faster R-CNN on a single image to detect boxes for electron events,
    then use finding maximum or pre-trained FCN to assign the entry positions
    The grid_predict and predict function works for image stack, but the locate is set to work for single image.
    Returns boxes as xmin, ymin, xmax,  ymax, x means horizontal and y means vertical.

    Args:
        fastrcnn_model: the loaded fast rcnn model
        device: torch.device('cpu') or torch.device('cuda')
        process_stride: divide the image into pieces when applying the fast rcnn, default 64.
        method: 'max' or 'fcn'
        dark_threshold: the intensity threshold for remove dark noise for image patches with density < 0.01.
        locating_model: the loaded fcn model for assigning entry position
        dynamic_param: bool, whether apply model tune for images with different electron density
        p_list: optional list of five multiplier for model tune, if none, will use default numbers: [6, 6, 1.3, 1.5, 23]
        meanADU: optional float for mean intensity per electron (ADU), if none, will use default 241 for 200kV.
    Example::

        >>>from  CountingNN.locator import Locator
        >>>counting = Locator(model_object, device, process_stride, method,
        >>>dark_threshold, locating_model, dynamic_param, p_list = p_list, meanADU=meanADU)
        >>>boxes_list = counting.predict(x) # x as the image array in shape [1,h,w]
        >>>filtered, coords, eventsize = counting.locate(x[0], boxes_list[0])

    """

    def __init__(self, fastrcnn_model, device, process_stride=64, method='max', dark_threshold=20, locating_model=None,
                 dynamic_param=False, **kwargs):
        super().__init__()
        self.fastrcnn_model = fastrcnn_model
        self.device = device
        self.dynamic_param = dynamic_param
        self.process_stride = process_stride
        self.method = method
        self.locating_model = locating_model
        self.dark_threshold = dark_threshold
        self.p_list = kwargs.get('p_list')
        if self.p_list is None:
            self.p_list = [8, 6, 1.5, 1, 50]
        self.meanADU = kwargs.get('meanADU')
        if self.meanADU is None:
            self.meanADU = 241.0

    def model_tune(self, arr):
        """
        Change the detection limits and thresholds of Fast R-CNN model by estimating the image sparsity
        """
        meanADU = self.meanADU * 4  # mean ADU * upsample_factor^2
        offset = 0
        # fit from 200kV Validation data, between a 64x64
        # up-sampled-by-2 image cell ans its original ground truth.
        limit = int(arr.sum() / meanADU + offset)
        if limit < 3:  # make the minimum limit as 3.
            limit = 3
        self.fastrcnn_model.rpn._pre_nms_top_n = {'training': limit * self.p_list[0], 'testing': limit * self.p_list[0]}
        self.fastrcnn_model.rpn._post_nms_top_n = {'training': limit * self.p_list[1],
                                                   'testing': limit * self.p_list[1]}
        self.fastrcnn_model.roi_heads.detections_per_img = int(limit * self.p_list[2])
        self.fastrcnn_model.roi_heads.score_thresh = self.p_list[3] / limit if limit < self.p_list[4] else 0
        self.fastrcnn_model.roi_heads.nms_thresh = 0.02  # smaller, delete more detections

        if limit > (0.004 * arr.shape[0] * arr.shape[1]):  # 0.002 is minimum for model13
            self.dark_threshold = 0  # for image that not quite sparse, lift the pre-thresholding.

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
            for i, j in list(itertools.product(range(int(max_size[1] / self.process_stride)),
                                               range(int(max_size[2] / self.process_stride)))):
                image_cell = divisible_img[:, i * self.process_stride:(i + 1) * self.process_stride,
                             j * self.process_stride:(j + 1) * self.process_stride]

                if self.dynamic_param:
                    self.model_tune(image_cell)

                # thresholding to remove dark noise before applying the model
                image_cell[image_cell < self.dark_threshold] = 0

                # image_cell = map01(image_cell)
                image_cell = (image_cell - whole_img.min()) / (whole_img.max() - whole_img.min())  # norm the image cells equally
                output = self.fastrcnn_model(image_cell[None, ...])[0]['boxes']
                increment = torch.zeros_like(output)
                increment[:, 0] = j * self.process_stride
                increment[:, 2] = j * self.process_stride
                increment[:, 1] = i * self.process_stride
                increment[:, 3] = i * self.process_stride
                boxes.append((output + increment))

                del image_cell, increment, output

            boxes_list.append(torch.cat(boxes, dim=0))

            del divisible_img

        return boxes_list

    def locate(self, image_array, boxes):

        width = 10
        filtered = np.zeros_like(image_array)
        boxes = boxes.round().int()
        coor = []
        eventsize = []

        if torch.cuda.is_available() and self.device == torch.device('cuda'):
            boxes = boxes.cpu()

        for box in boxes:
            xarea = image_array[box[1]:(box[3] + 1), box[0]:(box[2] + 1)]

            # # if the box is just 1~2 pxs, take the intensity value at four line edge instead of padding zero
            # if (xarea.shape[0]+xarea.shape[1]) <= 3 and self.ext_small:
            #     xarea_ext = image_array[box[1]-1:(box[3] + 2), box[0]-1:(box[2] + 2)]
            #     patch = np.pad(xarea_ext, ((0, width - xarea_ext.shape[0] + 2), (0, width - xarea_ext.shape[1] + 2)))

            # one more row and column added at four edges.
            if xarea.shape[0] > (width + 1) or xarea.shape[1] > (width + 1):
                patch = np.pad(xarea, ((1, width), (1, width)))
                patch = patch[:(width + 2), :(width + 2)]
            else:
                patch = np.pad(xarea, ((1, width - xarea.shape[0] + 1), (1, width - xarea.shape[1] + 1)))

            if self.method == 'fcn':
                if torch.cuda.is_available() and self.device == torch.device('cuda'):
                    self.locating_model.cuda()
                    image = torch.tensor(np.expand_dims(patch, axis=(0, 1)), dtype=torch.float).cuda()
                    image = image.cuda()
                    self.locating_model.eval()
                    with torch.no_grad():
                        res = self.locating_model.forward(image)
                    res = res.cpu()
                else:
                    image = torch.tensor(np.expand_dims(patch, axis=(0, 1)), dtype=torch.float)
                    self.locating_model.eval()
                    with torch.no_grad():
                        res = self.locating_model.forward(image)

                prob = torch.sigmoid(res)
                prob = prob.permute(0, 2, 3, 1)  # reshape with channel=last as in tf/keras
                prob = prob.numpy()[0, :, :, 0]

                (model_x, model_y) = np.unravel_index(np.argmax(prob), shape=(width + 2, width + 2))

            elif self.method == 'max':

                (model_x, model_y) = np.unravel_index(np.argmax(patch), shape=(width + 2, width + 2))

            else:
                raise ValueError("Use 'fcn' or 'max' to locate the entry position. ")

            cx = model_x + box[1] - 1
            cy = model_y + box[0] - 1
            if cx > (image_array.shape[0] - 1) or cy > (image_array.shape[1] - 1) or cx < 0 or cy < 0:
                continue
            coor.append((cx, cy))
            eventsize.append(np.sum(patch > 20))

        coords = np.array(coor).astype('int')
        eventsize = np.array(eventsize).astype('int')
        if len(coords):
            filtered[(coords[:, 0], coords[:, 1])] = 1

        return filtered, coords, eventsize

    def predict(self, x: np.ndarray) -> List[torch.tensor]:
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
            # image = map01(im)
            image = torch.tensor(im, dtype=torch.float32)
            image = image[None, None, ...]
            image = torch.nn.Upsample(scale_factor=2, mode='bilinear')(image)
            images.append(image[0])  # return dimension [C, H, W]

        inputs = list(im_.to(self.device) for im_ in images)
        boxes_list = self.grid_predict(inputs)
        for boxes in boxes_list:
            select = []
            for row, value in enumerate(boxes):
                if 0 < (value[2] - value[0]) < 30 and 0 < (value[3] - value[1]) < 30:
                    select.append(row)
            select = torch.as_tensor(select, dtype=torch.int, device=self.device)
            filtered_boxes = torch.index_select(boxes, 0, select)
            filtered_boxes = filtered_boxes / 2.0
            out_list.append(filtered_boxes)

        return out_list
