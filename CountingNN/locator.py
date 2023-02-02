from typing import List

import kornia
import torch
import torch.nn.functional as F


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='floor')

    return tuple(reversed(out))


def stich_windows(windows, k, cropx, cropy):
    if not torch.is_tensor(windows):
        windows = torch.as_tensor(windows)
    row0 = torch.cat([windows[0, 0][:-k, :-k]] +
                     [win[:-k, k:-k] for win in windows[0, 1:-1]] +
                     [windows[0, -1][:-k, k:]], dim=1)
    rows = []

    for r in range(windows.shape[0] - 1):
        r = r + 1
        rows.append(torch.cat([windows[r, 0][k:-k, :-k]] +
                              [win[k:-k, k:-k] for win in windows[r, 1:-1]] +
                              [windows[r, -1][k:-k, k:]], dim=1)
                    )

    row_last = torch.cat([windows[-1, 0][k:, :-k]] +
                         [win[k:, k:-k] for win in windows[-1, 1:-1]] +
                         [windows[-1, -1][k:, k:]], dim=1)

    final = torch.cat([row0] + rows + [row_last], dim=0)
    final = final[:cropx, :cropy]
    return final


class Locator:
    """
    Implements Faster R-CNN on a single image to detect boxes for electron events,
    then use finding maximum or pre-trained FCN to assign the entry positions
    The grid_predict and predict function works for image stack, but the locate is set to work for single image.
    Returns boxes as xmin, ymin, xmax,  ymax, x means horizontal and y means vertical.

    Args:
        fastrcnn_model: the loaded fast rcnn model
        device: torch.device('cpu') or torch.device('cuda')
        process_stride: divide the image into pieces when applying the fast rcnn, default 64 for static mode.
        None means no spliting and work on whole frames. Maximum stride 128 for mscdata gpu.
        method: 'max' or 'fcn'
        dark_threshold: the intensity threshold for remove dark noise for image patches with density < 0.01.
        locating_model: the loaded fcn model for assigning entry position
        mode: dynamic mode, whether apply model tune for images with different electron density.
        'static': static parameters, same threshold, no model tune
        'dynamic_window': apply model tune separately for each window
        'dynamic_frame': apply model tune equally the whole frame based on whole frame intensity.
        p_list: optional list of five multiplier for model tune, if none, will use default numbers: [6, 6, 1.3, 1.5, 23]
        meanADU: optional float for mean intensity per electron (ADU), if none, will use default 241 for 200kV.
        dynamic_thres: optional bool for wheather lift the threshold above some density within modeltune.
    Example::

        >>>from CountingNN.locator import Locator
        >>>counting = Locator(model_object, device, process_stride, method,
        >>>dark_threshold, locating_model, dynamic_param, p_list = p_list, meanADU=meanADU)
        >>>filtered = counting.predict_sequence(inputs)  # inputs as the image tensor in shape [N,h,w]

    """

    def __init__(self, fastrcnn_model, device, process_stride=64, method='max', dark_threshold=20, locating_model=None,
                 mode='static', **kwargs):
        super().__init__()
        self.fastrcnn_model = fastrcnn_model
        self.device = device
        self.mode = mode
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
        self.dynamic_thres = kwargs.get('dynamic_thres')
        if self.dynamic_thres is None:
            self.dynamic_thres = True
        self.fastrcnn_model = self.fastrcnn_model.to(self.device)

    def model_tune(self, arr):
        """
        Change the detection limits and thresholds of Fast R-CNN model by estimating the image sparsity
        """
        meanADU = self.meanADU * 4  # mean ADU * upsample_factor^2
        offset = 0
        # fit from 200kV Validation data, between a 64x64
        # up-sampled-by-2 image cell ans its original ground truth.
        limit = int(arr.sum() / meanADU + offset)
        arr_t = torch.as_tensor(arr[None, None, ...] > 30, dtype=torch.float32)
        limit_cca = kornia.contrib.connected_components(arr_t, num_iterations=10)
        limit = max(torch.unique(limit_cca).shape[0], limit)
        if limit < 1:  # make the minimum limit as 1.
            limit = 1
        self.fastrcnn_model.rpn._pre_nms_top_n = {'training': limit * self.p_list[0], 'testing': limit * self.p_list[0]}
        self.fastrcnn_model.rpn._post_nms_top_n = {'training': limit * self.p_list[1],
                                                   'testing': limit * self.p_list[1]}
        self.fastrcnn_model.roi_heads.detections_per_img = int(limit * self.p_list[2])
        self.fastrcnn_model.roi_heads.score_thresh = self.p_list[3] / limit if limit < self.p_list[4] else 0
        self.fastrcnn_model.roi_heads.nms_thresh = 0.02  # smaller, delete more detections

        if limit > (0.005 * arr.shape[0] * arr.shape[1]) and self.dynamic_thres:  # 0.002 is minimum for model13
            self.dark_threshold = 0  # for image that not quite sparse, lift the pre-thresholding.

    def images_to_window_lists(self, inputs: torch.tensor) -> List[torch.tensor]:
        """
        transform a batch of images(3D tensor) into windows of the images, with up-sampling by 2.
        if stride = None, not spliting and return whole images.
        """

        inputs = inputs.to(self.device)
        outputs = []
        maxs = []
        mins = []
        h, w = inputs.shape[1:]

        if self.process_stride is None:
            for image in inputs:
                windows = image[None, None, ...]
                windows = torch.nn.Upsample(scale_factor=2, mode='nearest')(windows)
                outputs = outputs + list(torch.flatten(windows, start_dim=0, end_dim=1))
                maxs = maxs + [image.max()] * (windows.shape[0] * windows.shape[1])
                mins = mins + [image.min()] * (windows.shape[0] * windows.shape[1])
        else:
            torch._assert((torch.as_tensor(inputs.shape[1:]) > self.process_stride).all(),
                          f"Your image dimension is {inputs.shape[1:]}, which is not larger than process stride, "
                          f"please use process_stride<{min(inputs.shape[1:])}"
                          )
            for image in inputs:
                pad = [torch.div(image.shape[0], (self.process_stride - 6), rounding_mode='floor') * (
                        self.process_stride - 6) + self.process_stride,
                       torch.div(image.shape[1], (self.process_stride - 6), rounding_mode='floor') * (
                               self.process_stride - 6) + self.process_stride]  # int works as floor for positive number
                image = F.pad(image,
                              (0, pad[1] - image.shape[1], 0, pad[0] - image.shape[0]))  # left, right, top, bottom !!!

                # the zero pad area make dim counting results due to dynamic modell tune, so fill with edge values
                image[h:, :w] = image[(2 * h - pad[0]):h, :w]
                image[:h, w:] = image[:h, (2 * w - pad[1]):w]
                image[h:, w:] = image[(2 * h - pad[0]):h, w:]

                windows = image.unfold(0, self.process_stride, self.process_stride - 6)
                windows = windows.unfold(1, self.process_stride, self.process_stride - 6)
                # up-sampling the windows
                windows = torch.nn.Upsample(scale_factor=2, mode='nearest')(windows)
                outputs = outputs + list(torch.flatten(windows, start_dim=0, end_dim=1))
                maxs = maxs + [image.max()] * (windows.shape[0] * windows.shape[1])
                mins = mins + [image.min()] * (windows.shape[0] * windows.shape[1])
        return outputs, windows.shape, maxs, mins

    @torch.no_grad()
    def predict_sequence(self, inputs: torch.tensor):
        """
        apply model on image one patch after another. The patches size equals the image size in training data, so that
        no need to tune the model detection limits for different limit sizes.
        """
        self.fastrcnn_model.transform.crop_max = max(inputs.shape[1], inputs.shape[2])*2
        # make size_divisible equals process_stride here to avoid inconsistent padding issue.
        # self.fastrcnn_model.transform.size_divisible = self.process_stride * 2
        self.fastrcnn_model.eval()

        counted_list = []
        eventsize_all = []
        inputs = torch.as_tensor(inputs, dtype=torch.float32)
        counted_images = torch.zeros_like(inputs)

        image_cell_list, windowshape, maxs, mins = self.images_to_window_lists(inputs)
        for i, image_cell in enumerate(image_cell_list):


            if self.mode =='dynamic_window':
                self.model_tune(image_cell)
            elif self.mode =='dynamic_frame':  # incorrect
                image_i = torch.div(i,  windowshape[0] * windowshape[1], rounding_mode='floor')
                self.model_tune(torch.nn.Upsample(scale_factor=2, mode='nearest')(inputs[image_i][None, None, ...]))
            elif self.mode == 'static':
                torch._assert(self.process_stride==64,
                              f"please use process_stride=64 for static mode."
                              )
                pass
            else:
                raise ValueError("Use mode = 'dynamic_window', dynamic_frame or 'static'. ")
                
            # thresholding to remove dark noise before applying the model
            image_cell[image_cell < self.dark_threshold] = 0
            
            image_cell_ori = image_cell

            image_cell = (image_cell - mins[i]) / (maxs[i] - mins[i])  # norm the image cells equally
            # boxes = self.fastrcnn_model([image_cell[None, ...]])[0]['boxes']
            boxes = self.fastrcnn_model(image_cell[None, None, ...])[0]['boxes']  # model direct input [N, C, H, W]

            select = []
            for row, value in enumerate(boxes):
                if 0 < (value[2] - value[0]) < 30 and 0 < (value[3] - value[1]) < 30:
                    select.append(row)
            select = torch.as_tensor(select, dtype=torch.int, device=self.device)
            filtered_boxes = torch.index_select(boxes, 0, select)
            filtered_boxes = filtered_boxes / 2.0
            image_cell_ori = F.interpolate(image_cell_ori[None, None, ...], scale_factor=0.5, mode='nearest')[0, 0]
            filtered, _, eventsize = self.locate(image_cell_ori, filtered_boxes)
            counted_list.append(filtered[None, ...])  # [1,w,h]
            eventsize_all = eventsize_all + eventsize

        image_num = int(len(counted_list) / windowshape[0] / windowshape[1])
        for index in range(image_num):
            counted_cells = counted_list[
                            index * windowshape[0] * windowshape[1]:(index + 1) * windowshape[0] * windowshape[1]]
            counted_cells = torch.cat(counted_cells)
            counted_cells = counted_cells.reshape(windowshape[0], windowshape[1], int(windowshape[2] / 2),
                                                  int(windowshape[3] / 2))
            counted_images[index] = stich_windows(counted_cells, k=3, cropx=inputs.shape[1], cropy=inputs.shape[2])

        return counted_images, eventsize_all

    def locate(self, image_array, boxes):

        width = 10
        filtered = torch.zeros_like(image_array)
        boxes = boxes.round().int()
        coor = []
        eventsize = []

        # if torch.cuda.is_available() and self.device == torch.device('cuda'):
        #     boxes = boxes.cpu()

        for box in boxes:
            xarea = image_array[box[1]:(box[3] + 1), box[0]:(box[2] + 1)]

            # # if the box is just 1~2 pxs, take the intensity value at four line edge instead of padding zero
            # if (xarea.shape[0]+xarea.shape[1]) <= 3 and self.ext_small:
            #     xarea_ext = image_array[box[1]-1:(box[3] + 2), box[0]-1:(box[2] + 2)]
            #     patch = np.pad(xarea_ext, ((0, width - xarea_ext.shape[0] + 2), (0, width - xarea_ext.shape[1] + 2)))

            # one more row and column added at four edges.
            if xarea.shape[0] > (width + 1) or xarea.shape[1] > (width + 1):
                patch = F.pad(xarea, (1, width, 1, width))
                patch = patch[:(width + 2), :(width + 2)]
            else:
                patch = F.pad(xarea, (1, (width - xarea.shape[1] + 1), 1, (width - xarea.shape[0] + 1)))

            if self.method == 'max':

                (model_x, model_y) = unravel_index(torch.argmax(patch), shape=(width + 2, width + 2))
            elif self.method == 'binary_com':
                patch[patch < 30] = 0
                patch[patch >= 30] = 1
                x = torch.linspace(0, patch.shape[0] - 1, patch.shape[0])
                y = torch.linspace(0, patch.shape[1] - 1, patch.shape[1])
                weights_x, weights_y = torch.meshgrid(x, y)
                model_x = (patch * weights_x).sum() / patch.sum()
                model_y = (patch * weights_y).sum() / patch.sum()
                model_x = int(torch.round(model_x))
                model_y = int(torch.round(model_y))
            elif self.method == 'com':
                x = torch.linspace(0, patch.shape[0] - 1, patch.shape[0])
                y = torch.linspace(0, patch.shape[1] - 1, patch.shape[1])
                weights_x, weights_y = torch.meshgrid(x, y)
                model_x = (patch * weights_x).sum() / patch.sum()
                model_y = (patch * weights_y).sum() / patch.sum()
                model_x = int(torch.round(model_x))
                model_y = int(torch.round(model_y))
            else:
                raise ValueError("Use 'max','com,'binary_com' to locate the entry position. ")

            cx = model_x + box[1] - 1
            cy = model_y + box[0] - 1
            if cx > (image_array.shape[0] - 1) or cy > (image_array.shape[1] - 1) or cx < 0 or cy < 0:
                continue
            coor.append((cx, cy))
            eventsize.append((torch.sum(patch > 20)).item())

        coords = torch.as_tensor(coor, dtype=torch.long).to(self.device)
        # eventsize = torch.as_tensor(eventsize, dtype=torch.long).to(self.device)
        if coords.shape[0]:
            filtered[coords[:, 0], coords[:, 1]] = 1
        return filtered, coords, eventsize
