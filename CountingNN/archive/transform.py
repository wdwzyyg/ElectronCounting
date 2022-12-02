import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class Transformer:

    def __init__(self, min_size, max_size, image_mean, image_std,size_divisible):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible

    def __call__(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
        ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:

        images = [img for img in images]
        if targets is not None:
            targets = [{k: v for k,v in t.items()} for t in targets]

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        ori_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))

        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        size = [round(s * scale_factor) for s in ori_image_shape]
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        box = target['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
        target['boxes'] = box

        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask

        return image, target

    # def batched_image(self, image, stride=32):
    #     size = image.shape[-2:]
    #     max_size = tuple(math.ceil(s / stride) * stride for s in size)
    #
    #     batch_shape = (1,) + max_size
    #     batched_img = image.new_full(batch_shape, 0)
    #     batched_img[:, :image.shape[-2], :image.shape[-1]] = image
    #
    #     return batched_img[None]

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    # def postprocess(self, result, image_shape, ori_image_shape):
    #     box = result['boxes']
    #     box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
    #     box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
    #     result['boxes'] = box
    #
    #     if 'masks' in result:
    #         mask = result['masks']
    #         mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
    #         result['masks'] = mask
    #
    #     return result

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

        return result


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)

    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)

    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)

        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask