from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model. Just keep crop boxes in targets here for training from the original GeneralizedRCNNTransform

    It returns a Tensor [N, C, H, W] for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
            self,
            size_divisible: int = 32,
            crop_max: int = 64,
            **kwargs: Any,
    ):
        super().__init__()
        self.crop_max = crop_max
        self.size_divisible = size_divisible

    def forward(
            self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        if targets is not None:
            targets = [{k: v for k, v in t.items()} for t in targets]

        for i in range(images.shape[0]):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            if self.crop_max < image.shape[-1]:
                image, target_crop = self.crop(image, target_index)
            else:
                target_crop = None
            images[i] = image
            if targets is not None and target_crop is not None:
                targets[i] = target_crop

        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets

    def crop(
            self,
            image: Tensor,
            target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        dtype, device = image.dtype, image.device
        image = image[:, :self.crop_max, :self.crop_max]

        if target is None:
            return image, target

        bbox = target["boxes"]
        ll = target["labels"]
        select = []
        crop_decision = bbox < self.crop_max
        for row, value in enumerate(crop_decision):
            if value.all():
                select.append(row)
        select = torch.as_tensor(select, dtype=torch.int, device=device)
        bbox = torch.index_select(bbox, 0, select)
        ll = torch.index_select(ll, 0, select)
        target["boxes"] = bbox
        target["labels"] = ll
        return image, target
