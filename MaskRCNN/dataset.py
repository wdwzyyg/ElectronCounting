import numpy as np
import torch
import torch.nn.functional as F


class GeneralizedDataset:
    """
    Returns:
    image: 256 x 256 tensor float32
    target: dict(image_id(str), boxes(tensor int32), masks(tensor uint8))
    """

    def __init__(self, data_dir, train=False, expandmask=False):
        self.data_dir = data_dir
        self.train = train
        self.expandmask = expandmask

        self.ids = ["%03d" % i + "%03d" % j for i in [*range(50)] for j in [*range(200)]]

    def __getitem__(self, i):
        img_id = self.ids[i]  # filename number 000-049 and index number 000-199
        image = self.get_image(img_id)
        target = self.get_target(img_id) if self.train else {}
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:3] + '_img.npz'
        image = np.load(path)['arr_' + str(int(img_id[3:]))]
        image = torch.tensor(image, dtype=torch.float32)

        return image

    def get_target(self, img_id):
        # boxes format is: x.min, y.min, x.max, y.max

        dir_b = self.data_dir + img_id[:3] + '_box.npz'
        dir_m = self.data_dir + img_id[:3] + '_mask.npz'

        boxes = np.load(dir_b)['arr_' + str(int(img_id[3:]))]
        masks = np.load(dir_m)['arr_' + str(int(img_id[3:]))]
        boxes = torch.tensor(boxes, dtype=torch.int)
        masks = torch.tensor(masks, dtype=torch.uint8)
        if self.expandmask:
            masks_e = torch.zeros(masks.size()[0], 256, 256)
            for i, box in enumerate(boxes):
                mask_e = masks[i]
                if box[0] < 0:
                    mask_e = mask_e[-box[0]:]
                if box[1] < 0:
                    mask_e = mask_e[:, -box[1]:]
                if box[2] > 256:
                    mask_e = mask_e[:(box[2] - 256)]
                if box[3] > 256:
                    mask_e = mask_e[:, :(box[3] - 256)]

                masks_e[i] = F.pad(mask_e, (max(0, box[1]), 256 - max(0, box[1]) - mask_e.size()[1], max(0, box[0]),
                                            256 - max(0, box[0]) - mask_e.size()[0]), "constant", 0)

            masks = masks_e

        target = dict(image_id=img_id, boxes=boxes, masks=masks)
        return target
