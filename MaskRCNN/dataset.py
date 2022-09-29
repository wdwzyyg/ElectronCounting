import numpy as np
import torch
import torch.nn.functional as F


def map01(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


class GeneralizedDataset:
    """
    Returns:
    image: imagesize x imagesize tensor int16
    target: dict(image_id(str), boxes(tensor int16), masks(tensor uint8))
    """

    def __init__(self, data_dir, train=False, filestart=0, filenum=25, upsample=1, getmask= False, expandmask=False, imagesize=256):
        self.data_dir = data_dir
        self.train = train
        self.expandmask = expandmask
        self.getmask = getmask
        self.imagesize = imagesize
        self.upsample = upsample
        self.ids = ["%03d" % i + "%03d" % j for i in [*range(filestart, filenum)] for j in [*range(200)]]

    def __getitem__(self, i):
        img_id = self.ids[i]  # filename number 000-049 and index number 000-199
        image = self.get_image(img_id)
        target = self.get_target(img_id)  # if self.train else {}
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        path = self.data_dir + img_id[:3] + '_img.npz'
        image = np.load(path)['arr_' + str(int(img_id[3:]))]
        image = map01(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image[None, None, ...]
        if self.upsample > 1:
            image = torch.nn.Upsample(scale_factor=self.upsample, mode='bilinear')(image)
        return image[0]  # return dimension [C, H, W]

    def get_target(self, img_id):
        # boxes format is: x.min, y.min, x.max, y.max

        dir_b = self.data_dir + img_id[:3] + '_box.npz'
        dir_m = self.data_dir + img_id[:3] + '_mask.npz'

        boxes = np.load(dir_b)['arr_' + str(int(img_id[3:]))]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones(size=(boxes.size()[0], 1), dtype=torch.int64).flatten()  # required to be int64 and 1D
        # labels = torch.cat((torch.zeros_like(labels), labels), dim=1)  # [0,1] for each box
        # in the original dataset, for box of one pixel, xmin=xmax, ymin=ymax, so add 1 to max to redefine
        # in order to meet requirement in generalized_rcnn.py line 95
        boxes[:, 2] = boxes[:, 2] + 1
        boxes[:, 3] = boxes[:, 3] + 1

        # the x and y in boxes seems to wrong. Swap here.
        boxes[:, [0, 1]] = boxes[:, [1, 0]]
        boxes[:, [2, 3]] = boxes[:, [3, 2]]

        if self.upsample > 1:
            boxes = boxes*self.upsample

        if self.getmask:

            masks = np.load(dir_m)['arr_' + str(int(img_id[3:]))]
            masks = masks.astype('int')
            masks = torch.tensor(masks, dtype=torch.uint8)
            if self.expandmask:
                masks_e = torch.zeros(masks.size()[0], self.imagesize, self.imagesize)
                for i, box in enumerate(boxes):
                    mask_e = masks[i]
                    box = box.type(torch.int)
                    if box[0] < 0:
                        mask_e = mask_e[-box[0]:]
                    if box[1] < 0:
                        mask_e = mask_e[:, -box[1]:]
                    if box[2] > self.imagesize:
                        mask_e = mask_e[:(box[2] - self.imagesize)]
                    if box[3] > self.imagesize:
                        mask_e = mask_e[:, :(box[3] - self.imagesize)]

                    masks_e[i] = F.pad(mask_e, (
                        int(max(0, box[1])), self.imagesize - int(max(0, box[1])) - mask_e.size()[1], int(max(0, box[0])),
                        self.imagesize - int(max(0, box[0])) - mask_e.size()[0]), "constant", 0)

                masks = masks_e
        if self.getmask:
            target = dict(image_ids=torch.tensor(int(img_id), dtype=torch.int), boxes=boxes, masks=masks, labels=labels)
        else:
            target = dict(image_ids=torch.tensor(int(img_id), dtype=torch.int), boxes=boxes, labels=labels)

        return target
