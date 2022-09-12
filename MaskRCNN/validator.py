import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops.boxes import box_iou


class Validator:
    """
    used for test model...
    Args:
        test_dataset: the GeneralizedDataset used for test
        model_object: the model to be tested with loaded weights
        test_part: only test rpn if 'rpn' and test the final boxed if 'all'.
    Returns:

    """

    def __init__(self, test_dataset=None, model_object=None, test_part='all'):
        self.model_object = model_object
        self.test_part = test_part
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def calculate_F1(self, threshold_IoU):
        self.model_object.eval()
        Plist = []
        Rlist = []
        F1list = []
        for i, (im, t) in enumerate(self.test_dataset):
            im = list(im_.to(self.device) for im_ in im)
            t = [{k: v.to(self.device) for k, v in t.items()} for t in t]
            if self.test_part=='rpn':
                images, targets = self.model_object.transform(im, t)
                y = (self.model_object.backbone(images.tensors))
                boxes = self.model_object.rpn(images, y, targets)[0][0]
            elif self.test_part=='all':
                boxes = self.model_object(im)[0]['boxes']

            nums_pred = boxes.size()[0]
            box_gt = t[0]['boxes']
            nums_gt = box_gt.size()[0]

            iou_matrix = box_iou(boxes, box_gt)
            tp = 0
            while np.any(iou_matrix > threshold_IoU):
                ind = np.argmax(iou_matrix)
                ind_col = ind % nums_gt
                ind_row = (ind - ind_col) // nums_gt
                tp += 1
                # set the corresponding row and col to zero exclude those already paired from future comparison
                iou_matrix[ind_row][:] = 0
                for ii in range(nums_pred):
                    iou_matrix[ii][ind_col] = 0  # set col to 0

            precision = 1.0 * tp / (np.sum(nums_gt))
            recall = 1.0 * tp / (np.sum(box_gt))
            F1 = 2.0 * recall * precision / (recall + precision)

            Plist.append(precision)
            Rlist.append(recall)
            F1list.append(F1)

        # Plotting Results
        fig = plt.figure()
        plt.plot(Rlist, Plist, marker='o', markerfacecolor='red', markersize=12, color='skyblue', linewidth=4,
                 label='precision')
        plt.title("PR Curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')

