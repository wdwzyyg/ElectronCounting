from collections import OrderedDict

import matplotlib.patches as patches
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

    def __init__(self, test_dataset=None, model_object=None, test_part='all', savepath=None, sample=1):
        self.model_object = model_object
        self.test_part = test_part
        self.test_dataset = test_dataset
        self.savepath = savepath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.boxes_list = []
        self.featuremap_list = []
        self.sample = sample
        self.im_list = []
        self.t_list = []

    @torch.no_grad()
    def predict(self):
        self.model_object.eval()
        for i, (im, t) in enumerate(self.test_dataset):
            if i == self.sample:
                break
            im = list(im_.to(self.device) for im_ in im)
            t = [{k: v.to(self.device) for k, v in tt.items()} for tt in t]

            if self.test_part == 'rpn':
                images, targets = self.model_object.transform(im, t)
                self.im_list.append(images)
                self.t_list.append(targets)
                # y = (self.model_object.backbone(images.tensors))
                y = self.model_object.backbone(images.tensors)
                if isinstance(y, torch.Tensor):
                    y = OrderedDict([("0", y)])
                boxes = self.model_object.rpn(images, y, targets)[0][0]
            elif self.test_part == 'all':
                boxes = self.model_object(im)[0]['boxes']
                images, targets = self.model_object.transform(im, t)
                self.im_list.append(images)
                self.t_list.append(targets)
                y = self.model_object.backbone(images.tensors)
                if isinstance(y, torch.Tensor):
                    y = OrderedDict([("0", y)])

            self.boxes_list.append(boxes)
            self.featuremap_list.append(y['0'][0, 0])

    def visualize(self, if_save, **kwargs):
        for i, (_, _) in enumerate(self.test_dataset):
            if i == self.sample:
                break
            if self.device == torch.device("cuda"):
                pred_ = self.boxes_list[i].detach().cpu().numpy()
                feature_ = self.featuremap_list[i].detach().cpu().numpy()
                box_gt = self.t_list[i][0]['boxes'].detach().cpu().numpy()
            else:
                pred_ = self.boxes_list[i]
                feature_ = self.featuremap_list[i]
                box_gt = self.t_list[i][0]['boxes']
            images = self.im_list[i]
            fig = plt.figure(figsize=(6, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(images.tensors[0][0].detach().cpu().numpy(), origin='lower')
            for box in pred_:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin - 0.5, ymin - 0.5), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax1.add_patch(rect)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(feature_, origin='lower')
            for box in box_gt:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin - 0.5, ymin - 0.5), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax2.add_patch(rect)
            plt.show()
            if if_save:
                fig.savefig(kwargs.get('savepath'))

    def calculate_F1(self, threshold_IoU):
        Plist = []
        Rlist = []
        F1list = []
        for i, (_, _) in enumerate(self.test_dataset):
            if i == self.sample:
                break
            nums_pred = self.boxes_list[i].size()[0]
            print('Detected: ', nums_pred)
            box_gt = self.t_list[i][0]['boxes']
            nums_gt = box_gt.size()[0]
            print('Ground truth: ', nums_gt)

            iou_matrix = box_iou(self.boxes_list[i], box_gt)

            if self.device == torch.device("cuda"):
                iou_matrix = iou_matrix.detach().cpu().numpy()
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

            precision = 1.0 * tp / nums_pred
            recall = 1.0 * tp / nums_gt
            if (precision + recall) == 0.0:
                print('Recall = ', recall, 'Precision = ', precision)
                F1 = None
            else:
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
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.ylabel('Precision')
        return Plist, Rlist, F1list
