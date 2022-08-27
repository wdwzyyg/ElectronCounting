import bisect
import glob
import math
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from MaskRCNN.model import faster_rcnn_fcn


# from MaskRCNN.gpu import collect_gpu_info


class Parameters:
    def __init__(self, batch_size, lr, momentum, weight_decay, lr_steps, epochs, warmup_iters, print_freq, iters,
                 **kwargs):
        """
        :param batch_size:
        :param lr:
        :param momentum:
        :param weight_decay:
        :param lr_steps:
        :param epochs:
        :param warmup_iters:
        :param print_freq:
        :param iters: max iters per epoch, -1 denotes auto for the data loader
        """

        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        self.epochs = epochs
        self.lr_lambda = lambda x: 0.1 ** bisect.bisect(self.lr_steps, x)
        self.warmup_iters = warmup_iters
        self.print_freq = print_freq
        self.iters = iters
        self.ckpt_path = kwargs.get('ckpt_path')
        self.result_path = kwargs.get('result_path')


def plot_train_history(checkpoint_paths):
    """
    plot the four loss curve for all saved check points
    different color for different check points/ epochs
    :param checkpoint_paths: wholes paths of the saved check points.
    """
    roi_box_loss, roi_classifier_loss, rpn_box_loss, rpn_objectness_loss = [], [], [], []
    for i, ckp in enumerate(checkpoint_paths):
        losses = torch.load(ckp)['losses']
        roi_box_loss.append([it['loss_box_reg'].item() for it in losses['train_loss']])
        roi_classifier_loss.append([it['loss_classifier'].item() for it in losses['train_loss']])
        rpn_box_loss.append([it['loss_rpn_box_reg'].item() for it in losses['train_loss']])
        rpn_objectness_loss.append([it['loss_objectness'].item() for it in losses['train_loss']])

    num = len(roi_box_loss[0])
    fig = plt.figure(figsize=(30, 8))
    fig.add_subplot(141)
    for j in range(len(roi_box_loss)):
        plt.scatter(np.arange(num) + j * num, roi_box_loss[j], s=2, alpha=1 - 0.5 * j / len(roi_box_loss))
    plt.xlabel('iters')
    plt.ylabel('roi_box_loss')
    fig.add_subplot(142)
    for j in range(len(roi_classifier_loss)):
        plt.scatter(np.arange(num) + j * num, roi_classifier_loss[j], s=2, alpha=1 - 0.5 * j / len(roi_classifier_loss))
    plt.xlabel('iters')
    plt.ylabel('roi_classifier_loss')
    fig.add_subplot(143)
    for j in range(len(rpn_box_loss)):
        plt.scatter(np.arange(num) + j * num, rpn_box_loss[j], s=2, alpha=1 - 0.5 * j / len(rpn_box_loss))
    plt.xlabel('iters')
    plt.ylabel('rpn_box_loss')
    fig.add_subplot(144)
    for j in range(len(rpn_objectness_loss)):
        plt.scatter(np.arange(num) + j * num, rpn_objectness_loss[j], s=2, alpha=1 - 0.5 * j / len(rpn_objectness_loss))
    plt.xlabel('iters')
    plt.ylabel('rpn_objectness_loss')


def train_one_epoch(model, optimizer, data_loader, device, epoch, param):
    for p in optimizer.param_groups:
        p["lr"] = param.lr_lambda(epoch) * param.lr
    train_loss = []
    iters = len(data_loader) if param.iters < 0 else param.iters
    print('iters per epoch:', iters)
    # iters = len(data_loader)

    model.train()
    A = time.time()
    for i, (images, targets) in enumerate(data_loader):
        num_iters = epoch * len(data_loader) + i
        if num_iters <= param.warmup_iters:
            r = num_iters / param.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * param.lr_lambda(epoch) * param.lr

        # image = image.to(device)
        # target = {k: v.to(device) for k, v in target.items()}
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)

        train_loss.append(losses)
        # total_loss = sum(losses.values())
        total_loss = sum(loss for loss in losses.values())

        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            # sys.exit(1)

        # elif training_mode=='rpn':
        #     total_loss[]
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % param.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        if i >= iters - 1:
            break

    return train_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, arg, generate=True):
    """
    make and save predictions
    """
    test_loss = []
    results = []
    model.eval()

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        res = {target["image_ids"]: output for target, output in zip(targets, outputs)}

    if generate:
        torch.save(res, arg.result_path)

    return res


def collate_fn(batch):
    return tuple(zip(*batch))


def fit(use_cuda, data_set, train_hp, mask_hp):
    """
    :param use_cuda: default True
    :param data_set: GeneralizedDataset(folderpath, True, filenum=1, expandmask=True)
    :param train_hp: training parameters
    """
    pms = Parameters(**train_hp)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    torch.manual_seed(1)
    indices = torch.randperm(len(data_set)).tolist()
    dataset_train = torch.utils.data.Subset(data_set, indices[:-int(0.3 * len(data_set))])
    dataset_test = torch.utils.data.Subset(data_set, indices[-int(0.3 * len(data_set)):])

    print('number of train data :', len(dataset_train))
    print('number of test data :', len(dataset_test))

    # define training and validation data loaders
    d_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=pms.batch_size, shuffle=True,num_workers=4,
        collate_fn=collate_fn)

    d_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # model = faster_rcnn_2conv(True, num_classes=2, weights_path='./modelweights/CNN_smoothl1.tar',
    #                           setting_dict=mask_hp).to(device)
    model = faster_rcnn_fcn(True, num_classes=2, weights_path='./modelweights/TinySegResNet_map01_im_metadict_final.tar',
                            setting_dict=mask_hp).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=pms.lr, momentum=pms.momentum, weight_decay=pms.weight_decay)

    start_epoch = 0
    since = time.time()
    for epoch in range(start_epoch, pms.epochs):
        print("\nepoch: {}".format(epoch + 1))
        A = time.time()
        lr_epoch = pms.lr_lambda(epoch) * pms.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(lr_epoch, pms.lr_lambda(epoch)))
        train_loss = train_one_epoch(model, optimizer, d_train, device, epoch, pms)

        test_res = evaluate(model, d_test, device, epoch, pms, generate=True)

        trained_epoch = epoch + 1

        # save checkpoint
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epochs": trained_epoch, "losses": {'train_loss': train_loss}}

        prefix, ext = os.path.splitext(pms.ckpt_path)
        ckpt_path = "{}-{}{}".format(prefix, trained_epoch, ext)
        torch.save(checkpoint, ckpt_path)

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(pms.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))

    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    plot_train_history(ckpts)  # plot train loss curve
    if start_epoch < pms.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
