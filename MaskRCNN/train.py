import bisect
import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from MaskRCNN import mask_rcnn
from MaskRCNN.gpu import collect_gpu_info


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


class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)


def plot_train_history(checkpoint_paths):
    """
    plot the four loss curve for all saved check points
    different color for different check points/ epochs
    :param checkpoint_paths: wholes paths of the saved check points.
    """
    roi_box_loss,roi_classifier_loss, rpn_box_loss, rpn_objectness_loss= [],[],[],[]
    for i, ckp in enumerate(checkpoint_paths):
        losses = torch.load(ckp)['losses']
        roi_box_loss.append([it['roi_box_loss'].item() for it in losses['train_loss']])
        roi_classifier_loss.append([it['roi_classifier_loss'].item() for it in losses['train_loss']])
        rpn_box_loss.append([it['roi_classifier_loss'].item() for it in losses['train_loss']])
        rpn_objectness_loss.append([it['rpn_objectness_loss'].item() for it in losses['train_loss']])

    num = len(roi_box_loss[0])
    fig = plt.figure(figsize=(30, 8))
    fig.add_subplot(141)
    for j in range(len(roi_box_loss)):
        plt.scatter(np.arange(num)+j*num, roi_box_loss[j], s=2, alpha=1-0.5*j/len(roi_box_loss))
    plt.xlabel('iters')
    plt.ylabel('roi_box_loss')
    fig.add_subplot(142)
    for j in range(len(roi_classifier_loss)):
        plt.scatter(np.arange(num)+j*num, roi_classifier_loss[j], s=2, alpha=1-0.5*j/len(roi_classifier_loss))
    plt.xlabel('iters')
    plt.ylabel('roi_classifier_loss')
    fig.add_subplot(143)
    for j in range(len(rpn_box_loss)):
        plt.scatter(np.arange(num)+j*num, rpn_box_loss[j], s=2, alpha=1-0.5*j/len(rpn_box_loss))
    plt.xlabel('iters')
    plt.ylabel('rpn_box_loss')
    fig.add_subplot(144)
    for j in range(len(rpn_objectness_loss)):
        plt.scatter(np.arange(num)+j*num, rpn_objectness_loss[j], s=2, alpha=1-0.5*j/len(rpn_objectness_loss))
    plt.xlabel('iters')
    plt.ylabel('rpn_objectness_loss')


def train_one_epoch(model, optimizer, data_loader, device, epoch, param):
    for p in optimizer.param_groups:
        p["lr"] = param.lr_lambda(epoch) * param.lr
    train_loss = []
    iters = len(data_loader) if param.iters < 0 else param.iters
    # iters = len(data_loader)

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= param.warmup_iters:
            r = num_iters / param.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * param.lr_lambda(epoch) * param.lr

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()

        losses = model(image, target)
        train_loss.append(losses)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)

        optimizer.step()
        optimizer.zero_grad()

        if num_iters % param.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg,
                                                                                1000 * m_m.avg, 1000 * b_m.avg))
    return A / iters, train_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, arg, generate=True):
    """
    make and save predictions
    """
    iters = len(data_loader) if arg.iters < 0 else arg.iters
    test_loss = []
    t_m = Meter("total")
    m_m = Meter("model")
    results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        # torch.cuda.synchronize()
        output, losses = model(image)
        test_loss.append(losses)
        if num_iters % arg.print_freq == 0:
            print("Test: {}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
        m_m.update(time.time() - S)

        target["image_ids"] = "%06d" % target["image_ids"]  # int to str
        prediction = {target["image_ids"]: {k: v.cpu() for k, v in output.items()}}
        results.extend(prediction)
        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg))
    if generate:
        torch.save(results, arg.result_path)

    return A / iters, test_loss


def fit(use_cuda, data_set, hp, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    indices = torch.randperm(len(data_set)).tolist()
    d_train, d_test = torch.utils.data.random_split(data_set, [int(len(data_set) / 2), int(len(data_set) / 2)])
    model = mask_rcnn.maskrcnn_2conv(True, num_classes=2, rpn_param=kwargs.get('rpn_param', None),
                                     roihead_param=kwargs.get('rpn_param', None),
                                     weights_path='./modelweights/CNN_smoothl1.tar').to(device)

    pms = Parameters(**hp)
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
        iter_train, train_loss = train_one_epoch(model, optimizer, d_train, device, epoch, pms)
        A = time.time() - A

        B = time.time()
        iter_eval, test_loss = evaluate(model, d_test, device, epoch, pms, generate=True)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        if torch.cuda.is_available():
            collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])

        # save checkpoint
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epochs": trained_epoch, "losses": {'train_loss': train_loss, 'test_loss': test_loss}}

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
    plot_train_history(ckpts) # plot train loss curve
    if start_epoch < pms.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))


def run(usecuda, data, hyperparams, **kwargs):
    """
    :param usecuda: default True
    :param data: GeneralizedDataset(folderpath, True, filenum=1, expandmask=True)
    :param hyperparams: {
    'batch_size': 1,
    'lr': 1 / 16 * 0.02,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr_steps': [6, 7],
    'epochs': 3,
    'warmup_iters': 1000,
    'print_freq': 1,
    'iters': 10,
    'result_path': 'G:/pycharm/pythonProject/results/model_results.pth'}
    :param : rpn_param: {
    'rpn_fg_iou_thresh': 0.7, 'rpn_bg_iou_thresh': 0.3,
    'rpn_num_samples': 256, 'rpn_positive_fraction': 0.5,
    'rpn_reg_weights': (1., 1., 1., 1.),
    'rpn_pre_nms_top_n_train': 2000, 'rpn_pre_nms_top_n_test': 1000,
    'rpn_post_nms_top_n_train': 2000, 'rpn_post_nms_top_n_test': 1000,
    'rpn_nms_thresh': 0.7,}
    :param : roihead_param: {
    'box_fg_iou_thresh': 0.5, 'box_bg_iou_thresh': 0.5,
    'box_num_samples': 512, 'box_positive_fraction': 0.25,
    'box_reg_weights': (10., 10., 5., 5.),
    'box_score_thresh': 0.1, 'box_nms_thresh': 0.6,
    'box_num_detections': 10,}

    """

    fit(use_cuda=usecuda, data_set=data, hp=hyperparams, **kwargs)
