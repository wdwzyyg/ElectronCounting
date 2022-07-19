import bisect
import glob
import os
import re
import sys
import time

import torch

from MaskRCNN import mask_rcnn
from MaskRCNN.dataset import GeneralizedDataset

usecuda = True
data = GeneralizedDataset('./data/', True, filenum=1, expandmask=True)
args = {
    'batch_size': 1,
    'lr': 1 / 16 * 0.02,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr_steps': [6, 7],
    'epochs': 3,
    'warmup_iters': 1000,
    'print_freq': 1,
    'iters': 10
}


class Parameters:
    def __init__(self, batch_size, lr, momentum, weight_decay, lr_steps, epochs, warmup_iters, print_freq, iters, **kwargs):
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, param):
    for p in optimizer.param_groups:
        p["lr"] = param.lr_lambda(epoch) * param.lr

    # iters = len(data_loader) if args.iters < 0 else args.iters
    iters = len(data_loader)

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
    return A / iters


# generate results file
@torch.no_grad()
def generate_results(model, data_loader, device, param):
    """
    make and save predictions
    """
    iters = len(data_loader) if param.iters < 0 else param.iters

    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        # torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)

        prediction = {target["image_ids"].item(): {k: v.cpu() for k, v in output.items()}}

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg))
    torch.save(prediction, param.result_path)

    return A / iters


def evaluate(model, data_loader, device, param, generate=True):
    # ref: https: // github.com / cocodataset / cocoapi / blob / master / PythonAPI / pycocotools / cocoeval.py
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, param)

    dataset = data_loader  #
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(param.result_path, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp

    return output, iter_eval


def fit(use_cuda=usecuda, data_set=data):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    indices = torch.randperm(len(data_set)).tolist()
    d_train, d_test = torch.utils.data.random_split(data_set, [int(len(data_set)/2), int(len(data_set)/2)])
    model = mask_rcnn.maskrcnn_2conv(True, num_classes=1, weights_path='./modelweights/CNN_smoothl1.tar').to(device)

    pms = Parameters(**args)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=pms.lr, momentum=pms.momentum, weight_decay=pms.weight_decay)

    start_epoch = 0

    for epoch in range(start_epoch, args.get('epochs')):
        print("\nepoch: {}".format(epoch + 1))
        A = time.time()
        lr_epoch = pms.lr_lambda(epoch) * pms.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(lr_epoch, pms.lr_lambda(epoch)))
        iter_train = train_one_epoch(model, optimizer, d_train, device, epoch, pms)
        A = time.time() - A

        B = time.time()
        eval_output, iter_eval = evaluate(model, d_test, device, pms)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        # print(eval_output.get_AP())

        save_ckpt(model, optimizer, trained_epoch, pms.ckpt_path, eval_info=str(eval_output))
        # save checkpoint
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epochs": trained_epoch, "eval_output": eval_output}

        prefix, ext = os.path.splitext(ckpt_path)
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
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))