"""
Heavily adapted from https://github.com/ziatdinovmax/atomai/blob/master/atomai/trainers/trainer.py

note for debug: line 158 step(self: int) correct?
self become numpy array in fit()
"""

import copy
import subprocess
import warnings
from collections import OrderedDict
from typing import Type, Union, Tuple, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from CountingNN import metrics
from CountingNN.FCN import SegResNet
from CountingNN.utils import set_train_rng, preprocess_training_image_data, average_weights, plot_losses


class Trainer:
    """
        Base trainer class for training semantic segmentation models

    Example:

    >>> # Initialize a trainer
    >>> t = Trainer()
    >>> # Compile trainer
    >>> t.fit(
    >>>     (images, labels, images_test_1, labels_test_1),
    >>>     loss="ce", training_cycles=25, swa=True)
    >>> t.save_model("my_model")
    """

    def __init__(self):
        set_train_rng(1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.compute_accuracy = False
        self.full_epoch = True
        self.swa = False
        self.perturb_weights = False
        self.running_weights = {}
        self.training_cycles = 0
        self.batch_idx_train, self.batch_idx_test = [], []
        self.batch_size = 1
        self.nb_class = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.train_loader = torch.utils.data.TensorDataset()
        self.test_loader = torch.utils.data.TensorDataset()
        self.data_is_set = False
        self.filename = "model"
        self.print_loss = 1
        self.meta_state_dict = dict()
        self.loss_acc = {"train_loss": [], "test_loss": [],
                         "train_accuracy": [], "test_accuracy": []}

    def weight_perturbation(self, e: int) -> None:
        """
        Time-dependent weights perturbation
        (role of time is played by "epoch" number)
        """
        a = self.perturb_weights["a"]
        gamma = self.perturb_weights["gamma"]
        e_p = self.perturb_weights["e_p"]
        if self.perturb_weights and (e + 1) % e_p == 0:
            var = torch.tensor(a / (1 + e) ** gamma)
            for k, v in self.net.state_dict().items():
                v_prime = v + v.new(v.shape).normal_(0, torch.sqrt(var))
                self.net.state_dict()[k].copy_(v_prime)
        return

    def select_loss(self, loss: str, nb_classes: int = None):
        """
        Selects loss for DCNN model training
        """
        if loss == 'ce' and nb_classes is None:
            raise ValueError("For cross-entropy loss function, you must" +
                             " specify the number of classes")
        # if loss == 'dice':
        #     criterion = dice_loss()
        # elif loss == 'focal':
        #     criterion = focal_loss()
        elif loss == 'ce' and nb_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == 'ce' and nb_classes > 2:
            criterion = torch.nn.CrossEntropyLoss()
        elif loss == 'mse':
            criterion = torch.nn.MSELoss()
        elif hasattr(loss, "__call__"):
            criterion = loss
        else:
            raise NotImplementedError(
                "Select Dice loss ('dice'), focal loss ('focal') "
                " cross-entropy loss ('ce') or means-squared error ('mse')"
                " or pass your custom loss function"
            )
        return criterion

    def dataloader(self,
                   batch_num: int,
                   mode: str = 'train') -> Tuple[torch.Tensor]:
        """
        Generates input training data with images/spectra
        and the associated labels (spectra/images)
        """
        if mode == 'test':
            features = self.X_test[batch_num][:self.batch_size]
            targets = self.y_test[batch_num][:self.batch_size]
        else:
            features = self.X_train[batch_num][:self.batch_size]
            targets = self.y_train[batch_num][:self.batch_size]

        return features, targets

    def step(self, e: int) -> None:
        """
        Single train-test step which passes a single
        mini-batch (for both training and testing), i.e.
        1 "epoch" = 1 mini-batch
        """
        features, targets = self.dataloader(
            self.batch_idx_train[e], mode='train')
        # Training step
        loss = self.train_step(features, targets)
        self.loss_acc["train_loss"].append(loss[0])
        features_, targets_ = self.dataloader(
            self.batch_idx_test[e], mode='test')
        # Test step
        loss_ = self.test_step(features_, targets_)
        self.loss_acc["test_loss"].append(loss_[0])
        if self.compute_accuracy:
            self.loss_acc["train_accuracy"].append(loss[1])
            self.loss_acc["test_accuracy"].append(loss_[1])

    def train_step(self,
                   feat: torch.Tensor,
                   tar: torch.Tensor) -> Tuple[float]:
        """
        Propagates image(s) through a network to get model's prediction
        and compares predicted value with ground truth; then performs
        backpropagation to compute gradients and optimizes weights.

        Args:
            feat: input features
            tar: targets
        """
        self.net.train()
        self.optimizer.zero_grad()
        feat, tar = feat.to(self.device), tar.to(self.device)
        prob = self.net(feat)
        loss = self.criterion(prob, tar)
        loss.backward()
        self.optimizer.step()
        if self.compute_accuracy:
            acc_score = self.accuracy_fn(tar, prob)
            return (loss.item(), acc_score)
        return (loss.item(),)

    def accuracy_fn(self,
                    y: torch.Tensor,
                    y_prob: torch.Tensor,
                    *args):
        iou_score = metrics.IoU(
                y, y_prob, self.nb_class).evaluate()
        return iou_score

    def test_step(self,
                  feat: torch.Tensor,
                  tar: torch.Tensor) -> float:
        """
        Forward pass for test data with deactivated autograd engine

        Args:
            feat: input features
            tar: targets
        """
        feat, tar = feat.to(self.device), tar.to(self.device)
        self.net.eval()
        with torch.no_grad():
            prob = self.net(feat)
            loss = self.criterion(prob, tar)
        if self.compute_accuracy:
            acc_score = self.accuracy_fn(tar, prob)
            return (loss.item(), acc_score)
        return (loss.item(),)

    def save_running_weights(self, e: int) -> None:
        """
        Saves running weights (for stochastic weights averaging)
        """
        swa_epochs = 30
        if self.training_cycles - e <= swa_epochs:
            i_ = swa_epochs - (self.training_cycles - e)
            state_dict_ = OrderedDict()
            for k, v in self.net.state_dict().items():
                state_dict_[k] = copy.deepcopy(v).cpu()
            self.running_weights[i_] = state_dict_
        return

    def print_statistics(self, e: int, **kwargs) -> None:
        """
        Print loss and (optionally) IoU score on train
        and test data, as well as GPU memory usage.
        """
        if torch.cuda.is_available():
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--id=' + str(torch.cuda.current_device()),
                    '--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            gpu_usage = [int(y) for y in result.split(',')][0:2]
        else:
            gpu_usage = ['N/A ', ' N/A']
        if self.compute_accuracy:
            print('Epoch {}/{} ...'.format(e + 1, self.training_cycles),
                  'Training loss: {} ...'.format(
                      np.around(self.loss_acc["train_loss"][-1], 4)),
                  'Test loss: {} ...'.format(
                      np.around(self.loss_acc["test_loss"][-1], 4)),
                  'Train {}: {} ...'.format(
                      'Accuracy',
                      np.around(self.loss_acc["train_accuracy"][-1], 4)),
                  'Test {}: {} ...'.format(
                      'Accuracy',
                      np.around(self.loss_acc["test_accuracy"][-1], 4)),
                  'GPU memory usage: {}/{}'.format(
                      gpu_usage[0], gpu_usage[1]))
        else:
            print('Epoch {}/{} ...'.format(e + 1, self.training_cycles),
                  'Training loss: {} ...'.format(
                      np.around(self.loss_acc["train_loss"][-1], 4)),
                  'Test loss: {} ...'.format(
                      np.around(self.loss_acc["test_loss"][-1], 4)),
                  'GPU memory usage: {}/{}'.format(
                      gpu_usage[0], gpu_usage[1]))

    def eval_model(self) -> None:
        """
        Evaluates model on the entire dataset
        """
        self.net.eval()
        running_loss_test, c = 0, 0
        if self.compute_accuracy:
            running_acc_test = 0
        if self.full_epoch:
            for features_, targets_ in self.test_loader:
                loss_ = self.test_step(features_, targets_)
                running_loss_test += loss_[0]
                if self.compute_accuracy:
                    running_acc_test += loss_[1]
                c += 1
            print('Model (final state) evaluation loss:',
                  np.around(running_loss_test / c, 4))
            if self.compute_accuracy:
                print('Model (final state) IoU:',
                      np.around(running_acc_test / c, 4))
        else:
            running_loss_test, running_acc_test = 0, 0
            for idx in range(len(self.X_test)):
                features_, targets_ = self.dataloader(idx, mode='test')
                loss_ = self.test_step(features_, targets_)
                running_loss_test += loss_[0]
                if self.compute_accuracy:
                    running_acc_test += loss_[1]
            print('Model (final state) evaluation loss:',
                  np.around(running_loss_test / len(self.X_test), 4))
            if self.compute_accuracy:
                print('Model (final state) IoU:',
                      np.around(running_acc_test / len(self.X_test), 4))

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            nb_class: int = 1,
            loss: str = 'ce',
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            training_cycles: int = 1000,
            batch_size: int = 32,
            #compute_accuracy: bool = False,
            swa: bool = False,
            perturb_weights: bool = False,
            **kwargs):

        self.training_cycles = training_cycles
        self.batch_size = batch_size
        #self.compute_accuracy = compute_accuracy
        self.swa = swa
        self.nb_class = nb_class
        #self.optimizer = optimizer
        #self.swa = swa
        self.perturb_weights = perturb_weights
        #self.filename = "model"
        #self.print_loss = kwargs.get("print_loss")
        #self.loss_acc = {"train_loss": [], "test_loss": [],
        #                 "train_accuracy": [], "test_accuracy": []}

        self.net = SegResNet()
        self.net.to(self.device)
        if self.device == 'cpu':
            warnings.warn(
                "No GPU found. The training can be EXTREMELY slow",
                UserWarning)
        weights = self.net.state_dict()

        #############
        # set data #
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=kwargs.get("test_size", .15),
                shuffle=True, random_state=kwargs.get("seed", 1))

        (self.X_train, self.y_train, self.X_test, self.y_test, nb_classes) = preprocess_training_image_data(
            X_train, y_train, X_test, y_test, batch_size, kwargs.get("memory_alloc", 4))

        if nb_classes != self.nb_class:
            raise AssertionError("Number of classes in initialized model" +
                                 " is different from the number of classes" +
                                 " contained in training data")
        #############
        # set ? #
        if self.perturb_weights:
            print("To use time-dependent weights perturbation, turn off the batch normalization layers")
            if isinstance(self.perturb_weights, bool):
                self.perturb_weights = {"a": .01, "gamma": 1.5, "e_p": 50}

        params = self.net.parameters()
        if self.optimizer is None:
            # will be over-witten by lr_scheduler (if activated)
            self.optimizer = torch.optim.Adam(params, lr=1e-3)
        else:
            self.optimizer = optimizer(params)
        self.criterion = self.select_loss(loss, nb_class)

        r = training_cycles // len(X_train)
        self.batch_idx_train = np.arange(
            len(X_train)).repeat(r + 1)[:training_cycles]
        r_ = training_cycles // len(X_test)
        self.batch_idx_test = np.arange(
            len(X_test)).repeat(r_ + 1)[:training_cycles]
        self.batch_idx_train = shuffle(
            self.batch_idx_train, random_state=kwargs.get("batch_seed", 1))
        self.batch_idx_test = shuffle(
            self.batch_idx_test, random_state=kwargs.get("batch_seed", 1))

        if self.print_loss is None:
            print_loss = 100

        self.filename = kwargs.get("filename", "./model")

        # def select_lr(e: int) -> None:
        #     lr_i = (lr_scheduler[e] if e < len(lr_scheduler)
        #             else lr_scheduler[-1])
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr_i

        # Train a neural network, prints the statistics,saves the final model weights.

        for e in range(self.training_cycles):
            self.step(e)
            if self.swa:
                self.save_running_weights(e)
            if perturb_weights:
                self.weight_perturbation(e)
            if any([e == 0, (e + 1) % print_loss == 0,
                    e == training_cycles - 1]):
                self.print_statistics(e)
        # save model weights
        torch.save(self.net.state_dict(), self.filename + '.tar')

        self.eval_model()
        if swa:
            print("Performing stochastic weight averaging...")
            self.net.load_state_dict(average_weights(self.running_weights))
            self.eval_model()

        plot_losses(self.loss_acc["train_loss"], self.loss_acc["test_loss"])

        return self.net
