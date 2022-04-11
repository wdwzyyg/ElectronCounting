import warnings
import copy
from typing import Union, Tuple, Dict, List
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_loss: Union[List[float], np.ndarray],
                test_loss: Union[List[float], np.ndarray]) -> None:
    """
    Plots train and test losses
    """
    print('Plotting training history')
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()


def set_train_rng(seed: int = 1):
    """
    For reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def num_classes_from_labels(labels: np.ndarray) -> int:
    """
    Gets number of classes from labels (aka ground truth aka masks)

    Args:
        labels (numpy array):
            ground truth (aka masks aka labels) for semantic segmentation

    Returns:
        number of classes
    """
    uval = np.unique(labels)
    if min(uval) != 0:
        raise AssertionError("Labels should start from 0")
    for i, j in zip(uval, uval[1:]):
        if j - i != 1:
            raise AssertionError("Mask values should be in range between "
                                 "0 and total number of classes "
                                 "with an increment of 1")
    num_classes = len(uval)
    if num_classes == 2:
        num_classes = num_classes - 1
    return num_classes


def check_image_dims(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     num_classes: int
                     ) -> Tuple[np.ndarray]:
    """
    Adds if necessary pseudo-dimension of 1 (channel dimensions)
    to images and masks
    """
    if X_train.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training images',
            UserWarning)
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to test images',
            UserWarning)
        X_test = X_test[:, np.newaxis]
    if num_classes == 1 and y_train.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training labels',
            UserWarning)
        y_train = y_train[:, np.newaxis]
    if num_classes == 1 and y_test.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to test labels',
            UserWarning)
        y_test = y_test[:, np.newaxis]

    return X_train, y_train, X_test, y_test


def average_weights(ensemble: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Averages weights of all models in the ensemble

    Args:
        ensemble (dict):
            Dictionary with trained weights (model's state_dict)
            of models with exact same architecture.

    Returns:
        Averaged weights (as model's state_dict)
    """
    ensemble_state_dict = copy.deepcopy(ensemble[0])
    names = [name for name in ensemble_state_dict.keys() if
             name.split('_')[-1] not in ["mean", "var", "tracked"]]
    for name in names:
        w_aver = []
        for model in ensemble.values():
            for n, p in model.items():
                if n == name:
                    w_aver.append(copy.deepcopy(p))
        ensemble_state_dict[name].copy_(sum(w_aver) / float(len(w_aver)))
    return ensemble_state_dict
