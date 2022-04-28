import copy
import warnings
from typing import Union, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def get_array_memsize(X_arr: Union[np.ndarray, torch.Tensor],
                      precision: str = "single") -> float:
    """
    Returns memory size of numpy array or torch tensor
    """
    if X_arr is None:
        return 0
    if isinstance(X_arr, torch.Tensor):
        X_arr = X_arr.cpu().numpy()
    arrsize = X_arr.nbytes
    if precision == "single":
        if X_arr.dtype in ["float64", "int64"]:
            arrsize = arrsize / 2
        elif X_arr.dtype in ["float32", "int32"]:
            pass
        else:
            warnings.warn(
                "Data type is not understood", UserWarning)
    elif precision == "double":
        if X_arr.dtype in ["float32", "int32"]:
            arrsize = arrsize * 2
        elif X_arr.dtype in ["float64", "int64"]:
            pass
        else:
            warnings.warn(
                "Data type is not understood", UserWarning)
    else:
        raise NotImplementedError(
            "Specify 'single' or 'double' precision type")
    return arrsize


def array2list_(x: Union[np.ndarray, torch.Tensor],
                batch_size: int, store_on_cpu: bool = False
                ) -> Union[List[torch.Tensor], List[np.ndarray]]:
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Provide data as numpy array or torch tensor")

    if isinstance(x, torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() and not store_on_cpu else 'cpu'
        x = x.to(device)
        print(x.shape)
    n_batches = int(np.divmod(x.shape[0], batch_size)[0])
    split = np.split if isinstance(x, np.ndarray) else torch.chunk
    return split(x[:n_batches*batch_size], n_batches)


def array2list(X_train: Union[np.ndarray, torch.Tensor],
               y_train: Union[np.ndarray, torch.Tensor],
               X_test: Union[np.ndarray, torch.Tensor],
               y_test: Union[np.ndarray, torch.Tensor],
               batch_size: int, memory_alloc: float = 4
               ) -> Union[Tuple[List[np.ndarray]], Tuple[List[torch.Tensor]]]:
    """
    Splits train and test numpy arrays or torch tensors into lists of
    arrays/tensors of a specified size. The remainders are not included.
    """
    all_data = [X_train, y_train, X_test, y_test]
    arrsize = sum([get_array_memsize(x) for x in all_data])
    store_on_cpu = (arrsize / 1e9) > memory_alloc
    X_train = array2list_(X_train, batch_size, store_on_cpu)
    y_train = array2list_(y_train, batch_size, store_on_cpu)
    X_test = array2list_(X_test, batch_size, store_on_cpu)
    y_test = array2list_(y_test, batch_size, store_on_cpu)
    return X_train, y_train, X_test, y_test


def preprocess_training_image_data_(images_all: Union[np.ndarray, torch.Tensor],
                                    labels_all: Union[np.ndarray, torch.Tensor],
                                    images_test_all: Union[np.ndarray, torch.Tensor],
                                    labels_test_all: Union[np.ndarray, torch.Tensor],
                                    ) -> Tuple[torch.Tensor]:
    """
    Preprocess training and test image data
    """
    all_data = (images_all, labels_all, images_test_all, labels_test_all)
    all_numpy = all([isinstance(i, np.ndarray) for i in all_data])
    all_torch = all([isinstance(i, torch.Tensor) for i in all_data])
    if not all_numpy and not all_torch:
        raise TypeError(
            "Provide training and test data in the form" +
            " of numpy arrays or torch tensors")
    num_classes = num_classes_from_labels(labels_all)
    (images_all, labels_all,
     images_test_all, labels_test_all) = check_image_dims(*all_data, num_classes)
    if all_numpy:
        images_all = torch.from_numpy(images_all)
        images_test_all = torch.from_numpy(images_test_all)
        labels_all = torch.from_numpy(labels_all)
        labels_test_all = torch.from_numpy(labels_test_all)
    images_all, images_test_all = images_all.float(), images_test_all.float()
    if num_classes > 1:
        labels_all, labels_test_all = labels_all.long(), labels_test_all.long()
    else:
        labels_all, labels_test_all = labels_all.float(), labels_test_all.float()

    return (images_all, labels_all, images_test_all,
            labels_test_all, num_classes)


def preprocess_training_image_data(images_all: Union[np.ndarray, torch.Tensor],
                                   labels_all: Union[np.ndarray, torch.Tensor],
                                   images_test_all: Union[np.ndarray, torch.Tensor],
                                   labels_test_all: Union[np.ndarray, torch.Tensor],
                                   batch_size: int, memory_alloc: float = 4
                                   ) -> Tuple[Union[List[np.ndarray], List[torch.Tensor]], int]:
    """
    Preprocess training and test image data
    """

    data_all = preprocess_training_image_data_(
        images_all, labels_all, images_test_all, labels_test_all)
    num_classes = data_all[-1]
    images_all, labels_all, images_test_all, labels_test_all = array2list(
        *data_all[:-1], batch_size, memory_alloc)

    return (images_all, labels_all, images_test_all,
            labels_test_all, num_classes)