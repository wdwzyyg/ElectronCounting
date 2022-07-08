import torch

from MaskRCNN.dataset import GeneralizedDataset

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

dataset = GeneralizedDataset('/content/drive/MyDrive/CNN e-detect for Celeritas/Wholeframes/QuantizedData/', 0.5, True, True)

indices = torch.randperm(len(dataset)).tolist()
d_train = torch.utils.data.Subset(dataset, indices)