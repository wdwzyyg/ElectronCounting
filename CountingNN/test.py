import time

import numpy as np
import torch

m = torch.load('G:/pycharm/pythonProject/CountingNN/modelweights/model13_final.pt')
from CountingNN.locator import Locator


def fastrcnn_predict(model, arr, device, process_stride, **kwargs):
    x = arr[None, ...]
    # device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #scriptmodel = torch.compile(model)
    counting = Locator(model, device, process_stride, 'max', 30, None, 'dynamic_window', meanADU=kwargs.get('meanADU'),
                       p_list=kwargs.get('p_list'))
    # scripted_counting = torch.jit.script(counting) # can not take instance with arguments
    filtered, event_sizes = counting.predict_sequence(x)
    filtered = filtered[0]
    return filtered


image = np.load('C:/Users/Jingrui Wei/OneDrive/Desktop/test.npy')
s = time.time()
r = fastrcnn_predict(m, image, torch.device('cpu'), 32)
print(r.sum(), time.time() - s)
