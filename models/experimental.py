# YOLOv5 experimental modules

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
# from utils.downloads import attempt_download

def attempt_download(file):  # from utils.google_utils import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))
    return str(file)

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights, map_location=torch.device('cpu'), inplace=True):
    from models.yolo import Model, Detect
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        m.inplace = inplace 

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
