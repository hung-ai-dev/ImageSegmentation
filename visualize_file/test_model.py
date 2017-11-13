import torch
import torch.nn as nn
from torch.autograd import Variable
import torchfcn

model = torchfcn.models.FCN8sAtOnce(12)
for m in model.modules():
    print(m)
    print('.....')