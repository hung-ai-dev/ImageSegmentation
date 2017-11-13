import torch
import torchvision
import torch.nn as nn

from vgg_clgt import FCN_ASP

asp = FCN_ASP()
asp.copy_params_from_vgg16()