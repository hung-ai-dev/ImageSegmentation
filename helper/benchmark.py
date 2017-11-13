#!/usr/bin/env python
import argparse
import os
import os.path as osp
import datetime
import pytz
import yaml
import time
from torch.autograd import Variable
import torch

import torchfcn

def main():
    model = torchfcn.models.FCN8sAtOnce(n_class=21)
    model.cuda()
    t0 = time.time()
    input = torch.rand(1,3,512,512).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    name = 'vgg_flattened'
    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    main()
