import torch
import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn
import time
from torchfcn.get_class_weights import median_frequency_balancing

resume = ''
cuda = True
data_folder = '/data/hungnd/CamVid'

model = torchfcn.models.FCN8sAtOnce(n_class=13)

if resume:
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration']

model.cuda()
root = data_folder
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
test_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.camvidLoader(root, split='test', is_transform=True),
    batch_size=4, shuffle=False, **kwargs)

folder = 'test'
import cv2
for idx, (data, target) in enumerate(test_loader):
    if cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)

    t0 = time.time()
    score = model(data)
    t1 = time.time()
    print('%f' % (t1 - t0))

    imgs = data.data.cpu()
    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    lbl_pred = lbl_pred.squeeze()
    cv2.imwrite(folder + '/' + str(idx) + 'png', lbl_pred)