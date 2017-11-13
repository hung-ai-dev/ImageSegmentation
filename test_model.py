import sys
import argparse
import yaml
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import math
import os
import os.path as osp
import shutil
import numpy as np
import pytz
import scipy.misc

import tqdm

import models
import datasets
import helper
from helper import utils
from helper.get_class_weights import median_frequency_balancing

weight = torch.Tensor(median_frequency_balancing()).cuda()
data_folder = '/data/hungnd/CamVid'
cuda = True

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-4,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=1000,
    ),
    2: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=5000,
    )
}

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def test(model, test_loader):
    n_class = len(test_loader.dataset.class_names)

    test_loss = 0
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(test_loader), total=len(test_loader), ncols=80, leave=False):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        loss = cross_entropy2d(score, target, weight)
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while validating')
        test_loss += float(loss.data[0]) / len(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = test_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)

    metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

    test_loss /= len(test_loader)

    with open('test_log.csv', 'a') as f:
        elapsed_time = datetime.datetime.now()
        log = [test_loss] + list(metrics) + [elapsed_time]
        log = map(str, log)
        f.write(','.join(log) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    resume = '/home/hungnd/Github/segmentation/logs/MODEL-vgg-camvid_CFG-001_MAX_ITERATION-100000_LR-0.0001_MOMENTUM-0.99_WEIGHT_DECAY-0.0005_INTERVAL_VALIDATE-1000_TIME-20171021-215528/checkpoint.pth.tar'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = data_folder
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    test_loader = torch.utils.data.DataLoader(
        datasets.camvidLoader(root, split='test', is_transform=True),
        batch_size=4, shuffle=False, **kwargs)

    # 2. model

    model = models.VGG_ASP61(n_class=13)


    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration']

    if cuda:
        model = model.cuda()
    
    test(model, test_loader)

if __name__ == '__main__':
    main()