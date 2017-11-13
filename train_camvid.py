#!/usr/bin/env python
import sys
import argparse
import os
import os.path as osp
import datetime
import json
import pytz
import yaml
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import datasets
import helper
import helper.utils
import models
from datasets import loader
from helper import get_class_weights

here = osp.dirname(osp.abspath(__file__))

configurations = {
    1: dict(
        max_iteration=100000,
        lr=5.0e-4,
        momentum=0.9,
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

def get_data_path(name):
    js = open(here + '/datasets/config.json').read()
    data = json.loads(js)
    return data[name]['data_path']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('-g', '--gpu', type=int, required=True, default = 0)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('-resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = helper.utils.get_log_dir(here, 'camvid', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if use_cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    data_path = get_data_path(args.dataset)
    data_loader = loader.get_loader(args.dataset)
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    # hand-crafted param
    img_size = [480, 640]    
    
    train_loader = torch.utils.data.DataLoader(
        data_loader(data_path, split='train', is_transform=True, img_size = img_size),
            batch_size=4, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        data_loader(data_path, split='val', is_transform=True, img_size = img_size),
            batch_size=4, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        data_loader(data_path, split='test', is_transform=True, img_size = img_size),
            batch_size=4, shuffle=False, **kwargs)

    # 2. model

    model = models.VGG_DECONV(n_class=data_loader.n_classes)
    start_epoch = 0
    start_iteration = 0

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        model.copy_params_from_vgg16()

    if use_cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'])
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'max', patience = 3, factor = 0.5)

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
    
    image_dir = data_path + "trainannot"
    weight = get_class_weights.median_frequency_balancing(image_dir, data_loader.n_classes)
    weight = torch.FloatTensor(weight)
    if use_cuda:
        weight = weight.cuda()

    trainer = helper.Trainer(
        cuda=use_cuda,
        model=model,
        weight = weight,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()

