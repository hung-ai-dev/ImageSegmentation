#!/usr/bin/env python
import sys
import argparse
import os
import os.path as osp
import datetime
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
import helper.joint_transforms as joint_transforms
import helper.transforms as extended_transforms
from datasets import cityscapes


here = osp.dirname(osp.abspath(__file__))
data_folder = '/data/hungnd/CamVid'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    
    out = helper.utils.get_log_dir('cityscapes', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    train_set = cityscapes.CityScapes('fine', 'train', joint_transform=train_joint_transform,
                                        transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=config['train_batch_size'], num_workers=8, shuffle=True)

    val_set = cityscapes.CityScapes('fine', 'val', joint_transform=val_joint_transform, transform=input_transform,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=config['val_batch_size'], num_workers=8, shuffle=False)


    # 2. model

    model = models.VGG_DEC(n_class=20)
    start_epoch = 0
    start_iteration = 0

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        model.copy_params_from_vgg16()

    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'])
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'max', patience = 2, factor=0.5, min_lr=1e-10)

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    weight = helper.get_class_weights(train_folder)

    trainer = helper.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
