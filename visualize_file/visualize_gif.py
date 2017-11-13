import cv2
import sys
import numpy as np
import argparse
from enet import Enet
import torch
from torch.autograd import Variable
from loader import get_loader
import time


mean = np.array([104.00699, 116.66877, 122.67892])


def get_tensor(img):
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img_size = img.shape
    img = torch.from_numpy(img).float().view(
        1, img_size[0], img_size[1], img_size[2])
    return img


def decode_segmap(temp, plot=False):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colours = np.array([Sky, Building, Pole, Road_marking, Road,
                              Pavement, Tree, SignSymbol, Fence, Car,
                              Pedestrian, Bicyclist])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 12):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def processImage(infile, args):
    n_classes = 12

    model = torchfcn.models.FCN8sAtOnce(n_class=13)
    resume = ''
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration']

    model.cuda()
    model.eval()

    gif = cv2.VideoCapture(infile)
    cv2.namedWindow('camvid')
    while (gif.isOpened()):
        ret, frame = gif.read()
        images = get_tensor(frame)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
        else:
            images = Variable(images)
        t1 = time.time()
        outputs = model(images)
        t2 = time.time()
        print(t2 - t1)
        pred = outputs.data.max(1)[1].cpu().numpy().reshape(360, 480)
        pred = decode_segmap(pred)
        vis = np.zeros((360, 960, 3), np.uint8)
        vis[:360, :480, :] = frame
        vis[:360, 480:960, :] = pred
        cv2.imshow('camvid', vis)
        cv2.waitKey(33)

def main():
