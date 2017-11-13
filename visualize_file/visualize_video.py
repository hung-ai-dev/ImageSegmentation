import cv2
import os
import PIL.Image
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np


def decode_segmap(temp, plot=False):
    temp = temp.astype(np.uint8)
    n_classes = 13
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
    Unlabelled = [0, 0, 0]

    label_colours = np.array([Sky, Building, Pole, Road_marking, Road,
                              Pavement, Tree, SignSymbol, Fence, Car,
                              Pedestrian, Bicyclist, Unlabelled]).astype(np.uint8)
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    
    return rgb.astype(np.uint8)


pred_folder = '/home/hungnd/test'
lbl_folder = '/media/hungnd/Data/LANE_DATASET/CamVid/testannot'
lbls = [f for f in os.listdir(lbl_folder)]

for n in range(232):
    pred = pred_folder + '/' + str(n) + '.png'
    lbl = lbl_folder + '/' + lbls[n]

    print(pred)
    print(lbl)

    pred_img = cv2.imread(pred, 0)
    pred_img = decode_segmap(pred_img, True)

    lbl_img = cv2.imread(lbl, 0)
    lbl_img = decode_segmap(lbl_img, True)
    print(pred_img.shape)
    cv2.imshow('pred', pred_img)
    cv2.imshow('lbl', lbl_img)

    q = cv2.waitKey(-1)
    if q == ord('q'):
        cv2.destroyAllWindows()
        break
