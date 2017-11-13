import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2
from torch.utils import data

ignore_label = 19
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32, 255, 255, 255]


class CityscapesLoader(data.Dataset):
    n_classes = 20

    def __init__(self, root, split='train', is_transform=False, img_size=[512, 1024]):
        self.img_size = img_size
        self.is_transform = is_transform
        self.mean = np.array([103.939, 116.779, 123.68])
        self.files = collections.defaultdict(list)
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        self.files = []
        img_path = os.path.join(root, 'leftImg8bit', split)
        mask_path = os.path.join(root, 'gtFine', split)
        categories = os.listdir(img_path)
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + '_gtFine_labelIds.png'))
                self.files.append(item)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path, lbl_path = self.files[index]
        img = m.imread(img_path)
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))    
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)
        lbl_copy = lbl.copy()
        for k, v in self.id_to_trainid.items():
            lbl[lbl_copy == k] = v
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        lbl[lbl == 255] = 0
        lbl = lbl.astype(float)
        lbl = m.imresize(
            lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
  
    def decode_segmap(self, temp, plot=False):
        label_colours = np.array(palette).reshape(-1, 3)
        r = np.zeros_like(temp)
        g = np.zeros_like(temp)
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        rgb = np.array(rgb, dtype=np.uint8)

        if plot:
            cv2.imshow('groudtruth', rgb)
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/media/hung/Data/Dataset'
    dst = CityscapesLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        gt = dst.decode_segmap(labels.numpy()[0], False)

        print('IMG', img)
        print('GT', gt)

        cv2.imshow('Image', img)
        cv2.imshow('Gt', gt)

        q = cv2.waitKey(-1)
        if q == ord('q'):
            cv2.destroyAllWindows()
            break