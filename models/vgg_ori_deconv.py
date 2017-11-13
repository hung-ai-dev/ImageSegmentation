import os.path as osp
import torch
import torch.nn as nn
import torchvision.models


class VGG_ORI_DEC(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        self.deconv6_1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.deconv6_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.deconv6_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu6_3 = nn.ReLU(inplace=True)

        self.deconv7_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.deconv7_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.deconv7_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu7_3 = nn.ReLU(inplace=True)

        self.deconv8_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.deconv8_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu8_2 = nn.ReLU(inplace=True)
        self.deconv8_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu8_3 = nn.ReLU(inplace=True)

        self.deconv9_1 = nn.ConvTranspose2d(64, n_class, 3, 2, 1, 1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.deconv9_2 = nn.Conv2d(n_class, n_class, 3, 1, 1)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.deconv9_3 = nn.Conv2d(n_class, n_class, 3, 1, 1)
        self.relu9_3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))

        h = self.relu6_1(self.deconv6_1(h))
        h = self.relu6_2(self.deconv6_2(h))
        h = self.relu6_3(self.deconv6_3(h))

        h = self.relu7_1(self.deconv7_1(h))
        h = self.relu7_2(self.deconv7_2(h))
        h = self.relu7_3(self.deconv7_3(h))

        h = self.relu8_1(self.deconv8_1(h))
        h = self.relu8_2(self.deconv8_2(h))
        h = self.relu8_3(self.deconv8_3(h))

        h = self.relu9_1(self.deconv9_1(h))
        h = self.relu9_2(self.deconv9_2(h))
        h = self.relu9_3(self.deconv9_3(h))

        return h
    
    def copy_params_from_vgg16(self):
        vgg16 = torchvision.models.vgg16(True)
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)