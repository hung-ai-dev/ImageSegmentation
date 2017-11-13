import os.path as osp
import torch
import torch.nn as nn
import torchvision.models

class Encode_A(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[1, 1]):
        super().__init__()
        dil_1 = dilation * grid[0]
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        # self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = dilation * grid[1]
        self.conv1_2 = nn.Conv2d(out_c, out_c, 3, padding=dil_2, dilation=dil_2)
        # self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

    def forward(self, x):
        h = x
        # h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        # h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        return h


class Encode_B(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[1, 1, 1], is_downsample = True):
        super().__init__()
        self.is_downsample = is_downsample

        dil_1 = dilation * grid[0]
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        # self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = dilation * grid[1]
        self.conv1_2 = nn.Conv2d(out_c, out_c, 3, padding=dil_2, dilation=dil_2)
        # self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        dil_3 = dilation * grid[2]
        self.conv1_3 = nn.Conv2d(out_c, out_c, 3, padding=dil_3, dilation=dil_3)
        # self.bn1_3 = nn.BatchNorm2d(out_c)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

    def forward(self, x):
        h = x
        # h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        # h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        # h = self.relu1_3(self.bn1_3(self.conv1_3(h)))
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.relu1_3(self.conv1_3(h))
        if self.is_downsample:
            h = self.pool1(h)

        return h


class Encode_ASP(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[0, 1, 1, 1]):
        super().__init__()
        # self.conv1_0 = nn.Conv2d(in_c, out_c, 1)
        # self.bn1_0 = nn.BatchNorm2d(out_c)
        # self.relu1_0 = nn.ReLU(inplace=True)

        dil_1 = int(dilation * grid[1])
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        # self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = int(dilation * grid[2])
        self.conv1_2 = nn.Conv2d(in_c, out_c, 3, padding=dil_2, dilation=dil_2)
        # self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        dil_3 = int(dilation * grid[3])
        self.conv1_3 = nn.Conv2d(in_c, out_c, 3, padding=dil_3, dilation=dil_3)
        # self.bn1_3 = nn.BatchNorm2d(out_c)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.conv1_4 = nn.Conv2d(out_c * 3, out_c, 1)
        # self.bn1_4 = nn.BatchNorm2d(out_c)
        self.relu1_4 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x
        # h0 = self.relu1_0(self.bn1_0(self.conv1_0(h)))
        # h1 = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        # h2 = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        # h3 = self.relu1_3(self.bn1_3(self.conv1_3(h)))
        # h = torch.cat((h1, h2, h3), 1)
        # h = self.relu1_4(self.bn1_4(self.conv1_4(h)))

        h1 = self.relu1_1(self.conv1_1(h))
        h2 = self.relu1_2(self.conv1_2(h))
        h3 = self.relu1_3(self.conv1_3(h))
        h = torch.cat((h1, h2, h3), 1)
        h = self.relu1_4(self.conv1_4(h))

        return h

class Decode(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        # self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        # self.bn3 = nn.BatchNorm2d(out_c)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x
        # h = self.relu1(self.bn1(self.deconv1(h)))
        # h = self.relu2(self.bn2(self.deconv2(h)))
        # h = self.relu3(self.bn3(self.deconv3(h)))
        h = self.relu1(self.deconv1(h))
        h = self.relu2(self.deconv2(h))
        h = self.relu3(self.deconv3(h))

        return h

class VGG_DECONV(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        # conv1
        conv1 = Encode_A(3, 64)

        # conv2
        conv2 = Encode_A(64, 128)

        # conv3
        conv3 = Encode_B(128, 256)

        # conv4
        conv4 = Encode_B(256, 512)

        # conv5
        conv5 = Encode_B(512, 512, is_downsample = False)


        deconv6 = Decode(512, 256)
    
        deconv7 = Decode(256, 128)
        
        deconv8 = Decode(128, 64)
        
        deconv9 = Decode(64, n_class)

        self.encoder = nn.Sequential(conv1, conv2, conv3, conv4, conv5)
        self.decoder = nn.Sequential(deconv6, deconv7, deconv8, deconv9)
        
    def forward(self, x):
        h = x
        h = self.encoder(h)
        score = self.decoder(h)
        
        return score

    def freeze_weight(self, limit = 5):
        self.encoder.requires_grad = False

    def __init_weight(self):
        for module in self.children():
            for l in module.children():
                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                     nn.init.kaiming_normal(l.weight.data)

    def copy_params_from_vgg16(self):
        vgg16 = torchvision.models.vgg16(True)
        features = vgg16.features
        idx = 0

        num_layer = 5
        for new in self.encoder.children():
            for l1 in new.children():
                if isinstance(l1, nn.Dropout2d):
                    continue

                l2 = features[idx]
                print("l1", l1)
                print("l2", l2)
                if (isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d)) \
                    or (isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d)):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data.copy_(l1.weight.data)
                    l2.bias.data.copy_(l1.bias.data)
                idx += 1

            num_layer -= 1
            if num_layer == 0:
                break
        print("Copy done")

