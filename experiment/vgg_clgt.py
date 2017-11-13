import os.path as osp
import torch
import torch.nn as nn
import torchvision.models


class Encode_A(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[1, 1]):
        super().__init__()
        dil_1 = dilation * grid[0]
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = dilation * grid[0]
        self.conv1_2 = nn.Conv2d(
            out_c, out_c, 3, padding=dil_2, dilation=dil_2)
        self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)
        h = self.dropout(h)

        return h


class Encode_B(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[1, 1, 1]):
        super().__init__()
        dil_1 = dilation * grid[0]
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = dilation * grid[0]
        self.conv1_2 = nn.Conv2d(
            out_c, out_c, 3, padding=dil_2, dilation=dil_2)
        self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        dil_3 = dilation * grid[0]
        self.conv1_3 = nn.Conv2d(
            out_c, out_c, 3, padding=dil_3, dilation=dil_3)
        self.bn1_3 = nn.BatchNorm2d(out_c)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.relu1_3(self.bn1_3(self.conv1_3(h)))
        h = self.pool1(h)
        h = self.dropout()

        return h


class Encode_ASP(nn.Module):
    def __init__(self, in_c, out_c, dilation=1, grid=[1, 1, 1, 1]):
        super().__init__()
        dil_0 = int(dilation * grid[0])
        self.conv1_0 = nn.Conv2d(in_c, out_c, 3, padding=dil_0, dilation=dil_0)
        self.bn1_0 = nn.BatchNorm2d(out_c)
        self.relu1_0 = nn.ReLU(inplace=True)

        dil_1 = int(dilation * grid[1])
        self.conv1_1 = nn.Conv2d(in_c, out_c, 3, padding=dil_1, dilation=dil_1)
        self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu1_1 = nn.ReLU(inplace=True)

        dil_2 = int(dilation * grid[2])
        self.conv1_2 = nn.Conv2d(in_c, out_c, 3, padding=dil_2, dilation=dil_2)
        self.bn1_2 = nn.BatchNorm2d(out_c)
        self.relu1_2 = nn.ReLU(inplace=True)

        dil_3 = int(dilation * grid[3])
        self.conv1_3 = nn.Conv2d(in_c, out_c, 3, padding=dil_3, dilation=dil_3)
        self.bn1_3 = nn.BatchNorm2d(out_c)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.conv1_4 = nn.Conv2d(out_c * 4, out_c, 1)
        self.bn1_4 = nn.BatchNorm2d(out_c)
        self.relu1_4 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x
        h0 = self.relu1_0(self.bn1_0(self.conv1_0(h)))
        h1 = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h2 = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h3 = self.relu1_3(self.bn1_3(self.conv1_3(h)))
        h = torch.cat((h0, h1, h2, h3), 1)
        h = self.relu1_4(self.bn1_4(self.conv1_4(h)))

        return h


class Upsample(nn.Module):
    def __init__(self, in_c, out_c, ks, stride, padding, output_padding):
        super().__init__()
        self.decode = nn.ConvTranspose2d(in_c, out_c, kernel_size=ks, stride=stride,
                                         padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.decode(x))


class FCN_ASP(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        # conv1
        self.encode1 = Encode_A(3, 64)

        # conv2
        self.encode2 = Encode_A(64, 128)

        # conv3
        self.encode3 = Encode_B(128, 256)

        # conv4
        self.encode4 = Encode_B(256, 512)

        # conv5
        self.encode5 = Encode_B(512, 512, grid=[1, 2, 4])

        # conv6
        self.encode6 = Encode_ASP(512, 512, 2, [0.5, 1, 2, 4])

        # decode
        self.decode6_1 = Upsample(512, 256, 3, 2, 1, 1)

        self.decode6_2 = Upsample(256, 128, 3, 2, 1, 1)

        self.decode4 = Upsample(512, 128, 3, 2, 1, 1)

        self.decode46 = Upsample(256, 64, 3, 2, 1, 1)

        self.decode3 = Upsample(256, 64, 3, 2, 1, 1)

        self.decode346 = Upsample(128, 32, 3, 2, 1, 1)

        self.decode2 = Upsample(128, 32, 3, 2, 1, 1)

        self.decode2346 = Upsample(64, n_class, 3, 2, 1, 1)
        

        self._init_weight()

    def forward(self, x):
        h1 = self.encode1(x)
        h2 = self.encode2(h1)
        h3 = self.encode3(h2)
        h4 = self.encode4(h3)
        h5 = self.encode5(h4)
        h6 = self.encode6(h5)

        decode6 = self.decode6_2(self.decode6_1(h6))
        decode4 = self.decode4(h4)
        decode46 = self.decode46(torch.cat((decode4, decode6), 1))

        decode3 = self.decode3(h3)
        decode346 = self.decode346(torch.cat((decode3, decode46), 1))

        decode2 = self.decode2(h2)
        out = self.decode2346(torch.cat((decode2, decode346), 1))

        return out

    def _init_weight(self):
        for new in self.children():
            for l1 in new.children():
                if isinstance(l1, nn.Conv2d) or isinstance(l1, nn.ConvTranspose2d):
                    nn.init.xavier_normal(l1.weight.data)
        print("Init weight done")

    def copy_params_from_vgg16(self):        
        vgg16_bn = torchvision.models.vgg16_bn(True)
        features = vgg16_bn.features
        idx = 0

        num_layer = 5
        for new in self.children():
            for l1 in new.children():
                if isinstance(l1, nn.Dropout2d):
                    continue

                l2 = features[idx]
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
