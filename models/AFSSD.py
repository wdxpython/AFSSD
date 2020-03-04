import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FEM(nn.Module):
    def __init__(self, inplanes, outplanes, change = None, downsample=None, stride=1, scales=4):
        super(FEM, self).__init__()
        self.conv1x1 = BasicConv(inplanes, inplanes // scales, kernel_size=1)
        self.conv = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.convs1 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(3, 1), stride=1, padding=(1, 0)))

        self.convs2 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1, dilation=(1,3), padding=(0, 3)))

        self.convs3 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1, dilation=(1, 7),
                      padding=(0, 7)))

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.scales = scales
        self.relu = nn.ReLU()


        self.out_channels = outplanes

        self.conv3 = BasicConv(inplanes, outplanes,  kernel_size=1)
        self.conv4 = BasicConv(inplanes, outplanes, kernel_size=3,stride=2,padding=1)
        self.change = change
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1x1(x)
        xs = []
        xs.append(self.conv(x))
        xs.append(x)
        xs.append(x)
        xs.append(x)
        ys = []

        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            if s == 1:
                ys.append(self.convs1(xs[0] + ys[-1]))
            if s == 2:
                ys.append(self.convs2(xs[s] + ys[-1]))
            if s == 3:
                ys.append(self.convs3(xs[s] + ys[-1]))

        out = torch.cat([ys[0], ys[1], ys[2], ys[3]], dim=1)
        out = self.relu(out + identity)
        if self.change:
            out = self.conv3(out)
        if self.downsample:
            out = self.conv4(out)
        return out

class FEMs(nn.Module):
    def __init__(self, inplanes, outplanes, change = None, downsample=None, stride=1, scales=4):
        super(FEMs, self).__init__()
        self.conv1x1 = BasicConv(inplanes, inplanes // scales, kernel_size=1)
        self.conv = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.convs1 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)))

        self.convs2 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1,
                      padding=(0, 1)))

        self.convs3 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(1, 3), stride=1, dilation=(1, 7),
                      padding=(0, 7)))

        self.convs4 = nn.Sequential(
            BasicConv(inplanes // scales, inplanes // scales, kernel_size=(3, 1), stride=1, dilation=(3,1), padding=(3, 0)))

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.scales = scales
        self.relu = nn.ReLU()

        self.convchange = BasicConv(640, inplanes, kernel_size=1)
        self.out_channels = outplanes

        self.conv3 = BasicConv(inplanes, outplanes,  kernel_size=1)
        self.conv4 = BasicConv(inplanes, outplanes, kernel_size=3,stride=2,padding=1)
        self.change = change
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1x1(x)
        xs = []
        xs.append(self.conv(x))
        xs.append(x)
        xs.append(x)
        xs.append(x)
        xs.append(x)
        ys = []

        for s in range(5):
            if s == 0:
                ys.append(xs[s])
            if s == 1:
                ys.append(self.convs1(xs[0] + ys[-1]))
                
            if s == 2:
                ys.append(self.convs2(xs[s] + ys[-1]))
                
            if s == 3:
                ys.append(self.convs3(xs[s] + ys[-1]))
               
            if s == 4:
                ys.append(self.convs4(xs[s] + ys[-1]))
               

        out = torch.cat([ys[0], ys[1], ys[2], ys[3],ys[4]],dim=1)
        out = self.convchange(out)
        out = self.relu(out + identity)
        if self.change:
            out = self.conv3(out)
        if self.downsample:
            out = self.conv4(out)
        return out



class AFSSD(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(AFSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 320:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm = FEMs(512,512,change=True)
        self.extras = nn.ModuleList(extras)
        self.CCE = nn.ModuleList([])
        self.CCE.append(BasicConv(1024, 512, kernel_size=1))
        self.CCE.append(BasicConv(512, 512, kernel_size=3, stride=1, padding=1))
        self.CCE.append(BasicConv(512, 512, kernel_size=1))
        self.CCE.append(BasicConv(512, 512, kernel_size=3, stride=1, padding=1))
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)
        p = x

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        ppp = self.CCE[0](x)
        ppp = self.CCE[1](ppp)
        pp = self.avg_pool(ppp)
        ppp = F.interpolate(ppp, p.size()[2:], mode='bilinear', align_corners=True)
        pppp = self.CCE[2](pp)
        x = p + ppp + pppp
        x = self.CCE[3](x)
        x = self.Norm(x + p)
        sources.append(x)
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if  k%2 !=0:
                sources.append(x)
       
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    layers = []
    layers += [BasicConv(512, 512, kernel_size=3, stride=2, padding=1)]
    layers += [FEMs(512, 512,change=True)]
    layers += [BasicConv(512, 512, kernel_size=3, stride=2, padding=1)]
    layers += [FEM(512, 512)]
    layers += [BasicConv(512, 512, kernel_size=3, stride=2, padding=1)]
    layers += [FEM(512, 256, change=True)]
    layers += [BasicConv(256, 256, kernel_size=3, stride=1)]
    layers += [FEM(256, 256)]
    layers += [BasicConv(256, 256, kernel_size=3, stride=1)]
    layers += [FEM(256, 256)]
    return layers

extras = {
    '320': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, 24, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)]

    conf_layers += [nn.Conv2d(512, 126, kernel_size=3, padding=1, stride=1)]
    conf_layers += [nn.Conv2d(512, 126, kernel_size=3, padding=1, stride=1)]
    conf_layers += [nn.Conv2d(512, 126, kernel_size=3, padding=1, stride=1)]
    conf_layers += [nn.Conv2d(256, 126, kernel_size=3, padding=1, stride=1)]
    conf_layers += [nn.Conv2d(256, 84, kernel_size=3, padding=1, stride=1)]
    conf_layers += [nn.Conv2d(256, 84, kernel_size=3, padding=1, stride=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '320': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 320 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return AFSSD(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
