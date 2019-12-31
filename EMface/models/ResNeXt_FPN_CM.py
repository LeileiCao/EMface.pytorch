#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import pretrainedmodels as pm
from layers import *
from data.config import cfg
import numpy as np
import math

def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y

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

class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, c2, c3, c4, c5):
        p5 = self.latlayer1(c5)
        p4 = _upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = _upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = _upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #out = self.se_module(out) + residual
        out = out +residual
        #out = self.relu(out)

        return out

class ResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups=64, stride=2,
                 downsample=None, base_width=4):
        super(ResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, dilation=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(planes * 4),)
        self.stride = stride

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SA(nn.Module):
    def __init__(self):
        super(SA,self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x*scale

class CM(nn.Module):
    """docstring for CM"""
    def __init__(self, in_planes):
        super(CM, self).__init__()
        self.branch1 = BasicConv(in_planes, in_planes, kernel_size=3, stride=1, padding=3, dilation=3, groups=32)
        self.branch2 = BasicConv(2*in_planes, in_planes, kernel_size=3, stride=1, padding=3, dilation=3, groups=32)            
        self.branch3 = BasicConv(3*in_planes, in_planes, kernel_size=3, stride=1, padding=3, dilation=3, groups=32)
        self.branch4 = nn.Conv2d(4*in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.SA=SA()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(torch.cat((x, x1), dim=1))
        x3 = self.branch3(torch.cat((x, x1, x2), dim=1))
        out = self.branch4(torch.cat((x, x1, x2, x3), dim=1))
        out=self.SA(out)
        return out

class S3FD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base_model, head, num_classes=2):
        super(S3FD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        '''
        self.priorbox = PriorBox(size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''
        # SSD network
        self.init_block = nn.Sequential(*list(base_model.children())[:4])
        self.res_block1 = nn.Sequential(*list(base_model.children())[4])
        self.res_block2 = nn.Sequential(*list(base_model.children())[5])
        self.res_block3 = nn.Sequential(*list(base_model.children())[6])
        self.res_block4 = nn.Sequential(*list(base_model.children())[7])
        self.layer6 = ResNeXtBottleneck(2048,64)                                     
        self.layer7 = ResNeXtBottleneck(256,64)
        self.fpn=FPN()
        self.SA1=CM(256)
        self.SA2=CM(256)
        self.SA3=CM(256)
        self.SA4=CM(256)
        self.SA5=CM(256)
        self.SA6=CM(256)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)
 


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        x = self.init_block(x)
        x = self.res_block1(x)
        c2 = x
        x = self.res_block2(x)
        c3 = x
        x = self.res_block3(x)
        c4 = x
        x = self.res_block4(x)
        c5 = x
        p2,p3,p4,p5 = self.fpn(c2,c3,c4,c5)
        p6 = self.layer6(c5)
        p7 = self.layer7(p6)
        sources.append(self.SA1(p2))
        sources.append(self.SA2(p3))
        sources.append(self.SA3(p4))
        sources.append(self.SA4(p5))
        sources.append(self.SA5(p6))
        sources.append(self.SA6(p7))


        # apply multibox head to source layers

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        '''
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        '''

        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]

        self.priorbox = PriorBox(size, features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

       
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()



def multibox():
    loc_layers = []
    conf_layers = []

    loc_layers += [nn.Conv2d(256, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 3 + 1, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256,4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256,4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256,4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256,4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_model(phase, num_classes=2):
    model = pm.resnext101_64x4d()
    base_model=list(model.children())[0]
    return S3FD(phase, base_model, multibox())

