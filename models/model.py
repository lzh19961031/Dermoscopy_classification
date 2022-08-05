import torchvision
import torch.nn.functional as F 
from torch import nn
import torchvision.models as models
from torchvision import transforms as T
from config import config
import itertools
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

import os
import torch.nn as nn
import torch.optim as optim
#from base_networks import *
from torchvision.transforms import *



class DisCA(Module):
    def __init__(self):
        super(DisCA, self).__init__()

        self.beta = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
            )

    def forward(self, x, x1, x2):
        m_batchsize, C, height, width = x1.size()
        F_ = self.conv(x1).view(m_batchsize, C, -1)
        F__ = self.conv(x2).view(m_batchsize, C, -1).permute(0, 2, 1)
        F = torch.bmm(F_, F__)
        attention = self.softmax(F)
        x_ = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, x_).view(m_batchsize, C, height, width)
        out = self.beta * out
        return out


class DisSA(Module):
    def __init__(self):
        super(DisSA, self).__init__()
        self.alpha = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
            )

    def forward(self, x, x1, x2):
        m_batchsize, C, height, width = x1.size()
        F_ = self.conv(x1).view(m_batchsize, C, -1).permute(0, 2, 1)
        F__ = self.conv(x2).view(m_batchsize, C, -1)
        F = torch.bmm(F_, F__)
        attention = self.softmax(F)
        x_ = x.view(m_batchsize, C, -1)
        out = torch.bmm(x_, attention).view(m_batchsize, C, height, width)
        out = self.alpha * out
        return out


class Distinguish(nn.Module):
    def __init__(self):
        super(Distinguish, self).__init__()

        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.dis = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
            )
        self.disca = DisCA()
        self.dissa = DisSA()
        self.fc = nn.Sequential(
            nn.ReLU(inplace = True),
	        nn.Dropout(),
            nn.Linear(512, 2)
            ) 
        self.avgpool = models.resnet18(pretrained = True).avgpool

    def forward(self, x1, x2):
        x = torch.cat([x1, x2],1)
        x = self.conv(x)
        x_ca = self.disca(x, x1, x2)
        x_sa = self.dissa(x, x1, x2)
        fea = x + x_ca + x_sa
        fea = self.avgpool(fea)
        fea = fea.view(fea.size(0), -1)
        disscore = self.fc(fea)
        return disscore




class CA(Module):
    def __init__(self):
        super(CA, self).__init__()
        self.alpha = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
            )
        self.avg_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.max_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_avg = self.avg_conv4(x)
        x_avg = torch.mean(x_avg, 1, True)
        x_avg = self.sigmoid(x_avg)
        x_max = self.max_conv4(x)
        x_max = torch.max(x_avg, 1).values.unsqueeze(1)
        x_max = self.sigmoid(x_max)
        attention = x_avg + x_max
        x_final = self.alpha * x0 * attention 
        return x_final


class SA(Module):
    def __init__(self):
        super(SA, self).__init__()

        self.beta = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
            )
        self.avg_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.max_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        self.sigmoid = Sigmoid()

        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self,x):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_avg = self.avg_conv4(x)
        x_avg = self.avgpool(x_avg)
        x_avg = self.sigmoid(x_avg)
        x_max = self.max_conv4(x)
        x_max = self.maxpool(x_max)
        x_max = self.sigmoid(x_max)
        attention = x_avg + x_max
        x_final = self.beta * x0 * attention 
        return x_final


class Attention(Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.sa = SA()
        self.ca = CA()

    def forward(self,x):
        x_ca = self.ca(x)
        x_sa = self.sa(x)
        output = x + x_ca + x_sa
        return output




class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.dis = Distinguish()
        self.attention = Attention()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace = True),
	    nn.Dropout(),
            nn.Linear(64,config.num_classes),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace = True),
	    nn.Dropout(),
            nn.Linear(64,config.num_classes),
            )
        self.fcdis = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace = True),
	    nn.Dropout(),

            nn.Linear(64,2),
            )
        self.avgpool = models.resnet18(pretrained = True).avgpool
        self.getfeature1 = nn.Sequential(
                                models.resnet18(pretrained = True).conv1,
                                models.resnet18(pretrained = True).bn1,
                                models.resnet18(pretrained = True).relu,
                                models.resnet18(pretrained = True).maxpool,
                                models.resnet18(pretrained = True).layer1,
                                models.resnet18(pretrained = True).layer2,
                                models.resnet18(pretrained = True).layer3,
       	                        models.resnet18(pretrained = True).layer4          
                                )
        self.getfeature2 = nn.Sequential(
                                models.resnet18(pretrained = True).conv1,
                                models.resnet18(pretrained = True).bn1,
                                models.resnet18(pretrained = True).relu,
                                models.resnet18(pretrained = True).maxpool,
                                models.resnet18(pretrained = True).layer1,
                                models.resnet18(pretrained = True).layer2,
                                models.resnet18(pretrained = True).layer3,
       	                        models.resnet18(pretrained = True).layer4          
                                )


    def forward(self, img1, img2):
        x1 = self.getfeature1(img1)
        x2 = self.getfeature2(img2)
        x1_ = self.attention(x1)
        x2_ = self.attention(x2)
        feature1_final = self.avgpool(x1_)
        feature1_final = feature1_final.view(feature1_final.size(0), -1)
        result1 = self.fc1(feature1_final)
        feature2_final = self.avgpool(x2_)
        feature2_final = feature2_final.view(feature2_final.size(0), -1)
        result2 = self.fc2(feature2_final)
        disscore = self.dis(x1_, x2_)
        return  result1, result2, disscore

