import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
from random import randint
from datetime import datetime
from torch import nn,optim
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed
import itertools
from IPython import embed
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from visdom import Visdom
from evaluate_and_curve import evaluate
from evaluate_and_curve import cauculate
from evaluate_and_curve import Get_Average
import matplotlib.pyplot as plt


#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


def main():
    fold = 0
    model = Model()
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
    weights = [1, 1]
    class_weights = torch.FloatTensor(weights).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(config.device)
    criterion1 = nn.L1Loss().to(config.device)
    start_epoch = 0
    fold = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False
    lr = config.lr

    train_data_list = get_files(config.train_data,"train")
    test_files = get_files(config.test_data,"test")
 
    train_dataset = Dataset(train_data_list)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(Datasetfortest(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
    scheduler =  optim.lr_scheduler.StepLR(optimizer,step_size = 20,gamma=0.1)
    
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()

    start = timer()
    for epoch in range(start_epoch,config.epochs):
        number = 1

        scheduler.step(epoch)
        train_progressor = ProgressBar(mode="Train",epoch=epoch,total_epoch=config.epochs,model_name=config.model_name0,total=len(train_dataloader))

        for iter,(img0 , img1, classlabel1, classlabel2,synlabel, dislabel11, dislabel12, dislabel21, dislabel22, number) in enumerate(train_dataloader):

            synlabel = synlabel.to(config.device)
            dislabel11 = dislabel11.to(config.device)
            dislabel12 = dislabel12.to(config.device)
            dislabel21 = dislabel21.to(config.device)
            dislabel22 = dislabel22.to(config.device)

            train_progressor.current = iter
            model.train()

            img0 = Variable(img0).to(config.device)
            img1 = Variable(img1).to(config.device)

            classlabel1 = classlabel1[0]
            classlabel2 = classlabel2[0]

            classlabel1,classlabel2 = classlabel1.to(config.device),classlabel2.to(config.device)     

            result1, result2, synscore = model(img0,img1)
            losssyn = criterion(synscore, synlabel)

            lossce1 = criterion(result1,classlabel1)
            lossce2 = criterion(result2,classlabel2)
            loss = lossce1 + lossce2 + 0.1 * losssyn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            number = number + 1



if __name__ =="__main__":
    main()


