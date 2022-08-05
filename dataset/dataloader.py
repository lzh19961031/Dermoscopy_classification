from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from dataset.aug import *
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 
from random import randint

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class Dataset(Dataset):
    def __init__(self,label_list,transforms=None,train=True,test=False):
        self.test = test 
        self.train = train 
        self.different_label = 0

        imgs = []


        for index,row in label_list.iterrows():
            imgs.append((row["filename"],row["label"]))
        self.imgs = imgs

        self.len = len(imgs)
        print('imgs',imgs)

        if transforms is None:
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])
                    ])
            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    #T.RandomRotation(30),
                    #T.RandomHorizontalFlip(),
                    #T.RandomVerticalFlip(),
                    #T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])
                    ])

        else:
            self.transforms = transforms

    def __getitem__(self,index):
        classlabel0 = [0]
        classlabel1 = [0]

        filename,label = self.imgs[index]
        filename0 = filename
        if label == 'benign':
            classlabel0[0] = 0
        else:
            classlabel0[0] = 1

        img0 = Image.open(config.train_data+filename0+'.jpg')
        img0 = self.transforms(img0)

        if index != self.len - 1:
            a = randint(1 ,self.len - index - 1)
            filename,label = self.imgs[index + a]
            filename1 = filename
            if label == 'benign':
                classlabel1[0] = 0
            else:
                classlabel1[0] = 1
        else:
            a = randint(1, self.len - 1)
            filename,label = self.imgs[index - a]
            filename1 = filename
            if label == 'benign':
                classlabel1[0] = 0
            else:
                classlabel1[0] = 1

        #print('filename0',filename0)
        #print('filename1',filename1)
        #print('classlabel0',classlabel0)
        #print('classlabel1',classlabel1)

        img1 = Image.open(config.train_data+filename1+'.jpg')
        img1 = self.transforms(img1)

        if classlabel0 == classlabel1:
            synlabel = 1
        else:
            synlabel = 0
            self.different_label = self.different_label + 1
            #print('different_label',self.different_label)#################记下来多少组不一样的
        #print('synlabel',synlabel)


        dislabel11 = 1
        dislabel12 = 0
        dislabel21 = 0
        dislabel22 = 1

        return img0,img1,classlabel0,classlabel1,synlabel, dislabel11, dislabel12, dislabel21, dislabel22, self.len


    def __len__(self):
        return self.len


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,mode):
    #for test
    if mode == "test":
        all_data_path,labels = [],[]

        df = pd.read_csv('ISBI2016_ISIC_Part3_Test_GroundTruth.csv') #返回一个DataFrame的对象，这个是pandas的一个数据结构
        print('df',df)
        df.columns=["A","B"]
 
        filename = df["A"]
        filename = np.array(filename)
        print(filename)

        labels = df["B"] #最后一列作为每行对应的标签label
        labels = np.array(labels) 
        print(labels)

        all_files = pd.DataFrame({"filename":filename,"label":labels})
        print('all_files',all_files)
        return all_files

    elif mode != "test": 
        #for train and val       
        all_data_path,labels = [],[]

        #filename = os.listdir(root)
        #filename.sort(key=lambda x:(x[:-10]))
        #print(filename)
        #all_images = [os.path.join(root, img) for img in filename]
        #for file in tqdm(all_images):
        #    all_data_path.append(file)
        #    print(all_data_path)

        df = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv') #返回一个DataFrame的对象，这个是pandas的一个数据结构
        print('df',df)
        df.columns=["A","B"]
 
        filename = df["A"]
        filename = np.array(filename)
        print(filename)

        labels = df["B"] #最后一列作为每行对应的标签label
        labels = np.array(labels) 
        print(labels)

        all_files = pd.DataFrame({"filename":filename,"label":labels})
        print('all_files',all_files)
        return all_files
    else:
        print("check the mode please!")
    
