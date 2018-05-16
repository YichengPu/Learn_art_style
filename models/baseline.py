from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
from scipy import misc
from os import listdir



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5,stride=3)
        self.conv2 = nn.Conv2d(6, 16, 5,stride=3)
        self.fc1   = nn.Linear(2704, 120)
#         16*5*5
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 30)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        return out


batch_size = 120
num_classes = 30
img_width, img_height = 512, 512  # input image dimensions

PATH = os.path.abspath('/home/yicheng/DS/scraper/styles_cropped')
class_lst = listdir(PATH) # all string filenames in a list
path_dic = {}
for Class in class_lst:
    SOURCE_IMAGES = os.path.join(PATH, Class)
    images = [SOURCE_IMAGES+'/'+f for f in listdir(SOURCE_IMAGES) ]
    path_dic[Class] = images

map_class_to_index={}
for i in range(len(class_lst)):
    map_class_to_index[class_lst[i]]=i

def data_loader(batch_size,class_lst,select_index):
    target_values=[]
    img_mats=[]
    counter=1
    for n in select_index:
        for style in class_lst:
            mat=misc.imread(path_dic[style][n])
            mat=np.array(mat)
            if len(mat.shape)<3:
                mat=np.expand_dims(mat, axis=2)
                mat=np.repeat(mat,3,axis=2)
            elif mat.shape[2]==1:
                mat=np.repeat(mat,3,axis=2)
            elif mat.shape[2]>3:
                mat=mat[:,:,:3]
            mat=np.swapaxes(mat,0,2)
            mat=np.expand_dims(mat, axis=0)
            img_mats.append(mat)
            target_values.append(map_class_to_index[style])
        
        if counter % 4==0:
            #yield (np.asarray(img_mats),np.asarray(target_values))
            yield (np.concatenate(img_mats,axis=0),np.array(target_values))
            target_values=[]
            img_mats=[]
        counter+=1


cuda=False
lr=0.001
momentum=0.5
log_interval=10
epochs=10
model = LeNet()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(epoch):
    model.train()
    for batch_idx, (input_data, input_target) in enumerate(train_loader):
        print(batch_idx)
        data=torch.from_numpy(input_data).float()

        target=torch.from_numpy(input_target).long()

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 750*30,
                100. * batch_idx / (750/4), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for input_data, input_target in test_loader:
        
        data=torch.from_numpy(input_data).float()

        target=torch.from_numpy(input_target).long()

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= 250*30
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 250*30,
        100. * correct / (250*30)))


for epoch in range(1, epochs + 1):
    train_loader=data_loader(10,class_lst,np.arange(750))
    train(epoch)
    test_loader=data_loader(10,class_lst,np.arange(751,1000))
    test()

