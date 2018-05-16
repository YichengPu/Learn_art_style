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

sub_number=1000
class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.glob_conv_1x1=nn.Conv2d(3,3,2,stride=2)

        self.glob_conv_3x3=nn.Conv2d(3,64,3)


        self.glob_conv_5x5=nn.Conv2d(3,64,5)


        #self.glob_conv_7x7=nn.Conv2d(3,64,7)



        self.detail_conv_3x3_s2_a=BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.detail_conv_3x3_s2_b=BasicConv2d(32, 64, kernel_size=3, stride=2)
        self.inception_1=Inception(64,256)
        self.inception_2=Inception(64,256)
        self.inception_3=Inception(64,256)
        self.inception_4=Inception_B(64,256)
#         self.detail_conv_3x3_c=BasicConv2d(64, 128, kernel_size=3)
#         self.detail_conv_3x3_d=BasicConv2d(128, 216, kernel_size=3)
        self.glob_fc_1=nn.Linear(3264,30)
#         self.glob_fc_2=nn.Linear(1000,300)
#         self.glob_fc_3=nn.Linear(300,30)
#         self.conv1 = nn.Conv2d(3, 6, 5,stride=3)
#         self.conv2 = nn.Conv2d(6, 16, 5,stride=3)
#         self.fc1   = nn.Linear(2704, 120)
# #         16*5*5
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 30)

    def forward(self, x):

        g_out=self.glob_conv_1x1(x)
        g_out_3=self.glob_conv_3x3(g_out)
        g_avg_3=g_out_3.mean(2,keepdim=True)
        g_avg_3=g_avg_3.mean(3,keepdim=True)
        g_avg_3=g_avg_3.squeeze(2)
        g_avg_3=g_avg_3.squeeze(2)


        g_out_5=self.glob_conv_5x5(g_out)
        g_avg_5=g_out_5.mean(2,keepdim=True)
        g_avg_5=g_avg_5.mean(3,keepdim=True)
        g_avg_5=g_avg_5.squeeze(2)
        g_avg_5=g_avg_5.squeeze(2)

#         g_out_7=self.glob_conv_7x7(g_out)
#         g_avg_7=g_out_7.mean(2,keepdim=True)
#         g_avg_7=g_avg_7.mean(3,keepdim=True)
#         g_avg_7=g_avg_7.squeeze(2)
#         g_avg_7=g_avg_7.squeeze(2)

        g_avg=torch.cat((g_avg_3,g_avg_5), 1)
#         g_avg=torch.cat((g_avg,g_avg_7), 1)

#        out=g_avg

        d_out=self.detail_conv_3x3_s2_a(x)
        d_out=self.detail_conv_3x3_s2_b(d_out)
        d_out = F.max_pool2d(d_out, kernel_size=2)

        d_out=self.inception_1(d_out)
        d_out=self.inception_2(d_out)
        d_out=self.inception_3(d_out)

        d_out = F.max_pool2d(d_out, kernel_size=2)

        d_out=self.inception_4(d_out)
        d_out = F.max_pool2d(d_out, kernel_size=2)
        d_out=self.inception_4(d_out)

        d_out = F.max_pool2d(d_out, kernel_size=2)

#         d_out=self.detail_conv_3x3_c(d_out)
#         d_out = F.max_pool2d(d_out, kernel_size=3)

#         d_out=self.detail_conv_3x3_d(d_out)
#         d_out = F.max_pool2d(d_out, kernel_size=3)
#         #print(d_out.size())
        d_out = d_out.view(d_out.size(0), -1)
        #print(d_out.size())
        final_vec=torch.cat((g_avg,d_out), 1)

        #print(final_vec.size())
        out=self.glob_fc_1(final_vec)
#         out=F.Sigmoid(out)
#         out=self.glob_fc_2(out)
        out=F.sigmoid(out)
#         out=self.glob_fc_3(out)
        return out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):

    def __init__(self,in_channels,out_channels=256,**kwargs):
        super(Inception, self).__init__()
        self.conv1x1_s=BasicConv2d(in_channels,64,kernel_size=1)
        self.conv3x3=BasicConv2d(64,out_channels,kernel_size=3,padding=1)
        self.conv5x5=BasicConv2d(64,out_channels,kernel_size=5,padding=2)
        self.conv1x1_e=BasicConv2d(576,64,kernel_size=1)
    def forward(self,x):
        m=self.conv1x1_s(x)
        b1=self.conv3x3(m)
        b2=self.conv5x5(m)
        b3=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)

        out=[b1,b2,b3]
        out=torch.cat(out,dim=1)
        out=self.conv1x1_e(out)
        return out

class Inception_B(nn.Module):

    def __init__(self,in_channels,out_channels=256,**kwargs):
        super(Inception_B, self).__init__()
        self.conv1x1_s=BasicConv2d(in_channels,64,kernel_size=1)
        self.conv3x3=BasicConv2d(64,out_channels,kernel_size=3,padding=1)
        self.conv1x5=BasicConv2d(64,out_channels,kernel_size=(1,5),padding=(0,2))
        self.conv5x1=BasicConv2d(out_channels,out_channels,kernel_size=(5,1),padding=(2,0))
        self.conv1x1_e=BasicConv2d(576,64,kernel_size=1)
    def forward(self,x):
        m=self.conv1x1_s(x)

        b1=self.conv3x3(m)

        b2=self.conv1x5(m)
        b2=self.conv5x1(b2)

        b3=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)

        out=[b1,b2,b3]
        out=torch.cat(out,dim=1)
        out=self.conv1x1_e(out)
        return out

batch_size = 1
num_classes = 30
img_width, img_height = 512, 512  # input image dimensions

PATH = os.path.abspath('styles_cropped')
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

        if counter % batch_size==0:
            #yield (np.asarray(img_mats),np.asarray(target_values))
            yield (np.concatenate(img_mats,axis=0),np.array(target_values))
            target_values=[]
            img_mats=[]
        counter+=1


cuda=True
lr=0.001
momentum=0.5
log_interval=10
epochs=50
model = SNet()
if cuda:
    model.cuda()

optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

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
        celoss=nn.CrossEntropyLoss()
        loss = celoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            with open("s2log.txt",'a') as file:
                file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), sub_number/4*3*30,
                100. * batch_idx / (sub_number/4*3/batch_size), loss.data[0]))

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
        celoss=nn.CrossEntropyLoss()
        test_loss += celoss(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= sub_number/4*30
    with open('s2out.txt','a') as file:
        file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, sub_number/4*30,
        100. * correct / (sub_number/4*30)))

#test_loader=data_loader(batch_size,class_lst,np.arange(int(sub_number*0.75)+1,sub_number))
#test()
for epoch in range(1, epochs + 1):
    train_loader=data_loader(batch_size,class_lst,np.arange(int(sub_number*0.75)))
    train(epoch)
    test_loader=data_loader(batch_size,class_lst,np.arange(int(sub_number*0.75)+1,sub_number))
    test()
