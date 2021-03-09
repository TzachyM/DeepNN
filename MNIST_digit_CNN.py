# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:32:51 2021

@author: Tzachy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class MNIST_(object):
    def __init__(self, transforms, data_kind):
        self.data =[]
        if type(transforms) == list:
            for transform in transforms:
                if data_kind == "train":
                    data_ = torchvision.datasets.MNIST(root='./MNIST_folder/Train', train=True, download=False ,transform=transform)
                else:
                    data_ = torchvision.datasets.MNIST(root='./MNIST_folder/Test', train=False, download=False ,transform=transform)
                for i in data_:
                    self.data.append(i)
        else:
             print("Bring me a list!")
    def __getitem__(self, index):
        return self.data[index]
            
    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, 32 , kernel_size, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32 , kernel_size, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)      
        self.conv4 = nn.Conv2d(32, 32 , kernel_size, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(32) 
        self.conv5 = nn.Conv2d(32, 32 , kernel_size, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(32) 
        self.conv6 = nn.Conv2d(32, 32 , kernel_size, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(32) 
        self.conv7 = nn.Conv2d(32, 32 , kernel_size, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(32) 
        self.fc1 = nn.Linear(32*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        
        
    def forward(self, x):

        x = self.conv1(x)        
        x = F.leaky_relu(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = F.leaky_relu(x)
        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 32*7*7)   
        x = self.fc1(x)   
        x = F.leaky_relu(x)
        x = F.dropout2d(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)    
        x = F.dropout2d(x)
        x = self.fc3(x)

        
        return x
    
    

    
#data loading
batch = 128
transform1 = transforms.Compose([transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                                transforms.ToTensor(),
                                transforms.Normalize(0, 1)])
transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0, 1)])
transform = [transform1, transform2]


train = MNIST_(transform, "train")
test = MNIST_(transform, "test")
train_loader = torch.utils.data.DataLoader(train, batch_size=batch,shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

#model
writer = SummaryWriter("runs/mnist")
kernel_size = 3
lr = 0.001
epochs = 10
in_channels = train[0][0].size()[0]
out_channels = 16
kernel_size = 3

model = Net(in_channels, out_channels, kernel_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
writer.add_graph(model,next(iter(train_loader))[0].to(device))
patience = 7
mean_loss = 0
counter = 0

running_loss=0
for epoch in range(epochs):
    old_loss = None

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #forward
        out = model(images)
        loss = criterion(out, labels)
        
        #backward propgation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        running_loss = loss.item()
        
        _, predict = torch.max(out,1)
        correct = (predict == labels).sum().item()
        

        writer.add_scalar("Training loss", running_loss, epoch*len(train_loader)+i)
        writer.add_scalar("Training Accuracy", 100*correct/batch, epoch*len(train_loader)+i)
            
        if i%1000 == 0:
            print(f"epoch {epoch} | mean loss {mean_loss/1000}")
            if old_loss==None:
                old_loss=mean_loss
            elif (mean_loss/1000) < (old_loss/1000):
                old_loss=mean_loss
            else:
                counter += 1
                if counter >= patience:
                    break
    if counter>=patience:
        break
print('Finished training')

correct = 0
num_sampels = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        #forward
        out = model(images)
        _, predict = torch.max(out,1)
        num_sampels += images.size(0)
        correct += (predict == labels).sum().item()

acc = 100.0 * (correct / num_sampels)
print(f'Accuracy of the network: {acc} %')
writer.close()

    
    

