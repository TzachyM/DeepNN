
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

class CifarPartialDataSet(torchvision.datasets.CIFAR10):
    def __init__(self, ind_range, train=True, transform=None, target_transform=None):
        self.train = train
        assert isinstance (ind_range, tuple)
        from_ind, to_ind = ind_range
        assert 0 <= from_ind < to_ind < 10
        self.transform = transform
        self.target_transform = target_transform
        if train:
            cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
            label_array = np.array(cifar10_train.targets, dtype=np.int64)
            indices = (from_ind <= label_array) & (label_array <= to_ind)
            self.data = cifar10_train.data[indices]
            self.targets = list(label_array[indices]-from_ind)
        else:
            cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
            label_array = np.array(cifar10_test.targets, dtype=np.int64)
            indices = (from_ind <= label_array) & (label_array <= to_ind)
            self.data = cifar10_test.data[indices]
            self.targets = list(label_array[indices]-from_ind)
            

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
cifar07_train = CifarPartialDataSet((0,7), True, transform=transform)
cifar07_train_loader = torch.utils.data.DataLoader(cifar07_train, batch_size=32,
                                                           shuffle=True, num_workers=0)
device = torch.device("cuda")
model = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad=False
num_feathers = model.fc.in_features
model.fc = nn.Linear(num_feathers, 8)
model = model.to(device)

       
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
epoch=5
correct = 0
total = 0
for _ in range(epoch):
    for i, (x,y) in enumerate(cifar07_train_loader):        
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            
            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
    print("loss",loss.item())
    print("acc",correct / total)
    exp_lr.step()
