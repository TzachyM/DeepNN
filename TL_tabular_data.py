import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split


class NN (nn.Module):
    def __init__(self, in_features):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.bnorm3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024,1)

    def forward(self,x):
        x = F.leaky_relu(self.bnorm1(self.drop(self.fc1(x))))
        x = F.leaky_relu(self.bnorm2(self.fc2(x)))   
        x = F.leaky_relu(self.bnorm3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))           
        return x
    
def split(x, y, batch_size):  
    shuffle = np.random.permutation(y.shape[0])
    idx = np.arange(batch_size, x.shape[0], batch_size)
    batches_x = np.split(x[shuffle, ...], idx)
    batches_y = np.split(y[shuffle].reshape(-1, 1), idx)
    return batches_x, batches_y

def train_net(epochs, batch_size, criterion, optimizer, x_train, y_train, model):
    for i in range(epochs):
        batches_x, batches_y = split(x_train, y_train, batch_size)
        for batch_x, batch_y in zip(batches_x, batches_y):

            optimizer.zero_grad()
            output=model(torch.tensor(batch_x).float().to(device))
            loss = criterion(output,torch.from_numpy(batch_y).float().to(device))
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss/epoch", loss, i)
        print(loss.item())
    writer.flush()

       
if __name__ == "__main__":
    
    # Preproccing
    data = pd.read_csv(r"indian_liver_patient.csv")
    data['Gender'].replace([0,1],['Female','Male'],inplace=True)
    data['Dataset'].replace(2,0,inplace=True)
    data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
    label = np.array(data['Dataset'])
    in_features = len(data.columns)
    data = data.drop(['Dataset'], axis=1)
    data = np.array(pd.get_dummies(data))
    device = torch.device("cuda")
    net = NN(in_features).to(device)
    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=1, test_size=0.2)

    # Training
    lr = 0.001
    epochs = 100
    batch_size = 32
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(r"logs")
    print("My simple deep net")
    train_net(epochs, batch_size, criterion, optimizer, x_train, y_train, net)
    
    # Transfer learning
    net.fc3 = nn.Linear(512, 2048)
    net.bnorm3 = nn.BatchNorm1d(2048) 
    net.fc4 = nn.Linear(2048, 1)
    transfer_model = net.to(device)
    t_lr = 0.0001
    t_optimizer = optim.Adam(transfer_model.parameters(), lr=t_lr)
    print("Transfer learning net")
    train_net(epochs, batch_size, criterion, t_optimizer, x_train, y_train, transfer_model)
