import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self,noise_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),   
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512,kernel_size=8,stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256,kernel_size=16,stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1,kernel_size=3,stride=1, bias=False),
            nn.Tanh()
             )

    def forward(self,x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64 , 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(28*28*128, 1),

            nn.Sigmoid(),
        )
        
    def forward(self,x):
        return self.main(x)
    
# def conv2d_size_out(size, kernel_size = 5, stride = 2):
#     return (size - (kernel_size - 1) - 1) // stride  + 1
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch = 32
noise_size = 100
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(0, 1)])
train_data = torchvision.datasets.MNIST(root='./MNIST/', train=True, download=True,
                                   transform=transforms)
train_load = DataLoader(train_data, batch_size=batch, shuffle=True, pin_memory=True)
w = train_data[0][0].shape[1]
h = train_data[0][0].shape[2]
fix_noise = torch.rand(size=(batch, noise_size, 1, 1)).to(device)


disc = Discriminator().to(device)
gene = Generator(noise_size).to(device)
opt_d = torch.optim.Adam(disc.parameters(), lr=0.0002)
opt_g = torch.optim.Adam(gene.parameters(), lr=0.0002)
criterion = nn.BCELoss()
img_list = []
epochs = 20
# first, zero grad for disc, send true image on Disc (label=1) and calc real img loss(all have the same criterion) and back prog
# second, gene fake img from noise, run on disc (label=0) and calc fake img loss and back prog
# third, add up fake and real loss, and run a step with disc optm
# forth, zero grad of gene, run the fake img on disc and calc loss of gene (label=1)
# fifth, back prog loss gene and run a step with gene optm
for epoch in range(epochs):
    for i, (real_images,_) in enumerate(train_load):
        disc.zero_grad()
        real_images = real_images.to(device)
        real_pred = disc(real_images)
        real_targets = torch.ones(batch, 1, device=device)
        real_loss = criterion(real_pred, real_targets)
        real_loss.backward()
        mean_real_loss = real_loss.mean().item()
        
        noise = torch.rand(size=(batch, 100, 1, 1)).to(device)
        fake_img = gene(noise).to(device)
        fake_pred = disc(fake_img.detach()).to(device)
        fake_target = torch.zeros(noise.size(0), 1, device=device)
        fake_loss = criterion(fake_pred, fake_target)
        fake_loss.backward()
        mean_fake_pred = fake_pred.mean().item()
        disc_loss = fake_loss+real_loss
        opt_d.step()
        mean_fake_loss = fake_loss.mean().item()
        
        gene.zero_grad()
        pred = disc(fake_img).to(device)
        targets = torch.ones(batch, 1, device=device)
        g_loss = criterion(pred, targets )
        g_loss.backward()
        opt_g.step()
        mean_g_loss = g_loss.mean().item()
        if i%32 ==0:
            print(f"mean_real_loss is:{mean_real_loss} \n mean_fake_loss is\
                  {mean_fake_loss} \n mean_g_loss is {mean_g_loss}")
    with torch.no_grad():
        fake = gene(fix_noise).detach()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
    

# =============================================================================
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.squeeze(train[1][0]))    
# =============================================================================
