# %% 
!cp -r /Users/tilman/Documents/projects/arena/arena-v1-ldn/w0d3 /Users/tilman/Documents/projects/arena/arena/w0d3

#%%
import torch as t
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import json
from pathlib import Path
from typing import Union, Tuple, Callable, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import w0d3.utils as utils

#%%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input is (28,28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1, stride=1)
        # Output of conv2 is (32,28,28)
        # hight = (height + 2* padding_h - kernel_height)//stride_h + 1
        #       = (28 + 2 * 1 - 3)/1 +1= 28
        # width = (width + 2* padding_w - kernel_width)//stride_w + 1
        #       = (28 + 2 * 1 - 3)/1 +1= 28
        self.ReLU = nn.ReLU()

        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        # height = (27 - 2)/2 +1 = 13.5 -> 14
        # (32,14,14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1)
        # output of conv2 will be (32,14,14)
        # hight = (height + 2* padding_h - kernel_height)//stride_h + 1
        #       = (13 + 2 * 1 - 3)/1 +1= 14
        # width = (width + 2* padding_w - kernel_width)//stride_w + 1
        #       = (13 + 2 * 1 - 3)/1 +1= 14
        
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        # height = (14 - 2)/2 +1= 7 
        
        self.flatten = nn.Flatten()
        # (64,7,7) -> (3136)

        # ...
        # correct would be (3136)
        # reverse engineered 3136/64 -> 49 -> 7*7
        self.linear1 = nn.Linear(in_features=3136, out_features=128)
        
        self.linear2 = nn.Linear(in_features=128, out_features=10)
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = ConvNet()
#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

testloader = DataLoader(testset, batch_size=64, shuffle=True)

#%%

import matplotlib.pyplot as plt 

plt.imshow(trainset.data[0])

#%%
from tqdm.notebook import tqdm_notebook
from tqdm import trange
import time
print('doesnt show')
for i in tqdm_notebook(range(100)):
    time.sleep(0.01)
print('shows')
for i in trange(100):
    time.sleep(0.01)

#%%
epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"

def train_convnet(trainloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")


    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list

loss_list = train_convnet(trainloader, epochs, loss_fn)

fig = px.line(y=loss_list, template="simple_white")
fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
fig.show()
#%%
epochs = 1
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
from tqdm import trange

def train_convnet(trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.

    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list, accuracy_list = [], []

    for epoch in trange(epochs):
        for (x,y) in testloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            y_hat = t.argmax(y_hat, dim=1)
            accuracy = t.sum(y_hat==y)/len(y)
            accuracy_list.append(accuracy.item())
        progress_bar = tqdm_notebook(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")
        # testing 
        

        
    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list, accuracy_list


loss_list, accuracy_list = train_convnet(trainloader, testloader, epochs, loss_fn)

utils.plot_loss_and_accuracy(loss_list, accuracy_list)

# %%
