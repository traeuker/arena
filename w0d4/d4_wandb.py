# %% 
#!cp -r /Users/tilman/Documents/projects/arena/arena-v1-ldn/w0d4 /Users/tilman/Documents/projects/arena/arena/w0d4
#!pip install wandb
# %%
import sys 
sys.path.append('/Users/tilman/Documents/projects/arena/arena/w0d3')


# %% 
import torch as t
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
import w0d4.utils as utils
#from w0d3 import *

device = "cuda" if t.cuda.is_available() else "cpu"

# %% 
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]
epoch = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indices = t.arange(1000)
trainset = Subset(trainset, indices)
testset = Subset(testset, indices)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,)

utils.show_cifar_images(trainset, rows=3, cols=5)
# %% 


def train(model, trainloader: DataLoader, testloader: DataLoader, num_epochs=epoch,):

    optimizer=t.optim.Adam(model.parameters())
    model.train()
    loss_list, acc_list = [], []

    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)

        for (x, y) in trainloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        
        acc, total = 0, 0
        with t.no_grad():
            for (x,y) in testloader:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_hat = t.argmax(y_hat, dim=1)
                acc += t.sum(y_hat == y).item()
                total += y.size(0)
            acc_list.append(acc/total)


    return model, loss_list, acc_list

model = models.resnet34(weights='IMAGENET1K_V1')

fine_model, loss_list, acc_list = train(model, trainloader=trainloader, testloader=testloader)


# %% 
# %%
def train_wnb_lame(model, trainloader: DataLoader, testloader: DataLoader, num_epochs=epoch,):

    optimizer=t.optim.Adam(model.parameters())
    model.train()
    examples_seen = 0
    
    wandb.init(project="w2d1_resnet")

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)


    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)

        
        for (x, y) in trainloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            examples_seen += len(y)
            wandb.log({"train_loss": loss}, step=examples_seen)

        
        acc, total = 0, 0
        with t.no_grad():
            for (x,y) in testloader:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_hat = t.argmax(y_hat, dim=1)
                acc += t.sum(y_hat == y).item()
                total += y.size(0)
            
            wandb.log({"test_accuracy": acc/total}, step=examples_seen)
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

    return model

model = models.resnet34(weights='IMAGENET1K_V1')

fine_model = train_wnb_lame(model, trainloader=trainloader, testloader=testloader)


# %% 
def train_wnb():

    wandb.init()
    
    num_epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = models.resnet34(weights='IMAGENET1K_V1').to(device=device)
    optimizer=t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()

    cifar_mean = [0.485, 0.456, 0.406]
    cifar_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    indices = t.arange(3000)
    trainset = Subset(trainset, indices)
    testset = Subset(testset, indices)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    examples_seen = 0
    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)
        
        for (x, y) in trainloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            examples_seen += len(y)
            wandb.log({"train_loss": loss}, step=examples_seen)

        with t.inference_mode():
            acc, total = 0, 0
            for (x,y) in testloader:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_hat = t.argmax(y_hat, dim=1)
                acc += t.sum(y_hat == y).item()
                total += y.size(0)
            
            wandb.log({"test_accuracy": acc/total}, step=examples_seen)
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

    return model

sweep_config = {
    'method': 'random',
    'name': 'w0d4_resnet_sweep_2',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [64, 128, 256]},
        'epochs': {'min': 1, 'max': 3},
        'lr': {'max': 0.1, 'min': 0.0001, 'distribution': 'log_uniform_values'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='w0d4_resnet')

wandb.agent(sweep_id=sweep_id, function=train_wnb, count=2)


# %% 




# %% 




# %% 




# %% 