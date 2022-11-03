# %% 
import torch as t
import torch.nn as nn
import fancy_einsum as einsum
from einops import rearrange

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

# %% 
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        init_weight_tensor = t.ones(num_features)
        self.weight = nn.Parameter(init_weight_tensor)

        init_bias_tensor = t.zeros(num_features)
        self.bias = nn.Parameter(init_bias_tensor)
                
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            # Running statistics 
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True) 
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked +=1
        else:
            mean = self.running_mean.reshape(1,self.num_features,1,1)
            var = self.running_var.reshape(1,self.num_features,1,1)
        
        weights = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")

        x_ = (x-mean)/t.sqrt(var+ self.eps) 

        x_ = x_ * weights + bias
        return x_

    def extra_repr(self) -> str:
        pass

utils.test_batchnorm2d_module(BatchNorm2d)
utils.test_batchnorm2d_forward(BatchNorm2d)
utils.test_batchnorm2d_running_mean(BatchNorm2d)


# %% 
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=(-2,-1))


# %% 
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. 
        Declare it second using another `Sequential`.
        '''
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
       
        self.left = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        # self.left = Sequential(
        #     nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1, bias=False),
        #     BatchNorm2d(out_feats),
        #     nn.ReLU(),
        #     nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        #     BatchNorm2d(out_feats)
        # )
        if first_stride > 1:
            self.right = nn.Sequential(
                nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, bias=False),
                nn.BatchNorm2d(out_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
            # self.right = Sequential(
            #     nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, bias=False),
            #     BatchNorm2d(out_feats)
            # )
        else:
            self.right = nn.Identity()
        self.relu = nn.ReLU()
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x_l = self.left(x)
        x_r = self.right(x)
        out = self.relu(x_l+x_r)
        return out
        
        # self.left = nn.Sequential(
        #     nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1, bias=False),
        #     BatchNorm2d(out_feats),
        #     nn.ReLU(),
        #     nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
        #     BatchNorm2d(out_feats)
        # )
        
        # if first_stride > 1:
        #     self.right = nn.Sequential(
        #         nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, bias=False),
        #         BatchNorm2d(out_feats)
        #     )


# %% 
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()

        blocks = []        
        blocks.append(
            ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)
            )
        for _ in range(n_blocks-1): 
            blocks.append(
                ResidualBlock(in_feats=out_feats, out_feats=out_feats)
                # only works if I dont pass the first_stride argument 
                )
        self.seq = nn.Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.seq(x)

# %% 
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),        
            )

        all_feats = [64] + out_features_per_group[:-1] 
        # shifting the information for on position 
        self.residual_layers = nn.Sequential()
        for idx, (n_blocks, in_f, out_f, first_stride) in enumerate(zip(
            n_blocks_per_group, all_feats, out_features_per_group, first_strides_per_group
        )):
            self.residual_layers.add_module(f"layer{idx+1}", 
            BlockGroup(n_blocks, in_f, out_f, first_stride))
        self.out_layers = nn.Sequential()
        self.out_layers.add_module("avgpool", AveragePool())
        self.out_layers.add_module("Flatten", nn.Flatten())
        self.out_layers.add_module("fc", nn.Linear(out_features_per_group[-1], n_classes, bias=True))
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        
        return x
      


# %% 
#og_weights = torchvision.models.resnet34(weights="DEFAULT")
my_resnet = ResNet34()

imported_resnet = torchvision.models.resnet34(weights='IMAGENET1K_V1')

utils.print_param_count(my_resnet,imported_resnet, use_state_dict=True)



# %%

'''Copy over the weights of `pretrained_resnet` to your resnet.'''

mydict = my_resnet.state_dict()
pretraineddict = imported_resnet.state_dict()

# Check the number of params/buffers is correct
assert len(mydict) == len(pretraineddict), "Number of layers is wrong. Have you done the prev step correctly?"

# Initialise an empty dictionary to store the correct key-value pairs
state_dict_to_load = {}

for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
    state_dict_to_load[mykey] = pretrainedvalue

my_resnet.load_state_dict(state_dict_to_load)




# %%
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = Path("./resnet_inputs")

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]


# %%
images[-3]
# %%
def prepare_data(images: list[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    t_images = []
    # t_images = t.Tensor()
    trans_to_tensor = transforms.ToTensor()
    trans_to_size = transforms.Resize((224, 224))
    trans_to_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    for i in images:
        i_ = trans_to_tensor(i)
        i_ = trans_to_size(i_)
        i_ = trans_to_norm(i_)
        t_images.append(i_)
    t_images = t.stack(t_images,0)
    return t_images

prepared_images = prepare_data(images)
print(prepared_images[0])


# %%
def predict(model, images):
    logits = model(images)
    return logits.argmax(dim=1)

res = predict(my_resnet, prepared_images)
with open("w0d3/imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())
for i in res:
    print(imagenet_labels[int(i)])
# %%

# %%


# %%