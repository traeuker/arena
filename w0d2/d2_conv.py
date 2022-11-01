# %%
import w0d2.utils as utils
import torch as t
from fancy_einsum import einsum
from typing import Optional
import functools
import numpy as np

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''

    batch_size, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape

    out_width = width - kernel_width + 1
    # x_ size : (batch_size, i)
    # x_ stride : batch, channel, width, width

    xsB, xsI, xsWi = x.stride()
    x_new_stride = (xsB, xsI, xsWi, xsWi)
    x_ = t.as_strided(x, (batch_size, in_channels, out_width, kernel_width), x_new_stride)

    out = einsum('b in_c out_w kern_w, out_c in_c kern_w -> b out_c out_w', x_, weights)

    # out size : (batch_sisze, out_channels, out_width)
    return out 

utils.test_conv1d_minimal(conv1d_minimal)
# %%
def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    out_width = width - kernel_width + 1
    out_hight = height - kernel_height + 1
    
    # x_ size : (batch_size, i)
    # x_ stride: batch, channel, height, width 

    xsB, xsI, xsHe, xsWi = x.stride()

    x_new_stride = (xsB, xsI, xsHe, xsWi, xsHe, xsWi)
    x_ = t.as_strided(x, (batch, in_channels, out_hight, out_width, kernel_height, kernel_width), x_new_stride)

    out = einsum('b in_c out_h out_w kern_h kern_w, out_c in_c kern_h kern_w -> b out_c out_h out_w', x_, weights)

    # out size : (batch, out_channels, output_height, output_width)
    return out 

utils.test_conv2d_minimal(conv2d_minimal)

# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    batch, in_channels, width = x.shape
    new_size = (batch, in_channels, left + right + width)
    
    out = x.new_full(new_size, fill_value=pad_value)

    out[..., left : left + width] = x
    return out 


utils.test_pad1d(pad1d)
utils.test_pad1d_multi_channel(pad1d)

# %%
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    batch, in_channels, height, width = x.shape
    new_size = (batch, in_channels, top + height + bottom, left + width + right)
    
    out = x.new_full(new_size, fill_value=pad_value)

    out[...,top : top + height, left : left + width] = x
    return out

utils.test_pad2d(pad2d)
utils.test_pad2d_multi_channel(pad2d)

# %%
def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    batch, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape

    if padding:
        x = pad1d(x, left=padding, right=padding, pad_value=0)
    
    out_width = 1 + (width + 2 * padding - kernel_width) // stride 

    xsB, xsI, xsWi = x.stride()

    x_new_stride = (xsB, xsI, xsWi * stride, xsWi)
    x_new_shape = (batch, in_channels, out_width, kernel_width)
    x_ = t.as_strided(x, x_new_shape, x_new_stride)

    out = einsum('b in_c out_w kern_w, out_c in_c kern_w -> b out_c out_w', x_, weights)

    return out # size : (batch_size, out_channels, out_width)

utils.test_conv1d(conv1d)
 # %%
from typing import Union
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:
#       force_pair((1, 2))     ->  (1, 2)
#       force_pair(2)          ->  (2, 2)
#       force_pair((1, 2, 3))  ->  ValueError

# %%

def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    if padding:
        x = pad2d(x, padding_w, padding_w, padding_h, padding_h, pad_value=0)
    out_width = int((width + 2* padding_w - kernel_width)//stride_w + 1)
    out_hight = int((height + 2* padding_h - kernel_height)//stride_h + 1)
    
    # x_ size : (batch_size, i)
    # x_ stride: batch, channel, height, width 

    xsB, xsI, xsHe, xsWi = x.stride()

    x_new_stride = (xsB, xsI, xsHe*stride_h, xsWi*stride_w, xsHe, xsWi)
    x_ = t.as_strided(x, (batch, in_channels, out_hight, out_width, kernel_height, kernel_width), x_new_stride)

    out = einsum('b in_c out_h out_w kern_h kern_w, out_c in_c kern_h kern_w -> b out_c out_h out_w', x_, weights)

    # out size : (batch, out_channels, output_height, output_width)
    return out

utils.test_conv2d(conv2d)



# %%
def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    batch, channels, height, width = x.shape

    kernel_height, kernel_width = force_pair(kernel_size)
    if stride is None:
        stride = kernel_size
    stride_h, stride_w = force_pair(stride)
    
    if padding:
        padding_h, padding_w = force_pair(padding)
        x = pad2d(x, padding_w, padding_w, padding_h, padding_h, pad_value=-t.inf)
    else:
        padding_h, padding_w = 0,0
    out_width = int((width + 2* padding_w - kernel_width)//stride_w + 1)
    out_hight = int((height + 2* padding_h - kernel_height)//stride_h + 1) 

    xsB, xsI, xsHe, xsWi = x.stride()

    x_new_stride = (xsB, xsI, xsHe*stride_h, xsWi*stride_w, xsHe, xsWi)
    x_ = t.as_strided(x, (batch, channels, out_hight, out_width, kernel_height, kernel_width), x_new_stride)
    
    out = t.amax(x_, dim=(-1, -2))
    return out 

utils.test_maxpool2d(maxpool2d)


# %%
import torch.nn as nn

# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_h, self.kernel_w = force_pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
        self.stride_h, self.stride_w = force_pair(stride)
        self.padding_h, self.padding_w = force_pair(padding)
        
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        batch, channels, height, width = x.shape
        out_width = int((width + 2* self.padding_w - self.kernel_w)//self.stride_w + 1)
        out_hight = int((height + 2* self.padding_h - self.kernel_h)//self.stride_h + 1) 

        x = pad2d(x, self.padding_w, self.padding_w, self.padding_h, self.padding_h, pad_value=-t.inf)
        xsB, xsI, xsHe, xsWi = x.stride()

        x_new_stride = (xsB, xsI, xsHe*self.stride_h, xsWi*self.stride_w, xsHe, xsWi)
        x_ = t.as_strided(x, (batch, channels, out_hight, out_width, self.kernel_h, self.kernel_w), x_new_stride)
    
        out = t.amax(x_, dim=(-1, -2))
        return out 
    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return """MaxPool2d is a nn.Module that will take a Tensor x and pools the maximum 
        values over the specified kernel size, with additional stride and padding.
        
        kernel_size: (int or pair), 
        stride: (int or pair), default None, 
        padding: (int or pair), default 1
        """

utils.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")

# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        x[x<0] = 0
        return x

utils.test_relu(ReLU)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        #return input.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
        shape = input.shape

        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        
        shape1 = shape[:start_dim]
        shape2 = shape[end_dim+1:]

        mid_shape = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1]) 
        # muliplies all the dimensions between start and end together to one big one
        # --> flattens these dimesions to one 

        new_shape = shape1+(mid_shape,)+shape2
        out = input.reshape(new_shape)
        return out

    def extra_repr(self) -> str:
        return """
        input: (Tensor) the input tensor.

        start_dim: (int) the first dim to flatten

        end_dim: (int) the last dim to flatten
        """

utils.test_flatten(Flatten)

# %%
from fancy_einsum import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.use_bias = bias

        N = 1 / np.sqrt(in_features)
        # dist = t.distributions.Uniform(-N, N)
        
        self.weight = N * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(self.weight)

        if bias is False:
            self.bias = None
        else:
            self.bias = N * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(self.bias)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        x = einsum('... in, out in -> ... out', x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        pass

utils.test_linear_forward(Linear)
utils.test_linear_parameters(Linear)
utils.test_linear_no_bias(Linear)


# %%

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels

        self.kernel_height, self.kernel_width = force_pair(kernel_size)
        self.stride_h, self.stride_w = force_pair(stride)
        self.padding_h, self.padding_w = force_pair(padding)    

        N = 1 / np.sqrt(in_channels * self.kernel_width * self.kernel_height)
        
        # weights -> kernel of shape (out_channels, in_channels, kernel_height, kernel_width))
        weight_tensor = t.rand(self.out_channels, self.in_channels, self.kernel_height, self.kernel_width)
        self.weight = N * (2 * weight_tensor - 1)
        self.weight = nn.Parameter(self.weight)
        
        if padding:
            self.padding = True


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        batch, in_channels, height, width = x.shape
        if self.padding:
            x = pad2d(x, self.padding_w, self.padding_w, self.padding_h, self.padding_h, pad_value=0)

        out_width = int((width + 2* self.padding_w - self.kernel_width)//self.stride_w + 1)
        out_hight = int((height + 2* self.padding_h - self.kernel_height)//self.stride_h + 1)
        
        # x_ size : (batch_size, i)
        # x_ stride: batch, channel, height, width 

        xsB, xsI, xsHe, xsWi = x.stride()

        x_new_stride = (xsB, xsI, xsHe*self.stride_h, xsWi*self.stride_w, xsHe, xsWi)
        x_new_shape = (batch, in_channels, out_hight, out_width, self.kernel_height, self.kernel_width)
        x_ = t.as_strided(x, x_new_shape, x_new_stride)

        out = einsum('b in_c out_h out_w kern_h kern_w, out_c in_c kern_h kern_w -> b out_c out_h out_w', x_, self.weight)

        # out size : (batch, out_channels, output_height, output_width)
        return out
    def extra_repr(self) -> str:
        pass

utils.test_conv2d_module(Conv2d)



# %%

# %%
