# %% 
!cp -r /Users/tilman/Documents/projects/arena/arena-v1/w0d2 /Users/tilman/Documents/projects/arena/arena/w0d2

# %%
import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import w0d2.utils as utils

arr = np.load("w0d2/numbers.npy")
# %%
utils.display_array_as_img(arr[5][2])
print(arr.shape) # (6, 3, 150, 150)


# %% Einops 1
arr1 = rearrange(arr, 'b c h w -> c h (b w)')
print(arr1.shape)
utils.display_array_as_img(arr1)

# %% Einops 2
arr2 = repeat(arr[0], 'c h w ->  c (2 h) w ')
print(arr2.shape)
utils.display_array_as_img(arr2)

# %% Einops 3 (me)
arr3 = repeat(arr[0:2], 'b c h w -> b c h (2 w) ')
print(arr3.shape)
arr3 = rearrange(arr3, 'b c h w -> c (b h) w')
print(arr3.shape)
utils.display_array_as_img(arr3)

# %% Einops 3 (correct)
arr3 = repeat(arr[0:2], 'b c h w -> c (b h) (2 w)')
utils.display_array_as_img(arr3)


# %% Einops 4
arr4 = repeat(arr[0], "c h w -> c (h 2) w")
utils.display_array_as_img(arr4)

# %% Einops 5
arr5 = rearrange(arr[0], "c h w -> h (c w)")
utils.display_array_as_img(arr5)
# %% Einops 6
arr6 = rearrange(arr, "(b1 b2) c h w-> c (b1 h) (b2 w)", b1=2, b2=3)
print(arr.shape)
print(arr6.shape)
utils.display_array_as_img(arr6)

# %% Einops 7
arr7 = reduce(arr, 'b c h w -> h (b w)', 'max')
print(arr7.shape)
utils.display_array_as_img(arr7)

# %%
arr8 = reduce(arr.astype('float'), 'b c h w -> h (b w)', 'mean')
utils.display_array_as_img(arr8)

# %%
arr9 = reduce(arr.astype('float'), 'b c h w -> h w', 'min')
utils.display_array_as_img(arr9)

# %%
arr10 = rearrange(arr[:2], 'b c h w ->  c h (b w)')
#print(arr10.shape) # -> (3, 150, 300) ->[1][2]

arr10 = rearrange(arr10, "c (h1 h2) w -> c h2 (h1 w) ", h1=2)
utils.display_array_as_img(arr10)


# %% 11
arr11 = rearrange(arr[1], "c h w -> c w h")
utils.display_array_as_img(arr11)
# %%
arr12 = rearrange(arr, "(b1 b2) c h w-> c (b1 w) (b2 h)", b1=2, b2=3)
utils.display_array_as_img(arr12)
# %% Didnt got this one
arr13 = rearrange(arr, "(b1 b2) c h w-> c (b1 h) (b2 w)", b1=2, b2=3)
#utils.display_array_as_img(arr13) #( 3, 300, 450)
arr13 = reduce(arr13, 'c (h1 h2) (w1 w2) -> c h2 w2', 'max', h1=2, w1=3)

utils.display_array_as_img(arr13, save=True)


# %% einops 13 (correct)
arr13 = reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
utils.display_array_as_img(arr13, save=True)

# %%
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    out = einsum("i i", mat)
    return out

utils.test_einsum_trace(einsum_trace)

# %%

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einsum('i j, j -> i', mat, vec)

utils.test_einsum_mv(einsum_mv)

# %%
def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einsum('i j, j k -> i k', mat1, mat2)
utils.test_einsum_mm(einsum_mm)

# %%
def einsum_inner(vec1, vec2):
    """
    Returns the same as `np.inner`.
    """
    return einsum('v, v', vec1, vec2)

utils.test_einsum_inner(einsum_inner)


# %%
def einsum_outer(vec1, vec2):
    """
    Returns the same as `np.outer`.
    """
    return einsum('v, w -> v w', vec1, vec2)
utils.test_einsum_outer(einsum_outer)

# %%

# %%
# %%

# %%
