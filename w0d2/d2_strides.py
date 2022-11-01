# %%
import torch as t
from collections import namedtuple

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)

# %%

TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(#0
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)),
    TestCase(#1
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=(5,), 
        stride=(1,)),
    TestCase(#2
        output=t.tensor([0, 5, 10, 15]), 
        size=(4,), 
        stride=(5,)),
    TestCase(#3
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=(2,3), 
        stride=(5,1)),
    TestCase(#4
        output=t.tensor([[0, 1, 2], [10, 11, 12]]), 
        size=(2,3), 
        stride=(10,1)),
    TestCase(#5
        output=t.tensor([[0, 0, 0], [11, 11, 11]]), 
        size=(2,3),
        stride=(11,0)),    
    TestCase(#6
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,), 
        stride=(6,)),
    TestCase(#7
        output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), 
        size=(2,1,3), 
        stride=(9,0,1)),
    TestCase(#8
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(2,2,2,2),
        stride=(12,4,2,1)),
]
for (i, case) in enumerate(test_cases):
    if (case.size is None) or (case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=case.size, stride=case.stride)
        if (case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {case.output}")
            print(f"Actual: {actual}")
        else:
            print(f"Test {i} passed!")


# %%
import w0d2.utils as utils

def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    mat_size = mat.shape
    out = t.as_strided(mat, (mat_size[0],), (mat_size[0]+1,)).sum()
    return out

utils.test_trace(as_strided_trace)
# %%
def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    
    v_stride = vec.stride()

    v_strided = t.as_strided(vec, mat.shape, (0, v_stride[0]))
    out = mat * v_strided

    return out.sum(dim =1)

utils.test_mv(as_strided_mv)
utils.test_mv2(as_strided_mv)

# %%
def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''

    new_shape = (matA.shape[0], matA.shape[1], matB.shape[1])


    mA = t.as_strided(matA, new_shape, (   matA.stride()[0], matA.stride()[1], 0))
    mB = t.as_strided(matB, new_shape, (0, matB.stride()[0], matB.stride()[1]   ))

    
    out = mA * mB

    return out.sum(dim=1)

utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)

# %%
a = t.Tensor([[1,2],[3,4]])
b = t.Tensor([[2,2],[2,2]])
t.matmul(a,b)
# %%

# %%


# %%

# %%


# %%


# %%
