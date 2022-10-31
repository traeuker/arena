# %% 
!cp -r /Users/tilman/Documents/projects/arena/arena-v1/w0d1 /Users/tilman/Documents/projects/arena/arena/w0d1

# %%
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import w0d1.utils 
            

# %%

# %% 1. DFT, 2. Inverse DFT
def DFT_1d(arr : np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    dim = len(arr)
    y = np.zeros(dim,dtype = 'complex_')
    arr = np.array(arr)
    
    
    out = np.ones((dim,dim),dtype = 'complex_')
    for i in range(1,dim):
        for k in range(1,dim):
            out[i,k]= 1*np.cos(-2*np.pi*i*k/dim)+ 1j*np.sin(-2*np.pi*i*k/dim)
    arr = np.transpose(arr)
    if inverse:
        y = np.linalg.inv(out) @ arr
    else:
        
        y = out @ arr

        
    return y


y=DFT_1d([1,2,3,4,5,5], inverse=True)
w0d1.utils.test_DFT_func(DFT_1d)
# %% Integration

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """
    space = np.linspace(x0,x1,n_samples)
    sum = 0
    for x in range(n_samples-1):
        diff = space[x+1]-space[x]
        hight = func(space[x])
        sum += diff*hight
    return sum

def test_integrate_function(integrate_function):
    
    func = lambda x: np.sin(x) ** 2
    x0 = -np.pi
    x1 = np.pi
    
    integral_approx = integrate_function(func, x0, x1)
    integral_true = 0.5 * (x1 - x0)
    
    np.testing.assert_allclose(integral_true, integral_approx, atol=1e-10)
    
w0d1.utils.test_integrate_function(integrate_function)

# %% 
def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    space = np.linspace(x0,x1,n_samples)
    sum = 0
    for x in range(n_samples-1):
        diff = space[x+1]-space[x]
        hight = func1(space[x])*func2(space[x])
        sum += diff*hight

    return sum
def test_integrate_product(integrate_product):
    
    func1 = np.sin
    func2 = np.cos 
    x0 = -np.pi
    x1 = np.pi
    
    integrate_product_approx = integrate_product(func1, func2, x0, x1)
    integrate_product_true = 0
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)
    
    integrate_product_approx = integrate_product(func1, func1, x0, x1)
    integrate_product_true = 0.5 * (x1 - x0)
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)
w0d1.utils.test_integrate_product(integrate_product)
# %% Fourier Series
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """
    a_0, A_n, B_n = 0, np.zeros(max_freq), np.zeros(max_freq)#np.ones(max_freq, dtype="complex_"), np.ones(max_freq, dtype="complex_")
    
    
    a_0 = 1/np.pi * integrate_function(func, -np.pi, np.pi)
    
    
    for i in range(1,max_freq):
        A_n[i] = 1/np.pi * integrate_product(func, lambda x: np.cos(i*x), -np.pi, np.pi)
        B_n[i] = 1/np.pi * integrate_product(func, lambda x: np.sin(i*x), -np.pi, np.pi)
    


    def func_approx(x):
        y = a_0 *0.5
        for n in range(max_freq):
             y += A_n[n]* np.cos(n*x) + B_n[n]* np.sin(n*x)
        return y

    func_approx = np.vectorize(func_approx)
                
    return ((a_0, A_n, B_n), func_approx)

step_func = lambda x: 1 * (x > 0)
w0d1.utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
# %% Basic Neural Networks - (II) Tensors 

import torch 

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 2000
LEARNING_RATE = 2e-6

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-np.pi, np.pi, 2000, dtype=dtype, device=device)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
# print(x_cos.shape) -> [2,2000]
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.rand((),device=device, dtype=dtype)
A_n = torch.rand(NUM_FREQUENCIES,device=device,dtype=dtype)
B_n = torch.rand(NUM_FREQUENCIES,device=device,dtype=dtype)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + A_n @ x_cos + B_n @ x_sin

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.square(y - y_pred).sum()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.numpy(), A_n.numpy(), B_n.numpy()])
        y_pred_list.append(y_pred.clone())

    # TODO: compute gradients of coeffs with respect to `loss`
    grad_y_pred = 2*(y_pred-y)

    grad_a_0 = 1/2 * grad_y_pred.sum() 
    grad_a = x_cos @ grad_y_pred 
    grad_b = x_sin @ grad_y_pred  
    #print(x_sin.shape, grad_y_pred.shape)

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= grad_a_0 * LEARNING_RATE
    A_n -= grad_a * LEARNING_RATE
    B_n -= grad_b * LEARNING_RATE

    
w0d1.utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %% (III) Autograd
x = torch.linspace(-torch.pi, torch.pi, TOTAL_STEPS, dtype=dtype, device=device)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.rand((),device=device, dtype=dtype,requires_grad = True)
A_n = torch.rand(NUM_FREQUENCIES,device=device,dtype=dtype,requires_grad = True)
B_n = torch.rand(NUM_FREQUENCIES,device=device,dtype=dtype,requires_grad = True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    
    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + A_n @ x_cos + B_n @ x_sin
    #print(a_0.grad_fn, A_n.grad_fn, B_n.grad_fn)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = torch.square(y - y_pred).sum()
    
    if step % 1400 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.detach().numpy(), A_n.detach().numpy(), B_n.detach().numpy()])
        y_pred_list.append(y_pred.detach().clone())
    
    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()    

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with torch.no_grad():
        a_0 -= a_0.grad * LEARNING_RATE
        A_n -= A_n.grad * LEARNING_RATE
        B_n -= B_n.grad * LEARNING_RATE

    a_0.grad = None
    A_n.grad = None
    B_n.grad = None
w0d1.utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)





# %% (IV) Models
import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.ln = torch.nn.Linear(2*NUM_FREQUENCIES,1)
        self.flat = torch.nn.Flatten(0, 1)

    def forward(self, x):
        x = self.ln(x)
        x = self.flat(x)
        return x   

b = Net()
x_cat = torch.cat((x_sin,x_cos)).T
assert x_cat.shape == (2000, 2 * NUM_FREQUENCIES)
loss_l = []

for step in range(TOTAL_STEPS):    
    y_pred = b(x_cat)
    #print(y.shape, y_pred.shape)

    loss = torch.square(y - y_pred).sum()
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([
            list(b.parameters())[1].detach().numpy().squeeze().copy(), # a_0
            list(b.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:].copy(), # A_n 
            list(b.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:].copy(), # B_n
            ]) 
        y_pred_list.append(y_pred.detach().clone())

    loss_l.append(loss)
   
    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()    

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with torch.no_grad():
        for p in b.parameters():
            p -= p.grad *LEARNING_RATE

    b.zero_grad()

w0d1.utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)



# %% Optimizer
import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.ln = torch.nn.Linear(2*NUM_FREQUENCIES,1)
        self.flat = torch.nn.Flatten(0, 1)

    def forward(self, x):
        x = self.ln(x)
        x = self.flat(x)
        return x   

b = Net()
optimizer = torch.optim.SGD(b.parameters(),lr=LEARNING_RATE)
x_cat = torch.cat((x_sin,x_cos)).T
assert x_cat.shape == (2000, 2 * NUM_FREQUENCIES)
loss_l = []

for step in range(TOTAL_STEPS):    
    y_pred = b(x_cat)
    #print(y.shape, y_pred.shape)

    loss = torch.square(y - y_pred).sum()
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([
            list(b.parameters())[1].detach().numpy().squeeze().copy(), # a_0
            list(b.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:].copy(), # A_n 
            list(b.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:].copy(), # B_n
            ]) 
        y_pred_list.append(y_pred.detach().clone())

    #loss_l.append(loss)
   
    # TODO: compute gradients of coeffs with respect to `loss`
    loss.backward()    

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    optimizer.step()
    optimizer.zero_grad()

w0d1.utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %%
# %%