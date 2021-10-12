---
jupytext:
  cell_metadata_json: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
from IPython.display import SVG
```

+++ {"slideshow": {"slide_type": "slide"}}

# Autodifferentiation

+++ {"slideshow": {"slide_type": "skip"}}

So far we have while performing the gradient descent we have used analitically derived formulas for the gradient. This is clearly impractical for larger systems. The core functionality of all of the neural networks libraries are *autodiferentiation* capabilities. We will explore them using a popular PyTorch library.

+++ {"slideshow": {"slide_type": "skip"}}

We start by loading and visualizing the same example as in the gradient descent:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data = np.load("../data/sgd_data.npy").astype('float32')
rxs = data[:50,0]
rys = data[:50,1]
```

+++ {"slideshow": {"slide_type": "skip"}}

Please note that we have explicitely changed the type to `float32`. This is a good practice when training neural networks. First the data will take less memory and second this will enable training on GPUs in the future.

```{code-cell} ipython3
plt.scatter(rxs,rys, alpha=0.7, color='none', edgecolor="black");
```

+++ {"slideshow": {"slide_type": "skip"}}

We will be fitting function

+++ {"slideshow": {"slide_type": "slide"}}

$$f(x|\omega, t) = \sin(\omega x +t)$$

+++ {"slideshow": {"slide_type": "skip"}}

by minimizing the mean squared error:

+++ {"slideshow": {"slide_type": "slide"}}

$$MSE(\omega,t|\textbf{x},\textbf{y}) = \frac{1}{2}\frac{1}{N}\sum_{i=1}^N \left(y_i-f(x_i|\omega, t)\right)^2 $$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def mse(f, x, y, o, t):
        err = f(x,o,t)-y
        return 0.5*np.sum(err*err, axis=-1)/len(x)
    
#Tensor version of the fit function described in GradientDescent notebook. 
def fitf_tensor(x,o,t):
    return np.moveaxis(np.sin(np.tensordot(np.atleast_1d(x),o,0)+t),0,-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
grid_size = 500
os = np.linspace(0, 2*np.pi, grid_size)
ts = np.linspace(-np.pi,np.pi,grid_size)
otg = np.meshgrid(os,ts)

vg = mse(fitf_tensor, rxs, rys, otg[0], otg[1])
```

```{code-cell} ipython3
cmap=plt.get_cmap('hot') 
target_fmt = {'c':'white', 'edgecolor':'red','s':80}    
traj_fmt = {'c':'white', 'edgecolor':'white','s':60}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from matplotlib import gridspec

fig = plt.figure(figsize=(9,8))
gs=gridspec.GridSpec(1,2, width_ratios=[4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
cs=ax1.contourf(otg[0], otg[1],vg, levels=40, cmap=cmap);
fig.colorbar(cs, cax=ax2);
ax1.scatter([2.188], [1],**target_fmt);
```

+++ {"slideshow": {"slide_type": "skip"}}

We will be using the [PyTorch](https://pytorch.org) package which is the second (and rising fast) most popular neural network library.  Let's start by importing it:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import torch 
```

+++ {"slideshow": {"slide_type": "skip"}}

Torch has its own implementation of tensors and we have to convert `numpy` arrays to torch tensors:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
t_rxs = torch.from_numpy(rxs)
t_rys = torch.from_numpy(rys)
```

+++ {"slideshow": {"slide_type": "skip"}}

We will now create the tensors for parameters to be optimized: `o` and `t`.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
o = torch.FloatTensor([3]) # One dimensional tensor of length 1 with value  3.0
t = torch.FloatTensor([1]) # One dimensional tensor of length 1 with value  1.0
```

+++ {"slideshow": {"slide_type": "skip"}}

The `torch` module has the ability to track the gradients of the functions. To this end we must mark the variables with respect to which the derivatives will be taken. In our case we want derivatives with respect to `o` and `t`. We do it by setting the `requires_grad` attribute of those tensors. This can be done using method `requires_grad_`:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
o.requires_grad_(True)
t.requires_grad_(True);
```

+++ {"slideshow": {"slide_type": "skip"}}

By convention  methods that modify tensors in place have names that end in `_`.

+++ {"slideshow": {"slide_type": "skip"}}

Now PyTorch will track the derivatives (gradients) with respect to `o` and `t` for every expression which  contains those variables. And we can use it to make our gradient descent:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax =  plt.subplots(figsize=(8,8))
ax.contourf(otg[0], otg[1],vg, levels=40, cmap=cmap)
p = [o.item(), t.item()]
ax.scatter([p[0]], [p[1]],**traj_fmt)
eta = 0.1
trajectory_list=[]
n_iter = 50
for i in range(n_iter):
    if not ( o.grad is None):
        o.grad.data.zero_()
    if not ( t.grad is None):
        t.grad.data.zero_()
        
    prediction = torch.sin(t_rxs*o+t)
    residual = t_rys-prediction;
    loss = 0.5*torch.mean(residual*residual)
    loss.backward()
    o.data.sub_(eta*o.grad)
    t.data.sub_(eta*t.grad)
    p =  [o.item(), t.item()]
 
    ax.scatter(p[0:1], p[1:],**traj_fmt)
    plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

So let's look closely at the core loop:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for i in range(n_iter):
    # attribute grad is not defined before we call backward for the first time
    # so we must check if it is None or not before stting it to zero. 
    # We have to set to to zero because backward() adds gradients to previous values.  
    if not ( o.grad is None):
        o.grad.data.zero_()
    if not ( t.grad is None):
        t.grad.data.zero_()
        
    prediction = torch.sin(t_rxs*o+t)
    residual = t_rys-prediction;
    loss = 0.5*torch.mean(residual*residual)
    # This call calculates the gradient of loss with respect to all tensors that have 
    # require_grad set to True
    loss.backward() 
    #This make the descent step. We use the underlying `data` attribute to modify parameters
    #because it's not alloved to change in-place tensors with require_grad set to True 
    o.data.sub_(eta*o.grad)
    t.data.sub_(eta*t.grad)    
```

+++ {"slideshow": {"slide_type": "skip"}}

We start by zeroing the gradients because default behaviour is to accumulate them. However at the begining the object that holds gradients does not exists so we have to check for that.

+++ {"slideshow": {"slide_type": "skip"}}

Next we just calculate the loss and call `backward` on it. This function traverse the *computations graph* backward and calculates gradients. Then the `grad` attribute of `o` and `t` contains the gradient of loss with respect to `o` and `t`.

+++ {"slideshow": {"slide_type": "skip"}}

 We substract those gradients from the values of `o` and `t` using method `sub_`. By convention  methods that modify the object in place have names that end in `_`. Another technical detail is that we are not allowed to modify in place a variable that requires grad, as this could potentially disrupt the  gradient calculations. That's why we do it by using the `data` attribute that holds reference to underlying tensor.

+++ {"slideshow": {"slide_type": "slide"}}

### Computations graph

+++ {"slideshow": {"slide_type": "skip"}}

All this magic depends on so called "computations graph" which describes all the computations done while calculating the loss. Below is a visualisation of actual  computation graph in this case.

```{code-cell} ipython3
SVG('../figures/call_graph.svg' )
```

+++ {"slideshow": {"slide_type": "skip"}}

This can be written as a series of operations:

+++ {"slideshow": {"slide_type": "slide"}}

$$v_0 = \frac{1}{2} v_1 \qquad v_1 = \operatorname{mean}(v_2)$$

$$v_2 = v_3 \cdot v_3\qquad v_3 = y - v_5$$

$$v_4=\sin(v_6) \qquad v_5= v_6 + v_7$$

$$v_6 = o*x \qquad v_7 = t$$

```{raw-cell}
---
slideshow:
  slide_type: skip
---
To differentiate we use the chain rule
```

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_0}{d t}=\frac{d v_0}{d v_1}\frac{d v_1}{d t} $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_0}{d v_1}=\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "skip"}}

Then  we do it again with the derivative of the next term

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_1}{d t}=\frac{d v_1}{d v_2}\frac{d v_2}{d t}$$

+++ {"slideshow": {"slide_type": "skip"}}

Because $v_2$ is a vector this is equal to

+++ {"slideshow": {"slide_type": "slide"}}

$$\sum_i \frac{d v_1}{d v_{2i}}\frac{d v_{2i}}{d t}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_1 = \frac{1}{N}\sum_{i=1}^N v_{2i}$$

+++ {"slideshow": {"slide_type": "skip"}}

so

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_1}{d v_{2i}}=\frac{1}{N}$$

+++ {"slideshow": {"slide_type": "skip"}}

and

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_1}{d t} = \sum_i \frac{d v_1}{d v_{2i}}\frac{d v_{2i}}{d t}=\frac{1}{N}\sum_i  \frac{d v_{2i}}{d t}$$

+++ {"slideshow": {"slide_type": "skip"}}

And we continue with next term:

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_{2i}}{d t} = \sum_{j}\frac{d v_{2i}}{d v_{3j}}\frac{d v_{3j}}{d t}+\sum_{j}\frac{d v_{2i}}{d v_{4j}}\frac{d v_{4j}}{d t}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{2i}}{d v_{3j}} = \delta_{ij} v_{4j}\qquad\frac{d v_{2i}}{d v_{4j}} = \delta_{ij} v_{3j}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{2i}}{d t} = v_{4i}\frac{d v_{3i}}{d t} + v_{3i}\frac{d v_{4i}}{d t}$$

+++ {"slideshow": {"slide_type": "skip"}}

and next one

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_{3i}}{d t}=
\sum_{j}
\frac{d v_{3i}}{d v_{5j}}\frac{d v_{5j}}{d t}$$

+++ {"slideshow": {"slide_type": "-"}}

$$\frac{d v_{4i}}{d t}=
\sum_{j}
\frac{d v_{4i}}{d v_{5j}}\frac{d v_{5j}}{d t}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{3i}}{d v_{5j}}=\frac{d v_{4i}}{d v_{5j}}=-1$$

+++

and so on:

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_{5i}}{d t}=\sum_{j}\frac{d v_{5i}}{d v_{6j}}\frac{d v_{6j}}{d t}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{5i}}{d v_{6j}}=\delta_{ij} \cos(v_{6j})$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{d v_{6i}}{d t}=\sum_j \frac{d v_{6j}}{d v_{7j}}\frac{d v_{7j}}{d t}
+ \sum_j \frac{d v_{6j}}{d v_{8j}}\frac{d v_{8j}}{d t}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{7j}}{d t}  = 0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{d v_{8j}}{d t}  = 1$$

+++ {"slideshow": {"slide_type": "skip"}}

What is important is that at every stage we need only the local values of the gradient and the variables at the given graph node. The final gradient is obtained by multiplying all the gradients  going backward through the graph.

+++ {"slideshow": {"slide_type": "slide"}}

## Optimizer

+++ {"slideshow": {"slide_type": "skip"}}

Finally let's add one more improvement. So far we were using a very simple gradient descent strategy. This will change and to this end PyTorch encapsulates the gradient steping in separate object called *optimizer*.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
o = torch.FloatTensor([3])
t = torch.FloatTensor([1])
o.requires_grad_(True)
t.requires_grad_(True);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
optim = torch.optim.SGD([o,t],lr=0.1)
```

+++ {"slideshow": {"slide_type": "skip"}}

SGD stands for "Stochastic Gradient Descent". The "stochastic" part will be explained later, but actually it just does the plain gradient descent. The parameter `lr`(learning rate)  is our `eta`.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax =  plt.subplots(figsize=(8,8))
ax.contourf(otg[0], otg[1],vg, levels=20)
p = [o.item(), t.item()]
ax.scatter([p[0]], [p[1]],c='none', s=20, edgecolor='red')
n_iter = 50
for i in range(n_iter):
    optim.zero_grad()
        
    prediction = torch.sin(t_rxs*o+t)
    residual = t_rys-prediction;
    loss = 0.5*torch.mean(residual*residual)
    loss.backward()
    optim.step()
    p =  [o.item(), t.item()]
 
    ax.scatter(p[0:1], p[1:],c='red', s=20, edgecolor='red')
    plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

## Loss function

+++ {"slideshow": {"slide_type": "skip"}}

We also do not have to write our own loss functions:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
o = torch.FloatTensor([3])
t = torch.FloatTensor([1])
o.requires_grad_(True)
t.requires_grad_(True);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
loss_f = torch.nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
optim = torch.optim.SGD([o,t],lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax =  plt.subplots(figsize=(8,8))
ax.contourf(otg[0], otg[1],vg, levels=20)
p = [o.item(), t.item()]
ax.scatter([p[0]], [p[1]],c='none', s=20, edgecolor='red')
n_iter = 50
for i in range(n_iter):
    optim.zero_grad()
        
    prediction = torch.sin(t_rxs*o+t)
    loss = loss_f(prediction, t_rys)
    loss.backward()
    optim.step()
    p =  [o.item(), t.item()]
 
    ax.scatter(p[0:1], p[1:],c='red', s=20, edgecolor='red')
    plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

As you can see thare is a slight difference, the steps seem to be longer. That's because the `MSELoss` does not contain the one  $\frac{1}{2}$ factor, so the gradient is twice as long.
