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
import matplotlib.text as text
```

+++ {"slideshow": {"slide_type": "slide"}}

## Linear regression

+++

In this notebook we will consider a simple linear regression model:

+++ {"slideshow": {"slide_type": "fragment"}}

$$ y_i = x_{ij} w_j + b$$

+++

We will be using the "summation conventions": when an index is repeated the summation over this index is implied:

+++

$$ 
x_{ij} w_j \equiv   \sum_j x_{ij} w_j 
$$

+++

#### Problem 1

+++

Implement function `linear(x,w,b)` that given feature matrix $\mathbf{x}$, weights $\mathbf{w}$ and bias $b$  returns $\mathbf{y}$. **Hint** Use matrix multiplication operator `@`.

```{code-cell} ipython3
def linear(x,w,b):
    return 
```

### Data

+++

#### Problem 2

+++ {"slideshow": {"slide_type": "-"}}

Generate a random feature matrix $\mathbf{x}$ witch 10000 samples and three features, such that first feature is drawn from normal distribution $\mathcal{N}(0,1)$, second feature from  uniform distribution on interval $[0,1)$ and third from $\mathcal{N}(1,2)$, where 
$N(\mu,\sigma)$ denotes normal distribution with mean $\mu$ and standard deviation $\sigma$. To generate random numbers you can use `numpy.random.normal` and `numpy.random.uniform` functions. To collect all features together you can use `numpy.stack` function.

+++ {"slideshow": {"slide_type": "-"}}

Then using $\mathbf{x}$, weights $w_{true}$  and  bias $b_{true}$:

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w_true = np.array([0.2, 0.5,-0.2])
b_true = -1
```

+++ {"slideshow": {"slide_type": "-"}}

generate output $\mathbf{y}$ assuming a normaly distributed $\mathcal{N}(0,0.1)$ noise $\mathbf{\epsilon}$.

+++ {"slideshow": {"slide_type": "-"}}

$$ y_i =  
x_{ij} w_j+b +\epsilon_i 
$$

+++ {"slideshow": {"slide_type": "slide"}}

### Loss

+++

#### Problem 3

+++

Given the means square loss

+++ {"slideshow": {"slide_type": "fragment"}}

$$ MSE(w,b|y,x) = \frac{1}{2}\frac{1}{N}\sum_{i=0}^{N-1} (y_i -  x_{ij} w_j -b  )^2$$

+++

write down the python function `mse(y,x,w,b)` implementing it:

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def mse(y,x,w,b):
    return 
```

### Gradient

+++

and implement functions `grad_w(y,x,w,b)` and `grad_b(y,x,w,b)` implementing those gradients.

```{code-cell} ipython3
:tags: []

def grad_w(y,x,w,b):
    return 

def grad_b(y,x,w,b):
    return 
```

### Gradient descent

+++

#### Problem 4

+++

Implement gradient descent for linear regression. Start from

```{code-cell} ipython3
w = np.asarray([0.0,0.0,0.0], dtype='float64')
b = 1.0 
```

How many epochs did you need to get MSE below 0.0075 ?

+++

### Pytorch

+++

#### Problem 5

+++

Implement gradient descent using pytorch. Start by just rewritting Problem 4 to use torch Tensors instead of numpy arrays.

+++

To convert frrom numpy arrays to torch tensors you can use ``torch.from_numpy()`` function:

```{code-cell} ipython3
import torch as t 
```

```{code-cell} ipython3
t_y = t.from_numpy(y)
t_x = t.from_numpy(x)
t_w = t.DoubleTensor([0,0,0])
t_b = t.DoubleTensor([1.0])
```

Then use the automatic differentiation capabilities of Pytorch. To this end the variable with respect to which the gradient will be calculated, `t_w` and `t_b` in this case, must have attribute
`requires_grad` set to `True`.

+++

The torch will automatically track any expression containing `t_w` and `t_b` and store its computational graph. The method `backward()` can be run on the final expression to back propagate the gradient. The gradient is then accesible as `t_w.grad`.

+++

Finally use  Pytorch  optimisers.
