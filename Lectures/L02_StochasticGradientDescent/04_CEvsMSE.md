---
jupytext:
  cell_metadata_json: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
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
from matplotlib import gridspec
from scipy.stats  import norm, multivariate_normal
from scipy.optimize import minimize
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Cross entropy

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

The Negative Logarithmic Likelihood loss

+++ {"slideshow": {"slide_type": "fragment"}}

$$-\sum_{i}\sum_{j} l_{ij} \log \tilde{P}(C_j|x_i) $$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

is an approximation of the quantity

+++ {"slideshow": {"slide_type": "fragment"}}

$$ -\int \text{d}X  P(X) \sum_j P(C_j|X)\log \tilde{P}(C_j|X) $$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

The expression

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$\sum_j P(C_j|X)\log \tilde{P}(C_j|X)$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

is the [_cross entropy_ ](https://en.wikipedia.org/wiki/Cross_entropy) between distributions $P(C_j|X)$: the true distribution of categories $C_i$ given features $X$ and approximating distribution $ \tilde{P}(C_j|X)$.

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

Cross entropy between two discrete distributions $p$ and $q$ is defined as the expactaion value of $\log q$ with respect to $p$:

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$H(p,q)\equiv -E[\log q]_p =  -\sum p_i \log q_i$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

One can show that this quantity is minimised if and only if $p_i=q_i$. Cross entropy has several properties that make it suitable loss function for working with neural networks. We will explore this below.

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Toy example

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

Let's consider a simple binary case with two categories 0 and 1. Let the true probability of 1 be $p$ and approximated probability be denoted $\tilde{p}$. We can look for $\tilde{p}$ by minimizing the _binary cross entropy_:

+++ {"slideshow": {"slide_type": "slide"}}

$$ \operatorname{BCE}(\tilde p, p) = - p\log \tilde{p} - (1-p) \log (1-\tilde{p})$$

+++ {"tags": ["problem"], "slideshow": {"slide_type": "skip"}}

__Problem__

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["problem"]}

Show that this function as a function of $\tilde{p}$ does have a minimum when $\tilde{p}=p$.

We will consider a simple case when the probability $\tilde p$ is given by simple logistic function of some parameter $x$:

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$\tilde{p} = \frac{1}{1+e^{-x}}, \quad 1-\tilde{p} = \frac{1}{1+e^x}$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

Then cross entropy is given by:

+++ {"slideshow": {"slide_type": "fragment"}}

$$p \log (1+e^{-x}) +  (1-p) \log (1+e^x)$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

and its  derivative by:

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{\text{d}}{\text{d}x}BCE(\tilde{p}(x),p)=-\frac{p}{1+e^{-x}}+\frac{1-p}{1+e^x} = \frac{1}{1+e^{-x}}-p$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

For the MSE error this is respectively

+++ {"slideshow": {"slide_type": "slide"}}

$$MSE(x,p) = \frac{1}{2}\left(\tilde{p}-p\right)^2 =  \frac{1}{2}\left(\frac{1}{1+e^{-x}}-p\right)^2$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{\text{d}}{\text{d}x}MSE(\tilde{p}(x),p)=-\left(\frac{1}{1+e^{-x}}-p\right) \frac{e^{-x}}{\left(1+e^{-x}\right)^2}$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

Let's plot the error functions

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def bce(pt,p):
    return -p*np.log(pt)+ -(1-p)*np.log(1-pt)

def mse(pt,p):
    return 0.5*(p-pt)*(p-pt)

def logistic(x):
    return 1/(1+np.exp(-x))

def logit(p):
    return np.log(p/(1-p))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
xs=np.linspace(-10,10,400)
p=0.9

fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].set_title('Cross entropy')
ax[0].plot(xs,bce(logistic(xs),p))
ax[0].axhline(0,c='black');
ax[0].axvline(logit(p), c='red');

ax[1].set_title('MSE')
ax[1].plot(xs,mse(logistic(xs),p));
ax[1].axhline(0,c='black');
ax[1].axvline(logit(p),c='red');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

What we can see from those plots is that the cross  entropy functions, contrary to MSE does not saturatefor large positive and negative values of parameter $x$. Actually it's behaviour is asymptoticaly  linear. That means that it will have non-zero gradients, while MSE gradients will be zero. This is verified by the derivative plots below.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
ys=-(1.0/(1+np.exp(-xs))-p)*np.exp(-xs)/(1+np.exp(-xs))**2
fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].set_title('Cross entropy')
ys=-(logistic(xs)-p)
ax[0].plot(xs,ys)
ax[0].axvline(logit(p),c='red');
ax[0].axhline(0,c='black');

ax[1].set_title('MSE')
ax[1].plot(xs,ys*np.exp(-xs)*logistic(xs)**2)
ax[1].axvline(np.log(p)-np.log(1-p),c='red');
ax[1].axhline(0,c='black');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Logistic regression

+++ {"slideshow": {"slide_type": "skip"}}

While this toy example is nor realy an example of machine learning, similar behaviour persists in more realistic scenarios. Consider a problem of separating two samples:

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
x1 =  multivariate_normal((7,7),(1,1)).rvs(size=100)
x2 = multivariate_normal((-7,-7), (1,1)).rvs(size=100)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
X = np.concatenate((x1,x2), axis=0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
Y = np.concatenate((np.ones(100), np.zeros(100)))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cols=np.array(['red','blue'])
fig,ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0],X[:,1],c=cols[Y.astype(int)]);
```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

We will use logistic regression for this task:

+++ {"slideshow": {"slide_type": "slide"}}

$$\tilde{y}_i = \beta_0x_{i0} +\beta_1x_{i1}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$ \tilde{p}_i = \frac{1}{1+e^{-y_i}}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def lin(x,b1,b2):
    return np.moveaxis(np.multiply.outer(x[:,0],b1) +  np.multiply.outer(x[:,1],b2),0,-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def logistic(x,b1,b2):
    logit = lin(x,b1,b2)
    return 1/(1+np.exp(-logit))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Means squared error

+++ {"slideshow": {"slide_type": "skip"}}

First we will plot the loss function with MSE error

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{1}{2}\sum_i (\tilde{p}_i-l_i)^2$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def mse(x, y, b1, b2):
        err = logistic(x,b1,b2)-y
        return 0.5*np.sum(err*err, axis=-1)/len(x)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
b_min = minimize(lambda x: mse(X,Y,*x),[0,0]).x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
b1s = np.linspace(-2,2,500)
b2s = np.linspace(-2,2,500)
grid  = np.meshgrid(b1s,b2s)
zs = mse(X,Y, grid[0], grid[1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
fig = plt.figure(figsize=(9,8))
gs=gridspec.GridSpec(1,2, width_ratios=[4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
cs=ax1.contourf(grid[0], grid[1],zs, levels=40);
ax1.plot([-2,2],[-2,2],c='red', linewidth=1, linestyle='--')
fig.colorbar(cs, cax=ax2);
ax1.scatter([b_min[0]], [b_min[1]], s=50, color='red')
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

And here are the values along the diagonal (red) line.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
phis = np.linspace(-2,2,500)
es = mse(X,Y, phis, phis)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
x_min = minimize(lambda x: mse(X,Y,x,x),[0]).x.item()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
plt.plot(phis,es)
plt.axvline(x_min,c='red');
```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

We can clearly see a "plateaux" on both sides.

+++ {"slideshow": {"slide_type": "slide"}}

### Cross entropy

+++ {"slideshow": {"slide_type": "skip"}}

And now we will plot the binary cross entropy loss

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sum_i l_i \log \tilde{p}_i + (1-l_i)\log (1-\tilde{p})$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def ce(x, y, b1, b2):
        logit =  logistic(x,b1,b2)
        return -np.sum(y*np.log(logit) + (1-y)*np.log(1-logit), axis=-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
b_min = minimize(lambda x: mse(X,Y,*x),[0,0]).x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
ce(X,Y,1,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
ces = ce(X,Y, grid[0], grid[1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig = plt.figure(figsize=(9,8))
gs=gridspec.GridSpec(1,2, width_ratios=[4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
cs=ax1.contourf(grid[0], grid[1],ces, levels=40);
ax1.plot([-2,2],[-2,2],c='red', linewidth=1, linestyle='--')
ax1.scatter([b_min[0]], [b_min[1]], s=50, color='red')
fig.colorbar(cs, cax=ax2);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
phis = np.linspace(-2,2,500)
es = ce(X,Y,phis, phis)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
x_min = minimize(lambda x: mse(X,Y,x,x),[0]).x.item()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(phis,es)
plt.axvline(x_min,c='green');
```

+++ {"slideshow": {"slide_type": "skip"}}

Now we can see that the "plateaux" is only on one side. But this is the right side! When we are there the value of error is already low. Contrary, when the loss is big we are on the slope with non-zero gradient.
