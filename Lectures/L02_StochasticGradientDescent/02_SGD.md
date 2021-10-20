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

+++ {"slideshow": {"slide_type": "slide"}}

# Stochastic Gradient Descent

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import utils

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

+++ {"slideshow": {"slide_type": "skip"}}

We retrace the steps from the `AutoDifferentiation` notebook. We will start with first 100 rows of data.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
data = np.load("../data/sgd_data.npy").astype('float32')
rxs = data[:100,0]
rys = data[:100,1]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.scatter(rxs,rys, alpha=0.7, color='none', edgecolor="black");
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def fitf(x,o,t):
    return np.sin(x*o+t)

def fitf_tensor(x,o,t):
    return np.moveaxis(np.sin(np.tensordot(np.atleast_1d(x),o,0)+t),0,-1)

def mse(f, x, y, o, t):
        err = f(x,o,t)-y
        return 0.5*np.sum(err*err, axis=-1)/len(x)

grid_size = 400
os = np.linspace(0, 7, grid_size)
ts = np.linspace(-np.pi,np.pi,grid_size)
otg = np.meshgrid(os,ts)

vg = mse(fitf_tensor, rxs, rys, otg[0], otg[1])

def grad(x,y, o, t):
    return np.array((
        -2*np.sum((y-np.sin(o*x+t))*np.cos(o*x+t)*x),
        -2*np.sum((y-np.sin(o*x+t))*np.cos(o*x+t))
    ))/len(x)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
t_rxs = torch.from_numpy(rxs)
t_rys = torch.from_numpy(rys)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
p = torch.FloatTensor([3,1])
p.requires_grad_(True);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
gd = torch.optim.SGD([p], lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
loss_f = torch.nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
fig_gd, ax =  plt.subplots(1,2,figsize=(16,8))
ax[0].contourf(otg[0], otg[1],vg, levels=20)
ax[0].scatter([p[0].item()], [p[1].item()],c='none', s=20, edgecolor='red')
eta = 0.1
trajectory_list=[]
n_iter = 50
for i in range(n_iter):
    gd.zero_grad()
    prediction = torch.sin(t_rxs*p[0]+p[1])
    loss = loss_f(prediction, t_rys)
    loss.backward()
    gd.step()
    np_p = p.detach().numpy()
    trajectory_list.append(np.concatenate((p.grad.numpy(),np_p,[mse(fitf,rxs, rys,*np_p)])))
    ax[0].scatter([np_p[0]], [np_p[1]],c='red', s=20, edgecolor='red')

trajectory_gd=np.stack(trajectory_list)
utils.plot_grad_and_trajectory(ax[1], trajectory_gd)
ax[1].set_xlabel("epoch")
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig_gd
```

+++ {"slideshow": {"slide_type": "skip"}}

This is a "plain vanilla" implementation of gradient descent. However a big drawback of this method is that the time for one iteration is proportional to the size of the training data set. This is not a problem in this case as the data set is very small, but soon our data will grow much bigger.

+++ {"slideshow": {"slide_type": "slide"}}

# Stochastic gradient

+++ {"slideshow": {"slide_type": "skip"}}

So far we have calculated the gradient as a sum over the whole data set:

+++ {"slideshow": {"slide_type": "slide"}}

$\newcommand{\b}[1]{\mathbf{#1}}$
$\newcommand{\grad}{\operatorname{grad}}$
$$\grad_\textbf{w}  L(\b{y},\b{x}|\b{w}) =\frac{1}{N}\sum_{i=0}^{N-1} \grad_\textbf{w}  L(\b{y}_i,\b{x}_i|\b{w}) 
$$

+++ {"slideshow": {"slide_type": "skip"}}

As you may recall from theory of probability  and statistics this sum is an approximation of the  real loss averaged over the  (unknown)  distribution of data $P(\b{y},\b{x})$.

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\frac{1}{N}\sum_{i=0}^{N-1} \grad_\textbf{w}  L(\b{y}_i,\b{x}_i|\b{w}) 
\approx \left\langle \grad_\textbf{w}  L(\b{y},\b{x}|\b{w}) \right\rangle_{P(\b{y},\b{x})}
$$

+++ {"slideshow": {"slide_type": "skip"}}

When $N$ becomes very large, we can sacrifice some precision by taking only a subset of the data $\{(\b{y}_{i_0}, \b{x}_{i_0}), \ldots,(\b{y}_{i_{M-1}}, \b{x}_{i_{M-1}}  )\}$ to calculate gradient:

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\left\langle \grad_\textbf{w}  L(\b{y},\b{x}|\b{w}) \right\rangle_{P(\b{y},\b{x})}\approx  \grad_\textbf{w} \frac{1}{M}\sum_{j=0}^{M-1} L(\b{y}_{i_j},\b{x}_{i_j}|\b{w}),\quad\text{where}\quad M\ll N
$$

+++ {"slideshow": {"slide_type": "skip"}}

This will be illustrated below. 
Instead of taking the whole data sample, we draw smaller *batches* from it. We then calculate the gradient over this batch. The red arrows represent gradient batches, and the blue arrow the whole sample gradient.

+++ {"slideshow": {"slide_type": "skip"}}

To draw random batch samples we use function `random.choice(sample_size, batch_size)` that draws `batch_size` numbers from range (0,sample_size-1) without replacement  and returns an array containing them. This array is then used to index the original data sample. This is called "fancy indexing" in numpy.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p=(3,1) # (o,t) parameters
sample_size = 100
batch_size = 25
fig, ax = plt.subplots(1,2, figsize=(12,6))
g  = grad(data[:sample_size,0], data[:sample_size,1],*p)
ax[0].set_xlim(-3,3);
ax[0].set_ylim(-3,3);  
ax[0].arrow(0,0,g[0],g[1],color='blue', width=0.01, head_width=0.05, length_includes_head=True, zorder=10)
ax[1].set_xlim(-1.2,1.2);
ax[1].set_ylim(-1.2,1.2);   
ng = g/np.linalg.norm(g)
ax[1].arrow(0,0, ng[0], ng[1],color='blue', width=0.01, head_width=0.05, length_includes_head=True,  zorder=10)
for i in range(12):
    batch_i = np.random.choice(sample_size, batch_size)
    g=grad(data[batch_i,0], data[batch_i,1],*p)
    ax[0].arrow(0,0,g[0],g[1],color='red', width=0.01, head_width=0.05, length_includes_head=True, zorder=1)
    ng = g/np.linalg.norm(g)
    ax[1].arrow(0,0,ng[0],ng[1],color='red', width=0.01, head_width=0.05, length_includes_head=True, zorder=1)
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

The figure on the left show the gradients, on the right we display same gradients but normalized to better compare their directions.

+++ {"slideshow": {"slide_type": "skip"}}

As you can see the gradients fluctuate but point more or less in the same direction. The fluctuations are getting smaller when batch size gets bigger. Please experiment with this number and see it yourself. The biggest possible sample size is 1000.

+++ {"slideshow": {"slide_type": "skip"}}

Below we take first four batches of data and display the mean square error function for each of them. As you can see while differening, they maintain the same structure. Again with bigger data sizes and bigger batches they will differ less.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
sample_size = 100
batch_size = 25
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig, ax = plt.subplots(2,2, figsize=(12,12))
os = np.linspace(0, 2*np.pi, 400)
ts = np.linspace(-np.pi,np.pi,400)
otg = np.meshgrid(os,ts)
for i in range(4):    
    vg = mse(fitf_tensor, 
             data[i*batch_size:(i+1)*batch_size,0], 
             data[i*batch_size:(i+1)*batch_size,1], otg[0], otg[1])
    ax.reshape(-1)[i].contourf(otg[0], otg[1],vg, levels=20)
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

## Stochastic gradient descent

+++ {"slideshow": {"slide_type": "skip"}}

Below we implement "plain vanilla" stochastics gradient descent (SGD). In each iteration we first shufle randomly the data and then split it into batches using function `array_split`. The for each  batch we calculate the gradient and update the parameters accordingly. That way for a single iteration we can get many more parameters updates then in simple gradient descent. This can be seen on the righthand side plot.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
t_data = torch.from_numpy(data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
p = torch.FloatTensor([3,1])
p.requires_grad_(True);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
sgd = torch.optim.SGD([p], lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
loss_f = torch.nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
sample_size = 100
batch_size  = 20
n_batches = (sample_size+batch_size-1)/batch_size
fig_sgd, ax =  plt.subplots(1,2,figsize=(16,8))
ax[0].contourf(otg[0], otg[1],vg, levels=20)
ax[0].scatter([p[0].item()], [p[1].item()],c='none', s=20, edgecolor='red')
eta = 0.1
trajectory_list=[]
n_iter = 50
for i in range(n_iter):
    perm = np.random.permutation(sample_size)
    batches = np.array_split(perm,n_batches)
    for b in batches:
        sgd.zero_grad()
        prediction = torch.sin(t_data[b,0]*p[0]+p[1])
        loss = loss_f(prediction, t_data[b,1])
        loss.backward()
        sgd.step()
        np_p = p.detach().numpy()
        trajectory_list.append(np.concatenate((p.grad.numpy(),np_p,[mse(fitf,rxs, rys,*np_p)])))
        ax[0].scatter([np_p[0]], [np_p[1]],c='red', s=20, edgecolor='red')

trajectory_sgd=np.stack(trajectory_list)
utils.plot_grad_and_trajectory(ax[1], trajectory_sgd)
ax[1].set_xlabel("epoch")
plt.close()
```

```{code-cell} ipython3
fig_sgd
```

+++ {"slideshow": {"slide_type": "skip"}}

In the gradient fluctuations you can clearly see the stochastic part of the algorith. Again those fluctuations will be smaller for bigger batch sizes.

+++ {"slideshow": {"slide_type": "skip"}}

A you can see comparing this to Gradient descent

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig_gd
```

+++ {"slideshow": {"slide_type": "skip"}}

 SGD algorithm converges more quickly. This can be better seen in the plot below. The horizontal  axis unit is one epoc, that is one pass over all data. The stochastic gradients descent makes more steps in one epoch in approximately same time as  gradient descent makes one step.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig_comp , ax = plt.subplots(figsize=(8,8))
utils.plot_trajectory(ax, trajectory_gd,1, label="Gradient Descent")
utils.plot_trajectory(ax,trajectory_sgd, n_batches, label="Stochastic Gradient Descent")
ax.legend()
ax.set_xlabel("epoch");
```

+++ {"slideshow": {"slide_type": "slide"}}

## Datasets utils

+++

PyTorch library provides also some utilities in form of the `Dataset` and `DataLoader` interfaces that facilitiate  iteration over batches.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3,1])
p.requires_grad_(True);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
sgd = torch.optim.SGD([p], lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
loss_f = torch.nn.MSELoss(reduction='sum')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
dataset = torch.utils.data.TensorDataset(t_data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train, test = torch.utils.data.random_split(dataset,(800,200))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
errors = []
n_iter = 50
verbose = 0
for i in range(n_iter):
    mse_train_error = 0
    train_count = 0
    for b, in train_loader:
        train_count+=len(b)
        sgd.zero_grad()
        prediction = torch.sin(b[:,0]*p[0]+p[1])
        loss = loss_f(prediction, b[:,1])
        mse_train_error += loss.item()
        loss /= len(b)
        loss.backward()
        sgd.step()
    mse_train_error /= train_count     
    with torch.no_grad():
        mse_test_error = 0.0
        test_count = 0
        for b, in train_loader:
            prediction = torch.sin(b[:,0]*p[0]+p[1])
            test_count +=len(b)
            mse_test_error += torch.nn.functional.mse_loss(prediction, b[:,1], reduction='sum').item()
        mse_test_error/=test_count
        errors.append((i,mse_train_error, mse_test_error))
        if verbose > 0:
            print(f"{i:4d} {loss.item():6.4f} {mse_test_error.item():6.4f}")
errors = np.stack(errors,0)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(errors[:,0], errors[:,1], '.', label='train');
plt.plot(errors[:,0], errors[:,2], '.', label='test');
plt.legend();
```

```{code-cell} ipython3

```
