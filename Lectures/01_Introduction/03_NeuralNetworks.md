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

+++ {"slideshow": {"slide_type": "slide"}}

# Neural networks

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
import torch
```

+++ {"slideshow": {"slide_type": "skip"}}

As we have learned in the previous notebook,  a key ingredient of the supervised learning is finding a mapping that minimizes loss over a given data set. As we cannot generally find a minimum in a set of all functions (and actually we do not want to) we are looking for the minimum in a familly of functions defined by some set of parameters.

+++ {"slideshow": {"slide_type": "skip"}}

The loss function the becomes the function of those parameters only.

+++ {"slideshow": {"slide_type": "skip"}}

The neural networks make up such a familly of functions. Those functions are made up by composing together many elementary simple functions. Those  elementary functions are usually called neurons.

+++ {"slideshow": {"slide_type": "slide"}}

# Neuron

+++ {"slideshow": {"slide_type": "skip"}}

A single neuron can have many inputs and only one output.

```{code-cell} ipython3
# For some reasons contrary to jupyter lab in jupyter notebooks we have to use the SVG function to display
# SVG files. In jupyter lab we can just use html <img> tag. 
SVG('../figures/perceptron.svg')
```

+++ {"slideshow": {"slide_type": "skip"}}

There is a number $w_i$, called *weight*, associated with each input. Each input value $x_i$ is multiplied by the weight and the results are added together and then another  parameter $b$ called 
*bias* is added to the sum:

+++ {"slideshow": {"slide_type": "fragment"}}

$$o = \sum_k w_k x_k +b$$

+++ {"slideshow": {"slide_type": "skip"}}

 and the result is used as an argument of an *activation function*.

+++ {"slideshow": {"slide_type": "fragment"}}

$$y = a(o) = a\left(\sum_k w_k x_k + b\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

Together weights, bias and activation function define the behaviour of the neuron. The activation function is chosen once and remains constant. The weights and bias are the parameters that  have to be optimized during learning.

+++ {"slideshow": {"slide_type": "slide"}}

# Activation functions

+++ {"slideshow": {"slide_type": "skip"}}

The simplest activation function would be the identity, which can be also considered as no activation function

+++ {"slideshow": {"slide_type": "slide"}}

### Identity

+++ {"slideshow": {"slide_type": "fragment"}}

$$a(x)=x$$

+++ {"slideshow": {"slide_type": "skip"}}

However this means that all that the neuron, or a collection of neurons can calculate are just affine functions. This is a much too small family for any practical use.
To be able to represent more complicated functions we need to add some *non-linearity*

+++ {"slideshow": {"slide_type": "slide"}}

### Step function

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\Theta(x) = \begin{cases}
0 & x\leq0 \\
1 & x>0
\end{cases}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
def step(x):
    return np.where(x>0,1,0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
xs = np.linspace(-10,10,100)
plt.plot(xs,step(xs),'-')
plt.grid();
```

+++ {"slideshow": {"slide_type": "slide"}}

### Sigmoid

+++ {"slideshow": {"slide_type": "fragment"}}

$$s(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def s(x):
    return 1.0/(1.0+np.exp(-x))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
xs = np.linspace(-10,10,100)
plt.plot(xs,s(xs),'-')
plt.grid();
```

+++ {"slideshow": {"slide_type": "slide"}}

### Softmax

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_i = \frac{\displaystyle e^{o_i}}{\displaystyle\sum_i e^{o_i}}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sum_i y_i = 1$$

+++ {"slideshow": {"slide_type": "slide"}}

### Tanh

+++ {"slideshow": {"slide_type": "fragment"}}

$$ \tanh(x) =\frac{e^{x}-e^{-x}}{e^x+e^{-x}}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.plot(xs,np.tanh(xs),'-');
```

+++ {"slideshow": {"slide_type": "skip"}}

$$\tanh(x) = 2 s(2 x) -1 $$

+++ {"slideshow": {"slide_type": "slide"}}

### Rectified Linear Unit ( ReLU)

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\newcommand{\relu}{\operatorname{relu}}
\relu(x) = \begin{cases}
0 & x<=0 \\
x & x>0
\end{cases}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def relu(x):
    return np.where(x>0,x,0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.plot(xs,relu(xs))
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Hiden layer

```{code-cell} ipython3
SVG('../figures/hidden_layer.svg')
```

+++ {"slideshow": {"slide_type": "slide"}}

$$h_i = a^{(1)}\left(\sum_{j}w^{(1)}_{ij}x_j + b^{(1)}_i\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$h = a^{(1)}\left(w^{(1)}x + b^{(1)}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$y =  a^{(2)}\left(\sum_{j}w^{(2)}_{ij}h_j + b^{(2)}_i\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$y =  a^{(2)}\left(w^{(2)}h + b^{(2)}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ 
y = a^{(2)}\left(
w^{(2)}a^{(1)}\left(w^{(1)}x + b^{(1)}
\right)+b^{(2)}
\right)$$

+++ {"slideshow": {"slide_type": "slide"}}

# Multilayer perceptron

```{code-cell} ipython3
SVG('../figures/MLP.svg')
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$o^{(k)}_j = a^{(k)}(\sum_i w^{(k)}_{ji} o^{(k-1)}_i + b^{(k)}),\quad o^{(0)}_j=i_j$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$\newcommand{\b}[1]{\mathbf{#1}}$
$$\b{o}^{(k)} = a^{(k)}(\b{w}^{(k)} \b{o}^{(k-1)} + b^{(k)}),\quad \b{o}^{(0)}=\b{i}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import torch.nn as nn
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
net = nn.Sequential(nn.Linear(in_features=1, out_features=128), nn.ReLU(),
                   nn.Linear(in_features=128, out_features=64), nn.ReLU(), 
                   nn.Linear(in_features=64, out_features=32),  nn.ReLU(), 
                   nn.Linear(in_features=32, out_features=1))
```

+++ {"slideshow": {"slide_type": "fragment"}}

How many parameters does this network have?

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data = np.load("../data/sgd_data.npy").astype('float32')
rxs = data[:50,0]
rys = data[:50,1]
rxs_valid = data[50:75,0]
rys_valid = data[50:75,1]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
t_rxs = torch.from_numpy(rxs).view(-1,1)
t_rys = torch.from_numpy(rys).view(-1,1)
t_rxs_valid = torch.from_numpy(rxs_valid).view(-1,1)
t_rys_valid = torch.from_numpy(rys_valid).view(-1,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
xs = np.linspace(-np.pi, np.pi, 200).astype('float32')
t_ys = net(torch.from_numpy(xs).view(-1,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.scatter(rxs, rys, color='none', edgecolors='black')
plt.scatter(rxs_valid, rys_valid, color='none', edgecolors='red')
plt.plot(xs,t_ys.detach().numpy());
```

```{code-cell} ipython3
print(np.sqrt(torch.nn.functional.mse_loss(net(t_rxs), t_rys).item()),
      np.sqrt(torch.nn.functional.mse_loss(net(t_rxs_valid), t_rys_valid).item()) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
loss_f = nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
optim = torch.optim.SGD(net.parameters(),lr=0.01)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time 
errors=[]
for epoch in range(5000):
    optim.zero_grad()
    pred = net(t_rxs)
    loss = loss_f(pred, t_rys)
    loss.backward()
    with torch.no_grad():
        pred_valid = net(t_rxs_valid)
        loss_valid = loss_f(pred_valid, t_rys_valid)
    optim.step()
    if epoch%25==0:
        errors.append((epoch,loss.detach(), loss_valid))
print(loss, loss_valid)
errors=np.asarray(errors)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(errors[:,0], errors[:,1]);
plt.plot(errors[:,0], errors[:,2]);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
xs = np.linspace(-np.pi, np.pi, 200).astype('float32')
t_ys = net(torch.from_numpy(xs).view(-1,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.scatter(rxs, rys, color='none', edgecolors='black')
plt.scatter(rxs_valid, rys_valid, color='none', edgecolors='red')
plt.plot(xs,t_ys.detach().numpy());
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(np.sqrt(torch.nn.functional.mse_loss(net(t_rxs), t_rys).item()),
      np.sqrt(torch.nn.functional.mse_loss(net(t_rxs_valid), t_rys_valid).item()) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---

```
