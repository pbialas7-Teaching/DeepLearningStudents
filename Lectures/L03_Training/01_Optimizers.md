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
%load_ext autoreload
%autoreload 2
from IPython.display import Image
```

```{code-cell} ipython3
Image('../img/boromir.png')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import torch, torchvision
import utils
```

+++ {"slideshow": {"slide_type": "slide"}}

# Optimizers

+++

[An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)

+++ {"slideshow": {"slide_type": "slide"}}

## Gradient Descent

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
\theta_{t+1}& = \theta^{t}-\eta\nabla_\theta L(\theta_t)
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data = np.load("../data/sgd_data.npy").astype('float32')
```

```{code-cell} ipython3
sin_example = utils.SinFitExample(data)
```

```{code-cell} ipython3
sin_example.display_data();
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
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
t_rxs = torch.from_numpy(data[:400,0])
t_rys = torch.from_numpy(data[:400,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
loss_f = torch.nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
rdataset = torch.utils.data.TensorDataset(t_rxs, t_rys)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
onebatchloader = torch.utils.data.DataLoader(rdataset, batch_size=len(rdataset), shuffle=False);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.SGD([p], lr=0.2)
sin_example.run_example(p, gd, onebatchloader);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
batch_data_loader = torch.utils.data.DataLoader(rdataset, batch_size=50, shuffle=True)
```

```{code-cell} ipython3
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.SGD([p], lr=0.2)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

## "Ravine"

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
rav_par = np.asarray([1.0,10.0]).astype('float32')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ravine_example = utils.RavineExample(rav_par)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.SGD([p], lr=0.02)
ravine_example.run_example(p, gd,100,1.0 );
```

+++ {"slideshow": {"slide_type": "slide"}}

## Gradient Descent with Momentum

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
v_{t+1}& = \mu v_{t} + (1-\beta)\nabla_\theta L(\theta_t)\\
\theta_{t+1}& = \theta_{t}-\eta v_{t+1}
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.SGD([p], lr=0.021, momentum=0.9)
ravine_example.run_example(p, gd,100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.SGD([p], lr=0.01, momentum=0.9)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

$$\begin{align}
v_{t+1}& = \mu v_{t} + (1-\beta)\nabla_\theta L(\theta_t)\\
\theta_{t+1}& = \theta_{t}-\eta v_{t+1}
\end{align}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_{t+1} = \mu v_{t} + (1-\beta)g_t$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_1 = (1-\beta)g_0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_2 = \mu (1-\beta)g_0+(1-\beta) g_1$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_3 = \mu\left(\mu (1-\beta)g_0+(1-\beta) g_1\right)+(1-\beta)g_2$$

+++ {"slideshow": {"slide_type": "slide"}}

$$v_3 = \mu^2 (1-\beta)g_0+\mu (1-\beta) g_1
+(1-\beta)g_2$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$v_t = (1-\beta)\sum_{i=1}^t \mu^{i-1}g_{t-i}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ns = np.arange(0,100)
for mu in [0.9, 0.7,0.5, 0.25]:
    plt.plot(ns,mu**ns,'.', label="%4.2f" % (mu,))
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}}

## Nesterov Accelerated Gradient Descent

```{code-cell} ipython3
Image('../figures/nesterov.jpg')
```

+++ {"slideshow": {"slide_type": "slide"}}

$$\begin{align}
v_{t+1}& = \mu v_{t} + \nabla_\theta L(\theta_t-\eta \mu v_t)\\
\theta_{t+1}& = \theta_{t}-\eta v_{t+1}
\end{align}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$r_t = \theta_t-\eta\mu v_t$$
$$\theta_t = r_t+\eta\mu v_t$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
v_{t+1}& = \mu v_{t} + \nabla_r L(r)\\
r_{t+1}& = r_{t}-\eta\left(\nabla_r L(r) +\mu v^{t+1}\right)
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.SGD([p], lr=0.01, momentum=0.9, nesterov=True)
ravine_example.run_example(p, gd,100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.SGD([p], lr=0.01, momentum=0.9, nesterov=True)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Adaptive gradient: Adagrad

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
v_{t+1}& =\nabla_\theta L(\theta^t)\\
G_{t+1}&=G_{t}+\left(\nabla_\theta L(\theta_t)\right)^2\\
\theta_{t+1}& = \theta_{t}-\frac{\eta}{\sqrt{G_{t+1}+\epsilon}} v_{t+1}
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.Adagrad([p], lr=2.0)
ravine_example.run_example(p, gd,100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.4,-0.4])
p.requires_grad_(True)
gd = torch.optim.Adagrad([p], lr=.1)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

## RMSProp

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
v_{t+1}& =\nabla_\theta L(\theta_t)\\
E[g^2]_{t+1}&=\gamma E[g^2]_t+(1-\gamma)\left(\nabla_\theta L(\theta_t)\right)^2\\
\theta_{t+1}& = \theta_{t}-\frac{\eta}{\sqrt{E[g^2]_{t+1}+\epsilon}} v_{t+1}
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.RMSprop([p], lr=1.0)
ravine_example.run_example(p, gd,100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.RMSprop([p], lr=0.1)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Adadelta

+++ {"slideshow": {"slide_type": "skip"}}

Units do not match!

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{align}
v^{t+1}& =\nabla_\theta L(\theta^t)\\
E[g^2]_{t+1}&=\gamma E[g^2]_t+(1-\gamma)\left(\nabla_\theta L(\theta^t)\right)^2\\
E[\Delta\theta^2]_t & = \gamma E[\Delta\theta]_{t-1} +(1-\gamma)\Delta\theta^2_t\\
\theta_{t+1}& = \theta_{t}-\frac{\eta E[\Delta\theta^2]_t}{\sqrt{E[g^2]_{t+1}+\epsilon}} v_{t+1}
\end{align}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.Adadelta([p], lr=4.0)
ravine_example.run_example(p, gd,2100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.Adadelta([p], lr=4.0)
sin_example.run_example(p, gd, batch_data_loader);
```

+++ {"slideshow": {"slide_type": "slide"}}

### Adam: Adaptive Momentum Estimation

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{split}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g^2_t \\
\end{split}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta_{t+1} = \theta_t -\frac{\eta}{\sqrt{v}+\epsilon}m_t $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([-8, 3])
p.requires_grad_(True);
gd = torch.optim.Adam([p], lr=0.1)
ravine_example.run_example(p, gd,2100,1.0 );
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = torch.FloatTensor([3.2,-0.4])
p.requires_grad_(True)
gd = torch.optim.Adam([p], lr=0.1)
sin_example.run_example(p, gd, batch_data_loader);
```

```{code-cell} ipython3

```
