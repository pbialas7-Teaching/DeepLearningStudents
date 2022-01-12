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
%load_ext autoreload
%autoreload 2
```

+++ {"slideshow": {"slide_type": "slide"}}

# Adversarial Examples by Fast Gradient Sign

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from IPython.display import clear_output, display
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10.0, 8.0]

import torch as t
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
import torchvision

import os

import utils as u
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%matplotlib inline
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
if t.cuda.is_available():
    if t.cuda.device_count()>1:
        device = t.device('cuda:1')
    else:
        device = t.device('cuda')   
    batch_size = 2**10    
else:
    device = t.device('cpu')
    batch_size = 64
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
#device=t.device('cpu')
```

+++ {"slideshow": {"slide_type": "slide"}}

## MNIST

```{code-cell} ipython3
dl_train = t.utils.data.DataLoader(
    torchvision.datasets.MNIST('../../data/mnist', train=True, download=True))

dl_test  = t.utils.data.DataLoader(
    torchvision.datasets.MNIST('../../data/mnist', train=False, download=True))
```

```{code-cell} ipython3
n_samples=12000
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
mnist_train_data   = dl_train.dataset.data.to(dtype=t.float32)[:n_samples].reshape(-1,1,28,28)/255.0
mnist_train_labels = dl_train.dataset.targets[:n_samples]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
mnist_test_data   = dl_test.dataset.data.to(dtype=t.float32).reshape(-1,1,28,28)/255.0
mnist_test_labels = dl_test.dataset.targets
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
train_dataset = t.utils.data.TensorDataset(mnist_train_data, mnist_train_labels)
test_dataset = t.utils.data.TensorDataset(mnist_test_data, mnist_test_labels)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=128)
```

```{code-cell} ipython3
retrain = True
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
if os.path.isfile('mnist_home_model_conv.pt') and not retrain:
    model = t.load('mnist_home_model_conv.pt')
    pretrained = True
else:        
    model = t.nn.Sequential( 
    #28x28
    t.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3, stride=1, padding=0), #26x26
    t.nn.ReLU(),
    t.nn.MaxPool2d(2,2),#13x13
    t.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=0), #11x11
    t.nn.ReLU(),
    t.nn.MaxPool2d(2,2), #5x5 
    t.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),#3x3
    t.nn.Flatten(),
    t.nn.Linear(in_features=9*64, out_features=64),
    t.nn.ReLU(),
    t.nn.Linear(in_features=64, out_features=10)
    )
    pretrained = False
    
```

```{code-cell} ipython3

```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
model.to(device)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(model, test_dataset[:][0], test_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ce = t.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
optimizer = t.optim.Adam(model.parameters(), lr=0.0002)
```

```{code-cell} ipython3
len(train_dataset)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%%time
if not pretrained:
    err_train=[]
    err_valid=[]
    for epoch in range(10):    
        for datum in train_loader:
            optimizer.zero_grad()
            (features,target) = datum
            pred = model(features)
            loss = ce(pred, target)
            loss.backward()
            optimizer.step()

        with t.no_grad():
            vpred  = model(test_dataset[:][0])
            vloss  = ce(vpred,test_dataset[:][1])
            err_valid.append(vloss)
            pred  = model(train_dataset[:][0])
            loss  = ce(pred,train_dataset[:][1])
            err_train.append(loss)
        clear_output()
        print("epoch %d %f %f %f %f" % 
              (epoch, loss, vloss, 
                u.accuracy(model, train_dataset[:][0],  train_dataset[:][1]),
                u.accuracy(model, test_dataset[:][0],  test_dataset[:][1])
                                       )   )

    plt.plot(err_train,c='b')
    plt.plot(err_valid,c='g')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
vpred  = model(test_dataset[:][0])
vloss  = ce(vpred,test_dataset[:][1])
pred  = model(train_dataset[:][0])
loss  = ce(pred,train_dataset[:][1])
print(loss.item(), vloss.item())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(u.accuracy(model, test_dataset[:][0],  test_dataset[:][1]) )
print(u.accuracy(model, train_dataset[:][0], train_dataset[:][1]) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
if not os.path.isfile('mnist_home_model_conv.pt') or retrain:
    t.save(model,"mnist_home_model_conv.pt")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
u.model_detach(model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
idx = 478
X = train_dataset[idx:idx+1][0].clone()
L = train_dataset[idx:idx+1][1].clone()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
plt.imshow(X.data.cpu().numpy().reshape(28,28), cmap='Greys')
print(L.item())
```

```{code-cell} ipython3
pred = model(X)
ce(pred,L)
```

```{code-cell} ipython3
u.prediction(model, X)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Noise

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
eta = t.empty_like(X).normal_(0,0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
X_noisy = t.clamp(X+eta,0,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(X_noisy.detach().numpy().reshape(28,28), cmap='Greys')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.prediction(model, X_noisy)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%time
test_noisy = test_dataset[:][0]+t.empty_like(test_dataset[:][0]).normal_(0.0, 0.1)
test_noisy = test_noisy.to(device);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(model, test_noisy, test_dataset[:][1])
```

+++ {"slideshow": {"slide_type": "slide"}}

## Linearty

+++

> "Explaining and Harnessing Adversarial Examples", Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)

+++ {"slideshow": {"slide_type": "slide"}}

$$\newcommand{\b}[1]{\mathbf{#1}}$$
$$  J(\b{x}) = \mathbf{w}\cdot \mathbf{x} $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$J(\b{x}+\b\delta)= \mathbf{w}\cdot \mathbf{x} + \mathbf{w}\cdot \mathbf{\delta}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
w = np.random.uniform(-0.25, 0.25,1000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
x = np.random.uniform(0, 1,1000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
w @ x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
epsilon = 0.2
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
noise = np.random.normal(0,epsilon, 1000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
w @(x+noise) - w @x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
J_sample =  np.random.normal(0,epsilon, (5000,1000))@w
plt.hist(J_sample, bins=100, histtype='step');
```

+++ {"slideshow": {"slide_type": "slide"}}

$$\newcommand{\sign}{\operatorname{sign}}$$
$$\eta = \epsilon\sign \b{w}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\b{w}\cdot\b{\eta} = \epsilon\sum_{i=1}^N |w_i|  \sim \epsilon N $$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
adv_noise = epsilon*np.sign(w)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
w @(x+adv_noise) - w @x
```

+++ {"slideshow": {"slide_type": "slide"}}

### But Neural Networks are highly non-linear? Right?

+++ {"slideshow": {"slide_type": "fragment"}}

### Wrong! Neural Networks are designed to be quite linear.

+++ {"slideshow": {"slide_type": "slide"}}

### Fast Gradient Sign

+++ {"slideshow": {"slide_type": "slide"}}

$$J(\b{X}+\b{\delta}) \approx J(\b{X})+\nabla_{\b{X}} J(\b{X})\cdot\b{\delta}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\newcommand{\grad}{\operatorname{grad}}$$
$$\nabla_{\b{X}} J(\b{X})=\grad_{\b{X}} J(\b{X})
\equiv\frac{\partial J(\b{X})}{\partial {X_i}},
\quad i=1,\ldots,N$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\delta = \epsilon \sign \nabla_{\b{X}} J(\b{X})$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
idx = 899
X = train_dataset[idx:idx+1][0].clone()
L = train_dataset[idx:idx+1][1]
print(L.item())
```

```{code-cell} ipython3
X.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(X.data.cpu().numpy().reshape(28,28), cmap='Greys')
plt.text(22,3,'%d' % (L,), fontsize=32);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def FGS(model,orig, label, eps):
    orig.requires_grad_(True);
    if orig.grad is not None:
        orig.grad.zero_()
    loss = ce(model(orig), label)
    loss.backward() 
    XG = orig.grad
    eta = eps*XG.sign()
    orig.requires_grad_(False)
    return (orig+eta)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
AdvX = FGS(model, X,L, 0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(AdvX.data.cpu().numpy().reshape(28,28), cmap='Greys')
plt.text(22,3,'%d' % (L,), fontsize=32);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.prediction(model, AdvX)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
t.save(X, 'real_5_conv.pt')
t.save(AdvX,'fake_2_conv.pt')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%time
test_adv = t.stack([FGS(model,x.view(-1,1,28,28),l.view(1),0.1) for x,l in zip(test_dataset[:][0], test_dataset[:][1])],dim=0)
test_adv=test_adv.reshape(-1,1,28,28).to(device);
```

```{code-cell} ipython3
test_adv.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(model, test_adv, test_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.imshow(test_adv[90].data.cpu().numpy().reshape(28,28), cmap='Greys')
u.prediction(model,test_adv[99:100])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
eps =  np.linspace(-0.1,0.1, 100)

X.requires_grad=True
X.grad.data.zero_()
out = model(X)
loss = ce(out,L)
loss.backward()
grad_X = X.grad.data
```

```{code-cell} ipython3
X.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ls = []
eta = grad_X.sign()
#eta = t.ones(28*28)
#print(eta@grad_X);
for e in eps:
    out = model(X.detach()+e*eta)
    loss = ce(out,L)
    ls.append(loss.item())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(eps, ls);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
outs =[]
eta = grad_X.sign()
for e in eps:
    out = model(X.detach()+e*eta)
    outs.append(out)

outs = t.stack(outs,0)
outs.squeeze_();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for i in range(10):
    plt.plot(eps,outs.numpy()[:,i], label='%d' % (i,))
plt.axvline(0)   
plt.axvline(0.05)
plt.legend()    
```

+++ {"slideshow": {"slide_type": "slide"}}

### But those examples are targeted for specific network?

+++ {"slideshow": {"slide_type": "fragment"}}

### Not really ...

+++ {"slideshow": {"slide_type": "slide"}}

## Generalisation

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
if os.path.isfile('another_mnist.pt'):
    another_model = t.load('another_mnist.pt')
    another_pretrained = True
else:
    another_model = u.make_model(64,0.2)

    another_model.apply(u.init_layer)
    another_pretrained = False
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
another_optimizer=t.optim.Adam(another_model.parameters(), lr=0.0002, betas=[0.5, 0.999])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
if not another_pretrained:
    err_train=[]
    err_valid=[]
    for epoch in range(20):    
        for datum in train_loader:
            another_optimizer.zero_grad()
            (features,target) = datum
            pred = another_model(features)
            loss = ce(pred, target)
            loss.backward()
            another_optimizer.step()

        with t.no_grad():
            vpred  = another_model(test_dataset[:][0])
            vloss  = ce(vpred,test_dataset[:][1])
            err_valid.append(vloss)
            pred  = another_model(train_dataset[:][0])
            loss  = ce(pred,train_dataset[:][1])
            err_train.append(loss)
        clear_output()
        print("epoch %d %f %f %f" % (epoch, loss, vloss,u.accuracy(another_model, test_dataset[:][0],  test_dataset[:][1])))   

    plt.plot(err_train,c='b')
    plt.plot(err_valid,c='g')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
if not os.path.isfile('another_mnist.pt'):
    t.save(another_model, 'another_mnist.pt')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
u.accuracy(another_model, test_dataset[:][0], test_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(another_model, test_adv, test_dataset[:][1])
```

+++ {"slideshow": {"slide_type": "slide"}}

## Adversarial trening

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%time
train_adv = t.stack([FGS(model,x,l,0.05) for x,l in zip(train_dataset[:][0], 
                                                                train_dataset[:][1])],dim=0)
train_adv = train_adv.to(device);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
adv_dataset = t.utils.data.TensorDataset(train_adv,train_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
combined_dataset = t.utils.data.ConcatDataset((train_dataset, adv_dataset))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
combined_loader = t.utils.data.DataLoader(combined_dataset, batch_size=128, shuffle=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.model_atach(model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
err_train=[]
err_valid=[]
for epoch in range(1,21):    
    for datum in combined_loader:
        optimizer.zero_grad()
        (features,target) = datum
        pred = model(features)
        loss = ce(pred, target)
        loss.backward()
        optimizer.step()

    with t.no_grad():
        vpred  = model(test_dataset[:][0])
        vloss  = ce(vpred,test_dataset[:][1])
        err_valid.append(vloss)
        pred  = model(train_dataset[:][0])
        loss  = ce(pred,train_dataset[:][1])
        err_train.append(loss)
    clear_output()
    print("epoch %d %f %f %f" % (epoch, loss, vloss,u.accuracy(another_model, test_dataset[:][0],  test_dataset[:][1])))   
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(err_train,c='b')
plt.plot(err_valid,c='g')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
u.accuracy(model, test_dataset[:][0], test_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(model, test_adv, test_dataset[:][1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
%time
test_adv = t.stack([FGS(model,x,l,0.05) for x,l in zip(test_dataset[:][0], test_dataset[:][1])],dim=0)
test_adv  =test_adv.to(device);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.accuracy(model, test_adv, test_dataset[:][1])
```

+++ {"slideshow": {"slide_type": "slide"}}

## Targeted Fast Gradient Sign (T-FGS)

+++

> "Adversarial examples in the physical world", Alexey Kurakin, Ian Goodfellow, Samy Bengio [arXiv:1607.02533](https://arxiv.org/abs/1607.02533)

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model = t.load('mnist_home_model.pt')
u.model_detach(model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
idx = 89
Y = train_dataset[idx:idx+1][0].clone()
L = train_dataset[idx:idx+1][1]
print(L.item())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(Y.numpy().reshape(28,28), cmap='Greys')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.prediction(model, Y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
T = t.LongTensor([3])
```

+++ {"slideshow": {"slide_type": "fragment"}}

$$ J(X, T) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ X-\epsilon \nabla_X J(X, T) $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def T_FGS(model,orig, label, target, eps):
    orig.requires_grad_(True);
    if orig.grad is not None:
        orig.grad.zero_()
    loss = ce(model(orig.reshape(1,-1)), target.view(1))
    loss.backward() 
    XG = orig.grad
    eta = eps*XG.sign()
    orig.requires_grad_(False)
    return t.clamp(orig-eta, 0,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
Y_adv = T_FGS(model, Y,L,T,0.15)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(Y_adv.numpy().reshape(28,28), cmap='Greys')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.prediction(model,Y_adv)
```

+++ {"slideshow": {"slide_type": "slide"}}

$$ X_0=X$$
$$ X_{i+1}-\epsilon \nabla_X J(X_i, T) $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def T_FGS_I(model,orig, label, target, eps, n_iter=3):
    i_eps = eps/n_iter
    adv = orig
    for i in range(n_iter):
        adv=T_FGS(model, adv, label, target, i_eps)
    return adv    
        
    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
Y_adv = T_FGS_I(model, Y,L,T,0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(Y_adv.numpy().reshape(28,28), cmap='Greys');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u.prediction(model,Y_adv)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
