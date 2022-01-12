---
jupytext:
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
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

```{code-cell} ipython3
if torch.cuda.is_available():
    batch_size=256
    if torch.cuda.device_count() > 1:
        device='cuda:1'
    else:
        device='cuda:0'
else:
    batch_size=64
    device='cpu'
```

```{code-cell} ipython3
device
```

+++ {"slideshow": {"slide_type": "slide"}}

## MNIST

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
dl_train = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/mnist', train=True, download=True))

dl_test  = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/mnist', train=False, download=True))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
train_data   = dl_train.dataset.data.to(dtype=torch.float32)
train_labels = dl_train.dataset.targets
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig_mnist, ax = plt.subplots(1,8, figsize=(8*4,4))
for i in range(8):
    ax[i].imshow(train_data[i].numpy(), cmap='Greys');
```

```{code-cell} ipython3
train_labels[0:8]
```

+++ {"slideshow": {"slide_type": "slide"}}

## Standarisation/Normalisation

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
train_dataset = torch.utils.data.TensorDataset( 
    (train_data/128.0-1.0).view(-1,28*28).to(device=device), 
    train_labels.to(device=device))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=128, 
                                           shuffle=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
test_data   = dl_test.dataset.data.to(dtype=torch.float32)
test_labels = dl_test.dataset.targets
test_dataset = torch.utils.data.TensorDataset(
    (test_data/128.8-1.0).view(-1,28*28).to(device=device), test_labels.to(device=device))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.Linear(1200,1200), nn.ReLU(),
    nn.Linear(1200,1200), nn.ReLU(),
    nn.Linear(1200,10),
)
model.to(device=device)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
```

```{code-cell} ipython3
loss_f = nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
errors = []
batches = 0
epochs = 0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
model.train()
for e in range(5):
    for d in train_loader:        
        optim.zero_grad()
        features, labels = d
        pred = model(features)
        loss = loss_f(pred, labels)
        errors.append(loss.item())
        loss.backward()
        optim.step()
        batches += 1
    epochs += 1   
print(loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(np.linspace(0,epochs, batches),errors)
```

```{code-cell} ipython3
model.eval()
with torch.no_grad():
    pred = torch.softmax(model(test_dataset[:][0]),1)
    ac = torch.sum(torch.argmax(pred,1)==test_dataset[:][1]).to(dtype=torch.float32)/len(test_dataset)
ac    
```

+++ {"slideshow": {"slide_type": "slide"}}

## Batch normalisation

+++ {"slideshow": {"slide_type": "fragment"}}

$$\hat{x}_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j},
\quad \mu_j=\frac{1}{N_{batch}}\sum_{i\in batch} x_{ij},
\quad \sigma_j = \sqrt{\frac{1}{N_{batch}}\sum_{i\in batch}(x_{ij}-\mu_j)^2}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_{ik}=\gamma_j \hat{x}_{ij}+\beta_j $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model_bnorm = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.BatchNorm1d(1200),
    nn.Linear(1200,1200), nn.ReLU(),
    nn.BatchNorm1d(1200),
    nn.Linear(1200,1200), nn.ReLU(),
    nn.BatchNorm1d(1200),
    nn.Linear(1200,10),
    nn.BatchNorm1d(10)
)
model_bnorm.to(device=device)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim_bnorm = torch.optim.SGD(model_bnorm.parameters(), lr=0.1, momentum=0.6)
```

```{code-cell} ipython3
loss_f = nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
errors_bn = []
batches_bn = 0
epochs_bn = 0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
model_bnorm.train()
for e in range(5):
    for d in train_loader:        
        optim_bnorm.zero_grad()
        features, labels = d
        pred = model_bnorm(features)
        loss = loss_f(pred, labels)
        errors_bn.append(loss.item())
        loss.backward()
        optim_bnorm.step()
        batches_bn += 1
    epochs_bn += 1   
print(loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(np.linspace(0,epochs, batches),errors)
plt.plot(np.linspace(0,epochs_bn, batches_bn),errors_bn)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model_bnorm.eval()
with torch.no_grad():
    pred = torch.softmax(model_bnorm(train_dataset[:][0]),1)
    ac = torch.sum(torch.argmax(pred,1)==train_dataset[:][1]).to(dtype=torch.float32)/len(train_dataset)
ac  
```

```{code-cell} ipython3
model_bnorm.eval()
with torch.no_grad():
    pred = torch.softmax(model_bnorm(test_dataset[:][0]),1)
    ac_bnorm = torch.sum(torch.argmax(pred,1)==test_dataset[:][1]).to(dtype=torch.float32)/len(test_dataset)
ac_bnorm    
```

```{code-cell} ipython3
(1-ac)
```

```{code-cell} ipython3
(1-ac_bnorm)
```

```{code-cell} ipython3
(1-ac_bnorm)/(1-ac)
```

```{code-cell} ipython3

```
