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

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import torch
import torchvision
from torch.cuda.amp import autocast
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

```{code-cell} ipython3
import scrapbook as sb
```

```{code-cell} ipython3
import sys
sys.path.append('../../modules')
```

```{code-cell} ipython3
import mnist
```

```{code-cell} ipython3
digits = mnist.MNIST('../data')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
n_samples = 50000
```

```{code-cell} ipython3
if torch.cuda.is_available():
    dev = torch.device(torch.cuda.device_count()-1)
else:
    dev = 'cpu'
```

```{code-cell} ipython3
:tags: [parameters]

device = dev
optimizer = 'SGD'
optimizer_parameters = {'lr': 0.01}
scheduler='none'
scheduler_parameters={}
batch_size = 512
n_epochs = 16
```

```{code-cell} ipython3
sb.glue('optimizer', optimizer)
sb.glue('optimizer_parameters', optimizer_parameters)
sb.glue('scheduler', scheduler)
sb.glue('scheduler_parameters',scheduler_parameters)
sb.glue('batch_size', batch_size)
sb.glue('n_epochs', n_epochs)
```

```{code-cell} ipython3
dataset = digits.flat_train_dataset(n_samples, device='cuda:1')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, (40000,10000))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                           batch_size = 128, 
                                           shuffle = True)
```

+++ {"slideshow": {"slide_type": "slide"}}

## The model

+++ {"slideshow": {"slide_type": "skip"}}

We will use a fully  four  fully connected layers with `ReLU` activation layers in between as our model and `softmax` as the last layer.  The model can be easily constructed using the PyTorch `nn.Sequential` class:

```{code-cell} ipython3
model = torch.nn.Sequential(
    nn.Linear(28*28,1152), nn.ReLU(),
    nn.Linear(1152,768), nn.ReLU(),
    nn.Linear(768,320), nn.ReLU(),
    nn.Linear(320,10)
).to(device=device)
```

```{code-cell} ipython3
def accuracy(pred, labels):
    return torch.sum(torch.argmax(pred,axis = 1)==labels).to(dtype=torch.float32).item()/len(labels)
```

```{code-cell} ipython3
def model_accuracy(model, dataset):
    features, labels = dataset[:]
    with torch.no_grad():
        pred = model(features)
    return accuracy(pred, labels)
```

+++ {"slideshow": {"slide_type": "skip"}}

Before we start training we need the loss function:

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
ce_loss = torch.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def kaiming_init_uniform(sigma=1):
    def init(layer): 
        if isinstance(layer,torch.nn.modules.linear.Linear):
            fan_in = layer.weight.size(1)
            s  = np.sqrt(6/fan_in)    
            torch.nn.init.uniform_(layer.weight,-sigma*s,sigma*s)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    return init  
```

```{code-cell} ipython3
model.apply(kaiming_init_uniform())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = getattr(torch.optim,optimizer)(model.parameters(), **optimizer_parameters)
if scheduler!='none':
    scheduler = torch.optim.lr_scheduler.StepLR(optim,**scheduler_parameters)
```

```{code-cell} ipython3
:tags: [parameters]

use_amp=False
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
for e in range(n_epochs):
    for features, labels in train_loader:        
        optim.zero_grad()
        with autocast(enabled=use_amp):
            pred = model(features)
            loss = ce_loss(pred, labels)
        loss.backward()
        optim.step()
        if scheduler!='none':
            scheduler.step()   
    print(e, loss.item())        
```

```{code-cell} ipython3
ac_train = model_accuracy(model, train_dataset)
print(ac_train)
sb.glue("ac_train", ac_train)
```

```{code-cell} ipython3
ac_test = model_accuracy(model, validation_dataset)
print(ac_test)
sb.glue("ac_test", ac_test)
```
