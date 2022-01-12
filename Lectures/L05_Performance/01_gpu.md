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

+++ {"slideshow": {"slide_type": "slide"}}

# Using GPU

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
import numpy
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cifar10_train_dataset = torchvision.datasets.CIFAR10("../data/cifar10", train=True, download=True, 
                                                     transform=torchvision.transforms.ToTensor() )
cifar10_test_dataset = torchvision.datasets.CIFAR10("../data/cifar10", train=False, download=True, 
                                                    transform=torchvision.transforms.ToTensor() )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
type(cifar10_train_dataset).mro()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
im, l = cifar10_train_dataset[0]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
type(im)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
im.dtype
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
im.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
type(cifar10_train_dataset.data[0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cifar10_train_dataset.data[0].shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cifar10_train_dataset.classes
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
rows=3
cols=5
size=4
classes = cifar10_train_dataset.classes
fig, axes = plt.subplots(rows,cols, figsize=(cols*size,rows*size))
for i in range(cols*rows):
    ax = axes.ravel()[i]
    im, lbl = cifar10_train_dataset[i]
    ax.imshow(im.numpy().transpose(1,2,0))
    ax.set_title(classes[lbl], fontsize=16)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
torch.cuda.is_available()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    prop = torch.cuda.get_device_properties(i)
    print(f"{prop.total_memory/2**30:4.1f}GB", prop.multi_processor_count, prop.is_integrated)
    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
if torch.cuda.is_available():
    dev = torch.device(torch.cuda.device_count()-1)
else:
    dev = 'cpu'
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cifar10_train_loader  = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=512)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=512)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
input_size = 3*32*32
print(input_size)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model = torch.nn.Sequential(
    nn.Linear(input_size,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10)
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
model.to(dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ce_loss = torch.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

+++ {"slideshow": {"slide_type": "fragment"}}

CPU took 43s

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
for e in range(5):
    for features, labels in cifar10_train_loader:  
        features = features.view(-1,input_size).to(dev)
        labels = labels.to(dev)
        optim.zero_grad()
        pred = model(features)
        loss = ce_loss(pred,labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def data_loader_accuracy(model, loader, dev):
    with torch.no_grad():
        good_count = 0
        count = 0
        for features, labels in loader:  
            features = features.view(-1,input_size).to(dev)
            labels = labels.to(dev)
            pred = model(features)
            classes = torch.argmax(pred,1)
            good_count += torch.sum(classes==labels).item()
            count+=len(labels)
        return good_count/count    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data_loader_accuracy(model,cifar10_train_loader, dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
data_loader_accuracy(model,cifar10_test_loader, dev)
```

+++ {"slideshow": {"slide_type": "slide"}}

## All on GPU

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_train_data   = cifar10_train_dataset.data
cf10_train_labels = cifar10_train_dataset.targets
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_test_data   = cifar10_test_dataset.data
cf10_test_labels = cifar10_test_dataset.targets
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
type(cf10_train_data)
```

```{code-cell} ipython3
type(cf10_train_labels)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cf10_train_data_t = torch.from_numpy(cf10_train_data.reshape(-1,input_size)/256.0).to(
    dtype=torch.float32, device = dev)
cf10_train_labels_t = torch.tensor(cf10_train_labels,dtype=torch.long, device=dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_train_dataset = torch.utils.data.TensorDataset(cf10_train_data_t, cf10_train_labels_t)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_train_loader = torch.utils.data.DataLoader(cf10_train_dataset, batch_size=512)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cf10_test_data_t = torch.from_numpy(cf10_test_data.reshape(-1,input_size)/256.0).to(
    dtype=torch.float32, device = dev)
cf10_test_labels_t = torch.tensor(cf10_test_labels,dtype=torch.long, device=dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_test_dataset = torch.utils.data.TensorDataset(cf10_test_data_t, cf10_test_labels_t)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
cf10_test_loader = torch.utils.data.DataLoader(cf10_test_dataset, batch_size=1024)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model = torch.nn.Sequential(
    nn.Linear(input_size,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10)
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
model.to(dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ce_loss = torch.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
for e in range(5):
    for features, labels in cf10_train_loader:  
        optim.zero_grad()
        pred = model(features)
        loss = ce_loss(pred,labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def data_loader_accuracy2(model, loader, dev):
    with torch.no_grad():
        good_count = 0
        count = 0
        for features, labels in loader:  
            pred = model(features)
            classes = torch.argmax(pred,1)
            good_count += torch.sum(classes==labels).item()
            count+=len(labels)
        return good_count/count    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data_loader_accuracy2(model,cf10_train_loader, dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data_loader_accuracy2(model,cf10_test_loader, dev)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Automatic Mixed Precision (Tensor cores)

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from torch.cuda.amp import autocast
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model = torch.nn.Sequential(
    nn.Linear(input_size,4096), nn.ReLU(),
    nn.Linear(4096,4096), nn.ReLU(),
    nn.Linear(4096,2048), nn.ReLU(),
    nn.Linear(2048,1024), nn.ReLU(),
    nn.Linear(1024,512), nn.ReLU(),
    nn.Linear(512,256), nn.ReLU(),
    nn.Linear(256,10)
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model.to(dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ce_loss = torch.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.9)
```

+++ {"slideshow": {"slide_type": "slide"}}

No AMP took 26.5

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
use_amp = False
for e in range(20):
    for features, labels in cf10_train_loader:  
        optim.zero_grad()
        with autocast(enabled=use_amp):
            pred = model(features)
            loss = ce_loss(pred,labels)
        loss.backward()
        optim.step()   
    #print(e, loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
use_amp = True
for e in range(20):
    for features, labels in cf10_train_loader:  
        optim.zero_grad()
        with autocast(enabled=use_amp):
            pred = model(features)
            loss = ce_loss(pred,labels)
        loss.backward()
        optim.step()   
    #print(e, loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data_loader_accuracy2(model,cf10_train_loader, dev)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
data_loader_accuracy2(model,cf10_test_loader, dev)
```
