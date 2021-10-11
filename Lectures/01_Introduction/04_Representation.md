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

# Representation/features selection

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

+++ {"slideshow": {"slide_type": "skip"}}

Closely tied with the concept of learning is a concept of *representation*.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
features = np.load("../data/ring_features.npy")
labels = np.load("../data/ring_labels.npy")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
r1=0.3
r2 = np.sqrt(r1**2 + 1/(2*np.pi))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
i_red  = labels==0
i_blue = labels==1

red  = features[i_red]
blue = features[i_blue]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
from matplotlib.patches import Circle
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(red[:,0], red[:,1],'.r')
ax.plot(blue[:,0], blue[:,1],'.b');
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y',  fontsize=14)
ax.add_patch(Circle((0,0),r1,color='k', fill=False))
ax.add_patch(Circle((0,0),r2,color='k', fill=False));
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Polar representation

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def polar(xy):
    phi = np.arctan2(xy[:,0], xy[:,1])
    r   = np.linalg.norm(xy, axis=1)
    return np.stack((phi,r), axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots(figsize=(8,8))
pol = polar(features)
ax.plot(pol[i_red,0],pol[i_red,1],'.r')
ax.plot(pol[i_blue,0],pol[i_blue,1],'.b');
ax.axhline(r1, color='k')
ax.axhline(r2, color='k')
ax.set_xlabel('$\\phi$', fontsize=14)
ax.set_ylabel('r',  fontsize=14);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots(figsize=(10,8))
ax.hist(pol[i_red,1], bins=50,color='r');
ax.hist(pol[i_blue,1], bins=50,color='b');
ax.set_xlabel('r', fontsize=14);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
if torch.cuda.is_available():
    device = f"cuda:{torch.cuda.device_count()-1}"
else:
    device = 'cpu'
print(device)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
device='cpu'
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
n_epochs=10000
batch_size = 250 
lr = 0.01  
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
size = 16
model = nn.Sequential(nn.Linear(in_features=2, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size//2),
                      nn.ReLU(), 
                      nn.Linear(in_features=size//2, out_features=1), 
                      nn.Sigmoid())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
model.to(device=device)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
t_data = torch.from_numpy(features).to(device=device, dtype=torch.float32)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
t_labels = torch.from_numpy(labels).to(device=device, dtype=torch.float32).view(-1,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
dataset = torch.utils.data.TensorDataset(t_data, t_labels)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
train_data, test_data = torch.utils.data.random_split(dataset,(750,250))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
loss_f = nn.BCELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
sgd = torch.optim.SGD(model.parameters(), lr = lr , momentum=0.2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
for epoch in range(n_epochs):
    for batch in train_loader:
        sgd.zero_grad()
        f,l = batch
        pred = model(f)
        loss =   loss_f(pred,l) 
        loss.backward()
        sgd.step()
print(loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
pred_valid = model(test_data[:][0])
pred_class = (pred_valid>0.5).long()
torch.sum(pred_class == test_data[:][1].long()).item()/len(test_data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
polar_model = nn.Sequential(nn.Linear(in_features=1, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size//2),
                      nn.ReLU(), 
                      nn.Linear(in_features=size//2, out_features=1), 
                      nn.Sigmoid())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
t_data_polar = torch.from_numpy(polar(features.astype('float32')))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
dataset_polar = torch.utils.data.TensorDataset(t_data_polar[:,1:], t_labels)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
train_data_polar, test_data_polar = torch.utils.data.random_split(dataset_polar,(750,250))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
train_loader_polar = torch.utils.data.DataLoader(train_data_polar, batch_size=batch_size)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
loss_f = nn.BCELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
sgd_polar = torch.optim.SGD(polar_model.parameters(), lr = lr , momentum=0.2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
for epoch in range(n_epochs):
    for batch in train_loader_polar:
        sgd_polar.zero_grad()
        f,l = batch
        pred = polar_model(f)
        loss =   loss_f(pred,l) 
        loss.backward()
        sgd_polar.step()
print(loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
pred_valid = polar_model(test_data_polar[:][0])
pred_class = (pred_valid>0.5).long()
torch.sum(pred_class == test_data_polar[:][1].long()).item()/len(test_data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
xs = torch.linspace(0,1,250).reshape(-1,1)
res = polar_model(xs).detach().numpy().ravel()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
plt.plot(xs,res);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Learning (forced)  representation

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
encoder =  nn.Sequential(nn.Linear(in_features=2, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size//2),
                      nn.ReLU(), 
                      nn.Linear(in_features=size//2, out_features=1)
                        )
decoder =nn.Sequential( nn.Linear(in_features=1, out_features=size),
                      nn.ReLU(), 
                      nn.Linear(in_features=size, out_features=size//2),
                      nn.ReLU(), 
                      nn.Linear(in_features=size//2, out_features=1), nn.Sigmoid()
                )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
rep_model = nn.Sequential(encoder, decoder)
                      
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
loss_f = nn.BCELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
sgd = torch.optim.SGD(rep_model.parameters(), lr = lr , momentum=0.2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
for epoch in range(2*n_epochs):
    for batch in train_loader:
        sgd.zero_grad()
        f,l = batch
        pred = rep_model(f)
        loss =   loss_f(pred,l) 
        loss.backward()
        sgd.step()
print(loss)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
pred_valid = rep_model(test_data[:][0])
pred_class = (pred_valid>0.5).long()
torch.sum(pred_class == test_data[:][1].long()).item()/len(test_data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
res = encoder(t_data)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
plt.hist(res.detach().numpy().ravel()[i_red], bins=50, color = 'red');
plt.hist(res.detach().numpy().ravel()[i_blue], bins=50, color ='blue');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(pol[i_red,1],res.detach().numpy().ravel()[i_red] , s=5,color='red');
ax.scatter(pol[i_blue,1],res.detach().numpy().ravel() [i_blue], s=5, color='blue');
ax.axvline(r1, color ='black')
ax.axvline(r2, color='black');
ax.set_xlabel('r', fontsize=14)
ax.set_ylabel('encoder output', fontsize=14);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
t_rs= torch.linspace(-4,5,500).view(-1,1).to(device=device)
res_dec = decoder(t_rs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(t_rs.cpu().numpy().ravel(), res_dec.detach().cpu().numpy().ravel())
ax.set_xlabel('encoder output', fontsize=14)
ax.set_ylabel('decoder output', fontsize=14)
```

```{code-cell} ipython3

```
