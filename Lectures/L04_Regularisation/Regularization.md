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
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
```

> Regularization is any modification we make to a learning algorithm that is intended  to reduce its generalization error but not its training error. 

```{code-cell} ipython3
np.random.seed(2346)
```

```{code-cell} ipython3
xs = np.linspace(-np.pi, np.pi,500)
```

```{code-cell} ipython3
rxs = np.random.uniform(-np.pi, np.pi,500)
rys = np.cos(rxs)+np.random.normal(0,.1,500)
```

```{code-cell} ipython3
data =np.stack((rxs, rys),1)
```

```{code-cell} ipython3
train_data_indices = np.random.choice(len(data),48,replace=False)
train_data = data[train_data_indices]
```

```{code-cell} ipython3
validation_data_indices = np.random.choice(len(data),24,replace=False)
validation_data = data[validation_data_indices]
```

```{code-cell} ipython3
plt.scatter(train_data[:,0], train_data[:,1],s=50, edgecolors='black', facecolors='none');
plt.scatter(validation_data[:,0], validation_data[:,1], s=50,edgecolors='red', facecolors='none');
plt.plot(xs, np.cos(xs));
```

```{code-cell} ipython3
train_data_t = torch.from_numpy(train_data).to(dtype=torch.float32)
validation_data_t = torch.from_numpy(validation_data).to(dtype=torch.float32)
```

```{code-cell} ipython3
model = nn.Sequential(
    nn.Linear(1,128),nn.ReLU(),
    nn.Linear(128,64),nn.ReLU(),
    nn.Linear(64,1)
)
```

```{code-cell} ipython3
loss_f = nn.MSELoss()
```

```{code-cell} ipython3
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

```{code-cell} ipython3
%%time
errs=[]
n_epochs =8000
for e in range(n_epochs):
    opt.zero_grad()
    pred = model(train_data_t[:,0].view(-1,1))
    loss = loss_f(pred, train_data_t[:,1].view(-1,1))
    loss.backward()
    opt.step()
    with torch.no_grad():
        validation_pred = model(validation_data_t[:,0].view(-1,1))
        validation_loss = loss_f(validation_pred, validation_data_t[:,1].view(-1,1))
    errs.append((loss.item(), validation_loss.item()))    
print(loss.item())   
errors= np.asanyarray(errs)
```

```{code-cell} ipython3
errors.min(0)
```

```{code-cell} ipython3
skip = 100
plt.plot(errors[skip:,0])
plt.plot(errors[skip:,1]);
plt.grid();
```

# Regularization

+++

## Early stopping

```{code-cell} ipython3
model = nn.Sequential(
    nn.Linear(1,128),nn.ReLU(),
    nn.Linear(128,64),nn.ReLU(),
    nn.Linear(64,1)
)
```

```{code-cell} ipython3
loss_f = nn.MSELoss()
```

```{code-cell} ipython3
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

```{code-cell} ipython3
min_model=None
min_validation_loss = 10000.0
min_e = None
```

```{code-cell} ipython3
%%time
errs=[]
n_epochs =8000
for e in range(n_epochs):
    opt.zero_grad()
    pred = model(train_data_t[:,0].view(-1,1))
    loss = loss_f(pred, train_data_t[:,1].view(-1,1))
    loss.backward()
    opt.step()
    with torch.no_grad():
        validation_pred = model(validation_data_t[:,0].view(-1,1))
        validation_loss = loss_f(validation_pred, validation_data_t[:,1].view(-1,1))
        vloss = validation_loss.item()
        if  vloss  <min_validation_loss:
            min_validation_loss = vloss
            e_min=e
            min_model = copy.deepcopy(model)
    errs.append((loss.item(), validation_loss.item()))    
print(loss.item())   
errors= np.asanyarray(errs)
```

```{code-cell} ipython3
errors.min(0)
```

```{code-cell} ipython3
skip = 100
plt.plot(errors[skip:,0])
plt.plot(errors[skip:,1]);
plt.grid();
```

```{code-cell} ipython3
xs_t = torch.linspace(-np.pi, np.pi,200)
```

```{code-cell} ipython3
with torch.no_grad():
    ys_t = model(xs_t.view(-1,1)).view(-1)
    ys_min_t = min_model(xs_t.view(-1,1)).view(-1)
```

```{code-cell} ipython3
plt.plot(xs_t.numpy(), ys_t.numpy());
plt.plot(xs_t.numpy(), ys_min_t.numpy(), c='red', alpha=0.75);
plt.scatter(train_data[:,0], train_data[:,1],s=50, edgecolors='black', facecolors='none');
plt.scatter(validation_data[:,0], validation_data[:,1], s=50,edgecolors='red', facecolors='none');
```

```{code-cell} ipython3
class Experiment:
    def __init__(self):
        self.min_model=None
        self.min_validation_loss = 10000.0
        self.min_e = None
        

    def run(self, model,optim,n_epochs,  train, validation, loss_f = nn.MSELoss() ):
        errs = []
        for e in range(n_epochs):
            optim.zero_grad()
            pred = model(train[:,0].view(-1,1))
            loss = loss_f(pred, train[:,1].view(-1,1))
            loss.backward()
            optim.step()
            with torch.no_grad():
                model.eval()
                validation_pred = model(validation[:,0].view(-1,1))
                validation_loss = loss_f(validation_pred, validation[:,1].view(-1,1))
                vloss = validation_loss.item()
            if  vloss  <self.min_validation_loss:
                self.min_validation_loss = vloss
                self.min_e = e
                self.min_model = copy.deepcopy(model)
            model.train()    
            errs.append((loss.item(), validation_loss.item()))    
        errors= np.asanyarray(errs)
        return errors, min_model
```

## Weight decay

```{code-cell} ipython3
model_wd = nn.Sequential(
    nn.Linear(1,128),nn.ReLU(),
    nn.Linear(128,64),nn.ReLU(),
    nn.Linear(64,1)
)
```

```{code-cell} ipython3
loss_f = nn.MSELoss()
```

```{code-cell} ipython3
opt_wd = torch.optim.SGD(model_wd.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)
```

```{code-cell} ipython3
wd_experiment = Experiment()
```

```{code-cell} ipython3
%%time
errors_wd, min_model_wd = wd_experiment.run(model_wd, opt_wd,8000, train_data_t, validation_data_t)
```

```{code-cell} ipython3
skip = 100
plt.plot(errors_wd[skip:,0])
plt.plot(errors_wd[skip:,1]);
plt.grid();
```

```{code-cell} ipython3
errors_wd.min(0)
```

```{code-cell} ipython3
xs_t = torch.linspace(-np.pi, np.pi,200)
```

```{code-cell} ipython3
with torch.no_grad():
    ys_t = model_wd(xs_t.view(-1,1)).view(-1)
    ys_min_t = min_model_wd(xs_t.view(-1,1)).view(-1)
```

```{code-cell} ipython3
plt.plot(xs_t.numpy(), ys_t.numpy());
plt.plot(xs_t.numpy(), ys_min_t.numpy());
plt.scatter(train_data[:,0], train_data[:,1],s=50, edgecolors='black', facecolors='none');
plt.scatter(validation_data[:,0], validation_data[:,1], s=50,edgecolors='red', facecolors='none');
```

## Bagging 

```{code-cell} ipython3
def random_choice(data):
    perm = np.random.choice(len(data), len(data),replace=True)
    return data[perm]
```

```{code-cell} ipython3
n_bags = 10
```

```{code-cell} ipython3
bag = [random_choice(train_data_t) for  i in range(n_bags)]
```

```{code-cell} ipython3
models_bag  = [nn.Sequential(
    nn.Linear(1,128),nn.ReLU(),
    nn.Linear(128,64),nn.ReLU(),
    nn.Linear(64,1))  for i in range(n_bags) ]
```

```{code-cell} ipython3
dp_experiment = Experiment()
```

```{code-cell} ipython3
for i in range(n_bags):
    print(i)
    m = models_bag[i]
    data = bag[i]
    opt_bag = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    errors_bag, min_model_bag = dp_experiment.run(m, opt_bag, 8000, data, validation_data_t)
```

```{code-cell} ipython3
skip = 100
plt.plot(errors_bag[skip:,0])
plt.plot(errors_bag[skip:,1]);
plt.grid();
```

```{code-cell} ipython3
xs_t = torch.linspace(-np.pi, np.pi,200)
```

```{code-cell} ipython3
bag_loss =0.0
with torch.no_grad():
    pred = torch.zeros(len(validation_data_t))
    for m in models_bag:
        out =  m(validation_data_t[:,0].view(-1,1)).view(-1)
        print(loss_f(out, validation_data_t[:,1]).item())
        pred += out
pred/=n_bags
print(loss_f(pred, validation_data_t[:,1]).item())
```

```{code-cell} ipython3

```

```{code-cell} ipython3
ys_t= torch.zeros_like(xs_t)
with torch.no_grad():
    for m in models_bag:
        out = m(xs_t.view(-1,1)).view(-1)
        plt.plot(xs_t.numpy(), out.numpy())
        ys_t +=out
    ys_t/=10.0    
```

```{code-cell} ipython3
plt.plot(xs_t.numpy(), ys_t.numpy());
#plt.plot(xs_t.numpy(), ys_min_t.numpy());
plt.scatter(train_data[:,0], train_data[:,1],s=50, edgecolors='black', facecolors='none');
plt.scatter(validation_data[:,0], validation_data[:,1], s=50,edgecolors='red', facecolors='none');
```

## Dropout

+++

![](dropout.png)

```{code-cell} ipython3
model_dp = nn.Sequential(
    nn.Linear(1,128),nn.ReLU(),nn.Dropout(0.25),
    nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.25),
    nn.Linear(64,1)
)
```

```{code-cell} ipython3
opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
```

```{code-cell} ipython3
dp_experiment = Experiment()
```

```{code-cell} ipython3
%%time
model_dp.train()
errors_dp, min_model_dp = dp_experiment.run(model_dp, opt_dp, 8000, train_data_t, validation_data_t)
```

```{code-cell} ipython3
skip = 100
plt.plot(errors_dp[skip:,0])
plt.plot(errors_dp[skip:,1]);
plt.grid();
```

```{code-cell} ipython3
errors_dp.min(0)
```

```{code-cell} ipython3
xs_t = torch.linspace(-np.pi, np.pi,200)
```

```{code-cell} ipython3
with torch.no_grad():
    model_dp.eval()
    min_model_dp.eval()
    ys_t = model_dp(xs_t.view(-1,1)).view(-1)
    ys_min_t = min_model_dp(xs_t.view(-1,1)).view(-1)
```

```{code-cell} ipython3
plt.plot(xs_t.numpy(), ys_t.numpy());
plt.plot(xs_t.numpy(), ys_min_t.numpy());
plt.plot(xs_t.numpy(), np.cos(xs_t.numpy()))
plt.scatter(train_data[:,0], train_data[:,1],s=50, edgecolors='black', facecolors='none');
plt.scatter(validation_data[:,0], validation_data[:,1], s=50,edgecolors='red', facecolors='none');
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
