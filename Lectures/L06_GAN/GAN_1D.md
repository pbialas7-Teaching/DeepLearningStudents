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
import numpy as np
import scipy
import scipy.stats
import torch as t
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout, Sigmoid
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
else:
    device = t.device('cpu')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
d1_dist = scipy.stats.norm(loc=3,scale=2)
d2_dist = scipy.stats.norm(loc=0,  scale=0.5)
```

```{code-cell} ipython3
d1 = d1_dist.rvs(size=50000).astype('float32')
d2 = d2_dist.rvs(25000).astype('float32')
```

```{code-cell} ipython3
def p_data(x):
    return 2/3*d1_dist.pdf(x)+1/3*d2_dist.pdf(x)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data =np.concatenate((d1,d2)) 
np.random.shuffle(data)
data_t = t.from_numpy(data).view(-1,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
data_t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
ys = np.linspace(-5,10,100).astype('float32')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(data, bins=100, density=True);
plt.plot(ys, p_data(ys))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
noise_dist=scipy.stats.uniform(loc=0, scale=1)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Inverse cumulant

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
us = np.linspace(0,1,100).astype('float32')
us_t = t.from_numpy(us).view(-1,1)
```

```{code-cell} ipython3
def f_generator(u):
    return t.log(u/(1-u))
```

+++ {"slideshow": {"slide_type": "slide"}}

$$ Y = \log\left(\frac{u}{1-u}\right)$$

+++

$$CDF_{Y}(x)=P(Y<y)$$ 

+++

$$P(\log\left(\frac{u}{1-u}\right)<y)$$ 

+++

$$P\left(\frac{u}{1-u}<e^y\right)$$ 

+++

$$P\left(u<e^y-u e^y\right)$$ 

+++ {"slideshow": {"slide_type": "slide"}}

$$P\left(u(1+ e^y)<e^y\right)$$ 

+++ {"slideshow": {"slide_type": "-"}}

$$P\left(u<\frac{e^y}{1+ e^y}\right)=\frac{e^y}{1+ e^y}=\frac{1}{1+ e^{-y}} $$ 

+++

$$PDF_Y(x) = \frac{\partial}{\partial y}\frac{1}{1+ e^{-y}}=\frac{e^{-y}}{(1+e^{-y})^2}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def p_gen(y):
    e = np.exp(-y)
    return e/((1+e)*(1+e))
```

```{code-cell} ipython3
p_ys = p_gen(ys).reshape(-1,1)
```

```{code-cell} ipython3
gen = f_generator(t.FloatTensor(10000,1).uniform_(0,1))
```

```{code-cell} ipython3
plt.hist(gen.data.numpy().reshape(-1), bins=100, density=True);
plt.plot(ys, p_ys);
```

+++ {"slideshow": {"slide_type": "slide"}}

### Data

```{code-cell} ipython3
n = 1000
xs = np.linspace(-5,10,n+1)
pdf_data = p_data(xs)
cum_data = np.cumsum(pdf_data)*15/n
cum_data[0]=0.0
cum_data[-1]=1.0
```

```{code-cell} ipython3
plt.plot(cum_data,xs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from scipy.interpolate import interp1d
```

```{code-cell} ipython3
inv_cum_data = interp1d(cum_data, xs,3)
```

```{code-cell} ipython3
fake_data = inv_cum_data(np.random.uniform(0,1,100000))
```

```{code-cell} ipython3
plt.hist(fake_data, bins=100, density=True);
plt.plot(xs, p_data(xs));
```

## Generative Adversarial Network

```{code-cell} ipython3
def makeNet(n_layers, n_neurons_in_last_layer):
    n = n_layers-1
    n_neurons_in_first_layer = n_neurons_in_last_layer*(2**(n-1))
    modules=[]
    modules.append(Linear(in_features=1, out_features=n_neurons_in_first_layer))
    modules.append(LeakyReLU())
    for i in range(n-1):
        modules.append(Linear(in_features=n_neurons_in_first_layer, out_features=n_neurons_in_first_layer//2))
        modules.append(LeakyReLU())
        n_neurons_in_first_layer//=2
    modules.append(Linear(in_features=n_neurons_in_last_layer, out_features=1))
   
    return Sequential(*modules)
```

```{code-cell} ipython3
discriminator = Sequential(Linear(1,512), LeakyReLU(0.2, inplace=True),
                           Linear(512,32), LeakyReLU(0.2, inplace=True),
                           Linear(32,1),  Sigmoid()
                                     )
```

```{code-cell} ipython3
discriminator=discriminator.to(device)
data_t = data_t.to(device)
```

```{code-cell} ipython3
d_optimizer = t.optim.Adam(discriminator.parameters(), lr=0.0002)
```

```{code-cell} ipython3
bce = t.nn.BCELoss()
```

```{code-cell} ipython3
d_out = discriminator(t.from_numpy(ys).view(-1,1).to(device))
```

```{code-cell} ipython3
plt.plot(ys,d_out.data.cpu().numpy().reshape(-1),c='r');
plt.plot(ys, p_data(ys)/(p_data(ys)+p_gen(ys)),c='b');
```

```{code-cell} ipython3
generator = makeNet(5,32)
```

```{code-cell} ipython3
print(generator)
```

```{code-cell} ipython3
generator= generator.to(device)
```

```{code-cell} ipython3
out_t = generator(t.FloatTensor(us_t).to(device));
```

```{code-cell} ipython3
plt.plot(us, out_t.data.cpu().numpy().reshape(-1))
```

```{code-cell} ipython3
g_optimizer = t.optim.Adam(generator.parameters(), lr=0.0002)
```

```{code-cell} ipython3
gen = generator(t.empty(10000,1).uniform_(-1,1).to(device))
```

```{code-cell} ipython3
plt.hist(gen.data.cpu().numpy().reshape(-1), bins=100, density=True);
#plt.plot(ys, p_data(ys));
```

```{code-cell} ipython3
-np.log(0.5)
```

```{code-cell} ipython3
mini_batch_size = 2048
k_discriminator = 4
k_generator = 1
for epoch in range(1,201):
    for batch in range(len(data)//mini_batch_size):
        for k_d in range(k_discriminator):
            d_optimizer.zero_grad()
            kr = np.random.randint(0,len(data)//mini_batch_size )
          
            d = data_t[kr*mini_batch_size:(kr+1)*mini_batch_size]
            real_labels = t.ones(mini_batch_size, 1, device=device)
            d_real_loss = bce(discriminator(d), 0.9*real_labels)
      
            z = t.empty(mini_batch_size,1, device=device).uniform_(-1,1)
            g_out = generator(z)
            fake_labels = t.zeros(mini_batch_size, 1, device=device)
            d_fake_loss = bce(discriminator(g_out), fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
    
        g_optimizer.zero_grad()
        z = t.empty(mini_batch_size,1, device=device).uniform_(-1,1)
        g_out = generator(z)
        g_loss = bce(discriminator(g_out), real_labels)
        g_loss.backward()
        g_optimizer.step()
        
    if epoch%5 == 0:
        with t.no_grad():
                z = t.empty(len(data_t),1, device=device).uniform_(-1,1)
                real_labels = t.ones(len(data_t),1, device=device)
                fake_labels = t.zeros(len(data_t),1, device=device)
                g_out =  generator(z)
                dg_out = discriminator(g_out)
                d_loss =  bce(discriminator(data_t), real_labels)
                d_loss +=  bce(dg_out, fake_labels)
                
                g_loss = bce(dg_out, real_labels)
                
                print(epoch, d_loss.item(), g_loss.item())     
```

```{code-cell} ipython3
gen = generator(t.empty(100000,1).uniform_(-1,1).to(device))
```

```{code-cell} ipython3
plt.hist(gen.data.cpu().numpy().reshape(-1), bins=100, density=True);
plt.plot(ys, p_data(ys));
```

```{code-cell} ipython3
d_out = discriminator(t.from_numpy(ys).to(device).view(-1,1))
```

```{code-cell} ipython3
plt.plot(ys,d_out.data.cpu().numpy().reshape(-1));
plt.axhline(0.5);
```

```{code-cell} ipython3
out_t = generator(t.linspace(-1,1,100).view(-1,1).to(device));
```

```{code-cell} ipython3
plt.plot(np.linspace(-1,1,100), out_t.data.cpu().numpy().reshape(-1))
plt.plot(2*cum_data-1, xs);
```

```{code-cell} ipython3

```
