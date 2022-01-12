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
import numpy as np
import scipy
import scipy.stats
import torch as t

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout, Sigmoid
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
if t.cuda.is_available():
    if t.cuda.device_count()>1:
        device = t.device('cuda:1')
    else:
        device = t.device('cuda')   
else:
    device = t.device('cpu')
```

```{code-cell} ipython3
device=t.device('cpu') #Overrride the above device choice
```

Generate the sample 2D distribution: uniform from unit circle.  

```{code-cell} ipython3
angle = np.random.uniform(-np.pi,np.pi,(1000,1)).astype('float32')
data = np.concatenate((np.cos(angle), np.sin(angle)),axis=1)
```

```{code-cell} ipython3
plt.scatter(data[:,0], data[:,1]);
```

```{code-cell} ipython3
data_t = t.from_numpy(data)
```

```{code-cell} ipython3
data_t
```

```{code-cell} ipython3
discriminator = Sequential(Linear(2,1),  Sigmoid()) #dummy discriminator: please subsitute you own implementation 
```

```{code-cell} ipython3
discriminator = discriminator.to(device) 
```

```{code-cell} ipython3
generator = Sequential(Linear(1,2))# dummy generator: please subsitute you own implementation 
```

```{code-cell} ipython3
generator= generator.to(device)
```

```{code-cell} ipython3
out_t = generator(t.empty(1000,1, device=device).uniform_(-1,1));
```

```{code-cell} ipython3
plt.scatter(out_t.data.cpu().numpy()[:,0],out_t.data.cpu().numpy()[:,1])
```

```{code-cell} ipython3
d_optimizer = t.optim.Adam(discriminator.parameters(), lr=0.0002)
```

```{code-cell} ipython3
g_optimizer = t.optim.Adam(generator.parameters(), lr=0.0002)
```

### Problem 1

+++

Implement the GAN train loop that will train GAN to generate from the sample distribution.  

+++

Update to Pegaz both the notebook and the trained generator. 

+++

### Problem 2

+++

Use sampling distribution below. 

```{code-cell} ipython3
n_samples = 10000
a = 2
b = 1
angle2 = np.random.uniform(-np.pi,np.pi,(n_samples,1)).astype('float32')
r = np.sqrt(np.random.uniform(0.5,1,(n_samples,1)))
data2 = np.stack((a*r*np.cos(3*angle2), b*r*np.sin(2*angle2)),axis=1)
```

```{code-cell} ipython3
plt.scatter(data2[:,0], data2[:,1], s=2, alpha=0.5);
```

Update to Pegaz both the notebook and the trained generator.
