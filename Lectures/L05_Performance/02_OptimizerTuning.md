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
  slide_type: slide
---
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
import re
from itertools import product
from IPython.display import Image
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def encode(f,frm="f",d=None):
    s = ("%"+frm) % (f,)
    if d is not None:
        s = s.replace('.',d)
    return s

floater = re.compile('\.\d*f$')


    
def encoder(params):
    prefix=""
    fmt_string = ""
    for name, fmt in params:
        fmt_string+=prefix
        fmt_string+=name
        fmt_string+="%"+fmt
        prefix="_"
    
    def encode(*pars):
        return (fmt_string % pars).replace('.','d')
    
    return encode
        
    
```

```{code-cell} ipython3
if torch.cuda.is_available():
    dev = torch.device(torch.cuda.device_count()-1)
else:
    dev = 'cpu'
```

## Grid search

```{code-cell} ipython3
lrs = [0.001, 0.01, 0.1, 0.2, 0.5]
moms = [0.0, 0.5, 0.9]
ns = [16,24]
```

```{code-cell} ipython3
enc = encoder((('lr',"4.3f"),('mom','3.2f'),('n',"02d")))
```

```{code-cell} ipython3
%%time 
for lr, mom, n in product(lrs, moms, ns):
        filename = f'optimizers/mnist_SGD_'+enc(lr,mom,n)+'.ipynb'
        print(filename)
        pm.execute_notebook('MNIST.ipynb',filename, progress_bar=False,
                         parameters={'optimizer':'SGD', 'optimizer_parameters':{'lr':lr, 'momentum':mom},'n_epochs':n, 'device':dev});
```

```{code-cell} ipython3
book = sb.read_notebooks('optimizers/')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
dfs = []
for nb,scraps in book.items():
    scraps = scraps.scraps
    df = scraps.dataframe
    rep = pd.DataFrame.from_records([df['data'].values], columns = [df['name'].values])
    #print(rep)
    dfs.append(rep)
summary = pd.concat(dfs,axis=0).reset_index(drop=True)

summary.columns = [x[0] for x in summary.columns.to_flat_index()]
```

```{code-cell} ipython3
summary.columns
```

```{code-cell} ipython3
summary.sort_values('ac_test', ascending=False)
```

## Random search

```{code-cell} ipython3
lrs = [0.001, 0.01, 0.1, 0.2, 0.5]
moms = [0.0, 0.5, 0.9]
ns = [16,24]
```

```{code-cell} ipython3
def gen_parameters():
    log_lr = np.random.uniform(np.log(0.001), np.log(0.5))
    lr = np.exp(log_lr)
    mom = np.random.uniform(0.0, 0.9)
    n =  np.random.randint(16,25)
    return lr,mom, n
```

```{code-cell} ipython3
gen_parameters()
```

```{code-cell} ipython3
enc = encoder((('lr',"4.3f"),('mom','3.2f'),('n',"02d")))
```

```{code-cell} ipython3
%%time
for i in range(30) :
        lr, mom, n = gen_parameters()
        filename = f'optimizers_rnd/mnist_SGD_'+enc(lr,mom,n)+'.ipynb'
        print(f"{i:3d} {filename:s}")
        pm.execute_notebook('MNIST.ipynb',filename, progress_bar=False,
                         parameters={'optimizer':'SGD', 'optimizer_parameters':{'lr':lr, 'momentum':mom}, 'n_epochs':n});
```

```{code-cell} ipython3
book_rnd = sb.read_notebooks('optimizers_rnd/')
```

```{code-cell} ipython3
dfs = []
for nb,scraps in book_rnd.items():
    scraps = scraps.scraps
    df = scraps.dataframe
    rep = pd.DataFrame.from_records([df['data'].values], columns = [df['name'].values])
    #print(rep)
    dfs.append(rep)
summary_rnd = pd.concat(dfs,axis=0).reset_index(drop=True)
summary_rnd.columns = [x[0] for x in summary_rnd.columns.to_flat_index()]
```

```{code-cell} ipython3
summary_rnd.sort_values('ac_test', ascending=False)
```

```{code-cell} ipython3
Image('random_search.png')
```
