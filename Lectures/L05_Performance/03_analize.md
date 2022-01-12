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

```

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
book = sb.read_notebooks('optimizers/')
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
summary.sort_values('ac_test', ascending=False)
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

```
