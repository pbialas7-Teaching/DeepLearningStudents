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

# Skin segmentation 

+++

In this assignement you will train classifier to assign colors to skin or no skin classes. The data is taken from [Skin Segmentation Data Set](http://archive.ics.uci.edu/ml/datasets/Skin+Segmentation#) in the UCI Machine Learning repository.

+++

The  data is in a plain text format and contains four columns. First three contain RGB color data  represented as integers in the range 0-255, and the last column is an integer label  with 1 representing skin and 2 representing no skin. This file we can load directly into a numpy array:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
```

```{code-cell} ipython3
data = np.loadtxt('data/Skin_NonSkin.txt')
```

```{code-cell} ipython3
rgb  = data[:,:3].astype('float32')
lbl = data[:,3].astype('float32') 
lbl = 2-lbl
```

```{code-cell} ipython3
len(data)
```

```{code-cell} ipython3
np.bincount(lbl.astype('int32'))
```

## Problem 1 

+++

Train the neural network to distinguish skin from no skin colors. Calculate the accuracy on train and validation sets. Calculate true positives rate and false positives rate.
