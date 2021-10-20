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
---
slideshow:
  slide_type: skip
---
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

+++ {"slideshow": {"slide_type": "slide"}}

# Classification

+++ {"slideshow": {"slide_type": "skip"}}

As the classification example we will use the "Hello World!" of machine learning: ["The MNIST Database of hanwritten digits"](http://yann.lecun.com/exdb/mnist/).

+++ {"slideshow": {"slide_type": "slide"}}

## MNIST

+++ {"slideshow": {"slide_type": "skip"}}

This dataset bundled in many machine learning libraries and PyTorch is no exception.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
train_data = torchvision.datasets.MNIST('../data/mnist', train=True, download=True)
test_data  = torchvision.datasets.MNIST('../data/mnist', train=False, download=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
train_features   = train_data.data.to(dtype=torch.float32)
train_labels = train_data.targets
```

+++ {"slideshow": {"slide_type": "skip"}}

The data consists of 28 by 28 pixels 8bit grayscale images of handwritten digits, the labels are integers denoting corresponding digits:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig_mnist, axes = plt.subplots(2,4, figsize=(16,8))
for i in range(8):
    ax=axes.ravel()[i]
    ax.imshow(train_features[i].numpy(), cmap='Greys');
    ax.set_title(train_labels[i].item(), fontsize=20)
```

+++ {"slideshow": {"slide_type": "skip"}}

For the purpose of this notebook I will load only a sybset of data. This will make the training of the network much quicker.

```{code-cell} ipython3
n_samples = 12000
```

+++ {"slideshow": {"slide_type": "skip"}}

Because we will be using the fully connected neural network as our classifier I will flatten each image to 28x28 long 1D array. I will also normalize the grayscale values to [0,1).

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
dataset = torch.utils.data.TensorDataset( 
    (train_features[:n_samples]/256.0).view(-1,28*28), 
    train_labels[:n_samples])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, (10000,2000))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size = 100, 
                                           shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                           batch_size = 100, 
                                           shuffle = True)
```

```{code-cell} ipython3
train_labels.dtype
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
test_features   = test_data.data.to(dtype=torch.float32)
test_labels = test_data.targets
test_dataset = torch.utils.data.TensorDataset(
    (test_features/256.0).view(-1,28*28), test_labels)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Cross entropy

+++ {"slideshow": {"slide_type": "skip"}}

For classification problems  with $M$  possible categories $C=\{0,\ldots,M-1\}$ the output of the model  is a 1D vector with $M$ entries corresponding to probabilities  for each class

+++ {"slideshow": {"slide_type": "fragment"}}

$\renewcommand{\b}[1]{\mathbf{#1}}$
$$o^l_i = P(i | \b{x}_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

where $l$ is the index of the  last layer. This is achieved by the _softmax_ activation function on the last layer:

+++ {"slideshow": {"slide_type": "fragment"}}

$$o^{l}_i = \frac{ e^{\displaystyle o^{l-1}_i}}{\sum_{i=0}^{M-1}e^{\displaystyle o^{l-1}_i}}$$

+++ {"slideshow": {"slide_type": "skip"}}

Where $o^{l-1}_i$ is the output of the previous layer.  I will  use the word _layer_ in a generalized sense. A layer is a single operation so for example the  activation function application is considered as a separate layer.

+++ {"slideshow": {"slide_type": "skip"}}

We will use the  _Negative Log Likelihood_ loss:

+++ {"slideshow": {"slide_type": "fragment"}}

$$-\sum_{i=0}^N\log P(c_i|\b{x}_i) = -\sum_{i=0}^N\log a_{ c_{\scriptstyle i}}$$

+++ {"slideshow": {"slide_type": "skip"}}

where $c_i$  is the category corresponding to features $\b{x}_i$.

+++ {"slideshow": {"slide_type": "skip"}}

This is often written in _cross entropy_ form:

+++ {"slideshow": {"slide_type": "skip"}}

$$-\sum_{i=0}^N
\sum_{j=0}^{M-1} l_{ij} \log a_{j}$$

+++ {"slideshow": {"slide_type": "skip"}}

where $l_{ij}$ are _one-hot_ encoded categories:

+++ {"slideshow": {"slide_type": "skip"}}

$$ l_{ij} =\begin{cases}
1 & c_i = j \\
0 & c_i\neq j
\end{cases}
$$

+++ {"slideshow": {"slide_type": "slide"}}

## The model

+++ {"slideshow": {"slide_type": "skip"}}

We will use a fully  four  fully connected layers with `ReLU` activation layers in between as our model and `softmax` as the last layer.  The model can be easily constructed using the PyTorch `nn.Sequential` class:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10), nn.Softmax(dim=1)
)
```

+++ {"slideshow": {"slide_type": "skip"}}

The network has 28x28=784 inputs and ten outputs which correspond to ten possible categories.  The model parameters: weights and biases are initalized randomly (more on this in other lectures). Let's check how this model performs, we will use   the `torch.no_grad()` _context manager_ to temporalily switch off gradient calculations

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with torch.no_grad():
    pred = model(train_dataset[:][0])
```

+++ {"slideshow": {"slide_type": "skip"}}

Tensor `pred` contains the predicted probabilities for each  digit for each input:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
pred[:4]
```

+++ {"slideshow": {"slide_type": "skip"}}

As we can see the distribution looks rather uniform. We can check that indeed the probabilities for each category sum to one:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
pred.sum(1)
```

+++ {"slideshow": {"slide_type": "skip"}}

The accuracy of  clasification can be calculated as follows:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def accuracy(pred, labels):
    return torch.sum(torch.argmax(pred,axis = 1)==labels).to(dtype=torch.float32).item()/len(labels)
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's break down this function. The `argmax` function with `axis` argument equal to one for each row returns the index of the column containing the largest value. This is next compared with actual labels.   Finally we use implicit conversion of `False` to zero and `True` to one to calculate the number of labels predicted correctly. We finally divide by the length of the dataset to obtain accuracy. The conversion to float is needed because otherwise the integer arthmetic is used.

+++ {"slideshow": {"slide_type": "skip"}}

Not suprisingly our accuracy is even worse then random guessing

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
accuracy(pred, train_dataset[:][1])
```

+++ {"slideshow": {"slide_type": "skip"}}

We will  define another  accuracy function for further convenience:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def model_accuracy(model, dataset):
    features, labels = dataset[:]
    with torch.no_grad():
        pred = model(features)
    return accuracy(pred, labels)
```

+++ {"slideshow": {"slide_type": "skip"}}

Before we start training we need the loss function:

```{code-cell} ipython3
def my_nll_loss(pred,labels):
    return -torch.mean(
        torch.log(0.0000001+pred[range(len(labels)),labels])
                      )
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's break it down: the `pred[range(len(labels)),labels]`  expression takes from each row $i$ of the `pred` tensor the value from the column $\mathtt{labels}_i$ which is the probability of the correct label. We then take the logarithm and average over all examples. The small value is added in case one of the entries would be zero.

+++ {"slideshow": {"slide_type": "skip"}}

After all this being said we can finally start training:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
for e in range(5):
    for features, labels in train_loader:        
        optim.zero_grad()
        pred = model(features)
        loss = my_nll_loss(pred, labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, train_dataset)
```

+++ {"slideshow": {"slide_type": "skip"}}

As you can see the accuracy has increased greatly. But really important is the accuracy on the test data set:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, test_dataset)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
.91**5
```

+++ {"slideshow": {"slide_type": "skip"}}

After training the model we can save it to file:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
torch.save(model,"mnist.pt")
```

+++ {"slideshow": {"slide_type": "skip"}}

and load later

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
copy = torch.load("mnist.pt")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with torch.no_grad():
    pred = torch.softmax(copy(train_dataset[:][0]),1)
    ac = torch.sum(torch.argmax(pred,1)==train_dataset[:][1]).to(dtype=torch.float32)/len(train_dataset)
ac 
```

+++ {"slideshow": {"slide_type": "slide"}}

## Using PyTorch loss functions

+++ {"slideshow": {"slide_type": "skip"}}

Our formulation of the loss function required calculation of the  logarithm of the softmax function. Doing those two operations separately is slower and numerically unstable. That's why PyTorch privides an implementation of the `log_softmax` function that does  both operations together. Please note that now the outputs of the model do not represent the probabilities.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10), nn.LogSoftmax(dim=1)
)
```

+++ {"slideshow": {"slide_type": "skip"}}

We can now use the provided  negative likelihood loss function that expects the logarithms of probabilities as its input:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
nll_loss = torch.nn.NLLLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
for e in range(5):
    for features, labels in train_loader:        
        optim.zero_grad()
        pred = model(features)
        loss = nll_loss(pred,labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

+++ {"slideshow": {"slide_type": "skip"}}

The accuracy functions will still work because the logarithm is monotonically increasing function.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, train_dataset)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, test_dataset)
```

+++ {"slideshow": {"slide_type": "skip"}}

And finally we can drop the last activation layer

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10)
)
```

+++ {"slideshow": {"slide_type": "skip"}}

and use the cross entropy loss function that  calculates the log softmax internally

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ce_loss = torch.nn.CrossEntropyLoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
for e in range(5):
    for features, labels in train_loader:        
        optim.zero_grad()
        pred = model(features)
        loss = ce_loss(pred,labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

+++ {"slideshow": {"slide_type": "skip"}}

The accuracy functions will still work as before, because softmax  does not change the relative order of the input values.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, train_dataset)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, test_dataset)
```

+++ {"slideshow": {"slide_type": "slide"}}

## MSE loss

+++ {"slideshow": {"slide_type": "skip"}}

And finally we will use the MSE loss for comparison. To this end we have to one-hot encode the labels:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
one_hot_labels = torch.zeros(n_samples, 10).to(dtype=torch.float32)
one_hot_labels[range(n_samples),train_labels[:n_samples]] =  1.0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
one_hot_labels
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
one_hot_dataset = torch.utils.data.TensorDataset( 
    (train_features[:n_samples]/256.0).view(-1,28*28), 
    one_hot_labels)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
one_hot_train_dataset, one_hot_validation_dataset = torch.utils.data.random_split(one_hot_dataset,(10000,2000))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
one_hot_train_loader = torch.utils.data.DataLoader(one_hot_train_dataset, batch_size=100)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
model = torch.nn.Sequential(
    nn.Linear(28*28,1200), nn.ReLU(),
    nn.Linear(1200,600), nn.ReLU(),
    nn.Linear(600,300), nn.ReLU(),
    nn.Linear(300,10), nn.Softmax(dim=1)
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
optim = torch.optim.SGD(model.parameters(), lr=0.1)
```

```{code-cell} ipython3
mse_loss = torch.nn.MSELoss()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
%%time
for e in range(5):
    for features, labels in one_hot_train_loader:        
        optim.zero_grad()
        pred = model(features)
        loss = mse_loss(pred,labels)
        loss.backward()
        optim.step()   
    print(e, loss.item())        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, train_dataset)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
model_accuracy(model, test_dataset)
```

+++ {"slideshow": {"slide_type": "skip"}}

As we can see the accuracy is much smaller: the convergence is slower. In the next notebook we will take a look at why it is so.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---

```
