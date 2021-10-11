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

# Learning

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../modules/')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
```

+++ {"slideshow": {"slide_type": "skip"}}

For all the visualisations we will be using the matplotlib library:

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

+++ {"slideshow": {"slide_type": "skip"}}

The second line above is an example of jupyter "magic command". This one ensures that the figures produced by matplotlib commands will be automatically displayed, without need for an explicit `show()` command. Third line sets the default figure size.

+++ {"slideshow": {"slide_type": "slide"}}

## Supervised learning

+++ {"slideshow": {"slide_type": "skip"}}

Before we dive into deep learning we must establish what we mean by learning. Throughout this lecture we will be concerned only with one form of machine learning (ML) know as **supervised learning**. This is by far most popular  form of ML, one of the reasons being that it is the easiest, which does not mean easy.

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

With supervised learning we are presented with a sets of examples consistings of features and labels:

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$\newcommand{b}[1]{\mathbf{#1}}$

$$\begin{array}{lcccccl}
\text{features} & & & & \text{labels}\\
\b{X}_0 & = &(x_{0,0},x_{0,1},\ldots,x_{0,N_f-1}) &\rightarrow &  \left(y_{0,0},y_{0,1},\ldots,y_{0,N_l-1}\right) & = & \b{Y}_0\\
\b{X}_1 & = &(x_{1,0},x_{1,1},\ldots,x_{1,N_f-1}) &\rightarrow &  \left(y_{1,0},y_{1,1},\ldots,y_{1,N_l-1}\right)& = & \b{Y}_1\\
\b{X}_2 & = &(x_{2,0},x_{2,1},\ldots,x_{2,N_f-1}) &\rightarrow &  \left(y_{2,0},y_{2,1},\ldots,y_{2,N_l-1}\right)& = & \b{Y}_2\\
&&&\vdots& & & \\
\b{X}_{N_s-1} & = &(x_{N_s-1,0},x_{N_s-1,1},\ldots,x_{N_s,N-1}) &\rightarrow &  \left(y_{N_s-1,0},y_{N_s-1,1},\ldots,y_{N_s-1,M-1}\right)& = & \b{Y}_{N_s-1}
\end{array}
$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

$\b{X}$ and $\b{Y}$ are matrices where each row corresponds to on observation with $\b{X}$ conatining features and $\b Y$ containing labels.

+++ {"slideshow": {"slide_type": "skip"}}

 The goal is to learn the unknow _mapping_ or _function_ from features to labels:

+++ {"slideshow": {"slide_type": "fragment"}}

$$ \b{y} = f(\b{x})$$

+++ {"slideshow": {"slide_type": "skip"}}

Often such mapping will be called a *model*.

+++ {"slideshow": {"slide_type": "slide"}}

## Loss function

+++ {"slideshow": {"slide_type": "skip"}}

This is accomplished by defining a loss function that measures  how bad is our mapping  in predicting the labels. The loss is usually a sum on losses of each individual feature/label pair

+++ {"slideshow": {"slide_type": "fragment"}}

$\newcommand{\loss}{\operatorname{loss}}$

$$\loss\left(\mathbf{Y},\;f(\mathbf{X})\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\loss\left(\mathbf{Y},\;f(\mathbf{X})\right)=  \sum_{i} \loss(\b{Y}_i, f(\b{X}_i)) $$

+++ {"slideshow": {"slide_type": "skip"}}

Then learning can be formaly reframed as problem of finding a mapping that minimizes the loss

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$\newcommand{\argmin}{\operatorname{argmin}}$

$$f = \argmin_f\loss\left(\mathbf{Y},\,f(\mathbf{X})\right)$$ (eq:learning)

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

This formulation is strictly formal. In general we cannot  perform minimalisation over the space of all functions. Other problem is that we could define $f$ as

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$
f(\b x) = \begin{cases}
\b y_i & \exists_i\,\b{x}=\b{X}_i\\
0  & \text{otherwise}
\end{cases}
$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

it assumes that each $\b{X}_i$ maps into unique $\b{Y}_i$ _i.e._ 
$\b{Y}_i\neq\b{Y}_j\implies \b{X}_i\neq\b{X}_j$ but for continous features and labels this condition is often fulfiled. Then function $f$ achieves a minimal loss but is completely useless as it cannot predict labels not present in the training set. This is a drastic example of *overfiting*.

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

```{Attention} 
From this example we can infere that loss function minimisation is **not** the goal of learning. The goal is to find such function $f$ that will perform well on the examples not provided in the training set. Equation  {eq}`eq:learning` is only a mean to this end, but in it's unrestricted form is not useful. We will address this problem later in this lecture.
```

+++ {"slideshow": {"slide_type": "slide"}}

### Mean Square Error (MSE)

+++ {"slideshow": {"slide_type": "skip"}}

An example of loss function would be the Mean Square Error

+++ {"slideshow": {"slide_type": "fragment"}}

$$\operatorname{MSE}\left(\mathbf{y}_i,\,f(\mathbf{x}_i)\right)=\frac{1}{N_l}\sum_{j=0}^{N_l-1}\left(y_{ij}-f(\b{x}_i)_j\right)^2$$

+++ {"slideshow": {"slide_type": "skip"}}

or Root Mean Square Error

+++ {"slideshow": {"slide_type": "fragment"}}

$$\operatorname{RMSE}\left(\mathbf{y}_i,\,f(\mathbf{x}_i)\right)=\sqrt{\frac{1}{N_s}\sum_{i=0}^{N_-1} \operatorname{MSE}(\b{y}_i, f(\b{x}_i))}$$

+++ {"slideshow": {"slide_type": "slide"}}

## Distributions

+++ {"slideshow": {"slide_type": "skip"}}

The above formulation is too simplistic for most of the real world examples. In reality the mapping from features to labels is seldom deterministic. So instead of mapping, the relation between features and labels is better descibed by a conditional distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\b{y}|\b{x})$$

+++ {"slideshow": {"slide_type": "skip"}}

that represenst the probability that given features $\b{x}$ labels will have value $\b{y}$.

+++ {"slideshow": {"slide_type": "skip"}}

Then we proceed in the same way as before  looking for a distribution that mininimilizes the loss with respect to the data

+++ {"slideshow": {"slide_type": "fragment"}}

$$P = \argmin_\tilde{P}\loss\left(\b{Y},\b{X},\tilde{P}\right)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Negative Logarithm Likelihood (NLL)

+++ {"slideshow": {"slide_type": "skip"}}

Given a distribution $P$ and the data, one measure of goodness of fit would be  the likelihood: the probability that we observed our data under this distribution. The bigger this probability the better this distribution describes the data.

+++ {"slideshow": {"slide_type": "skip"}}

We assume all our examples to be independent so the resulting probability is the product of probabilities of observing each example

+++ {"slideshow": {"slide_type": "fragment"}}

$$\tilde{P}(\b{y}|\b{x})=\prod_{i}\tilde{P}(\b{y}_i| \b{x}_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

Taking the logarithm changes the product into a sum. As logarithm is a monotonic function it does not change the location of minima and maxima.

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log \tilde{P}(\b{y}|\b{x})=\sum_{i}\log \tilde{P}(\b{y}_i| \b{x}_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

And finally taking the negative assures us that the results is non negative and makes it a loss function as now a bigger the value indicates a worst match

+++ {"slideshow": {"slide_type": "fragment"}}

$$\operatorname{NNL}(\b Y, \b X) =  -\sum_{i}\log \tilde P(\b{Y}_i| \b{X}_i)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Cross entropy

+++ {"slideshow": {"slide_type": "skip"}}

The NNL loss is closely tied with the notion of *cross entropy*.

+++ {"slideshow": {"slide_type": "fragment"}}

$$H(P,Q)=-\int\text{d}\b x\, P(\b x) \log \tilde{P}(\b x) =-\operatorname{E}_P[\tilde{P}]$$

+++ {"slideshow": {"slide_type": "skip"}}

For large sample we can treat NNL as an approximate average with respect to the distribution $P(\b y| \b x)$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
-\sum_{i}\log \tilde P(\b{y}_i| \b{x}_i)\approx -\left\langle\log \tilde P(\b{y}| \b{x})\right\rangle_{P(\b y, \b x)}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
-\left\langle\log \tilde P(\b{y}| \b{x})\right\rangle_{P(\b y, \b x)}\equiv -\int\text{d}\b{y}\text{d}\b{x} P(\b{y},\b{x})\log \tilde P(\b{y}| \b{x})=
-\int\text{d}\b{x} P(\b{x})\int\text{d}\b{y} P(\b{y}| \b{x})\log \tilde P(\b{y}| \b{x})
$$

+++ {"slideshow": {"slide_type": "skip"}}

where we have used  the product rule $P(\b y , \b x)= P(\b y| \b x) P(\b x)$. Integral

+++ {"slideshow": {"slide_type": "fragment"}}

$$
-\int\text{d}\b{y}
P(\b{y}| \b{x})\log \tilde P(\b{y}| \b{x})= H(P(\cdot|\b x),\tilde P(\cdot|\b{x}))
$$

+++ {"slideshow": {"slide_type": "skip"}}

is the cross entropy between distributions $P(\b{y}| \b{x})$ and $\tilde P(\b{y}| \b{x})$.

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Kullback–Leibler divergence

+++ {"slideshow": {"slide_type": "skip"}}

In turn cross entropy is closely tied to *Kullback–Leibler divergence* defined for two distributions $P(\b x)$ and $\tilde P(\b x)$ as

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$\newcommand{\tP}{\tilde{P}}$
$$D_{KL}(P|Q) = -\int\text{d}\b x P(\b x) \log\left(\frac{\tP(\b x)}{P(\b x)}\right) = \int\text{d}\b x P(\b x) \log\left(\frac{P(\b x)}{\tP(\b x)}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$D_{KL}(P|\tP)\ge 0,\quad D_{KL}(P|\tP)=0\implies \tP=P$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$D_{KL}(P|Q) = -\int\text{d}\b x P(\b x) \log\left(\frac{Q(\b x)}{P(\b x)}\right) 
= \underbrace{-\int\text{d}\b x P(\b x) Q(\b x)}_{H(P,\tP)}
+\underbrace{\int\text{d}\b x P(\b x) \log P(\b x)}_{-H(P)}
$$

+++ {"slideshow": {"slide_type": "slide"}}

## Regression

+++ {"slideshow": {"slide_type": "skip"}}

The maximal likelihood is a general method of finding loss functions. Let's see some examples. We will start with the regression.
Assume that we are given noisy data

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_i = f(x_i)+\epsilon_i\quad \epsilon_i \sim \mathcal{N}(0,\sigma)$$

+++ {"slideshow": {"slide_type": "skip"}}

in other words $y_i$ is drawn from the distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_i \sim \mathcal{N}\left(f(x_i),\sigma\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

so

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\b{y}|\b{x}) = \prod_i \frac{1}{\sqrt{2\pi}\sigma}e^{\displaystyle-\frac{\left(y_i-f(x_i)\right)^2}{2\sigma^2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

and

+++ {"slideshow": {"slide_type": "fragment"}}

$$-\log P(\b{y}|\b{x}) = N\frac{1}{2}\log\left(2 \pi \sigma^2\right) + \frac{1}{2 \sigma^2}\sum_{i}\left(y_i-f(x_i)\right)^2  $$

+++ {"slideshow": {"slide_type": "skip"}}

For simplicity let's assume that we know $\sigma$, then optimizing NNL is same as optimizing MSE. This remain true even if we don't know $\sigma$ (why?).

+++ {"slideshow": {"slide_type": "slide"}}

## Classification

+++ {"slideshow": {"slide_type": "skip"}}

In classification the labels form  a discrete set of classes:

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$y_i\in \{C_0,\ldots,C_{N_l-1}\}$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

Without any loss of generality we can assume that labels are consequtive natural number *i.e.* $C_k=k$.

+++ {"slideshow": {"slide_type": "fragment"}}

$$\begin{array}{lccc}
\text{features} & & & & \text{labels}\\
\b{X}_0 & = &(x_{0,0},x_{0,1},\ldots,x_{0,N_f-1}) &\rightarrow & y_0 \\
\b{X}_1 & = &(x_{1,0},x_{1,1},\ldots,x_{1,N_f-1}) &\rightarrow & y_1 \\
\b{X}_2 & = &(x_{2,0},x_{2,1},\ldots,x_{2,N_f-1}) &\rightarrow & y_2 \\
&&&\vdots& & & \\
\b{X}_{N_s-1} & = &(x_{N_s-1,0},x_{N_s-1,1},\ldots,x_{N_s,N-1}) &\rightarrow & y_{N_s-1}
\end{array}
$$

+++ {"slideshow": {"slide_type": "skip"}}

The goal  is to predict the class $j$ given the feature vector $\b{x}$. Instead of predicting the labels directlty we will be predicting the conditional probability that an exemplar with given features vector $\b x$ belongs to given class:

+++ {"slideshow": {"slide_type": "slide"}}

$\newcommand{\tp}{\tilde{p}}$
$$
\tp_k(\b x) =  \tP( k | \b x)
$$

+++

$$f(\b x) \rightarrow \left(\tp_{0}(\b x),\ldots,\tp_{N_l-1}(\b x)\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

Having the set of features and labels the probaility that we have observed the labels under distribution $\tP$ is

+++ {"slideshow": {"slide_type": "fragment"}}

$$\tP(\b y|\b X)=\prod_i p_{y_i}(\b{x}_i) $$

+++ {"slideshow": {"slide_type": "skip"}}

which converts to negative likelihood loss:

+++ {"slideshow": {"slide_type": "slide"}}

$$-\sum_{i=1}^{N_s-1} \log p_{y_i}(\b{x}_i) =-\sum_{i=0}^{N_s-1} \sum_{k=0}^{N_l-1} l_{ik} \log p_k(\b{x}_i) = \sum_{i=0}^{N_s-1} \operatorname{CELoss} (y_i,\b x_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

Vector $\b{l}_i$ is a one-hot encoding of label $y_i$:

+++ {"slideshow": {"slide_type": "fragment"}}

$$l_{ik}= \begin{cases}
1 & y_i= k\\
0 & \text{otherwise}
\end{cases}
$$

+++ {"slideshow": {"slide_type": "slide"}}

### Binary cross entropy

+++ {"slideshow": {"slide_type": "skip"}}

When we have only two labels the we need only one value of probability

+++ {"slideshow": {"slide_type": "fragment"}}

$$(\tp_0(\b x),\,\tp_1(\b x))=(1-p,p)$$

+++ {"slideshow": {"slide_type": "skip"}}

and cross entropy loss simplifies to:

+++ {"slideshow": {"slide_type": "fragment"}}

$$-\sum_{i=0}^{N_s-1}\left( y_i \log p(\b x_i) + (1-y_i)\log (1-p(\b x_i)) \right)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Regression example

+++ {"slideshow": {"slide_type": "skip"}}

We will load some prepared data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
data = np.load("../data/sgd_data.npy")
```

+++ {"slideshow": {"slide_type": "skip"}}

and take first 50 rows of it. We will also take another 25 rows as the validation set which will not be used for training

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
train_set = data[:50]
validation_set = data[50:75]
```

+++ {"slideshow": {"slide_type": "skip"}}

That's how the data looks like:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.scatter(train_set[:,0],train_set[:,1], alpha=0.7, color='none', edgecolor="black", label='train');
ax.scatter(validation_set[:,0],validation_set[:,1], alpha=0.7, color='none', edgecolor="blue", label='validate');
ax.legend();
```

+++ {"slideshow": {"slide_type": "skip"}}

To find the mapping corresponding to this data we will use the MSE loss. Untill now all was rather abstract, we were talking about optimizing over a space of all possible functions. That's obviously not possible. The way to proceed is to take a familly of functions parametrized with some set of parameters and optimize over the space of those parameters.

+++ {"slideshow": {"slide_type": "slide"}}

## Model capacity, underfitting and overfitting

+++ {"slideshow": {"slide_type": "skip"}}

Judging by the look of the date we will choose the trigometric series

+++ {"slideshow": {"slide_type": "fragment"}}

$$f_N(x|\b c,\b s) = c_0 + \sum_{i=1}^{N-1}\left(s_i \sin(i\cdot x)+ c_i \cos(i\cdot x)\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

as the familly of functions that will be used to "learn" the data. Learning will consist of finding such parameters $c_i$ and $s_i$ as to minimize the MSE loss functions

+++ {"slideshow": {"slide_type": "fragment"}}

$$L(\b y, \b x| \b c, \b s)=\sum_i (y_i -f_N(x_i|\b c, \b s))^2$$

+++ {"slideshow": {"slide_type": "skip"}}

In this example this can be done analiticaly  and is coded in the function `fourier` of my `fit` module.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import fit
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
f = fit.fourier(train_set[:,0], train_set[:,1],12,0.1)
```

+++ {"slideshow": {"slide_type": "skip"}}

We will now train the model changing the value of $N$ and plotting the  resulting loss both on the train and validation sets.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [hide-input]
---
errs = []
l=0.0
for n in range(1,26):
    f = fit.fourier(train_set[:,0], train_set[:,1],n,l)
    t_err = fit.rmse(train_set[:,0], train_set[:,1],f)
    v_err = fit.rmse(validation_set[:,0], validation_set[:,1],f)
    errs.append((n,t_err, v_err))
errs = np.asarray(errs)   

fig, ax = plt.subplots(1,1 , figsize = (12,8))


ax.set_xlabel('n', fontsize=14)
ax.set_ylabel('MSRE', fontsize=14)
ax.set_ylim(0,4)
ax.set_xlim(0,25)
ax.scatter(errs[:,0], errs[:,1], label='train');
ax.scatter(errs[:,0], errs[:,2], label='validation');
ax.axhline(0);
ax.legend();
ax.grid()
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

As you can see the error on the train set is high for  small $N$ and then falls sharply around $N=3$ (could you predict that looking at the data?) and continues to decrease, albeit slowly,  as $N$ increases, getting very close to zero around $N=25$.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: [hide-input]
---
fig, ax = plt.subplots()
xs = np.linspace(-np.pi, np.pi,400)
ax.set_ylim(-2,2)
ax.scatter(train_set[:,0],train_set[:,1], alpha=0.7, color='black', edgecolor="black");
ax.scatter(validation_set[:,0],validation_set[:,1], alpha=0.7, color='blue', edgecolor="blue");
for n in [1,2,3]:
    f = fit.fourier(train_set[:,0], train_set[:,1],n)
    ax.plot(xs,f(xs), linewidth=1,  label=f"{n}");
ax.legend();    
```

+++ {"slideshow": {"slide_type": "skip"}}

This is a typical behaviour. $N$ is a measure of the model *capacity*. Clearly with $N=0$ (constant term) or $N=1$ (trigonometric function with period equal to $2\pi$) we are  capable of representing the data. We say that the model *underfits*. As $N$ increases the model can represent a larger class of functions and the error gets smaller. Eventually the capacity is big enough to represent training data perfectly.

+++ {"slideshow": {"slide_type": "skip"}}

But this is not what we want! We have to look at the validation error which  follows same pattern up to $N\sim 10$ and then starts to grow "blowing up" around $N=20$. This unfortunatelly is also a typical behaviour. 
As the capacity of model increases it approximates the training data better and better. But it has **no incentive** to keep error low in the regions outside data points. We say that model *overfits* and does not generalise.

+++ {"slideshow": {"slide_type": "skip"}}

```{Attention}
That's why the minimisation of loss on the training set is **NOT** the goal of  training! This is a mean to an end, which is generalisation, that is low error on data that model was not trained with.
```

+++ {"slideshow": {"slide_type": "skip"}}

Actually if we wanted the network to be 100% accurate on the data we are traning it on we could just memorise them :)

+++ {"slideshow": {"slide_type": "skip"}}

This shows what happens as we fit using bigger values of $N$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: [hide-input]
---
fig, ax = plt.subplots()
xs = np.linspace(-np.pi, np.pi,400)
ax.set_ylim(-2,2)
ax.scatter(train_set[:,0],train_set[:,1], alpha=0.7, color='black', edgecolor="black");
ax.scatter(validation_set[:,0],validation_set[:,1], alpha=0.7, color='blue', edgecolor="blue");
for n in [8,16,20]:
    f = fit.fourier(train_set[:,0], train_set[:,1],n)
    ax.plot(xs,f(xs), linewidth=1,  label=f"{n}");
ax.legend();    
```

+++ {"slideshow": {"slide_type": "slide"}}

## Loss vs error

+++ {"slideshow": {"slide_type": "skip"}}

We reserve the name loss to functions that we will optimize (minimize) during the learning. By error we mean any  measure of deviance of the model prediction from the "groud-truth". So MSE is both loss function and an error, while NLL is not usually denoted as an error. On contrary, the percentage of misclasified examples is an error but not a loss function that could be used in gradient descent (see next notebook).

+++ {"slideshow": {"slide_type": "skip"}}

Loss and error are of course strongly correlated: lower loss usually implies lower error, but at certain point, especially in classification, this may be not the case: reducing loss does not have to decrease the error.

+++ {"slideshow": {"slide_type": "skip"}}

That said from the previous regression example we can infer that we have several kinds of error.

+++ {"slideshow": {"slide_type": "slide"}}

## Training error

+++ {"slideshow": {"slide_type": "skip"}}

This is the error achieved on the training set i.e. the set used for training the model. The training loss is used as the function to be optimized. As we have seen that does not guarantees us that the model will perform well on other data.

+++ {"slideshow": {"slide_type": "fragment"}}

## Validation error

+++ {"slideshow": {"slide_type": "skip"}}

This  is the  error on a set of data never seen by the learning model. It is an approximates of the generalisation error which itself is impossible to measure. This is the error that we would like to keep low.

+++ {"slideshow": {"slide_type": "fragment"}}

## Bayes error

+++ {"slideshow": {"slide_type": "skip"}}

Let's check the error on the function that was used to produce the data:

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
rys_true =  np.sin(2.188*train_set[:,0]+1)
residue = rys_true-train_set[:,1]
np.sqrt(np.mean(residue*residue) )
```

+++ {"slideshow": {"slide_type": "skip"}}

As you can see it is not zero. That is because the data is noisy and the mapping is not deterministic. This is an error that we will never get rid of: it is inherent in the problem. Sometimes it's called the Bayes error.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ys_true = np.sin(2.188*xs+1)
plt.scatter(train_set[:,0],train_set[:,1], alpha=0.7, color='none', edgecolor="black", label="training");
plt.scatter(validation_set[:,0],validation_set[:,1], alpha=0.7, color='none', edgecolor="blue", label="validation");
plt.legend()
plt.plot(xs,ys_true, c='grey');
```

+++ {"slideshow": {"slide_type": "slide"}}

## Generalisation

+++ {"slideshow": {"slide_type": "skip"}}

As we have stressed the real goal of training is to train a model that generalises well. However be warned that this is not a well posed problem. To illustrate this please look at the data below

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data_resonance = np.load('../data/resonance.npy')
ros = data_resonance[:,0]
rAs = data_resonance[:,1]
#plt.plot(os, As);
plt.scatter(ros,rAs);
```

+++ {"slideshow": {"slide_type": "skip"}}

The points seem to lie on a rather simple curve and we can try to approximate it be a polynomial of fourth degree

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p  = np.polyfit(ros, rAs,5)
xs = np.linspace(-1,1,100)
plt.scatter(ros,rAs);
plt.plot(xs,np.polyval(p, xs),'-r');
```

+++ {"slideshow": {"slide_type": "skip"}}

However the real function used to generate the data looks like this

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import norm
def func(x,a=1,mu=0, sigma=1):
    return np.sqrt(1-x*x)+a*norm(mu, sigma).pdf(x) 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
a=0.02
mu =0.35333
sigma = 0.005
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs = np.linspace(-1,1,1000)
plt.scatter(ros,rAs);
plt.plot(xs,np.polyval(p, xs),'-r')
plt.plot(xs,func(xs,a,mu, sigma),'-g');
```

+++ {"slideshow": {"slide_type": "skip"}}

I admit this is a contrived example but illustrates an important point. The real loss that we would like to optimize is this

+++ {"slideshow": {"slide_type": "slide"}}

## Real generalisation loss

+++ {"slideshow": {"slide_type": "-"}}

$$\int\!\text{d}\b{x} P(\b{x})\,\text{d}\b{y}P(\b{y}|\b{x})\loss\left(\b{y},\b{x},\tilde{P}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

In the above $P$ stands for the real distribution and $\tilde{P}$ for the model. Calculating  this error requires the knowledge of the real $P$. If we new it we would not be training :(

+++ {"slideshow": {"slide_type": "skip"}}

The training and validation losses are an approximation to this formula. But those approximation are good as long as we sample the whole space of $\b x$. Please note that in general $\b{X}$ is of very high dimensions.  Sampling the whole space is impossible and we can only sample some "regions of interest". In principle the data collected by definition samples the most probable regions, but we may have some hidden "peaks": low probability events with big influence on the model.

+++ {"slideshow": {"slide_type": "skip"}}

As an example please imagine that you are collecting data while you drive _e.g._  from a camera. How many times did a pedestrian jump in front of you? This is an example of "peak" in the data: a low probability event with very big consequences. And we know that pedestrains sometimes walk in front of a car. Sometimes we have no idea that some region of data distribution has been left over.

+++ {"slideshow": {"slide_type": "skip"}}

You can find many, not so dramatic, real examples of unsufficient sampling. Just search for "data biases".

+++ {"slideshow": {"slide_type": "skip"}}

This is not intended to sound pesimistic. We just want you to remember that with supervised learning **your model will be only as good as your data**.

+++ {"slideshow": {"slide_type": "slide"}}

## Takeaways

+++ {"slideshow": {"slide_type": "fragment"}}

Supervised learning consists of finding a model that best describes a mapping between sets of features and labels.

+++ {"slideshow": {"slide_type": "fragment"}}

This is achieved by minimizing a suitably choosen loss function

+++ {"slideshow": {"slide_type": "fragment"}}

The goal is to have low validation error, not only the training error.

+++ {"slideshow": {"slide_type": "fragment"}}

The model is only as good as the collected data is representative.

+++ {"slideshow": {"slide_type": "slide"}}

## What comes next

+++ {"slideshow": {"slide_type": "-"}}

The next notebook will go into technical details how to find the model that minimizes loss function.
