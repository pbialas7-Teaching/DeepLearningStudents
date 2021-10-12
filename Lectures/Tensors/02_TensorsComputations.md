---
jupytext:
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

```{code-cell} ipython3
import numpy as np

import matplotlib.image as img
import matplotlib.pyplot as plt
%matplotlib inline
```

## Arithmetical operations

+++

So far all the operations on tensors did not change their elements. Time to change this. 

`numpy` supports all arithmetic operations and many functions in form of elementwise operations. For example  for multiplication

+++

$$ \forall_{ijk}\quad  z_{ijk}=x_{ijkl} \cdot y_{ijk}$$ 

+++

Please note several different ways used to time the commands below. 

```{code-cell} ipython3
x = np.random.normal(0,1,(3,5,2))
y = np.random.normal(0,1,(3,5,2))
%time z = x * y 
```

Please note that arithmetic operations create a new array. 

```{code-cell} ipython3
print(z.base)
```

This is equivalent to the following loop but faster

```{code-cell} ipython3
%%time
zloop = np.zeros((3,5,2))
for i in range(3):
    for j in range(5):
        for k in range(2):
            z[i,j,k]=x[i,j,k] * y[i,j,k]
```

Time difference in this case is not  very big, but for bigger arrays it can becomes very large:

```{code-cell} ipython3
import timeit
```

```{code-cell} ipython3
xb = np.random.normal(0,1,(30,50,20))
yb = np.random.normal(0,1,xb.shape)
start_time = timeit.default_timer()
zb = xb * yb 
end_time = timeit.default_timer()
elapsed_implicit = end_time-start_time
print("Took %s " % (elapsed_implicit,))
```

```{code-cell} ipython3
s = xb.shape
start_time = timeit.default_timer()
zbloop = np.empty_like(xb)
for i in range(s[0]):
    for j in range(s[1]):
        for k in range(s[2]):
            zbloop[i,j,k]=xb[i,j,k] * yb[i,j,k]
end_time = timeit.default_timer()            
elapsed_explicit = end_time-start_time
print("Took %fs which is %f times longer!" %(elapsed_explicit, elapsed_explicit/elapsed_implicit))
```

As you can see the difference is of the order of  hundreds!  That is the main reason you shoudl become fluent in tensor operations.

+++

Similarly we can apply a numpy function to every element of the tensor just by calling it with tensor argument:

+++

$$\forall_{ijk}\quad s_{ijk} = \sin(x_{ijk})$$ 

```{code-cell} ipython3
%time s = np.sin(x)
```

Please compare yourself the time of the execution of this operation to an explicit loop. 

+++

You can also use a scalar argument in tensor operations with the common sense interpretation:

```{code-cell} ipython3
grumpy = img.imread("GrumpyCat.jpg")
```

```{code-cell} ipython3
normalized_grumpy = grumpy/255
```

## Reduction

+++

Another common operations are  reductions. Those are the functions that can be applied to a subset of dimensions "reducing" them  to a single number. Using our freshly acquired skills in array manipulations we will build an array where every column will contain 1000 numbers draw from a different distribution. 

```{code-cell} ipython3
n = 1000
d1 = np.random.normal(0,1, n)
d2 = np.random.normal(1,0.5, n)
d3 =np.random.uniform(0,1,n)
data = np.stack((d1,d2,d3), axis=1)
data.shape
```

A common reduction operation is sum. Without any additional parameters sum sums all the element of the array

```{code-cell} ipython3
np.sum(data)
```

But we can specify the dimension(s) along which the reduction operation will be applied. 

```{code-cell} ipython3
row_sum = np.sum(data, axis=1)
row_sum.shape
```

As we can see the dimension 1 was "reduced". 

In the same way we can calculate the mean of every column:

```{code-cell} ipython3
np.mean(data, axis=0)
```

or standard deviation

```{code-cell} ipython3
np.std(data, axis=0)
```

We can reduce more then one dimension at the time. Below we calculate the mean value of each chanel in grumpy

```{code-cell} ipython3
np.mean(grumpy, axis=(0,1))
```

or max and min  values

```{code-cell} ipython3
np.min(grumpy, axis=(0,1))
```

```{code-cell} ipython3
np.max(grumpy, axis=(0,1))
```

## Contractions -- inner product

+++

Another class of operations are contraction. In contraction we sum over two dimensions of a product of two arrays. The examples include the dot (scalar) product

+++

$$ x\cdot y =\sum_{i} x_{i} \cdot y_{i}$$ 

+++

matrix vector multiplication:

+++

$$ v_j =\sum_{i} A_{ji} \cdot w_{i} \quad \forall_{i}$$ 

+++

and matrix multiplication

+++

$$   z_{ij}=\sum_{k} x_{ik} \cdot y_{kj} \quad \forall_{ij}$$ 

+++

`numpy` has special operators for both operations but we can use more general `inner` and `tensordot`. 

`inner` takes two arrays and contracts last dimensions in each of them. That means that the sizes of those dimensions must match. 

When both arrays are vectors this is normal scalar product:

```{code-cell} ipython3
x = np.random.normal(0,1,10)
y = np.ones_like(x)
np.inner(x,y)
```

When first is  a matrix and other a vector this is matrix vector multiplication:

```{code-cell} ipython3
m = np.asarray([[1,-1],[-1,1]])
v = np.array([0.5, -0.5])
np.inner(m,v)
```

Can you tell what the operation below is doing? 

```{code-cell} ipython3
w =np.asarray([0.3, 0.59, 0.11])
G = np.inner(grumpy,w)
```

Similar to `inner` is `dot`.Please check out its documentatio [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html). 

+++

Matrix multiplication requires contraction of last and first dimension. That's why it's more convenient to use `tensordot`

```{code-cell} ipython3
A = np.random.normal(0,1,(2,3))
B = np.random.normal(0,2,(3,4))
C = np.tensordot(A,B,1)
```

```{code-cell} ipython3
print(C.shape)
C
```

If we want to do matrix multiplication it's better to use 
`matmul` function which is described [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul). This function can be invokde using operator `@`

```{code-cell} ipython3
A@B
```

`tensordot(A,B,n)` is more general contracts last `n` dimensions of array `A` with first `n` dimensions of array `B`. 

```{code-cell} ipython3
A2 = np.random.normal(0,1,(4,3))
B2 = np.random.normal(0,2,(4,3))
C2 = np.tensordot(A2,B2,2)
print(C2.shape)
C2
```

In the above expression `C2` is calculated as: 
$$ C = \sum_{ij}A_{ij} B_{ij}$$

+++

We can also specify which dimensions will be contracted, by providing lists of dimensions in each array:

```{code-cell} ipython3
A3 = np.random.normal(0,1,(4,3))
B3 = np.random.normal(0,2,(3,4))
C3 = np.tensordot(A3,B3,[[0,1], [1,0]])
print(C3.shape)
C3
```

Which corresponds to 
$$ C = \sum_{ij}A_{ij} B_{ji}$$
which is more intuitive.

+++

You have a matrix 3x4 matrix W and a set of N 4-vectors in a form of array X of shape (N,4). How to produce an array of shape (N,3) where each row is the product of matrix W and corresponding row of X ? 

+++

## Outer product

+++

What happens when we request zero dimension contraction in `tensordot`? For two vectors this should correspond to
$$ z_{ij} = x_i \cdot y_j\quad \forall_{ij} $$
Let's check this. 

```{code-cell} ipython3
x = np.arange(4)
y = np.arange(5)
z  = np.tensordot(x,y,0)
print(z.shape)
z
```

This operation is called outer or tensor product. We can achieve same result with function `outer`

```{code-cell} ipython3
x = np.arange(4)
y = np.arange(5)
z  = np.outer(x,y)
print(z.shape)
z
```

However those two functions behave the same same only for 1-dimensional arrays. 

+++

## "Degenerate" dimensions

This a technical but a quite important point. It concerns dimensions with size one. While it may seem that such dimensions are spurious or "degenerate" they nevertheless change the dimensionality of the array and can impact the result of the operations.

+++

Let's start by creating a vector

```{code-cell} ipython3
vector = np.random.normal(0,1,(4,))
print(vector.shape)
vector
```

and reshape it to one row matrix 

```{code-cell} ipython3
vector_row = np.reshape(vector,(1,4))
print(vector_row.shape)
vector_row
```

and one column matrix:

```{code-cell} ipython3
vector_column = np.reshape(vector,(4,1))
print(vector_column.shape)
vector_column
```

Now make some experiments:

```{code-cell} ipython3
np.inner(vector, vector)
```

```{code-cell} ipython3
np.inner(vector_row, vector_row)
```

```{code-cell} ipython3
np.inner(vector_column, vector_column)
```

This actually the outer product:

```{code-cell} ipython3
np.outer(vector, vector)
```

The only two other combinations that will match are: 

```{code-cell} ipython3
np.inner(vector, vector_row)
```

```{code-cell} ipython3
np.inner(vector_row, vector)
```

Please explain the results of all the above operations. Write down using indices what each operation actually does.
