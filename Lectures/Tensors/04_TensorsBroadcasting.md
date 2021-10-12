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
```

## Broadcasting

+++

Elmentwise operation on arrays require the tensors to be of the same shape:

```{code-cell} ipython3
mat1 = np.ones((3,4))
mat2 = 2*np.ones_like(mat1)
```

```{code-cell} ipython3
mat1+mat2
```

However creating a whole array when we just want to add same  number to all elemnts like above would be tedious. That's why `numpy` provides a much more convenient way:

```{code-cell} ipython3
mat1 + 2
```

This is in example of *broadcasting*. Value 2 in this case is broadcast along all missing dimensions. This notion extends beyond the scalars.

Let's try to add a vector. Experimenting with the size you will find out that only size (except for 1) that does not give an error is 4:

```{code-cell} ipython3
mat1 + np.arange(4)
```

and the result is to add this vector to all rows. This is also convenient. Bradcasting is a very powerful technique but can be easy to get wrong. 
That's why sooner or later you should invest some time and familiarize yourself with numpy [broadcast semantics](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html). 

+++

For broadcast to work the dimensions of the two arrays must be compatible. The dimensions are matched from the last dimenion. Two dimensions are compatible if 
  * they have the same size
  * one of them has size one

+++

In our example 

    3 x 4 
        4 
        
The missing dimensions are  assumed to have size one so this is equivalent to   

    3 x 4 
    1 x 4 

When one dimension has size one, the size of the other one is taken as te size of this dimension in the result. So the result of this operation is 

    3 x 4


We could say that the vector od size 4 was first reshaped  to size (1,4) and then its copies were concatenated along the first dimension to form a (3,4) array that was finally added to the original.  

+++

So how we add a vector to every column? 

+++

Broadcasting can have suprising effects

```{code-cell} ipython3
v1 = np.ones(5)
v2 = np.arange(5)
```

This works as expected:

```{code-cell} ipython3
v1+v2
```

However reshaping the first vector to column vector, will produce something more akin to tensor product.  

```{code-cell} ipython3
v1.reshape(-1,1) + v2 
```

This happens because according to bradcasting rules, the original shapes

      5 x 1
          5
          
are broadcasted to 

      5 x 5   

And this can get compouned in higher dimensions. 

```{code-cell} ipython3
(mat1.reshape(1,3,1,4)+ mat1.reshape(3,1,4,1)).shape
```

As mentioned before broadcasting is a very powerful technique. However because it is applied automatically it can  allow some errors to go unnnotices. You should always check if the results agree with your intentions. 

```{code-cell} ipython3

```
