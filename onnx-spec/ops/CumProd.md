# CumProd

First introduced in opset **26**

## Description

Performs cumulative product of the input elements along the given axis.
By default, it will do the product inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform product in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 2, 6]
exclusive=1
output = [1, 1, 2]
exclusive=0
reverse=1
output = [6, 6, 3]
exclusive=1
reverse=1
output = [6, 3, 1]
```

## Attributes

- **exclusive** (INT, optional): If set to 1 will return exclusive product in which the top element is not included. In other terms, if set to 1, the j-th output element would be the product of the first (j-1) elements. Otherwise, it would be the product of the first j elements.
- **reverse** (INT, optional): If set to 1 will perform the products in reverse direction.

## Inputs (2 - 2)

- **x** (T): An input tensor that is to be processed.
- **axis** (T2): A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means counting dimensions from the back.

## Outputs (1 - 1)

- **y** (T): Output tensor of the same type as 'x' with cumulative products of the x's elements

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to numeric tensors.
- **T2**: tensor(int32), tensor(int64)
  axis tensor can be int32 or int64 only
