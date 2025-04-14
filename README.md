# Tensor

A basic tensor class (~numpy like)

## Usage
```python
from tensor import Tensor

t = Tensor([[1, 2], [3, 4]])
t = Tensor([1, 2, 3, 4], shape=(2, 2))
t = Tensor(shape=(3, 5))
```

Operations like elt-wise add (`+`), mult (`*`), matmul (`@`), indexing (`[]`), and reshaping (`.reshape()`) work (~) similarly to NumPy arrays, with broadcasting and shape infer support

See `test_tensor.py` for examples. 