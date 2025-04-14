import math
import numbers
import itertools


class Tensor():

    def _infer_shape_and_flatten(self, nested_list):
        """
        Infer shape from a nested array, return the 1D data and shape.
        ValueError if nedted list and shape inconsistent.
        """
        shape = []
        data = []

        def _recurse(sublist, current_level):

            # sublists must not be none or empty
            if not sublist:
                raise ValueError("can't infer shape from empty lists")

            # not a list, should be a scalar && stop recursing 
            if not isinstance(sublist, list):
                if not isinstance(sublist, numbers.Number):
                     raise TypeError(f"Tensor class only numeric types, this was not: {sublist}")
                data.append(sublist)
                return 1 # found a scalar

            if len(shape) <= current_level:
                shape.append(len(sublist))
            elif shape[current_level] != len(sublist):
                raise ValueError(f"Inconsistent list lengths in dimension {current_level}")

            is_scalar_level = False
            for i, item in enumerate(sublist):
                item_type = _recurse(item, current_level + 1)
                if i == 0 and item_type == 1:
                    is_scalar_level = True
                elif is_scalar_level and item_type != 1:
                    raise ValueError("Can't mix scalars and lists")
                elif not is_scalar_level and item_type == 1:
                     raise ValueError("Can't mix scalars and lists")

            return 0 # found a list 

        _recurse(nested_list, 0)

        if math.prod(shape) != len(data):
             raise ValueError("ndarray flattening error")

        return data, tuple(shape)


    def __init__(self, data=None, shape=None, dtype='fp32'):

        if data is None and shape is None:
            raise ValueError("Either data or shape needed for initializaiton ")

        inferred_shape = None
        is_nested = False

        if shape is None and isinstance(data, list):
             try:
                 if any(isinstance(i, list) for i in data):
                     is_nested = True
                     flat_data, inferred_shape = self._infer_shape_and_flatten(data)
                     data = flat_data
                     shape = inferred_shape
                 else: # Flat list
                     if not all(isinstance(x, numbers.Number) for x in data):
                          raise TypeError("Tensor class only stores numeric types")
                     shape = (len(data),)
             except (ValueError, TypeError) as e:
                 raise ValueError(f"Could not infer shape from initialization data: {e}")

        if shape is not None:
             if not isinstance(shape, tuple):
                 if isinstance(shape, int):
                     shape = (shape,)
                 else:
                    raise TypeError("Shape must be a tuple or an integer")

             expected_size = math.prod(shape)

             if data is None:
                 self.data = [0] * expected_size # TODO: Use dtype to initialize to fp, int, unint etc...
             elif isinstance(data, list):
                 if len(data) != expected_size:
                     origin = "inferred" if is_nested else "provided"
                     raise ValueError(f"Data of {origin} size {len(data)} doesn't match shape {shape}, expected size = {expected_size}")
                 if not all(isinstance(x, numbers.Number) for x in data):
                      raise TypeError("Data list contains non-numeric elements")
                 self.data = data
             else:
                 raise TypeError(f"Initialization data must be list of scalars or nested lists of scalatrs")

        else:
            raise ValueError("failed to process initialization shape or data")


        self.shape = shape
        self.dtype = dtype
        self.stride = tuple([
            math.prod(self.shape[i:]) for i in range(1, len(shape))
            ] + [1]) if len(shape) > 0 else () # Handle 0-dim tensors

    
    def _get_indices(self, key): 
        if not isinstance(key, tuple):
            if len(self.shape) == 1 and isinstance(key, int):
                key = (key,)
            else:
                raise TypeError(f"Tensor indices must be integers or tuples of integers")

        if len(key) != len(self.shape):
            raise IndexError(f"indices should be of length {len(self.shape)}, got {len(key)}")

        indices = []
        for i, idx in enumerate(key):
            if not isinstance(idx, int):
                raise TypeError(f" Indices must be integers")
            if not (0 <= idx < self.shape[i]):
                raise IndexError(f"Index {idx} is out of bounds for dimension {i} with size {self.shape[i]}")
            indices.append(idx)
        return indices
        
    def __getitem__(self, key):
        indices = self._get_indices(key)
        flat_index = sum(indices[i] * self.stride[i] for i in range(len(self.shape)))
        return self.data[flat_index]

    def __setitem__(self, key, item):
        indices = self._get_indices(key)
        flat_index = sum(indices[i] * self.stride[i] for i in range(len(self.shape)))
        self.data[flat_index] = item

    def reshape(self, shape):
        """
        Returns the *same* tensor data with a new shape.
        Similar to numpy.reshape, allows one dimension to be inferred using -1.
        Note: This modifies the tensor in-place and returns self.
        """
        original_elements = math.prod(self.shape) if self.shape else 1

        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple or an integer")

        minus_one_count = shape.count(-1)
        if minus_one_count > 1:
            raise ValueError("only one dimension as -1, meaning it will be inferred")
        elif minus_one_count == 1:
            if original_elements == 0:
                 raise ValueError("can't reshape empty tensor")

            known_dims_prod = math.prod(d for d in shape if d > 0)

            if known_dims_prod == 0:
                 if original_elements != 0:
                     raise ValueError(f"can't reshape array of size {original_elements} into shape {shape} containing zero unless the array is empty")
                 inferred_dim = 0
            elif original_elements % known_dims_prod != 0:
                raise ValueError(f"can't reshape array of size {original_elements} into shape {shape}")
            else:
                 inferred_dim = original_elements // known_dims_prod

            shape = tuple(inferred_dim if d == -1 else d for d in shape)

        new_elements = math.prod(shape) if shape else 1

        if len(shape) > 0 and any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError("new shape must tuple of non-negative integers")

        if new_elements != original_elements:
            raise ValueError(f"can't reshape array of size {original_elements} into shape {shape} (size {new_elements})")

        self.shape = shape
        self.stride = tuple([
            math.prod(self.shape[i:]) for i in range(1, len(shape))
            ] + [1]) if len(shape) > 0 else () # Handle 0-dim tensors

        return self

    def _calculate_broadcast_shape(self, other_shape):
        
        shape1 = list(self.shape)
        shape2 = list(other_shape)
        n1, n2 = len(shape1), len(shape2)

        if n1 < n2:
            shape1 = [1] * (n2 - n1) + shape1
        elif n2 < n1:
            shape2 = [1] * (n1 - n2) + shape2

        result_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2:
                result_shape.append(dim1)
            elif dim1 == 1:
                result_shape.append(dim2)
            elif dim2 == 1:
                result_shape.append(dim1)
            else:
                raise ValueError(f"could not broadcast shapes {self.shape} and {other_shape}")

        return tuple(result_shape)

    def _apply_elementwise_op(self, other, op_func):
        if isinstance(other, numbers.Number):
            result_data = [op_func(s, other) for s in self.data]
            return Tensor(result_data, self.shape, self.dtype)

        elif isinstance(other, Tensor):
            try:
                target_shape = self._calculate_broadcast_shape(other.shape)
            except ValueError as e:
                 raise e
            
            # elt wise op
            if self.shape == other.shape: 
                result_data = [op_func(s, o) for s, o in zip(self.data, other.data)]
                return Tensor(result_data, self.shape, self.dtype)

            target_size = math.prod(target_shape)
            if target_size == 0:
                return Tensor([], target_shape, self.dtype)

            # broadcasting
            result_data = [0] * target_size

            ndim_target = len(target_shape)
            shape1 = ([1] * (ndim_target - len(self.shape))) + list(self.shape)
            strides1 = ([0] * (ndim_target - len(self.shape))) + list(self.stride)
            shape2 = ([1] * (ndim_target - len(other.shape))) + list(other.shape)
            strides2 = ([0] * (ndim_target - len(other.shape))) + list(other.stride) 

            target_strides = [0] * ndim_target
            temp_stride = 1
            for i in range(ndim_target - 1, -1, -1):
                target_strides[i] = temp_stride
                temp_stride *= target_shape[i]

            for target_coords in itertools.product(*[range(d) for d in target_shape]):
                target_flat_index = sum(target_coords[i] * target_strides[i] for i in range(ndim_target))
                idx1 = sum((target_coords[i] if shape1[i] != 1 else 0) * strides1[i] for i in range(ndim_target))
                idx2 = sum((target_coords[i] if shape2[i] != 1 else 0) * strides2[i] for i in range(ndim_target))
                result_data[target_flat_index] = op_func(self.data[idx1], other.data[idx2])

            return Tensor(result_data, target_shape, self.dtype)

        else:
            return NotImplemented

    def __add__(self, other):
        return self._apply_elementwise_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
         return self._apply_elementwise_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented

        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(f"cant matmul {self.shape} and {other.shape}")

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"cant matmul {self.shape} @ {other.shape}")

        out_shape = (self.shape[0], other.shape[1])
        out_size = math.prod(out_shape)
        result_data = [0] * out_size

        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                sum_val = 0
                for k in range(self.shape[1]):
                    self_val = self[i, k]
                    other_val = other[k, j]
                    sum_val += self_val * other_val

                result_flat_index = i * out_shape[1] + j
                result_data[result_flat_index] = sum_val

        return Tensor(result_data, out_shape, self.dtype)

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, dtype='{self.dtype}')"

