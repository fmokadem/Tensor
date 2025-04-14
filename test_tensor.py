from tensor import Tensor

def main():
    # init
    print("--- Init ---")
    t1 = Tensor([1, 2, 3, 4], shape=(2, 2))
    t2 = Tensor([[5, 6], [7, 8]])
    print("t1:", t1)
    print("t2:", t2)
    print("-" * 20)

    # set/get Item
    print("--- Item Access ---")
    print("t1[1, 0]:", t1[1, 0])
    t1[1, 0] = 99
    print("t1[1, 0] = 99:", t1)
    t1d = Tensor([10, 20, 30])
    print("t1d:", t1d)
    print("t1d[1]:", t1d[1])
    t1d[1] = 25
    print("t1d[1] = 25:", t1d)
    print("-" * 20)

    # elt-wise Ops
    print("--- elt-wise ops ---")
    t1 = Tensor([1, 2, 99, 4], shape=(2, 2))
    t2 = Tensor([[5, 6], [7, 8]])
    t_add = t1 + t2
    t_mul = t1 * t2
    print("t1 + t2:", t_add)
    print("t1 * t2:", t_mul)
    print("-" * 20)

    # matmul
    print("--- MatMul ---")
    t_matmul = t1 @ t2
    print("t1 @ t2:", t_matmul)

    t3 = Tensor(list(range(6)), shape=(2, 3))
    t4 = Tensor(list(range(12)), shape=(3, 4))
    print("t3 (2x3):", t3)
    print("t4 (3x4):", t4)
    t3_t4_matmul = t3 @ t4
    print("t3 @ t4 (2x4):", t3_t4_matmul)
    print("-" * 20)

    # reshape
    print("--- Reshape ---")
    t5 = Tensor(list(range(12)), shape=(3, 4))
    print("t5 (3x4):", t5)

    t5.reshape((2, -1))
    print("t5 -> (2, -1):", t5)
    t5.reshape((-1, 3))
    print("t5 -> (-1, 3):", t5)
    t5.reshape((3, 4))
    print("t5 -> (3, 4):", t5)

    try:
        t5.reshape((-1, -1))
    except ValueError as e:
        print("Error (-1, -1):", e)

    try:
        t5.reshape((5, -1))
    except ValueError as e:
        print("Error (5, -1):", e)

    t_empty = Tensor([], shape=(0, 5))
    print("t_empty:", t_empty)
    t_empty.reshape((5, 0))
    print("t_empty -> (5, 0):", t_empty)
    try:
        t_empty.reshape((2, -1))
    except ValueError as e:
        print("Error empty -> (2, -1):", e)

    print("-" * 20)

    # infer shape
    print("--- infer shape ---")
    t_infer = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("Infer:", t_infer)
    print("Shape:", t_infer.shape)

    try:
        Tensor([[1, 2], [3]])
    except ValueError as e:
        print("Error inconsistent list:", e)

    try:
        Tensor([1, [2]])
    except ValueError as e:
        print("Error mixed types:", e)

    print("-" * 20)

    # broadcasting
    print("--- Broadcast ---")
    t_vec = Tensor([1, 2, 3])
    t_mat = Tensor([[10], [20]])
    t_row = Tensor([[100, 200]])
    t_col = Tensor([[10], [20], [30]])
    t_scalar_like = Tensor([5])
    scalar = 5

    print(f"{t_vec.shape} + {scalar}:", t_vec + scalar)
    print(f"{scalar} + {t_vec.shape}:", scalar + t_vec)
    print(f"{t_mat.shape} * {scalar}:", t_mat * scalar)
    print(f"{scalar} * {t_mat.shape}:", scalar * t_mat)
    print(f"{t_scalar_like.shape} * {scalar}:", t_scalar_like * scalar)
    print("-" * 10)

    print(f"{t_vec.shape} + {t_mat.shape}:")
    try: print(t_vec + t_mat)
    except ValueError as e: print("Error:", e)

    print(f"{t_vec.shape} * {t_mat.shape}:")
    try: print(t_vec * t_mat)
    except ValueError as e: print("Error:", e)
    print("-" * 10)

    print(f"{t_row.shape} + {t_col.shape}:")
    try: print(t_row + t_col)
    except ValueError as e: print("Error:", e)
    print("-" * 10)

    print(f"{t_scalar_like.shape} * {t_vec.shape}:")
    try: print(t_scalar_like * t_vec)
    except ValueError as e: print("Error:", e)
    print("-" * 10)

    print("Incompatible:")
    t_incompat1 = Tensor([1, 2])
    t_incompat2 = Tensor([10, 20, 30])
    try:
        print(f"{t_incompat1.shape} + {t_incompat2.shape}")
        t_incompat1 + t_incompat2
    except ValueError as e:
        print("Error:", e)

    t_incompat3 = Tensor([[1, 2], [3, 4]])
    t_incompat4 = Tensor([[1, 2, 3], [4, 5, 6]])
    try:
        print(f"{t_incompat3.shape} * {t_incompat4.shape}")
        t_incompat3 * t_incompat4
    except ValueError as e:
        print("Error:", e)

    print("-" * 20)


if __name__ == '__main__':
    main()
