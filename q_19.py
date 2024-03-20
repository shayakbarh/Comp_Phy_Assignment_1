import numpy as np
import time

# Define the matrices
matrices = [
    np.array([[2, 1], [1, 0]]),
    np.array([[2, 1], [1,0], [0,1]]),
    np.array([[2,1], [-1,.1], [1,1], [2,-1]]),
    np.array([[1, 1, 0], [-1, 0, 1], [0, 1, -1], [1, 1, -1]]),
    np.array([[1, 1, 0], [-1, 0, 1], [0, 1, -1], [1, 1, -1]]),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1]])
]

# Compute SVD for each matrix
for idx, matrix in enumerate(matrices):
    print(f"Matrix {idx + 1}:")
    start_time = time.time()
    U, S, VT = np.linalg.svd(matrix)
    end_time = time.time()

    print(f"\n U matrix: {np.around(U, 3)},\n  \n S matrix: {np.around(S, 3)},\n  \n VT matrix: {np.around(VT, 3)}")
    print(" \n Time taken:", end_time - start_time, "seconds")
    print()

