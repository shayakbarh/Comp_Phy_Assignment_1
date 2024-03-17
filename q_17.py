# Question number 17

# QR decomposition

import numpy as np

# making the A matrix
A = np.array([[5,-2],[-2,8]])

# Q, R decomposition using numpy.linalg.qr
Q, R = np.linalg.qr(A) 

# print the Q and R matrix
print("\n The Q matrix: \n", Q, "\n The R matix: \n",R)



# calculate eigenvalues using the decomposition
def qr_decomp(A, iteration = 10, tol=10**-3):

    # initially setting V = I
    V = np.eye(len(A))

    for k in range(iteration):

        # QR decomposition of A
        Q, R = np.linalg.qr(A)

        A_k = np.dot(R, Q)

        V = np.dot(V, Q)    # columns of V are eigenvectors of A

        eigenval = np.diagonal(A_k)

        A = A_k

    return eigenval, V


# Eigenvalues and eigenvectors produced by QR decomposition
eigenval_qr, V = qr_decomp(A)  
print("\n Eigenvalues of A using QR decomposition: ", eigenval_qr)


# Eigenvalues and eigenvectors produced by numpy.linalg.eigh
eigenval_eigh, eigenvec_eigh = np.linalg.eigh(A)
print("\n Eigenvalues of A using np.linalg.eigh: ", eigenval_eigh, "\n")
