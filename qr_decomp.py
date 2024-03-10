# Question number 17

# QR decomposition
import numpy as np

# making the A matrix
A = np.array([[5,-2],[-2,8]])

# Q, R decomposition using numpy.linalg.qr
Q, R = np.linalg.qr(A) 

# print the Q and R matrix
print("The Q matrix: \n", Q, "\n The R matix: \n",R)


# calculate eigenvalues using the decomposition
eigenval_qr = np.diag(R)
print("Eigenvalues of A using QR decomposition: ", eigenval_qr)


# Eigenvalues and eigenvectors produced by numpy.linalg.eigh
eigenval_eigh, eigenvec_eigh = np.linalg.eigh(A)
print("Eigenvalues of A using np.linalg.eigh: ", eigenval_eigh)
