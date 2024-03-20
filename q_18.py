# Question number 18

import numpy as np

# the given matrix 
matrix = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])

# choosing the initial guess
y_0 = np.ones(len(matrix))

# Empty array to store eigenvalues after every iterations
eigval_array = []


# Function to calculate dominant eigenvalue of matrix A with initial guess x_0 and tolerance 1%

def power_method(A, x_0, iteration = 100, tol = 10**-2):

    for k in range(iteration):

        # making (A^k x_0) matrix
        x = np.dot(A, x_0)

        # making the (A^k+1 x_0) matix
        Ax = np.dot(A,x)       

        # calculate the dominat eigenvalue
        eigval = (np.dot(Ax.T, x))/(np.dot(x.T, x))

        # store the eigenvalue in that empty array
        eigval_array.append(eigval)

        # eigenvector of the dominat eigenvalue
        eigvec = x/np.linalg.norm(x)

        # check the tolerance
        if k > 0:
            if np.abs(eigval_array[k-1] - eigval_array[k]) < tol:
                return eigval,eigvec, k

        x_0 = x

    return eigval,eigvec

dom_eig, eig_vec, no_of_iterations = power_method(matrix,y_0)

print("\n The dominat eigen value: ", dom_eig,"\n The corresponding eigen vector: ", eig_vec,"\n The required no. of iterations to reach tolerance: ", no_of_iterations,"\n")