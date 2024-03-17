# Question 16

import numpy as np

matrix = np.array([[0.2, 0.1, 1, 1, 0], [0.1, 4, -1, 1, -1], [1, -1, 60, 0, -2], [1, 1, 0, 8, 4], [0, -1, -2, 4, 700]])

B = np.array([1, 2, 3, 4, 5])

# initial guess
y_0 = np.zeros(len(matrix))

# known solution
given_solution = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])

# Jacobi method
def jacobi_method(A, b, x_0, sol, max_iterations = 100, tol = 10**-2):

    x = np.zeros(len(x_0))
    x_prev = x_0

    for k in range(1, max_iterations):
        
        for i in range(len(A)):

            s2 = 0
            s3 = 0
            s1 = 0

            s1 = s1 +  b[i]/A[i][i] 

            for j1 in range(i):
                s2 = s2 - (A[i][j1]*x_prev[j1])/ A[i][i]
            

            for j2 in range(i+1,len(A)):
                s3 = s3 - (A[i][j2]*x_prev[j2] )/ A[i][i]           

            x[i] = s1 + s2 + s3 

        # setting up the tolerence
        if np.abs(np.linalg.norm(x - sol)) < tol:
            return x, k
        
        x_prev = x.copy()

    return x  

# Print the result from Jacobi method
jacobi_solution, jacobi_iteration = jacobi_method(matrix, B, y_0, given_solution)
print(f"\n Solution using Jacobi method: {jacobi_solution} \n and required iteration: {jacobi_iteration}")



# ------------------------------------------------------------------------
# Gauss-Seidel method( GS method )

def gauss_seidel_method(A, b, x_0, sol, max_iterations = 100, tol = 10**-2):

    x = np.zeros(len(x_0))
    x_prev = x_0

    for k in range(1, max_iterations):
        
        for i in range(len(A)):

            s2 = 0
            s3 = 0
            s1 = 0

            s1 = s1 +  b[i]/A[i][i] 

            for j1 in range(i):
                s2 = s2 - (A[i][j1]*x[j1])/ A[i][i]
            

            for j2 in range(i+1,len(A)):
                s3 = s3 - (A[i][j2]*x_prev[j2] )/ A[i][i]
            

            x[i] = s1 + s2 + s3 


        # setting up the tolerence
        if np.abs(np.linalg.norm(x - sol)) < tol:
            return x, k
        
        x_prev = x.copy()

    return x

# Print the result from GS method
gs_solution, gs_iteration = gauss_seidel_method(matrix, B, y_0, given_solution)
print(f"\n Solution using Gauss-Seidel method: {gs_solution} \n and required iteration: {gs_iteration}")


# ----------------------------------------------------------------------------
# Relaxation method
def relaxation_method(A, b, x_0, w, sol, max_iterations = 100, tol = 10**-2):
    x = np.zeros(len(x_0))
    x_prev = x_0

    for k in range(1,max_iterations):
        
        for i in range(len(A)):

            s2 = 0
            s3 = 0
            
            s1 = (1 - w) * x[i] + (w * b[i])/A[i][i]
            
            for j1 in range(i):
                s2 = s2 - A[i][j1]*x[j1] * w / A[i][i]
            

            for j2 in range(i+1,len(A)):
                s3 = s3 - A[i][j2]*x_prev[j2] * w / A[i][i]

            x[i] = s1 + s2 + s3
             
        

        if np.abs(np.linalg.norm(x - sol)) < tol:
            return x, k
        
        x_prev = x.copy()

    return x  

# Print the result from Relaxation method
relax_solution, relax_iteration = relaxation_method(matrix, B, y_0, 1.25, given_solution)
print(f"\n Solution using Relaxation method: {relax_solution} \n and required iteration: {relax_iteration}")



# --------------------------------------------------------
# Conjugate Gradient method (CG method)
def conjugate_gradient_method(A, b, sol, max_iterations = 100000, tol = 10**-2):

    x = np.zeros(len(A))

    for k in range(1, max_iterations):

        r = b - np.dot(A, x)

        Ar = np.dot(A, r)

        t1 = np.dot(r.T, r)/ np.dot(r.T, Ar)

        x = x + t1*r

        if np.abs(np.linalg.norm(x - sol)) < tol:
            return x, k

    return x, k

# Print the result from Conjugate Gradient method
cg_solution, cg_iteration = conjugate_gradient_method(matrix, B, given_solution)
print(f"\n Solution using Conjugate Gradient method: {cg_solution} \n and iteration: {cg_iteration}")





