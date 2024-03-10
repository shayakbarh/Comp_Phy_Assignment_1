# Question number 11

# Solving system of linear equations using np.linalg.solve
import numpy as np

# solving the 1st system:  A1.x_1 = b1

A1 = np.array([[3,-1,1],[3,6,2],[3,3,7]])
b1 = np.array([1,0,4])
x_1 = np.linalg.solve(A1, b1)
# print the solution for 1st system
print("Solution of the 1st system: ", x_1)


# solving the 2nd system: A2.x_2 = b2

A2 = np.array([[10,-1,0],[-1,10,-2],[0,-2,10]])
b2 = np.array([9,7,6])
x_2 = np.linalg.solve(A2, b2)
# print the solution for 2nd system
print("Solution of the 2nd system: ", x_2)



# solving the 3rd system: A3.x_3 = b3

A3 = np.array([[10,5,0,0], [5,10,-4,0], [0,-4,8,-1],[0,0,-1,5]])
b3 = np.array([6,25,-11,-11])
x_3 = np.linalg.solve(A3, b3)
# print the solution for 3rd system
print("Solution of the 3rd system: ", x_3)


# solving the 4th system: A4.x_4 = b4

A4 = np.array([[4,1,1,0,1], [-1,-3,1,1,0], [2,1,5,-1,-1], [-1,-1,-1,4,0], [0,2,-1,1,4]])
B4 = np.array([6,6,6,6,6])
x_4 = np.linalg.solve(A4, B4)
print("Solution of the 4th system: ", x_4)


