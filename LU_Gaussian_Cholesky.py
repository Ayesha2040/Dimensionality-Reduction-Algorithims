import numpy as np
import scipy.linalg as la
from numpy import linalg as npla

#-------Question 1 ---------------------

print("Question 1")

print("\n***(a)***")


def LUFactorization(A):
    m = len(A)
    L = np.eye(m, m)
    U = A
    for k in range(m - 1):
        for row in range(k + 1, m):
            L[row, k] = U[row, k] / U[k, k]
            U[row, k:] = U[row, k:] - L[row, k] * U[k, k:]

    return L, U

def GaussianElimination_PartialPivoting(A, b, doPivoting = True):
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between"+
                     "A & b.", b.size, n)
    for k in range(n-1):
        if doPivoting:
            maxindex = abs(A[k:,k]).argmax() + k
            if A[maxindex, k] == 0:
                raise ValueError("Matrix is singular.")
            if maxindex != k:
                A[[k,maxindex]] = A[[maxindex, k]]
                b[[k,maxindex]] = b[[maxindex, k]]
        else:
            if A[k, k] == 0:
                raise ValueError("Pivot element is zero. Try setting doPricing to True.")
        for row in range(k+1, n) :
           multiplier = A[row,k]/A[k,k]
           A[row, k:] = A[row, k:] - multiplier*A[k, k:]
           b[row] = b[row] - multiplier*b[k]
    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
    return x

#Constructing A and b
A = np.array([[0, 1, 1, -2], [1, 2, -1, 0], [2, 4, 1, -3], [1, -4, -7, -1]])
b = np.array([[ -3],[2],[-2], [-19]])

#in order for LU decompostion to work the diagonal must not be 0, therefore we interchange the rows
A = np.array([[1.0, 2.0, -1.0, 0.0], [0.0, 1.0, 1.0, -2.0], [2.0, 4.0, 1.0, -3.0], [1.0, -4.0, -7.0, -1.0]])
b = np.array([[2.0], [-3.0], [-2.0], [-19.0]])

#finding least square x using Gaussian
X_gauss = GaussianElimination_PartialPivoting(np.copy(A), np.copy(b), doPivoting = False)
print("\n x using gaussian:", X_gauss)

#comparing with scipy
X_scipy = la.solve(np.copy(A), np.copy(b))
print("\n x using scipy:", X_scipy.T)

#perturbing A
t = 1e-5
perturb = 1 + t

A_pert = np.zeros((4, 4))
for i in range(len(A)):
    for j in range(len(A)):
        A_pert[i, j] = A[i, j] * perturb



print("\n***(b)***")

#exploring the stability with respect to L U
#without pertubation
L,U = LUFactorization(np.copy(A))
A_not_pert = np.matmul(L,U)
print("\n Unpeturbed A constructed with LU \n", A_not_pert)

L1,U1 = LUFactorization(np.copy(A_pert))
A_pert_LU = np.matmul(L,U)
print("\n peturbed A constructed with LU \n", A_pert_LU)

print("\nLeast squares found using A constructed using LU")
#finding the least squares of perturbed A
X_gauss_pertubed = GaussianElimination_PartialPivoting(np.copy(A_pert_LU), np.copy(b), doPivoting = True)
print("\n x using gaussian with unperturbed A:", X_gauss_pertubed)

'''
 x using gaussian with unperturbed A: [-1.  2.  1.  3.]

 x using gaussian with perturbed A: [-1.  2.  1.  3.]

Here we can see that LU factorization is highly stable, as it damps out the small approximation erroes as we can see gives us 
almost (because there may be variation in 0.0000000th decimal) the same result 

'''''

print("\nLeast squares Stability")

#finding the least squares of perturbed A
X_gauss_pertubed = GaussianElimination_PartialPivoting(np.copy(A_pert), np.copy(b), doPivoting = False)
print("\n x using gaussian with perturbed A:", X_gauss_pertubed)

X_scipy_pertubed = la.solve((A_pert), np.copy(b))
print("\n x using scipy with perturbed A:", X_scipy.T)

'''
From seeing x using SciPy and Gauss, we can conclude that this is Least squares is stable. if this was unstable, x would deviate to a large extent
its original value since the change but here we can see change in x is almost the same as the perturbation of A
'''''
print("\nQuestion 2")

print("\n***(a)***")

#Symmetrical positive definite matrix
np.random.seed(3)

def function_A(n):
    A = np.random.randint(n)
    A = 0.5 * (A + np.transpose(A))
    A = A + (n * np.identity(n))
    return A

A = function_A(4)
print("Positive Definite Matrix:\n", A)


def cholesky(A):
    R = A
    m = A.shape[0]
    for k in range(m):
        for j in range(k+1,m):
            R[j,j:m] =  R[j,j:m] - (np.dot(R[k,j:m],R[k,j]))/R[k,k]
        R[k,k:m] =  R[k,k:m]/np.sqrt(R[k,k])
        for i in reversed(range(m)):
         R[k,i:k] = 0
    return R


print("\n***(c)***")
#comparing solution obtained to numpy solution
L = npla.cholesky(A)
print("\ncholesky using numpy:\n", L) #this is a lower triangular matrix

R = cholesky(A)
print("\ncholesky using algorithim:\n", R) #this is upper triangular matrix

'''
For cholesky decomposition
            A = LL^T 
        Where L is lower triangular and L^T is upper triangular
'''''

A_numpy = np.matmul(L,L.T)  #L.T is just the upper triamgular of L
print("\nConstructed A using numpy cholesky \n", A_numpy)

A_cholesky = np.matmul(R.T,R)  #R.T is now lower triangular and R is upper triangular
print("\nConstructed A using textbook cholesky algorithim \n", A_cholesky)


print("\n***(d)***")

'''
Explanation of Least squares method
Ax = b 
We know that R*R = A     //where R* is the transpose of R
therefore  R*Rx = b
we need to find a y s.t R*y = b     // y = Rx     
                        R*_invR*y = (R*_inv)b    //R*_invR* = I
                                y =  (R*_inv)b
       
    As Rx = y
        (R_inv)Rx = (R_inv)y     
        x = (R_inv)y
        hence:
            x =  (R_inv)(R*_inv)b
            
    
'''''
def leastsqCholesky(A,b):
    R = cholesky(A)
    T = npla.inv(R)
    P = npla.inv(R.T)
    x = T @ P @ b
    return x

A = function_A(4)
b = np.array([[2], [-1], [1], [1]])

print("\nx computed using textbook alg\n", leastsqCholesky(np.copy(A),b))

x_s_numpy = npla.solve(np.copy(A), b)
print("\nx computed using numpy\n",x_s_numpy)

print("Question 3")


A = np.array([[0.5, 0.4], [-0.104, 1.1]])
#A square matrix is only diagnalizable if there exists an invertible P such that A = PDP^-1
vals,vecs = la.eig(A)
print("\nEigenvalues:\n", vals)

P = vecs
D = np.diag(vals)
inv_P = la.inv(P)

reconstructed_A = P @ D @ la.inv(P)
print("reconstructed A\n", reconstructed_A)

check_if_equal = np.allclose(reconstructed_A,A)
print("\ncheck if Reconstructed A and A are equal:", check_if_equal)





