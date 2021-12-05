
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

#----------Question-1--------------
print("Question 1")

A = np.mat("3 -7 -2;-3 5 1; 6 -4 0" )

w, v = np.linalg.eig(A)
print( "Eigenvalues\n",w)
print( "\nEigenvectors\n",v)

U,s,Vt = np.linalg.svd(A)
print("\nU\n",U)
print("\nSigma Matrix\n", np.diag(s))
print('\nV^T\n',Vt)

Q,R  = np.linalg.qr(A)
print("\nQ\n",Q)
print("\nR\n",R)



#----------Question-2--------------


print("\nQuestion 2")

A = imread("selfie.PNG")
X=np.mean(A,-1) #averaging over 3rd dimension for 2D
img = plt.imshow(X)

img.set_cmap('gray')
plt.title("Original image")
plt.show()

#--Q2-(i)--

def Rank(M):
    print("shape", np.shape(M))
    (U, s, Vt) = np.linalg.svd(M, full_matrices=False)
    rank = len(s)
    return print("rank",rank)

Rank(X)
print("Rank = min(m,n), this is a full rank, as number of columns is 1242 and so is the rank hence all columns are linearly independent")

#--Q2-(ii)--

def Plot_sigma(M):
    (U, s, Vt) = np.linalg.svd(X, full_matrices=False)
    plt.subplot(221)
    plt.plot(s)
    plt.title("Singular values vs index")
    plt.xlabel("i")
    plt.ylabel('sigma_i')
    plt.subplot(222)
    plt.plot(s)
    plt.title("Singular values vs index")
    plt.xlabel("i")
    plt.ylabel('sigma_i')
    plt.xlim(right=10)
    plt.xlim(left=0)
    plt.show()


Plot_sigma(X)
#shows that the contribution of first few singular values is higher than the latter, the other sigma values have negligible contribution, nearly equal to 0
#and their corresponding singular vectors will scale accordingly
#in the second graph it is clear that the first 10 sigma values contain the most information and hence are most important, the downward trend is
#in sigma values is clearly visible

#--Q2-(iii & iv)--

def reconstruct_w_lower_Sigma(M,rank):
    (U, s, Vt) = np.linalg.svd(M, full_matrices=False)
    S = np.diag(s)
    i = 0
    for k in rank:
        Rec_pic = U[:, :k].dot(S[:k, :k]).dot(Vt[:k, :])
        i += 1
        plt.figure(i+1)
        img = plt.imshow(Rec_pic)
        img.set_cmap('gray')
        plt.title('k=' + str(k))
        plt.show()

rank=[10,20,30,35,45,70]
reconstruct_w_lower_Sigma(X,rank)
''''
From the values around 35 sigma values are enough to create a blurry image and 45 is enough to have almost all of my face's features
clearly distinguishable, by 70 sigma values a fairly sharp image is recreated
out of 1242 sigma values and columns only 45 were needed to create a clear enough image
'''''

#----------Question-3--------------


print("Question 3")


#-----------Classical Gramschdmit--------


def Classical_GramScdmit(A):
    V = np.zeros([len(A), len(A[0])])
    Q =  np.zeros([len(A), len(A[0])])
    r =  np.zeros([len(A[0]), len(A[0])])
    for j in range(len(A[0])):
        V[:, j] = A[:, j]
        for i in range(len(A)-1):
            r[i][j]  = np.dot((Q[:, i].T),A[:, j])
            V[:, j] = V[:, j] -  r[i][j]*Q[:, i]
        r[j][j] = la.norm(V[:, j], ord=2)
        Q[:, j] = V[:, j] / r[j][j]
    print("Q\n",Q)
    print("\nr\n",r)
    return print("\nReconstructed A\n", np.dot(Q,r))

#------------------------ Modified Gramschdmit-----------------------


def Modified_GramSchdmit(A):
    V = np.zeros([len(A), len(A[0])])
    Q =  np.zeros([len(A), len(A[0])])
    r =  np.zeros([len(A[0]), len(A[0])])
    for j in range(len(A[0])):
        V[:, j] = A[:, j]
        for i in range(len(A[0])):
            r[i][i] = la.norm(V[:, i], ord=2)
            Q[:, i] = V[:, i] / r[i][i]
            for j in range(i+2,len(A[0])):
                r[i][j]  = np.dot((Q[:, i].T),V[:, j])
                V[:, j] = V[:, j] -  r[i][j]*Q[:, i]


    print("Q\n", Q)
    print("\nr\n", r)
    return print("\nReconstructed A\n", np.dot(Q, r))

#------------------------ Numpy Gramschdmit-----------------------


def Numpy_Gram(A):
    Q, r = np.linalg.qr(A)
    print("Q\n", Q)
    print("\nr\n", r)
    return print("\nReconstructed A\n", np.dot(Q, r))

epsilon = 1.E-03
A = np.array([[1, 1, 1],
                [epsilon, 0, 0],
                [0, epsilon, 0 ],
                [0, 0,  epsilon ]])

print("Classical Gramschdmit")
Classical_GramScdmit(A)

print("\nModified Gramschdmit")
Modified_GramSchdmit(A)

print("\nNumpy Gramschdmit")
Numpy_Gram(A)

''''
Of all the three methods, Reconstruction of A using Modified GramSchmidt is closest to actual A
*Modified GM
**only one component is very far off from actual value
** 2.9591628e-20 instead of 0 in the 2nd row third column
**Classical GM
** all components in the second row and 3rd row third column are imprecise
*Numpy Gramschmidt
** 3 components are far off from actual value
The loss of orthogonality or Numerical Stability from eye balling is highest in Modifed GM

'''''