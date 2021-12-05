
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy import array
import numpy as np
from scipy import linalg
from numpy import array
from numpy.linalg import eig
from scipy.linalg import hessenberg
import time
from numpy import *
#----------Question-1--------------

print("Question 1")

A = imread("selfie.PNG")
A=np.mean(A,-1) #averaging over 3rd dimension for 2D
A = A[1:720, 1:720]

img = plt.imshow(A)
img.set_cmap('gray')
plt.title("Original image")
plt.show()

#part 1: hessenberg
#print(hessenberg(A))

#Pure QR - part 2
error=np.zeros(100)
def Pure_QR(A, maxIter =100, tolerance = 1.e-5):
    val, vec = eig(A)
    for i in range(maxIter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        error[i] = (np.linalg.norm(np.diag(A - val), 2) / np.linalg.norm(val, 2))
    return A


start_time = time.time()
Pure_QR(A)
print("\n %s Pure QR \n" % (time.time() - start_time))
plt.plot(error)
plt.show()


#Two phase QR

B = hessenberg(A)
error=np.zeros(100)
def QR_w_Hess(A, maxIter =100, tolerance = 1.e-5):
    val, vec = eig(A)
    for i in range(maxIter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        error[i] = (np.linalg.norm(np.diag(A - val), 2) / np.linalg.norm(val, 2))

    return A

start_time = time.time()
QR_w_Hess(B)
print("\n %s - QR with Hess \n" % (time.time() - start_time))
plt.plot(error)
plt.show()

#From this we easily see that that that 2 Phase QR takes on average 2 seconds less to compute(for 100 iterations)
#and ther is stark difference in convergence rate, where 2 phase QR converges at around 10 and Pure  QR converges at 30 iterations


def reconstruct_w_lower_Eigen(M,rank):
    (U, Q) = np.linalg.eig(M)
    S = np.diag(U)
    i = 0
    QT = np.linalg.inv(Q)
    for k in rank:
        Rec_pic = Q[:, :k].dot(S[:k, :k]).dot(QT[:k, :])
        i += 1
        plt.figure(i+1)
        img = plt.imshow(np.real(Rec_pic))
        img.set_cmap('gray')
        plt.title('k=' + str(k))
        plt.show()

rank=[10,20,30,35,45,70,100,150]
reconstruct_w_lower_Eigen(A,rank)

#From this we can see that we only need 70 eigenmodes to have "decent" reconstruction - in the
#sense that the features are clearly idenifiable all the main charectoristics can be seen
#by 100 eigenmodes its nearly the same as the Origina; but just not as sharp(the switch in the background's feature
#are also distinguashable

#--------------Question 2--------------------

t = [1200, 1300, 1400,1500]
stu = []
logy = []
def Viscosity(A, E,t0):
    for i in range(len(t)):
        stu.append(-E/(t[i] - t0))
        logy.append(np.log(A)-stu[i])
    return logy
A_og = 50
E_og = 20
t0_og = 1000
y = Viscosity(A = A_og,E = E_og,t0 = t0_og)
print("\nY-values\n", y)

#the equations are
#ln(A)*t + y(t0) + ln(A)T0 - E = yt
# the parameter estimations or x would be [ln(A), t0, ln(A)T0 - E)
#for E since we cannot get a direct answer we will use back substitution
t = np.array([1200, 1300, 1400,1500])
A = np.zeros( (t.size, 3))
y = y
yt = []
for i in range(4):
    yt.append(y[i] * t[i])


for i in range(3):
    A[:,0] = [1200, 1300, 1400, 1500]
    A[:,1] = [4.012023005428146, 3.978689672094813, 3.962023005428146, 3.952023005428146]
    A[:, 2] = 1

print("\nMatrix A\n", A)

Q,R = np.linalg.qr(A)
m = Q.T @ yt
x = np.linalg.inv(R) @ m
print("\nParameters\n", x)

A_p = np.exp(x[0])
print("\nA-Reconstructed:", A_p)
T0_p = x[1]
print("\nT0-Reconstructed:",T0_p)
E_p = x[0]*x[1] + x[2]
print("\nEp-Reconstructed:",E_p)

error_A = np.linalg.norm(A_p - A_og)/np.linalg.norm(A_og)
print("\nError in A:", error_A)
error_t0 = np.linalg.norm(T0_p  - t0_og )/np.linalg.norm(t0_og)
print("\nError in T0:",error_t0)
error_E = np.linalg.norm(E_p  - E_og )/np.linalg.norm(E_og)
print("\nError in E:",error_E)

stu = []
logy = []
def Viscosity(A, E,t0):
    for i in range(len(t)):
        stu.append(-E/(t[i] - t0))
        logy.append(np.log(A)-stu[i])
    return logy
logy = []
y_new = np.array(Viscosity(A_p,E_p,T0_p))

noise = np.linalg.norm(y_new  - y )/np.linalg.norm(y)
print("\nnoise:",noise)


N = 4
max_val, min_val = 1, -1
range_size = (max_val - min_val)
s = np.random.rand(N) * range_size + min_val
y = np.array([4.012023005428146, 3.978689672094813, 3.962023005428146, 3.952023005428146])
for i in range(4):
    pert_y = y +  (noise * s[i])
print("\nPerturbed Y", pert_y)

t = np.array([1200, 1300, 1400,1500])
A = np.zeros( (t.size, 3))
pert_y = pert_y
pert_yt = []
for i in range(4):
    pert_yt.append(pert_y[i] * t[i])


for i in range(3):
    A[:,0] = [1200, 1300, 1400, 1500]
    A[:,1] = [4.01202301, 3.97868967, 3.96202301, 3.95202301]
    A[:, 2] = 1

print("\nMatrix A- Perturbed\n", A)

Q,R = np.linalg.qr(A)
m = Q.T @ pert_yt
x = np.linalg.inv(R) @ m
print("\nParameters\n", x)

A_pp = np.exp(x[0])
print("\nA-Reconstructed-pertubrd:", A_pp)
T0_pp = x[1]
print("\nT0-Reconstructed-pertubrd:",T0_pp)
E_pp = x[0]*x[1] + x[2]
print("\nEp-Reconstructed-pertubrd:",E_pp)

error_A = np.linalg.norm(A_pp - A_og)/np.linalg.norm(A_og)
print("\nError in A-pert:", error_A)
error_t0 = np.linalg.norm(T0_pp  - t0_og )/np.linalg.norm(t0_og)
print("\nError in T0-pert:",error_t0)
error_E = np.linalg.norm(E_pp  - E_og )/np.linalg.norm(E_og)
print("\nError in E-pert:",error_E)


#the relative errors are larger with the perturbed but still gives a pretty good estimate

#--------------Question 3--------------------

print("\nQuestion 3")

def QR_with_shifts(A, maxIter =500):
    n = A.shape[0]
    I = np.eye(n)
    QQ = np.eye(n)
    Q, R = np.linalg.qr(A)
    AK = [9]
    for i in range(maxIter):
        A  = A - AK*I
        Q, R  = np.linalg.qr(A)
        A = R @ Q + AK*I
        QQ = QQ @ Q
    return A, QQ


A = np.array([[1, 1, -2],[-1, 5, -1] , [0, -1, 9]])
A, Q = QR_with_shifts(A)
print("\nEigen-values", (np.diag(A)))
print("\nEigen-Vectors", Q)

A = np.array([[1, 1, -2],[-1, 5, -1] , [0, -1, 9]])
Values,Vecs = eig(A)
print("\nEigen-values Numpy\n", Values)
print("\nEigen-Vectors Numpy\n", Vecs)

#--------------------why practical QR or QR with shifts----------

#is backward stable
#converges cubicallly
#opeation count is O(4/3m^#)
#It is better than pure because in this it guranteed to converge to an upper triangular matrix
#and the eigen vectors generated depend upon which intial eigenvalue estimate you take
#in this case could have taken 1,5,9 and would the Q would then have the eigenvector to which it converges

#--------------Question 4--------------------
print("\nQuestion 4")


'''
a) realistic propotion for p,q,r
p = 0.3
q = 0.06
r = 0.08

P + E + B = 1
P(0.1) + B(0.06) + E(0.08) + = 0.08
p = 0.3

'''''


# Least-squares solution using SVD

A = np.array([[1,1,1], [0.1,0.06,0.08], [1,0,0]])
print('A={}' .format(A))

y= np.array([1,0.08,0.3])
print('y={}' .format(y))
#Pinv is the generalized inverse of a matrix using its singular-value decomposition (SVD)
x=np.linalg.pinv(A).dot(y.T)
print('x={}' .format(x))

#-----------part b----------------------
#c) x is the weigtages, A portfolio with 30% in Cryptocurrency, 30% in EQUITIES and 40% in BONDS.


#----------part d---------------
#why choose this algorithim
# it is extremely stable and matrices like could be rank deficent but this method is numerically stable and can handle rank
#deficiency other than that it usually not because computationally expensive but since we are dealing with a very small matrix it is insignificant
#and is a very robust approach



