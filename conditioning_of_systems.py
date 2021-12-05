import numpy as np
from sympy import Symbol
import sympy as sym
from sympy.polys.polytools import poly_from_expr
from sympy import symbols,solve
import numpy.linalg as la
import math
from scipy.misc import derivative
import matplotlib.pyplot as plt

#------------------------------Question 1 ----------------------------

print("\nQuestion 1")

co_eff = np.poly1d([1,-3,3,1])
P_roots = co_eff.r
deri = np.polyder(co_eff)

val = [2,1,0]

for i in range(3):
    for j in range(3):
        a = abs(co_eff[i]*pow(P_roots[j],val[i]))
        b = abs(deri(P_roots[j]))
        condition_num = np.abs(a/b)
        if condition_num < 2:
            print("\nCoefficent",i+1,"Root",j+1,"\nCondition number:",condition_num,"\nWell Conditioned" )
        else:
            print("\nCoefficent", i+1, "Root", j+1, "\nCondition number:", condition_num, "\nill Conditioned")


#------------------------------Question 2 ----------------------------

print("\nQuestion 2")


x = Symbol('x')

W = 1
for i in range(1, 21):
    W = W * (x-i)

P,d = poly_from_expr(W.expand())
Coeff = P.all_coeffs()
Roots = np.sort(np.roots(Coeff))

np.random.seed(4)
s = np.random.normal(0, 1, 1)
perturb = np.zeros(21)
perturb[1]= 1e-10*s
perturbed_coeffs = Coeff + perturb
P_roots = np.sort(np.roots(perturbed_coeffs))


def f(x):
    return x**20 - 210*x**19 + 20615*x**18 - 1256850*x**17 + 53327946*x**16 - 1672280820*x**15 + 40171771630*x**14 - 756111184500*x**13 + 11310276995381*x**12 - 135585182899530*x**11 + 1307535010540395*x**10 - 10142299865511450*x**9 + 63030812099294896*x**8 - 311333643161390640*x**7 + 1206647803780373360*x**6 - 3599979517947607200*x**5 + 8037811822645051776*x**4 - 12870931245150988800*x**3 + 13803759753640704000*x**2 - 8752948036761600000*x + 2432902008176640000

con = []
for i in range(20):
    a = np.abs(Coeff[1]*math.pow(P_roots[i],18))
    b = np.abs(derivative(f,P_roots[i], dx = 1e-7))
    con.append((a/b))
    print("Root",i+1,"-", con[i])

print("\nRoot most sensitive to pertubation:", con.index(max(con))+1,"\n Condition number:",max(con))
print("\nRoot least sensitive to pertubation:", con.index(min(con))+1,"\n Condition number:",min(con))


np.random.seed(4)

real_parts = []
img_parts = []
for i in range(100):
    for j in range(20):
        s = np.random.normal()
        pert = 1 + (1e-10 * s)
        Coeff[j] = Coeff[j] * pert
        Per_root = np.roots(Coeff)
    real_parts.append(Per_root.real)
    img_parts.append(Per_root.imag)


y = np.zeros(20)
plt.scatter(real_parts, img_parts, s=10, c='black')
plt.scatter(Roots,y,s = 40, c='blue')
plt.xlabel("Real", fontsize=12)
plt.ylabel('Img', rotation=0, fontsize=12)
plt.title("Graphical representation of ill-conditioning", fontsize=12)
plt.show()






#--------------Question 3 ----------------------
#Plot the condition number of the matrix of this form as a function of its size, m, for m = 1, 2, ..., 100
print("\nQuestion 3")

np.random.seed(1)

Condition_num = []
for i in range(1,101):
    R = np.random.random((i,i))
    R *=np.tri(*R.shape)
    R= R.T
    Condition_num.append(la.cond(R))

plt.plot(Condition_num)
plt.title("Graphical representation conditioning of R", fontsize=14)
plt.show()

#For m = 50 define a ’true’ problem: y =rand(m, 1), b = R ∗ y.

n = 50
y = np.random.randn(n)
R = np.random.randn(50, 50) * np.tri(50).T
b = np.dot(R, y)  # True problem

#Now, solve this system by back substitution
def Back_sub(n,B):
    y_comp = np.zeros(n)
    for i in range(n - 1, -1, -1):
        t = B[i]
        for j in range(n - 1, i, -1):
            t -= y_comp[j] * R[i, j]

        y_comp[i] = t / R[i, i]
    return y_comp

y_new = Back_sub(50,b)
 # y computed using back subsitution

#compute the relative error of your solution
error_rel_y = la.norm(y_new-y)/la.norm(y)
print("Relative error of solution",'{:.2e}'.format(error_rel_y))

#What is the condition number of your problem
con_no = la.cond(R)
print("\nCondition number w respect to pertubation in R", '{:.2e}'.format(con_no))

est = la.norm(R)*la.norm(y)/la.norm(b)
con_b = con_no/est
print("\nCondition number w respect to pertubation in b",'{:.2e}'.format(con_b))

residual = la.norm(np.dot(R, y_new) - b)
d = residual/la.norm(b)

bound = con_no * d
print("\nBound with respect to R ", '{:.3e}'.format(bound))

bound2 = con_b * d
print("\nBound with respect to b",'{:.3e}'.format(bound2) )


'''''
let A be an arbitrary matrix. The closer κ(A) is to 1, the smaller thebound can become. On the other hand, an ill-conditioned A would allow for large
variations in the relative error. Here because the condition number is larger for R, the bound is also large and allows for larger variation in relative error 

'''
#Compare any components of the true and computed solution. How many digits were lost?

decimals_lost = [1e-4,1e-3,1e-2,1e-1,]
for i in range(3):
    res = np.allclose(y_new, y, error_rel_y,decimals_lost[i])
    if res == True:
        print("\nAnswer accurate up to", decimals_lost[i], "decimal places")
        break


digit_lost = math.log(la.cond(R),10)
print("\nEstimated_digits_lost:", round(digit_lost))

b2 = b + 1.e-11*np.random.randn(50)
b3 = b + 1.e-11*np.random.randn(50)

change_rel_b2 = la.norm(b2-b)
change_rel_b3 = la.norm(b3-b)
print("\nchange_relaltive_b2", '{:.2e}'.format(change_rel_b2))
print("\nchange_relaltive_b3", '{:.2e}'.format(change_rel_b3))


y2 = Back_sub(50,b2)
y3 = Back_sub(50,b3)

error_rel_y2 = la.norm(y2-y)/la.norm(y)
error_rel_y3 = la.norm(y3-y)/la.norm(y)
error_rel_y3_y2 = la.norm(y3-y2)/la.norm(y2)

print("\nerror_relaltive_y2", round(error_rel_y2,3))
print("\nerror_relaltive_y3", round(error_rel_y3,3))
print("\nerror_relaltive_y2_y3", round(error_rel_y3_y2,3))



