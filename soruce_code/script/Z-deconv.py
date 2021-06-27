import numpy as np
from scipy.optimize import minimize, LinearConstraint
#from scipy import optimize
#from scipy.optimize import NonlinearConstraint
import scipy as sp

# t = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
t = np.array([[1,2,255,4],[113,84,5,87],[9,110,171,212],[3,54,15,16]])
print(t)
print("shape of t: ",t.shape)

f = np.array([[2,1,0],[0,1,2],[1,0,2]])

print(f)
print("shape of f: ",f.shape)

#kernel = 3*3 , padding =0 , stride =1

a = t.shape[0] - f.shape[0] + 1
b = t.shape[0] - f.shape[1] + 1

result = []

for rn in range(a):
    for cn in range(b):
        result1 = t[rn:rn+f.shape[0], cn:cn+f.shape[1]]*f
        result.append(np.sum(result1))


result = np.array(result).reshape(a,b)

print("shape of result")
print(result)


def function(x):
    return 2*x[0] + 1*x[1] + 0*x[2] + 0*x[3] + 1*x[4] + 2*x[5] + 1*x[6] + 0*x[7] + 2*x[8] - 449

# 얘는 제약 조건
def eq_constraint(x):
    return 2*x[0] + 1*x[1] + 0*x[2] + 0*x[3] + 1*x[4] + 2*x[5] + 1*x[6] + 0*x[7] + 2*x[8] - 449


Init_Point = np.array([1.,2.,255.,113.,84.,5.,9.,110.,171.])

op = sp.optimize.fmin_slsqp(function, np.array([1.,2.,3.,5.,6.,7.,9.,10.,12.]), eqcons=[function])
print(op)

