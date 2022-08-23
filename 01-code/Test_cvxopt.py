from cvxopt import matrix, solvers

from cvxopt import matrix, solvers

c = matrix([2.0, 1.0])
G = matrix([[-1., -1., 0, 1], [1., -1., -1., -2.]])
h = matrix([1., -2., 0., 4.])
# A = matrix([1.], [2.])
A = matrix([1., 2.], (1,2))
b = matrix(3.5)
sol=solvers.lp(c,G,h,A,b)
print(sol['x'])



A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
c = matrix([ 2.0, 1.0 ])
sol=solvers.lp(c,A,b)
print(sol['x'])


Q = 2*matrix([ [2, .5], [.5, 1] ])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)
sol=solvers.qp(Q, p, G, h, A, b)
print(sol['x'])