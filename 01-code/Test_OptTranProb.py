import numpy as np
from cvxopt import matrix, solvers
import datetime

# sol=solvers.lp(c,G,h,A,b)
begin = datetime.datetime.now()

delta = 0.1
chooseP = np.arange(0, 1.+delta, delta)
# 变量个数
length = chooseP.size
tranP = np.zeros((length, length))
# tranP.size == 121

# 隐私强度 epsP
epsP = 4
W = np.exp(epsP)

# 先验概率 先设置为均匀的 之后从数据集中抽取统计
ProbP = np.ones((length, 1))/length
# ProbP[5, 0] = ProbP[5, 0] + 10
# ProbP = ProbP/sum(ProbP)

# ProbP = np.zeros((length, 1))
# ProbP[5, 0] = 1.

print('set s.t. ...')
# 由a c 导出 tranP转成行矩阵对应的位置 c*11+a
num_st1 = 0
list_st1 = []
# s.t.1 隐私保护带来的限制条件
for c in range(length):
    # 选取a,b
    for a in range(length):
        for b in range(a + 1, length):
            num_st1 = num_st1+1
            list_st1.append((c, a, b))
num_st1 = num_st1 * 2
print(num_st1)

num_st2 = 0
list_st2 = []
# s.t.2 prob的天然条件 [0,1]
for a in range(length):
    # 选取a,b
    for c in range(length):
        num_st2 = num_st2 + 1
        list_st2.append((c, a))
num_st2 = num_st2 * 2
print(num_st2)
num_neq_st = num_st1 + num_st2


list_st3 = []
num_st3 = 0
for a in range(length):
    # 选取a,b
    num_st3 = num_st3 + 1
    list_st3.append(a)
print(num_st3)

# 不等式条件，矩阵
label = 0
Gnp = np.zeros((num_neq_st, length * length))
hnp = np.zeros((num_neq_st, 1))
for ele in list_st1:
    (c, a, b) = ele
    # p_{a,c} - W p_{b,c} <=0
    Gnp[label, a * length + c] = 1.
    Gnp[label, b * length + c] = -W
    label = label + 1
    # -W p_{a,c} + p_{a,c} <=0
    Gnp[label, a * length + c] = -W
    Gnp[label, b * length + c] = 1.
    label = label + 1

for ele in list_st2:
    (c, a) = ele
    # -p_{a,c} <= 0
    Gnp[label, a * length + c] = -1.
    label = label + 1
    # p_{a, c} <= 1
    Gnp[label, a * length + c] = 1.
    hnp[label, 0] = 1.
    label = label + 1

print(label)

# 等式条件，矩阵
Anp = np.zeros((num_st3, length * length))
bnp = np.ones((num_st3, 1))
label = 0
for ele in list_st3:
    (a) = ele
    Anp[label, a * length:(a + 1) * length] = 1.
    label = label + 1

print(label)

def getCost(c,a):
    return (delta*c - delta*a)**2

print('set obj ...')
# Pr(A) * p_{a,c} * (从a变成c偏差)
cnp = np.zeros((length*length, 1))
label = 0
for a in range(length):
    for c in range(length):
        cnp[label, 0] = ProbP[a, 0] * getCost(c, a)
        label = label + 1

Gcx = matrix(Gnp)
hcx = matrix(hnp)
Acx = matrix(Anp)
bcx = matrix(bnp)

ccx = matrix(cnp)

end = datetime.datetime.now()
print(end-begin)

begin = datetime.datetime.now()

print('cal ...')
sol = solvers.lp(ccx, Gcx, hcx, Acx, bcx)
# print(sol['x'])

xSolution = np.array(sol['x'])
x = xSolution.reshape((length, length))
print(x)
print(np.sum(x, axis=1))
print('hi')

end = datetime.datetime.now()
print(end-begin)

