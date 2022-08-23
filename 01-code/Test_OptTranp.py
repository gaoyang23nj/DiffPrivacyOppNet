import numpy as np
from cvxopt import matrix, solvers
import datetime

begin = datetime.datetime.now()

oneset = np.zeros(4)
length = len(oneset)
ll = []
choose = np.arange(0, 1.1, 0.2)

# 建立映射表
def getone(a, i):
    if i == a.size:
        if sum(a) == 1.:
            ll.append(a.copy())
        return
    for j in range(choose.size):
        a[i:] = 0
        a[i] = choose[j]
        if sum(a) > 1.:
            continue
        getone(a, i+1)


getone(oneset, 0)
print(len(ll))
length = len(ll)
tranP = np.zeros((length, length))

# 隐私强度 epsP
epsp = 0.1
w = np.exp(epsp)


# 先验概率 先设置为均匀的 之后从数据集中抽取统计
Probp = np.ones((length, 1))/length
# ProbP[5, 0] = ProbP[5, 0] + 10

print('set s.t. ...')
# 由a c 导出 tranp转成行矩阵对应的位置 c*length+a
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
    Gnp[label, b * length + c] = -w
    label = label + 1
    # -W p_{a,c} + p_{a,c} <=0
    Gnp[label, a * length + c] = -w
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

print('set obj ...')
# Pr(A) * p_{a,c} * (从a变成c偏差)
cnp = np.zeros((length*length, 1))
label = 0
for a in range(length):
    for c in range(length):
        diff = ll[c] - ll[a]
        cnp[label, 0] = Probp[a, 0] * sum(diff * diff)
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

xSolution = np.array(sol['x'])
x = xSolution.reshape((length, length))
print(x)
print(np.sum(x, axis=1))
print('hi')

end = datetime.datetime.now()
print(end-begin)
