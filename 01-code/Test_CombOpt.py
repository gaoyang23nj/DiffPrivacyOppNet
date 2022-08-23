import numpy as np
from cvxopt import matrix, solvers
import datetime

# [TranP tranp]
begin = datetime.datetime.now()

def getTranBigP():
    chooseP = np.arange(0, 1.1, 0.1)
    # 变量个数
    length = chooseP.size
    tranP = np.zeros((length, length))
    # tranP.size == 121

    # 隐私强度 epsP
    epsP = 4
    W = np.exp(epsP)

    # 先验概率 先设置为均匀的 之后从数据集中抽取统计
    ProbP = np.ones((length, 1)) / length
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
                num_st1 = num_st1 + 1
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
        Gnp[label, a * 11 + c] = 1.
        Gnp[label, b * 11 + c] = -W
        label = label + 1
        # -W p_{a,c} + p_{a,c} <=0
        Gnp[label, a * 11 + c] = -W
        Gnp[label, b * 11 + c] = 1.
        label = label + 1

    for ele in list_st2:
        (c, a) = ele
        # -p_{a,c} <= 0
        Gnp[label, a * 11 + c] = -1.
        label = label + 1
        # p_{a, c} <= 1
        Gnp[label, a * 11 + c] = 1.
        hnp[label, 0] = 1.
        label = label + 1

    print(label)

    # 等式条件，矩阵
    Anp = np.zeros((num_st3, length * length))
    bnp = np.ones((num_st3, 1))
    label = 0
    for ele in list_st3:
        (a) = ele
        Anp[label, a * 11:(a + 1) * 11] = 1.
        label = label + 1

    print(label)
    print('end set s.t. for BigP')
    return Gnp, hnp, Anp, bnp


def getTranSmallp():
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
            getone(a, i + 1)

    getone(oneset, 0)
    print(len(ll))
    length = len(ll)
    tranP = np.zeros((length, length))

    # 隐私强度 epsP
    epsp = 0.1
    w = np.exp(epsp)

    # 先验概率 先设置为均匀的 之后从数据集中抽取统计
    Probp = np.ones((length, 1)) / length
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
                num_st1 = num_st1 + 1
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
    print('end set s.t. for Smallp')
    return Gnp, hnp, Anp, bnp, ll


G1, h1, A1, b1 = getTranBigP()
G2, h2, A2, b2, ll = getTranSmallp()

print(G1.shape)
print(G2.shape)
G = np.zeros((G1.shape[0] + G2.shape[0], G1.shape[1] + G2.shape[1]))
h = np.zeros((G1.shape[0] + G2.shape[0], 1))
G[:G1.shape[0], :G1.shape[1]] = G1
G[G1.shape[0]:, G1.shape[1]:] = G2

A = np.zeros((A1.shape[0] + A2.shape[0], A1.shape[1] + A2.shape[1]))
b = np.ones((A1.shape[0] + A2.shape[0], 1))
A[:A1.shape[0], :A1.shape[1]] = A1
A[A1.shape[0]:, A1.shape[1]:] = A2


# def PrioP(A, a, len1, len2):
#     return 1./(len1*len2)

def PrioP(BigA, Smalla, len1, len2):
    if BigA == 5 and Smalla == 60:
        return 1.
    else:
        return 0.


def getCost(BigA, BigC, Smalla, Smallc):
    tmp1 = (0.1*BigA-0.1*BigC)**2
    diff = ll[Smallc] - ll[Smalla]
    tmp2 = sum(diff * diff)
    return tmp1+tmp2

# QP问题
print('set QP')
lenQ = A1.shape[0]*A1.shape[0] + A2.shape[0]*A2.shape[0]
Q = np.zeros((lenQ, lenQ))
shifting = A1.shape[0]*A1.shape[0]
for BigA in range(A1.shape[0]):
    for BigC in range(A1.shape[0]):
        for Smalla in range(A2.shape[0]):
            for Smallc in range(A2.shape[0]):
                Pr = PrioP(BigA, Smalla, A1.shape[0], A2.shape[0])
                tmpCost = getCost(BigA, BigC, Smalla, Smallc)
                tmpx = BigA * A1.shape[0] + BigC
                tmpy = Smalla * A2.shape[0] + Smallc + shifting
                Q[tmpx, tmpy] = 0.5 * Pr * tmpCost
                Q[tmpy, tmpx] = 0.5 * Pr * tmpCost
Q = 2*Q
p = np.zeros((lenQ, 1))

Qcx = matrix(Q)
pcx = matrix(p)
Gcx = matrix(G)
hcx = matrix(h)
Acx = matrix(A)
bcx = matrix(b)

end = datetime.datetime.now()
print(end-begin)

begin = datetime.datetime.now()

print('cal...')
sol = solvers.qp(Qcx, pcx, Gcx, hcx, Acx, bcx)

print(sol['x'])

end = datetime.datetime.now()
print(end-begin)
