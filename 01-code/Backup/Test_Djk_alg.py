import numpy as np
import sys

# 最大浮点数
# sys.float_info.max

n = 100
matrix = np.random.random((n,n))
for i in range(n):
    matrix[i,i]=0
print(matrix)

# 记录距离
dis = np.ones((n))*sys.float_info.max
# 记录上一个节点
pre = np.ones((n))*-1
# 标记是否已经访问过 1表示访问过
vis = np.ones((n))*0
# 本节点 上一个节点
# 路径权重
# 1. init
for i in range(n):
    dis[i] = matrix[0,i]
    pre[i] = 0
vis[0] = 1
count = 1
while count != n:
    # dis最小的那个下标
    tmp_idx = 0
    min = sys.float_info.max
    for i in range(n):
        if vis[i]!=1 and dis[i]<min:
            min = dis[i]
            tmp_idx = i
    vis[tmp_idx] = 1
    count = count + 1
    for i in range(n):
        if vis[i]!=1 and dis[tmp_idx]+matrix[tmp_idx][i] < dis[i]:
            dis[i] = dis[tmp_idx]+matrix[tmp_idx][i]
            pre[i] = tmp_idx

print(dis)
print(pre)
print(vis)
