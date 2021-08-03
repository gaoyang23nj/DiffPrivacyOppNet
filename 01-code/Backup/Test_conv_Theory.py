import numpy as np

def __research_next(targetmatrix, matrix_size, tier, begin, tmp_prob):
    # 达到最后一层
    if tier == matrix_size[0]:
        return tmp_prob
    tmp_res = 0.
    for i in range(begin, matrix_size[1]):
        tmp_prob = tmp_prob * targetmatrix[tier, i]
        tmp_res = tmp_res + __research_next(targetmatrix, matrix_size, tier+1, i+1, tmp_prob)
    return tmp_res

sum_prob = 0.
tmp_prob = 1.
targetmatrix = np.ones((3, 10))
matrix_size = targetmatrix.shape

res = __research_next(targetmatrix, matrix_size, 0, 0, tmp_prob)
print(res)

