import numpy as np

path_set = np.random.random((2,6))
num_compare = 3
tmplist = [(0., -1)]*num_compare
for tunple_index in range(path_set.shape[1]):
    (onepath, onevalue) = (path_set[0, tunple_index], path_set[1, tunple_index])
    insert_index = -1
    for i in range(num_compare):
        if tmplist[i][0] < onevalue:
            tmplist.insert(i, (onevalue, tunple_index))
            tmplist.pop()
            break
print(path_set)
print(tmplist)
