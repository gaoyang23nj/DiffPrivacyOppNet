import numpy as np

oneset = np.zeros(4)
length = len(oneset)
ll = []
choose = np.arange(0, 1.1, 0.2)


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
# print(ll)