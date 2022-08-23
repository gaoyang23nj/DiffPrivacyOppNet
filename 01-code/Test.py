import numpy as np
a = np.zeros(4)
length = len(a)
ll = []
choose = np.arange(0, 1.1, 0.1)

a = np.zeros(4)
for j1 in range(choose.size):
    a[0:] = 0
    a[0] = choose[j1]
    for j2 in range(choose.size):
        a[1:] = 0
        a[1] = choose[j2]
        if sum(a) > 1.:
            continue
        for j3 in range(choose.size):
            a[2:] = 0
            a[2] = choose[j3]
            if sum(a) > 1.:
                continue
            for j4 in range(choose.size):
                a[3] = choose[j4]
                if sum(a) != 1.:
                    continue
                ll.append(a.copy())
print(ll)