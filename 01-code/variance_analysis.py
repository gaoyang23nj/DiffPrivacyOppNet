# variance analysis for the uniform distribution of P
import matplotlib.pyplot as plt
import numpy as np

# K=9

def var_dodp(eps, K):
    # 从1到K-1
    tmp_res = 0.
    for i in range(1, K):
        tmp_res = tmp_res + i*i*(K-i)
    term3 = tmp_res
    term1 = 1 / np.power(K, 3)
    term2 = 1 / (np.exp(-eps)+(K-1))
    term4 = 1/(12*K*K)
    res = term3*term1*term2*2+term4
    return res

def var_lap(eps):
    term1 = 2 / np.power(eps, 3) * np.exp(-eps)
    term2 = 2 / np.power(eps, 2) * np.exp(-eps)
    term3 = 2 / np.power(eps, 3)
    term4 = 2 / np.power(eps, 2)
    res = term1 + term2 - term3 + term4
    return res

x = np.arange(0.1, 3.1, 0.1)
x_len = len(x)
y_lap = np.zeros(x_len)
y_dodp_K9 = np.zeros(x_len)
y_dodp_K5 = np.zeros(x_len)
y_dodp_K3 = np.zeros(x_len)
for i in range(x_len):
    y_lap[i] = np.log10(var_lap(x[i]))
    y_dodp_K9[i] = np.log10(var_dodp(x[i], 9))
    y_dodp_K5[i] = np.log10(var_dodp(x[i], 5))
    y_dodp_K3[i] = np.log10(var_dodp(x[i], 3))


plt.plot(x,y_lap,'b',marker = 'o',label='Lap')
plt.plot(x,y_dodp_K9,'r',marker = '^',label='DODP_K9')
plt.plot(x,y_dodp_K5,'k',marker = '+',label='DODP_K5')
plt.plot(x,y_dodp_K3,'g',marker = 'v',label='DODP_K3')
plt.legend(loc='upper right')
plt.show()