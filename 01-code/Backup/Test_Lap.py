from scipy.stats import laplace
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = laplace.stats(moments='mvsk')
x = np.linspace(laplace.ppf(0.01),
                laplace.ppf(0.99), 100)
ax.plot(x, laplace.pdf(x),
        'r-', lw=5, alpha=0.6, label='laplace pdf')
rv = laplace()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = laplace.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], laplace.cdf(vals))

r = laplace.rvs(size=1000)

ax.hist(r, density=True, histtype='stepfilled', alpha=0.05)
ax.legend(loc='best', frameon=False)
plt.show()
print('oooo')


r = laplace.rvs(size=1000)
plt.hist(r, bins = 50, density=True, histtype='stepfilled', alpha=0.9)