import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# part a 
w_truth = [0.3, -4.2, -1.6, 2.5, -3.2, 2.1]
noise_variance_truth = 1.8
n = 30

# part b
def gen_sample():
    x = np.empty((0, 6))
    noise = np.random.normal(0, np.sqrt(noise_variance_truth), size=n) 
    for i in range(n):
       x_i = np.random.uniform(-1, 1, 5)
       x_i = np.insert(x_i, 0, 1)
       x = np.vstack([x, x_i])

    y = x @ np.matrix(w_truth).T + np.matrix(noise).T
    return x, y

# part c and d
def lin_regression(x, y):
    w = np.linalg.inv(x.T @ x) @ x.T @ y
    w1 = np.linalg.lstsq(x, y)
    residuals = np.sum(np.array((y - (x @ w))) ** 2)
    noise_variance = residuals/(len(x) - 5 - 1)

    w_variance = noise_variance * np.linalg.inv(x.T @ x)
    var_first_coeff = w_variance[1][1]

    return w, noise_variance, var_first_coeff

hist_values = []
for i in range(10):
    x, y = gen_sample()
    w, noise_variance, var_w1 = lin_regression(x, y)
    t = (w[1, 0] - w_truth[1]) / np.sqrt(var_w1)
    hist_values.append(t)

x = np.linspace(-3, 3, 10000)
t_dist = stats.t.pdf(x, n - 6)

plt.hist(hist_values, color='blue', bins=70, density = True)
plt.plot(x, t_dist, 'r-', label='t-distribution with df=24')
plt.xlabel('t-values')
plt.ylabel('Frequency')
plt.title('Frequency of t-values vs. t-distribution with df=24')
plt.legend()
plt.show()







