import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

w_truth = [0.3, -4.2, -1.6, 2.5, -3.2, 2.1]
noise_variance_truth = 1.5
n = 20
df = 5

def gen_sample():
    x = np.empty((0, 6))
    noise = np.random.normal(0, np.sqrt(noise_variance_truth), size=n) 
    for i in range(n):
       x_i = np.random.uniform(-1, 1, 5)
       x_i = np.insert(x_i, 0, 1)
       x = np.vstack([x, x_i])

    y = x @ np.matrix(w_truth).T + np.matrix(noise).T
    return x, y

def lin_regression(x, y):
    w = np.linalg.lstsq(x, y)
    residuals = np.sum(np.array((y - (x @ w[0]))) ** 2)
    noise_variance = residuals/(n - df - 1)

    return noise_variance

hist_values = []
for i in range(10000):
    x, y = gen_sample()
    noise_variance = lin_regression(x, y)
    hist_values.append(noise_variance)

plt.hist(hist_values, bins=30, density=True, alpha=0.6, color='b', label='Estimate Noise Variances')

x = np.linspace(0, 10, 1000)
# pdf = chi2.pdf(x, n - df - 1, scale=(noise_variance_truth/(n - df - 1)))
pdf = chi2.pdf(x * (n - df - 1) / noise_variance_truth, n - df - 1) * (n - df - 1) / noise_variance_truth

plt.plot(x, pdf, 'r-', lw=2, label='Chi-squared PDF')

plt.xlabel('Estimated Noise Variance')
plt.ylabel('Probability Density')
plt.legend()

plt.title(f'Chi-squared Distribution (df={n - df - 1}) and Histogram of Estimated Noise Variances')
plt.grid(True)
plt.show()