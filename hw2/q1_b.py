import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

noise_variance = 1.5
noise_mean = 1
n = 20

def gen_sample_with_mean():
    noise = np.random.normal(noise_mean, np.sqrt(noise_variance), size=n)
    noise = noise - noise_mean
    noise_squared = noise ** 2
    rss = np.sum(noise_squared)
    return rss/(n - 1)

def gen_sample_without_mean():
    noise = np.random.normal(0, np.sqrt(noise_variance), size=n)
    noise_squared = noise ** 2
    rss = np.sum(noise_squared)
    return rss/n

hist_vals = []
for i in range(10000):  
    hist_vals.append(gen_sample_with_mean())
    # hist_vals.append(gen_sample_without_mean())


plt.hist(hist_vals, bins=30, density=True, alpha=0.6, color='b', label='Histogram')

x = np.linspace(0, 10, 1000)

# for with mean
pdf = chi2.pdf(x, n, scale=(noise_variance/(n - 1)))

# for without mean
# pdf = chi2.pdf(x, n, scale=(noise_variance/(n)))

plt.plot(x, pdf, 'r-', lw=2, label='Chi-squared PDF')
plt.xlabel('Estimated Noise Variance')
plt.ylabel('Probability Density')
plt.legend()

plt.title(f'Chi-squared Distribution (df={n}) and Histogram of Estimated Noise Variances')
plt.grid(True)
plt.show()