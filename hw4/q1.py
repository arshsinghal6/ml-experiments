import numpy as np

mean = np.array([-1, -1, -1])
covariance = np.array([[1.0, 0, -0.2], [0, 0.8, 0.3], [-0.2, 0.3, 1.1]])
n = 5

samples = np.random.multivariate_normal(mean, covariance, n)

prior_cov = np.eye(3)
prior_mu = np.array([0, 0, 0])
posterior_mean = prior_cov @ np.linalg.inv(prior_cov + covariance / n) @ np.mean(samples, axis=0).T
posterior_mean = posterior_mean + 1/n * covariance @ np.linalg.inv(prior_cov + covariance / n) @ prior_mu.T
posterior_cov = prior_cov @ np.linalg.inv(prior_cov + covariance / n) @ (covariance / n)

print(posterior_mean)
print(posterior_cov)
print("\n")

curr_cov = np.eye(3)
curr_mean = np.array([0, 0, 0])
for i in range(5):
    post_mean = curr_cov @ np.linalg.inv(curr_cov + covariance) @ samples[i].T
    post_mean = post_mean + covariance @ np.linalg.inv(curr_cov + covariance) @ curr_mean.T
    post_cov = curr_cov @ np.linalg.inv(curr_cov + covariance) @ (covariance)

    curr_mean, curr_cov = post_mean, post_cov

print(curr_mean)
print(curr_cov)


# maxval = float('-inf')
# mu = None
# for i in np.linspace(curr_mean[0] - 0.005, curr_mean[0] + 0.005, 100):
#     for j in np.linspace(curr_mean[1] - 0.005, curr_mean[1] + 0.005, 100):
#         for k in np.linspace(curr_mean[2] - 0.005, curr_mean[2] + 0.005, 100):
#             tmu = np.array([i, j, k])
#             exponent = -0.5 * (tmu @ tmu.T + np.trace((samples - tmu) @ np.linalg.inv(covariance) @ (samples - tmu).T))
#             #print(exponent)
#             val = np.exp(exponent)
#             # print(val)
#             if val > maxval:
#                 maxval = val
#                 mu = tmu

# print(mu)
            
        