import numpy as np
import matplotlib.pyplot as plt

matrix = np.genfromtxt('hw4_q2.csv', delimiter=',')
x = matrix[:, 0:3]
y = matrix[:, 3:]
noise_variance = 0.1


ones = np.ones((x.shape[0], 1))
x_ones = np.hstack((ones, x))

# part a
w = np.linalg.lstsq(x_ones, y)[0]
w_ridge = np.linalg.inv(x_ones.T @ x_ones + np.eye(4)) @ x_ones.T @ y 

print("weights ", w)
print("ridge ", w_ridge)

def bayesian_line_func(x1, weights):
    fixed_x = np.array([1, 0, 0.5, 0.5])
    intercept = fixed_x @ weights
    return intercept + weights[1] * x1


# part b
for alpha in [0.1, 1, 2, 5, 7, 10, 15]:
    # w_map = np.linalg.inv(x_ones.T @ x_ones + alpha * noise_variance * np.eye(4)) @ x_ones.T @ y 
    Sn = np.linalg.inv(alpha * np.eye(4) + x_ones.T @ x_ones / noise_variance)
    mu_n = Sn @ x_ones.T @ y / noise_variance

    x1 = np.linspace(0, 1, 1000)

    # part d
    if alpha == 0.1:
        samples = np.random.multivariate_normal(mu_n.T[0], Sn, 20)

        for weights in samples:
            y_sample = bayesian_line_func(x1, weights.T)
            plt.plot(x1, y_sample, 'r-')
        
        x1_orig = x[:, 0]
        plt.scatter(x1_orig, y)
        plt.xlabel('x1')
        plt.ylabel('y')
        plt.title('20 Samples Posterior Weight Distribution, Alpha=0.1')
        plt.show()

    y_bayes = bayesian_line_func(x1, mu_n)

    plt.plot(x1, y_bayes, label=f'alpha={alpha}')

    print(f'alpha = {alpha}, mu = {mu_n}')

# part c
x1_orig = x[:, 0]
plt.scatter(x1_orig, y)
plt.xlabel('x1')
plt.ylabel('y')
plt.title('MAP Linear Fits with Different Alpha Values')
plt.legend()
plt.show()