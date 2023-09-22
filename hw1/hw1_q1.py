import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_a = pd.read_csv('hw1_q1_d1.csv')
data_b = pd.read_csv('hw1_q1_d2.csv')
data_c = pd.read_csv('hw1_q1_d3.csv')
data_d = pd.read_csv('hw1_q1_d4.csv')

x_a = data_a['x']
y_a = data_a['y']

x_b = data_b['x']
y_b = data_b['y']

x_c = data_c['x']
y_c = data_c['y']

x_d = data_d['x']
y_d = data_d['y']


def calc_coefficients(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    numerator = np.sum((x - xbar) * (y - ybar))
    denominator = np.sum((x - xbar) ** 2)
    m = numerator/denominator
    b = ybar - m * xbar

    return (m, b)

def RSS(x, y, m, b):
    return np.sum((y - b - m*x) ** 2)

def noise_variance(n, rss):
    return (rss/(n - 1 - 1))

def plot(x, y, m, b):
    plt.scatter(x, y, label='Dataset #3')
    bestFit = m * x + b
    plt.plot(x, bestFit, color='red', label='Best Fit Line')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.show()

# change the x, y dataset accordingly 
m, b = calc_coefficients(x_a, y_a)
rss = RSS(x_a, y_a, m, b)
noise_var = noise_variance(len(x_a), rss)
plot(x_a, y_a, m, b)

w, residuals, rank, singular_values = np.linalg.lstsq(x_a, y_a, rcond=None)


# def calc_coeff(x, y):
#     x = np.matrix(x)
#     y = np.matrix(y)
#     ones = np.ones((x.shape[0], 1))
#     x_ones = np.hstack((ones, x))

#     w = np.linalg.inv(x_ones.T @ x_ones) @ x_ones.T @ y
#     return w








