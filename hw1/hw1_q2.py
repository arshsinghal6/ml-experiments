import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

matrix = np.genfromtxt('hw1_q2.csv', delimiter=',')
x = matrix[:, 0:5]
y = matrix[:, 5:]

ones = np.ones((x.shape[0], 1))
x_ones = np.hstack((ones, x))

def calc_coeff(x, y):
    w = np.linalg.inv(x.T @ x) @ x.T @ y
    return w

# part a
w = calc_coeff(x_ones, y)

# part b
y_hat = np.matrix([1, 0.4, 0.4, 0.4, 0.4, 0.4]) @ w

# part c
residuals = np.sum((y - (x_ones @ w)) ** 2)
print(residuals)
print(len(x))
noise_variance = residuals/(len(x) - 5 - 1)
print(noise_variance)

# part d
w_variance = noise_variance * np.linalg.inv(x_ones.T @ x_ones)
var_first_coeff = w_variance[1][1]




