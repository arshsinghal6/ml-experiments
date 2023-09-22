import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('hw2_q2.csv', delimiter=",")

x = data[:, 0]
y = data[:, 1]

reordered_indices = np.random.permutation(len(x))
x_train = x[reordered_indices[:len(x)//2]]
x_test = x[reordered_indices[len(x)//2:]]
y_train = y[reordered_indices[:len(y)//2]]
y_test = y[reordered_indices[len(y)//2:]]

mses_train = []
mses_test = []

for j in range(1, 11):
    order = j
    design_matrix_train = np.empty((x_train.shape[0], order + 1))
    design_matrix_test = np.empty((x_test.shape[0], order + 1))

    for i in range(order + 1):
        column_train = x_train**i
        design_matrix_train[:, i] = column_train
        column_test = x_test**i
        design_matrix_test[:, i] = column_test
        

    w = np.linalg.lstsq(design_matrix_train, y_train)
    residuals_train = np.sum(np.array((y_train - (design_matrix_train @ w[0].T))) ** 2)
    mse_train = residuals_train/(len(x_train))
    mses_train.append(mse_train)

    residuals_test = np.sum(np.array((y_test - (design_matrix_test @ w[0].T))) ** 2)
    mse_test = residuals_test/(len(x_test))
    mses_test.append(mse_test)


plt.plot(range(1, 11), mses_train, color='blue', linestyle='-', label='Training MSE')
plt.plot(range(1, 11), mses_test, color='red', linestyle='-', label='Testing MSE')


plt.xlabel('Order of Polynomial')
plt.ylabel('Mean Squared Error')
plt.legend()

plt.grid(True)
plt.title('Order of Polynomial vs. MSE')
plt.show()
