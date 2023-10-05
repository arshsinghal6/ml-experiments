import numpy as np
import matplotlib.pyplot as plt

noise_variance = 0.5
k = 10
n = 1000
u = np.random.normal(0, np.sqrt(noise_variance), size=n)
x = np.random.uniform(-1, 1, n)
y = np.sin(2 * np.pi * x) + u

def design_matrix(x, order):
    design_matrix = np.empty((x.shape[0], order + 1))
    for i in range(order + 1):
        column = x**i
        design_matrix[:, i] = column
    return design_matrix


mses_test = []
variances = []
squared_biases = []

for j in range(1, 10):
    order = j

    w_models = []
    mse_sum = 0
    
    for foldNum in range(1, k + 1):
        foldSize = n / k
        startIndex = int(foldSize * (foldNum - 1))
        endIndex = int(foldSize * foldNum)

        x_test = x[startIndex:endIndex]
        x_train = np.concatenate([x[:startIndex], x[endIndex:]])
        y_test = y[startIndex:endIndex]
        y_train = np.concatenate([y[:startIndex], y[endIndex:]])

        design_matrix_train = design_matrix(x_train, order)
        design_matrix_test = design_matrix(x_test, order)
            
        w = np.linalg.lstsq(design_matrix_train, y_train)
        w_models.append(w[0])
        y_pred = design_matrix_test @ w[0].T
        residuals_test = np.sum(np.array((y_test - y_pred)) ** 2)
        mse_test = residuals_test/(len(x_test))
        mse_sum += mse_test
    
    mse = mse_sum / k
    mses_test.append(mse)
    w_models = np.reshape(w_models, (len(w_models), len(w_models[0])))
    avg_model = np.average(w_models, axis=0)

    eval_x = np.linspace(-1, 1, n)
    eval_y = np.sin(2 * np.pi * eval_x)
    design_x_eval = design_matrix(eval_x, order)
    y_preds = w_models @ design_x_eval.T
    avg_y_pred = np.average(y_preds, axis=0)
    diff_y = (y_preds - avg_y_pred)**2
    
    variance = np.average(np.average(diff_y, axis=1))
    squared_bias = np.average((eval_y - avg_y_pred)**2)

    variances.append(variance)
    squared_biases.append(squared_bias)
    
sum_var_bias = np.array(variances) + np.array(squared_biases)

plt.plot(range(1, 10), variances, color='blue', linestyle='-', label='Variance')
plt.plot(range(1, 10), squared_biases, color='red', linestyle='-', label='Bias^2')
plt.plot(range(1, 10), mses_test, color='black', linestyle='-', label='MSE')
plt.plot(range(1, 10), sum_var_bias, color='green', linestyle='-', label='Variance + Bias^2')


plt.xlabel('Order of Polynomial')
plt.ylabel('Error')
plt.legend()

plt.grid(True)
plt.title('Order of Polynomial vs. Error')
plt.show()


