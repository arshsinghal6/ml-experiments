import numpy as np
import matplotlib.pyplot as plt

def design_matrix(x, order):
    design_matrix = np.empty((x.shape[0], order + 1))
    for i in range(order + 1):
        column = x**i
        design_matrix[:, i] = column
    return design_matrix

alphas_order = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
alphas_lambdas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
lambdas = [1, 10, 20, 50, 100, 500, 1000, 10000, 50000, 100000]
x = np.linspace(-10, 10, 1000)
kernel_funcs_ridge = []    

# change range of lambdas based on what is investigated
for lmd in lambdas: 
    # change range of order based on what is investigated 
    for order in range(4, 5):
        phi = design_matrix(x, order)
        S_inv = np.linalg.inv(phi.T @ phi)
        S_inv_ridge = np.linalg.inv(phi.T @ phi + lmd * np.eye(order + 1))
        kernel_funcs = []


        test_x = [-7, -3, -1, 2, 4, 8]
        for val in test_x:
            phi_val = design_matrix(np.array([val]), order)
            kernel_func = phi_val @ S_inv @ phi.T
            kernel_func_ridge = phi_val @ S_inv_ridge @ phi.T
            kernel_funcs.append(np.reshape(kernel_func, 1000))
            
            if val == test_x[0]:
                kernel_funcs_ridge.append(np.reshape(kernel_func_ridge, 1000))

        # for k in range(len(kernel_funcs)):
        #     plt.plot(x, kernel_funcs[k], color='blue', linestyle='-', alpha=alphas_order[k])

        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.grid(True)
        # plt.title(f'Equivalence Kernel Function of Polynomial Regression Order {order}')
        # plt.show()
    
for k in range(len(kernel_funcs_ridge)):
    plt.plot(x, kernel_funcs_ridge[k], color='red', linestyle='-', alpha=alphas_lambdas[k])

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.title(f'Equivalence Kernel Function of Polynomial Regression Order {order}')
plt.show()


