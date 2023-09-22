import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

data = np.genfromtxt('hw2_q2.csv', delimiter=",")

x = data[:, 0]
y = data[:, 1]

order = 5
design_matrix = np.empty((x.shape[0], 6))

for i in range(order + 1):
    column = x**i
    design_matrix[:, i] = column

w = np.linalg.lstsq(design_matrix, y)
residuals = np.sum(np.array((y - (design_matrix @ w[0].T))) ** 2)
unbiased_noise_var = residuals/(len(x) - 5 - 1)
mle_noise_var = residuals/len(x)

print(w[0])
print(unbiased_noise_var)
print(mle_noise_var)

var_values = np.linspace(50, 250, 500)
joint_prob = []
for var in var_values:
    prod = 1
    for i in range(len(x)):
        prod *= stats.norm.pdf(y[i], loc=(design_matrix[i] @ w[0].T), scale=np.sqrt(var))

    joint_prob.append(prod)

plt.plot(var_values, joint_prob, color='blue', linestyle='-')
plt.axvline(x=unbiased_noise_var, color='red', linestyle='--', label=f'Unbiased estimated noise variance')
plt.axvline(x=mle_noise_var, color='green', linestyle='--', label=f'MLE estimated noise variance')

plt.xlabel('Estimated Noise Variance')
plt.ylabel('Joint Conditional Probability')
plt.legend()

plt.grid(True)
plt.title('Joint Conditional Probability of Dataset vs. Possible Noise Variances')
plt.show()

# x_values = np.linspace(0, 10, 1000)
# def poly(x):
#     return (w[0][0] + w[0][1] * x + w[0][2] * x**2 + w[0][3] * x**3 + w[0][4] * x**4 + w[0][5] * x**5)

# y_vals = poly(x_values)

# plt.plot(x_values, y_vals, label='Polynomial Function', color='blue', linestyle='-')
# plt.scatter(x, y, label='Data Points', color='red', marker='o')

# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.legend()

# # Add grid lines
# plt.grid(True)

# # Set the plot title
# plt.title('Polynomial Function and Data Points')

# # Show the plot
# plt.show()
