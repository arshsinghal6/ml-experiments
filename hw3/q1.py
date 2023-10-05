import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x = np.array([0, -1, 1])
y = np.array([1.1, -9.1, 11.0])

# part a
A = np.vstack([np.ones(len(x)), x]).T

w = np.linalg.lstsq(A, y, rcond=None)[0]
residuals = np.sum((y - (A @ w)) ** 2)

# parts e and f
w_ridge = np.linalg.inv(A.T @ A + 0.5268 * np.eye(2)) @ A.T @ y 
residuals_ridge = np.sum((y - (A @ w_ridge)) ** 2)

w_lasso = [0, 8]
residuals_lasso = np.sum((y - (A @ w_lasso)) ** 2)


# part c
fig, ax = plt.subplots()

for i in [2, 4, 6, 8]:  # selected scaling factors that seemed appropriate for the plot
    ellipse = patches.Ellipse((w[0], w[1]), width=(0.57735 * i), height=(0.70711 * i), fill=False, color='blue', linewidth=2)
    ax.add_patch(ellipse)

# diamond = patches.Polygon([[0, 8], [-8, 0], [0, -8], [8, 0]], fill=False, color='red', linewidth=2)
# ax.add_patch(diamond)
circle = patches.Circle((0, 0), 8, fill=False, color='red', linewidth=2)
ax.add_patch(circle)

ax.set_xlim(-10, 10)
ax.set_ylim(-9, 14)
ax.set_aspect('equal')

plt.xlabel('w0-axis')
plt.ylabel('w1-axis')
# plt.title('LASSO, 1-ball, radius = 8')
plt.title('Ridge Regression, 2-ball, radius = 8')


plt.grid(True)
plt.show()