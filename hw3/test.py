import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.rand(100)

# Perform K-fold cross-validation for polynomial orders 1 through 9
degrees = list(range(1, 10))
mse_scores = []
bias_squared_scores = []
variance_scores = []

for degree in degrees:
    # Create a polynomial regression model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Perform 5-fold cross-validation
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    
    # Calculate MSE, Bias^2, and Variance
    mse = -scores.mean()
    bias_squared = mse - np.var(y)
    variance = np.var(y)
    
    mse_scores.append(mse)
    bias_squared_scores.append(bias_squared)
    variance_scores.append(variance)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.plot(degrees, mse_scores, marker='o', label='MSE')
plt.title('Mean Squared Error')
plt.xlabel('Polynomial Degree')
plt.grid(True)
plt.xticks(degrees)
plt.legend()

plt.subplot(132)
plt.plot(degrees, bias_squared_scores, marker='o', label='Bias^2')
plt.title('Bias Squared')
plt.xlabel('Polynomial Degree')
plt.grid(True)
plt.xticks(degrees)
plt.legend()

plt.subplot(133)
plt.plot(degrees, variance_scores, marker='o', label='Variance')
plt.title('Variance')
plt.xlabel('Polynomial Degree')
plt.grid(True)
plt.xticks(degrees)
plt.legend()

plt.tight_layout()
plt.show()
