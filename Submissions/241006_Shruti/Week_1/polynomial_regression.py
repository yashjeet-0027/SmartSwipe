import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear dataset curves
np.random.seed(42)
X = np.random.uniform(-3, 3, (100, 1))
y = 0.5 * (X**2) + X + 2 + np.random.normal(0, 1, (100, 1))

# Helper function to generate polynomial feature representations
def get_polynomial_features(X, degree):
    X_poly = np.hstack([X**i for i in range(1, degree + 1)])
    return X_poly

# Scratch Normal Equation Implementation to guarantee precision at high polynomial degrees
class ClosedFormRegression:
    def fit(self, X, y):
        # Insert column of ones to handle the bias term cleanly
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        # Compute parameters via the normal equation: (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0][0]
        self.weights = theta[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Set up visualization domain intervals
X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.7, label='Data points')

degrees = [1, 2, 5]
colors = ['steelblue', 'forestgreen', 'darkorange']
linestyles = ['-', '--', '-.']

for deg, color, ls in zip(degrees, colors, linestyles):
    X_train_poly = get_polynomial_features(X, deg)
    X_plot_poly = get_polynomial_features(X_plot, deg)
    
    model = ClosedFormRegression()
    model.fit(X_train_poly, y)
    predictions = model.predict(X_plot_poly)
    
    plt.plot(X_plot, predictions, color=color, linestyle=ls, linewidth=2.5, 
             label=f'Polynomial Curve (Degree {deg})')

plt.title('Polynomial Curve Fitting & Overfitting Analysis Comparison')
plt.xlabel('Feature Variable (X)')
plt.ylabel('Target Metric (y)')
plt.ylim(min(y) - 1, max(y) + 1)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Overfitting Reflection Summary ---
# Question Answered: Which degree overfits and how can you tell?
#
# Observation: 
# The Degree 5 polynomial overfits the dataset. This is visible near the boundaries 
# (e.g., when X < -2.5 or X > 2.5), where the curve begins to fluctuate wildly and 
# chase statistical noise variations rather than following the true underlying quadratic trend. 
# While its training error might be low, its test generalization performance breaks down.