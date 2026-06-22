import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Generate artificial linear regression data to validate math execution properties
np.random.seed(42)
X_raw = np.random.rand(200, 1) * 10
y_raw = 3.5 * X_raw + 4.2 + np.random.randn(200, 1) * 2

# Split 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

class ScratchLinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            # 1. Forward Pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 2. Compute Mean Squared Error Loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # 3. Compute Gradients
            dw = (2 / m) * np.dot(X.T, (y_pred - y))
            db = (2 / m) * np.sum(y_pred - y)

            # 4. Step Updates (Gradient Descent)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the scratch model
model_scratch = ScratchLinearRegression(lr=0.01, iterations=1500)
model_scratch.fit(X_train, y_train)
predictions_scratch = model_scratch.predict(X_test)

# Evaluation Function Metrics
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, r2

mse_scratch, r2_scratch = calculate_metrics(y_test, predictions_scratch)

# Validation Benchmark Comparison using Sklearn
model_sklearn = SklearnLinearRegression()
model_sklearn.fit(X_train, y_train)
predictions_sklearn = model_sklearn.predict(X_test)
mse_sklearn, r2_sklearn = calculate_metrics(y_test, predictions_sklearn)

# Output Results Summary Comparisons
print("=== Performance Evaluation Metric Results ===")
print(f"Scratch Model  -> MSE: {mse_scratch:.4f} | R²: {r2_scratch:.4f}")
print(f"Sklearn Model  -> MSE: {mse_sklearn:.4f} | R²: {r2_sklearn:.4f}")

# Plot loss optimization tracking curve matrix
plt.figure(figsize=(7, 4))
plt.plot(model_scratch.loss_history, color='crimson', linewidth=2)
plt.title('Scratch Linear Regression Optimization Path')
plt.xlabel('Gradient Descent Iteration Counter')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()