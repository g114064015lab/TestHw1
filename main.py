import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# Data generation
# ----------------------------
def generate_data(a: float, n_points: int, noise: float, seed: int = 42):
    """Generate synthetic data for linear regression."""
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 10, n_points)
    y = a * X + rng.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

# ----------------------------
# Train / evaluate
# ----------------------------
def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Train a linear regression model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_train, y_train, X_test, y_test, y_pred

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Simple Linear Regression")

# Sidebar controls (sliders)
st.sidebar.header("Parameters")
a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
noise = st.sidebar.slider("Noise (std dev)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
n_points = st.sidebar.slider("Number of points", min_value=10, max_value=1000, value=50, step=1)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
seed = st.sidebar.number_input("Random state", value=42, step=1)

# Generate data and train model
X, y = generate_data(a, n_points, noise, seed=int(seed))
model, mse, X_train, y_train, X_test, y_test, y_pred = train_and_evaluate_model(
    X, y, test_size=float(test_size), seed=int(seed)
)

# ----------------------------
# Results
# ----------------------------
st.subheader("Results")
st.write(f"**Mean Squared Error (Test):** {mse:.4f}")

# Model coefficients (intercept & slope)
st.subheader("Model Coefficients")
st.write(
    f"- **Intercept:** {model.intercept_:.4f}\n"
    f"- **Slope (coef for X):** {model.coef_[0]:.4f}"
)

# ----------------------------
# Plot: data & fitted line
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train, y_train, label='Training data', alpha=0.7)
ax.scatter(X_test, y_test, label='Testing data', alpha=0.7)

# Draw regression line across the full X-range for a clean line
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, linewidth=2, label='Regression line')

ax.set_title('Simple Linear Regression')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
st.pyplot(fig)

# ----------------------------
# Top 5 Outliers (by absolute residuals on TEST set)
# ----------------------------
st.subheader("Top 5 Outliers (Test Set)")
residuals = y_test - y_pred
outlier_df = pd.DataFrame({
    "X": X_test.flatten(),
    "y_true": y_test,
    "y_pred": y_pred,
    "residual": residuals,
    "abs_residual": np.abs(residuals)
}).sort_values("abs_residual", ascending=False).head(5)

st.dataframe(outlier_df.reset_index(drop=True))
