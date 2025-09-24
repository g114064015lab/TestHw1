import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def generate_data(a, n_points, noise):
    """Generate synthetic data for linear regression."""
    X = np.linspace(0, 10, n_points)
    y = a * X + np.random.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

def train_and_evaluate_model(X, y):
    """Train a linear regression model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_train, y_train, X_test, y_test, y_pred

# Streamlit UI
st.title("📈 Simple Linear Regression")

# Sidebar for inputs
st.sidebar.header("Parameters")
a = st.sidebar.number_input("Slope (a)", value=2.0)
noise = st.sidebar.number_input("Noise", value=1.0)
n_points = st.sidebar.number_input("Number of points", value=50, min_value=10, max_value=1000)

# Generate data and train model
X, y = generate_data(a, n_points, noise)
model, mse, X_train, y_train, X_test, y_test, y_pred = train_and_evaluate_model(X, y)

# Show results
st.subheader("Results")
st.write(f"**Mean Squared Error:** {mse:.4f}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train, y_train, color='blue', label='Training data')
ax.scatter(X_test, y_test, color='green', label='Testing data')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
ax.set_title('Simple Linear Regression')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()

st.pyplot(fig)
