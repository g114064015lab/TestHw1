import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# Config & Intro (README-aligned)
# ----------------------------
st.set_page_config(page_title="Linear Regression (CRISP-DM)", page_icon="📈", layout="centered")
st.title("📈 Simple Linear Regression — CRISP-DM Demo")

with st.expander("1) Business Understanding"):
    st.markdown(
        """
本應用用於探索 **單一數值特徵 X 與目標 y** 的線性關係，並以**簡單線性回歸**建立預測模型。  
典型情境如：預算→銷售、坪數→房價。此互動式介面可快速體驗並理解線性回歸概念。
"""
    )

with st.expander("2) Data Understanding (Synthetic)"):
    st.markdown(
        """
資料由方程式 **y = a·x + b**（此處固定 b=0 以簡化）加上高斯雜訊所生成，  
你可以即時調整 **斜率 a**、**雜訊大小**、**資料點數** 來觀察學習難易度的變化。
"""
    )


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Parameters")
a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
noise = st.sidebar.slider("Noise (std dev)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
n_points = st.sidebar.slider("Number of points", min_value=20, max_value=2000, value=200, step=10)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)  # README 預設 0.2
seed = st.sidebar.number_input("Random state", value=42, step=1)

# ----------------------------
# Data generation
# ----------------------------
def generate_data(a: float, n_points: int, noise: float, seed: int = 42):
    """
    產生合成資料：
      X ~ linspace(0, 10, n_points)
      y = a * X + Normal(0, noise^2)
    回傳形狀：
      X: (n, 1), y: (n,)
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 10, n_points)
    y = a * X + rng.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

# ----------------------------
# Train / evaluate
# ----------------------------
def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42
):
    """
    切分資料、訓練線性回歸，輸出模型與測試集 MSE 及各分割資料。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return model, mse, X_train, y_train, X_test, y_test, y_pred_test

# ----------------------------
# Outlier detection (no extra UI)
# ----------------------------
def detect_outliers_by_zscore(residuals: np.ndarray, z_thresh: float = 2.5) -> np.ndarray:
    """
    以殘差的 Z 分數偵測離群點。回傳布林遮罩（True 表示離群）。
    - 僅用「測試集殘差」計算平均與標準差。
    """
    mu = float(np.mean(residuals))
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    if sigma == 0:
        return np.zeros_like(residuals, dtype=bool)
    z = (residuals - mu) / sigma
    return np.abs(z) > z_thresh

# ----------------------------
# Main flow
# ----------------------------
# 產生資料並訓練
X, y = generate_data(a, n_points, noise, seed=int(seed))
model, mse, X_train, y_train, X_test, y_test, y_pred_test = train_and_evaluate_model(
    X, y, test_size=float(test_size), seed=int(seed)
)

# 取得測試殘差與離群點遮罩（自動）
residuals_test = y_test - y_pred_test
outlier_mask = detect_outliers_by_zscore(residuals_test, z_thresh=2.5)

# ----------------------------
# Results & coefficients
# ----------------------------
st.subheader("Results (Evaluation)")
st.write(f"**Mean Squared Error (Test):** {mse:.4f}")

st.subheader("Model Coefficients")
st.write(
    f"- **Intercept:** {model.intercept_:.4f}\n"
    f"- **Slope (coef for X):** {model.coef_[0]:.4f}"
)

# ----------------------------
# Plot: training/testing + regression line + highlighted outliers
# ----------------------------
fig, ax = plt.subplots(figsize=(8.4, 6.2))

# Training / Testing
ax.scatter(X_train, y_train, label='Training data', alpha=0.75)
ax.scatter(X_test, y_test, label='Testing data', alpha=0.75)

# Regression line (fit on training)
x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, linewidth=2.2, label='Regression line')

# Highlight & annotate outliers on TEST set (no extra UI)
if np.any(outlier_mask):
    ax.scatter(
        X_test[outlier_mask],
        y_test[outlier_mask],
        s=90,
        facecolor="#6A0DAD",
        edgecolor="black",
        alpha=0.95,
        label="Detected Outliers"
    )
    # 簡短標註：Outlier #k
    for i, (xx, yy) in enumerate(zip(X_test[outlier_mask, 0], y_test[outlier_mask])):
        ax.annotate(
            f"Outlier",
            (xx, yy),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=9,
            color="#6A0DAD",
            weight="bold",
        )

ax.set_title('Simple Linear Regression (Train/Test & Auto-Highlighted Outliers)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
ax.grid(alpha=0.15)
st.pyplot(fig)

# ----------------------------
# Top 5 Outliers table (by |residual| on TEST)
# ----------------------------
st.subheader("Top 5 Outliers (Test Set)")
outlier_df = pd.DataFrame({
    "X": X_test.flatten(),
    "y_true": y_test,
    "y_pred": y_pred_test,
    "residual": residuals_test,
    "abs_residual": np.abs(residuals_test),
    "flagged_by_zscore": outlier_mask
}).sort_values("abs_residual", ascending=False).head(5).reset_index(drop=True)

st.dataframe(outlier_df)

# ----------------------------
# Helpful notes (concise)
# ----------------------------
st.caption(
    "Notes: Outliers are detected automatically from **test residuals** using z-score > 2.5. "
    "If many points are flagged, consider reducing noise, increasing sample size, or checking data quality."
)
