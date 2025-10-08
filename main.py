import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------
# Data generation
# ---------------------------------------
def generate_clean_data(a: float, n_points: int, noise: float, seed: int = 42):
    """
    Make clean linear data: x in [0,10], y = a*x + epsilon, epsilon ~ N(0, noise^2).
    Returns X (n,1), y (n,)
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 10, n_points)
    y = a * X + rng.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

def inject_outliers(
    X: np.ndarray,
    y: np.ndarray,
    n_outliers: int,
    magnitude: float,
    seed: int = 123,
):
    """
    Inject 'n_outliers' extreme points by adding +/- magnitude * std(y) to y
    at randomly chosen x positions. Returns:
      X_all, y_all, outlier_mask (len=n_total)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n_outliers <= 0:
        return X, y, np.zeros(n, dtype=bool)

    # 隨機抽 outlier 的位置（也可以生成全新點，這裡採「覆寫部分 y」的方式）
    idx = rng.choice(np.arange(n), size=min(n_outliers, n), replace=False)
    y_new = y.copy()
    bump = magnitude * np.std(y)  # 以資料尺度決定幅度
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(idx))
    y_new[idx] = y_new[idx] + signs * bump

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return X, y_new, mask

# ---------------------------------------
# Fit helpers
# ---------------------------------------
def fit_line(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="Linear Regression — Outliers Effect", page_icon="📉")
st.title("📉 Linear Regression — Effect of Outliers")

with st.sidebar.expander("About"):
    st.markdown(
        "- This demo shows **how a few outliers can tilt the regression line**.\n"
        "- The model is fit on the **entire dataset** (no train/test split).\n"
        "- Purple points are injected outliers; compare coefficients with/without them."
    )

st.sidebar.header("Data (clean)")
a = st.sidebar.slider("Slope (a)", -10.0, 10.0, 2.0, 0.1)
noise = st.sidebar.slider("Noise (std)", 0.0, 10.0, 2.0, 0.1)
n_points = st.sidebar.slider("Number of points", 50, 1000, 400, 10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.header("Outliers")
n_outliers = st.sidebar.slider("Number of outliers", 0, 10, 5, 1)
magnitude = st.sidebar.slider("Outlier magnitude (× std of y)", 2.0, 20.0, 8.0, 0.5)
out_seed = st.sidebar.number_input("Outlier seed", value=123, step=1)

# Generate + inject outliers
X_clean, y_clean = generate_clean_data(a, n_points, noise, seed=int(seed))
X_all, y_all, out_mask = inject_outliers(
    X_clean, y_clean, n_outliers=int(n_outliers), magnitude=float(magnitude), seed=int(out_seed)
)

# Fit models
model_clean = fit_line(X_clean, y_clean)
model_out = fit_line(X_all, y_all)

# Coefficients table
coef_df = pd.DataFrame(
    {
        "Scenario": ["Without Outliers", "With Outliers"],
        "Intercept": [model_clean.intercept_, model_out.intercept_],
        "Slope": [model_clean.coef_[0], model_out.coef_[0]],
    }
)
coef_df["Δ Intercept"] = coef_df["Intercept"] - coef_df["Intercept"].iloc[0]
coef_df["Δ Slope"] = coef_df["Slope"] - coef_df["Slope"].iloc[0]

st.subheader("Model Coefficients (Comparison)")
st.dataframe(coef_df.style.format({"Intercept": "{:.4f}", "Slope": "{:.4f}", "Δ Intercept": "{:+.4f}", "Δ Slope": "{:+.4f}"}))

# ---------------------------------------
# Plot: scatter + regression line (with outliers)
# ---------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# base points (non-outliers)
base_mask = ~out_mask
ax.scatter(X_all[base_mask, 0], y_all[base_mask], s=35, alpha=0.7, label="Generated Data", zorder=1)

# outliers
if out_mask.any():
    ax.scatter(
        X_all[out_mask, 0], y_all[out_mask], s=80, facecolor="#6A0DAD", edgecolor="black",
        alpha=0.95, label="Outliers", zorder=2
    )
    # annotate each outlier
    for i, (xx, yy) in enumerate(zip(X_all[out_mask, 0], y_all[out_mask])):
        ax.annotate(
            f"Outlier",
            (xx, yy),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
            color="#6A0DAD",
            weight="bold",
        )

# regression line (fitted on ALL points -> 展示被拉歪的線)
x_line = np.linspace(float(X_all.min()), float(X_all.max()), 400).reshape(-1, 1)
y_line_out = model_out.predict(x_line)
ax.plot(x_line[:, 0], y_line_out, color="crimson", linewidth=2.5, label="Linear Regression (with outliers)", zorder=3)

ax.set_title("Linear Regression with Outliers (fit on whole dataset)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc="upper left")
ax.grid(alpha=0.15)
st.pyplot(fig)

# ---------------------------------------
# Optional: also show the clean line for visual delta
# ---------------------------------------
show_clean_line = st.checkbox("Overlay regression line without outliers (for reference)", value=True)
if show_clean_line:
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    # all points still shown，方便比較
    ax2.scatter(X_all[base_mask, 0], y_all[base_mask], s=35, alpha=0.7, label="Generated Data")
    if out_mask.any():
        ax2.scatter(X_all[out_mask, 0], y_all[out_mask], s=80, facecolor="#6A0DAD", edgecolor="black", alpha=0.95, label="Outliers")

    y_line_clean = model_clean.predict(x_line)
    y_line_out2 = model_out.predict(x_line)
    ax2.plot(x_line[:, 0], y_line_clean, linewidth=2.5, label="Line (without outliers)")
    ax2.plot(x_line[:, 0], y_line_out2, linewidth=2.5, color="crimson", label="Line (with outliers)")

    ax2.set_title("Overlay: clean vs outlier-affected regression lines")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.15)
    st.pyplot(fig2)


