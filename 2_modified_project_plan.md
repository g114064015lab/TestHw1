# Prompt: Build a Streamlit App with CRISP-DM Workflow

- **Goal**: Create a **single-file** Streamlit app (`app.py`) that demonstrates **simple linear regression** and follows the **CRISP-DM methodology** end-to-end.

---

## 1. Overview & Business Understanding
- At the **top of the app**, briefly explain:
  - **Business goal**.
  - **Success criteria**.
- In the **sidebar**, include an **“About” expander** that:
  - Summarizes **possible use cases** (e.g., education, model exploration, deployment demo).

---

## 2. Data Source Options
- **Synthetic Data**:
  - Controls for:
    - **Slope**.
    - **Noise**.
    - **Number of points**.
- **CSV Upload**:
  - Allow user to:
    - Upload a CSV file.
    - Select **one numeric feature column**.
    - Select **one numeric target column**.

---

## 3. Data Understanding
- Display:
  - **Dataset shape**.
  - **Missing values**.
  - **Descriptive statistics**.
  - **Pearson correlation**.
- Visualizations (matplotlib only):
  - **Scatter plot** of feature vs target.
    - Optional fitted line overlay.
  - **Histogram** of feature.
  - **Histogram** of target.

---

## 4. Data Preparation
- Handle **missing values**:
  - Remove rows with NaNs in selected columns.
  - Report how many rows were dropped.
- **Optional IQR-based winsorization** (checkbox):
  - For outlier handling.
- **Sidebar controls**:
  - **Test size** (for train/test split).
  - **Random state**.
- **Polynomial feature expansion** (slider):
  - Default degree = 1.
  - Avoid data leakage.
- **Build a scikit-learn pipeline**:
  - `PolynomialFeatures` (if degree > 1).
  - `LinearRegression`.

---

## 5. Modeling
- **Fit the pipeline** on the training set.
- Display:
  - **Intercept**.
  - **Coefficients**.
- **Statsmodels OLS** on the training set:
  - Show **p-values**.
  - **Confidence intervals**.
  - **Durbin–Watson statistic**.
- **Downloadable OLS summary** as `.txt`.

---

## 6. Evaluation
- Compute metrics on **train** and **test** sets:
  - **MAE**.
  - **RMSE**.
  - **R²**.
- Residual diagnostics (matplotlib):
  - **Residuals vs fitted values**.
  - **QQ plot**.
  - **Histogram of residuals**.
  - **Residuals vs feature**.
- Add **interpretation notes**:
  - Mention issues like **heteroscedasticity**.
  - If linear fit is poor, suggest **increasing polynomial degree**.

---

## 7. Deployment & Reporting
- **Downloads**:
  - **Trained pipeline** (`model.joblib`).
  - **OLS summary** (text file).
  - **Markdown report** summarizing:
    - Data source.
    - Preprocessing.
    - Model parameters.
    - Coefficients.
    - Metrics.
    - Residual checks.
    - Suggested next steps.
- **Prediction utility**:
  - User inputs a single feature value.
  - App outputs prediction from the trained pipeline.

---

## 8. Technical Requirements
- **Allowed libraries only**:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `statsmodels`, `scipy` (if needed), `joblib`, `streamlit`.
- **No seaborn**, **no internet calls**.
- **Code structure**:
  - Use **functions**, **docstrings**, **type hints**.
  - Apply **caching** (`st.cache_data`) where appropriate.
- **Error handling**:
  - Handle invalid inputs, missing columns, or model fitting errors gracefully.

---

