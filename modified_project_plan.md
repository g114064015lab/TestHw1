## CRISP-DM Requirements (what the app must DO & SHOW)

### 1) Business Understanding
- At the top, render a concise purpose statement and success criteria (e.g., “maximize test R², minimize RMSE, well-behaved residuals”).
- Provide a sidebar “About” expander summarizing use cases.

### 2) Data Understanding
- Two data sources:
  - **Synthetic generator** (keep your existing slope/noise/points).
  - **CSV upload** (user selects one numeric feature column and one numeric target column).
- Show:
  - Shape, NA counts, descriptive stats.
  - Pearson correlation (feature vs target).
  - Plots (use matplotlib only; no seaborn):
    - Scatter (feature vs target) with optional fitted line toggle.
    - Histograms for feature and target.

### 3) Data Preparation
- Handle missing values by dropping rows containing NA in the chosen columns; report how many removed.
- Optional **outlier handling**: IQR-based winsorization checkbox; report bounds and how many values affected.
- Train/test split with sidebar controls: `test_size` (default 0.2), `random_state` (default 42).
- Optional **PolynomialFeatures** degree slider (default 1 = off). **No leakage**: fit transforms on train only.
- Build a **scikit-learn Pipeline**: [PolynomialFeatures?] → `LinearRegression(fit_intercept=True)`.

### 4) Modeling
- Fit on train; show learned intercept and slope(s).
- Also fit a **statsmodels OLS** on the train set to provide:
  - p-values, confidence intervals, and Durbin–Watson statistic.
- Provide a downloadable text file of the OLS summary.

### 5) Evaluation
- Report metrics on **train** and **test**: MAE, RMSE, R².
- Residual diagnostics on test:
  - Residuals vs fitted
  - QQ plot
  - Histogram of residuals
  - Residuals vs feature
- Add short interpretation bullets (e.g., heteroscedasticity hints).
- If diagnostics look poor and `poly_degree == 1`, show a hint to try higher degree.

### 6) Deployment (lightweight)
- Button to **download the trained pipeline** as `model.joblib`.
- Text input to **predict from a single feature value** using the trained pipeline; display the prediction.
- Button to **download a Markdown report** summarizing: data source, parameters, coefficients, metrics, residual checks, and recommended next steps.
