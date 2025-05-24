import numpy as np
import os

save_dir = r"C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\Data\Figures 2025\Hebbian index fits"
os.makedirs(save_dir, exist_ok=True)

# Save lists of variable-length arrays using dtype=object
np.savez_compressed(
    os.path.join(save_dir, "hebbian_fit_inputs.npz"),
    HIb=np.array(HIb, dtype=object),
    THR=np.array(THR, dtype=object),
    RT=np.array(RT, dtype=object),
    RPE=np.array(RPE, dtype=object),
    HIT=np.array(HIT, dtype=object)
)

print("Saved input data to:", os.path.join(save_dir, "hebbian_fit_inputs.npz"))

#%%
fit_inputs = np.load(
    r"C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\Data\Figures 2025\Hebbian index fits\hebbian_fit_inputs.npz",
    allow_pickle=True
)

HIb = list(fit_inputs["HIb"])
THR = list(fit_inputs["THR"])
RT  = list(fit_inputs["RT"])
RPE = list(fit_inputs["RPE"])
HIT = list(fit_inputs["HIT"])


# %% Preprocessing
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Construct full dataframe
X_list = []
y_list = []
session_ids = []

for i in range(len(HIb)):
    n_trials = len(HIb[i])
    thr = THR[i]
    thr = (thr - thr[2]) / thr[2]
    df = pd.DataFrame({
        'RT': RT[i],
        'RPE': RPE[i],
        'THR': thr,
        'HIT': HIT[i],
        'session': i
    })
    X_list.append(df)

    hi = HIb[i]
    hi_zscore = (hi - np.nanmean(hi)) / np.nanstd(hi)  # Normalize HI within session
    y_list.append(hi_zscore)

    session_ids.extend([i] * n_trials)

X = pd.concat(X_list, ignore_index=True)
y = np.concatenate(y_list)


# Drop rows with any NaNs
X_numeric = X.drop(columns='session')
mask = ~X_numeric.isna().any(axis=1) & ~np.isnan(y)

X_clean = X_numeric[mask]
y_clean = y[mask]

# %% Fit model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Define model
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Get cross-validated predictions
y_cv_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)

# Evaluate R² and Pearson r
r2 = r2_score(y_clean, y_cv_pred)
r, p = pearsonr(y_clean, y_cv_pred)

print(f"Cross-validated R² = {r2:.3f}")
print(f"Cross-validated Pearson r = {r:.3f}, p = {p:.2e}")


# %% Diagnostics
# %% Diagnostics
plt.figure(figsize=(5, 5))
sns.regplot(x=y_clean, y=y_cv_pred, line_kws={"color": "red"})
plt.xlabel("True HI (z-scored)")
plt.ylabel("Predicted HI")
plt.title(f"Cross-Validated Predictions (R² = {r2:.2f})")
plt.tight_layout()
plt.show()

# Residuals
residuals = y_clean - y_cv_pred
plt.figure(figsize=(5, 3))
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel("Residual (True - Predicted HI)")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()


# %% SHAP analysis
# Fit your model first
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
model.fit(X_clean, y_clean)  # Ensure this line runs before SHAP

# Then use SHAP
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_clean)  # SHAP values for the entire (clean) dataset

# Plot
shap.summary_plot(shap_values, X_clean, plot_type="bar")
shap.summary_plot(shap_values, X_clean)
shap.dependence_plot("RPE", shap_values, X_clean, interaction_index="THR")
#%%
X_clean['RPE_SHAP'] = shap_values[:, X_clean.columns.get_loc("RPE")]
X_clean['THR_bin'] = pd.qcut(X_clean['THR'], q=3, labels=['low', 'med', 'high'])

sns.boxplot(data=X_clean, x='THR_bin', y='RPE_SHAP')
plt.title("RPE SHAP values by THR level")
from sklearn.linear_model import LinearRegression

from scipy.stats import linregress

thr_vals = X_clean['THR'].to_numpy()
shap_rpe_vals = shap_values[:, X_clean.columns.get_loc("RPE")]

slope, intercept, r_value, p_value, stderr = linregress(thr_vals, shap_rpe_vals)

print(f"Slope = {slope:.3f}")
print(f"R² = {r_value**2:.3f}")
print(f"p-value = {p_value:.2e}")

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Define the random forest
rf_model = RandomForestRegressor(
    n_estimators=100,     # number of trees
    max_depth=None,       # let trees grow deep
    random_state=0,
    n_jobs=-1             # use all available cores
)

# Cross-validated predictions
kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_cv_pred_rf = cross_val_predict(rf_model, X_clean, y_clean, cv=kf)

# Evaluate
r2_rf = r2_score(y_clean, y_cv_pred_rf)
r_rf, p_rf = pearsonr(y_clean, y_cv_pred_rf)

print(f"Random Forest Cross-validated R² = {r2_rf:.3f}")
print(f"Random Forest Cross-validated Pearson r = {r_rf:.3f}, p = {p_rf:.2e}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 3))
#sns.scatterplot(x=y_clean, y=y_cv_pred_rf, s=20)
sns.regplot(x=y_clean, y=y_cv_pred_rf, line_kws={"color": "red"})
plt.xlabel("True HI (normalized)")
plt.ylabel("Predicted HI (Random Forest)")
plt.title(f"Random Forest CV Predictions (R² = {r2_rf:.2f}, r = {r_rf:.2f})")
#plt.axline((0, 0), slope=1, linestyle='--', color='gray')
plt.tight_layout()
plt.show()



#%% MLP, doesn't work very well
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Drop non-numeric columns
X_numeric = X.select_dtypes(include=[np.number]).drop(columns='session')

# Clean: drop rows with NaNs or infs
mask = ~X_numeric.isna().any(axis=1) & ~np.isnan(y)
X_clean = X_numeric[mask]
y_clean = y[mask]

# === Scale input features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# === Define the MLP ===
mlp_model = MLPRegressor(
    hidden_layer_sizes=(10,),     # One hidden layer with 10 units
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=0
)

# === Cross-validated predictions ===
kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_cv_pred_mlp = cross_val_predict(mlp_model, X_scaled, y_clean, cv=kf)

# === Evaluation ===
r2_mlp = r2_score(y_clean, y_cv_pred_mlp)
r_mlp, p_mlp = pearsonr(y_clean, y_cv_pred_mlp)

print(f"MLP (scaled) Cross-validated R² = {r2_mlp:.3f}")
print(f"MLP (scaled) Cross-validated Pearson r = {r_mlp:.3f}, p = {p_mlp:.2e}")





import matplotlib.pyplot as plt
import seaborn as sns

# === Scatter plot: True vs Predicted ===
plt.figure(figsize=(5, 5))
sns.scatterplot(x=y_clean, y=y_cv_pred_mlp, s=20)
plt.xlabel("True HI (normalized)")
plt.ylabel("Predicted HI (MLP)")
plt.title(f"MLP CV Predictions (R² = {r2_mlp:.2f}, r = {r_mlp:.2f})")
plt.axline((0, 0), slope=1, linestyle='--', color='gray')
plt.tight_layout()
plt.show()

# === Residuals histogram ===
residuals_mlp = y_clean - y_cv_pred_mlp
plt.figure(figsize=(5, 3))
sns.histplot(residuals_mlp, kde=True, bins=20)
plt.xlabel("Residual (True - Predicted HI)")
plt.title("Residuals (MLP Cross-Validated)")
plt.tight_layout()
plt.show()






























