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
#%%
for i in range(len(THR)):
    THR[i][0] = THR[i][1]
    ind = np.where(np.isnan(THR[i]))[0]
    if len(ind) > 0:
        THR[i][ind] = THR[i][ind[0]-1]

#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === Helper: Exponential kernel smoothing ===
def exp_kernel_smooth(x, tau):
    kernel = np.exp(-np.arange(0, 10*tau) / tau)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')

# === Tau values to grid over ===
tau_rpe_list = [1, 10]
tau_hit_list = [1, 10]
tau_thr_list = [1, 10]

results = []

# === Grid search loop ===
for tau_rpe in tau_rpe_list:
    for tau_hit in tau_hit_list:
        for tau_thr in tau_thr_list:
            print(f"Trying tau_rpe={tau_rpe}, tau_hit={tau_hit}, tau_thr={tau_thr}")

            X_list = []
            y_list = []
            session_ids = []

            for i in range(len(HIb)):
                n = len(HIb[i])
                if n == 0 or np.nanstd(HIb[i]) == 0:
                    continue

                rt = np.asarray(RT[i])[:n]
                rpe = np.nan_to_num(RPE[i])[:n]
                hit = np.asarray(HIT[i])[:n]
                thr = np.asarray(THR[i])[:n]
                if len(thr) <= 2 or thr[2] == 0:
                    continue
                thr_norm = (thr - thr[2]) / thr[2]

                delta_hit = np.concatenate([[0], np.diff(hit)])[:n]
                delta_thr = np.concatenate([[0], np.diff(thr_norm)])[:n]

                rpe_smooth = exp_kernel_smooth(rpe, tau_rpe)[:n]
                dhit_smooth = exp_kernel_smooth(delta_hit, tau_hit)[:n]
                dthr_smooth = exp_kernel_smooth(delta_thr, tau_thr)[:n]

                if not all(len(arr) == n for arr in [rt, rpe_smooth, dhit_smooth, dthr_smooth, hit, thr_norm]):
                    continue

                df = pd.DataFrame({
                    'RT': rt,
                    'RPE': rpe_smooth,
                    'Delta_HIT': dhit_smooth,
                    'Delta_THR': dthr_smooth,
                    'HIT': hit,
                    'THR': thr_norm,
                    'session': i
                })
                X_list.append(df)

                hi = np.asarray(HIb[i])
                hi_zscore = (hi - np.nanmean(hi)) / np.nanstd(hi)
                y_list.append(hi_zscore[:n])
                session_ids.extend([i] * n)

            if not X_list:
                print("No valid sessions for this tau combo.")
                continue

            X = pd.concat(X_list, ignore_index=True)
            y = np.concatenate(y_list)

            X_numeric = X.drop(columns='session')
            mask = ~X_numeric.isna().any(axis=1) & ~np.isnan(y)
            X_clean = X_numeric[mask]
            y_clean = y[mask]

            if len(X_clean) < 5:
                print(f"Too few samples after cleaning: {len(X_clean)}")
                continue

            model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            y_cv_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)

            r2 = r2_score(y_clean, y_cv_pred)
            r, p = pearsonr(y_clean, y_cv_pred)

            print(f"→ R² = {r2:.3f}, Pearson r = {r:.3f}, p = {p:.2e}")
            results.append({
                'tau_rpe': tau_rpe,
                'tau_hit': tau_hit,
                'tau_thr': tau_thr,
                'r2': r2,
                'r': r,
                'p': p
            })

# Optional: convert results to DataFrame
results_df = pd.DataFrame(results)
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Get best tau combo
best = results_df.loc[results_df['r'].idxmax()]
print("Best tau combo:", best)

# Rebuild X and y using best taus
tau_rpe = best['tau_rpe']
tau_hit = best['tau_hit']
tau_thr = best['tau_thr']

# [Insert the full X_list, y_list loop from earlier with those taus here...]

# Clean and assemble
X = pd.concat(X_list, ignore_index=True)
y = np.concatenate(y_list)
X_numeric = X.drop(columns='session')
mask = ~X_numeric.isna().any(axis=1) & ~np.isnan(y)
X_clean = X_numeric[mask]
y_clean = y[mask]

# Fit model with cross-validation
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)

# Plot: predicted vs. true
r2 = r2_score(y_clean, y_pred)
plt.figure(figsize=(5, 5))
sns.regplot(x=y_clean, y=y_pred, line_kws={"color": "red"})
plt.xlabel("True HI (z-scored)")
plt.ylabel("Predicted HI")
plt.title(f"Cross-Validated Predictions (R² = {r2:.2f})")
plt.tight_layout()
plt.show()

# Plot: residuals
residuals = y_clean - y_pred
plt.figure(figsize=(5, 3))
sns.histplot(residuals, kde=True, bins=20)
plt.xlabel("Residual (True - Predicted HI)")
plt.title("Residual Distribution (Cross-Validated)")
plt.tight_layout()
plt.show()
#%%
import shap
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
model.fit(X_clean, y_clean)

explainer = shap.TreeExplainer(model)
shap_values_wrapped = explainer(X_clean)

# Plot global feature importance
shap.plots.bar(shap_values_wrapped)

# Beeswarm for detailed view
shap.plots.beeswarm(shap_values_wrapped)

# Example interaction: does THR modulate RPE?
shap.plots.scatter(shap_values_wrapped[:, "RPE"], color=shap_values_wrapped[:, "THR"])
#%%
import shap

explainer = shap.TreeExplainer(model)
interaction_values = explainer.shap_interaction_values(X_clean)
rpe_index = X_clean.columns.get_loc("RPE")

for fname in X_clean.columns:
    idx = X_clean.columns.get_loc(fname)
    interaction = interaction_values[:, rpe_index, idx]
    shap.dependence_plot((rpe_index, idx), interaction_values, X_clean)
#%%
sign_rpe = np.sign(interaction_values[:, rpe_index, rpe_index])
for i, fname in enumerate(X_clean.columns):
    if i == rpe_index:
        continue
    flips = np.sum((interaction_values[:, rpe_index, i] * sign_rpe) < 0)
    print(f"{fname}: {flips} sign-modulating interactions")
#%%
explainer = shap.TreeExplainer(model)
interaction_values = explainer.shap_interaction_values(X_clean)
rpe_index = X_clean.columns.get_loc("RPE")
modulation_strength = {}

for fname in X_clean.columns:
    idx = X_clean.columns.get_loc(fname)
    
    # Ignore RPE itself (main effect)
    if idx == rpe_index:
        continue

    # Compute mean absolute interaction value across all samples
    interaction = interaction_values[:, rpe_index, idx]
    modulation_strength[fname] = np.mean(np.abs(interaction))

# Sort by modulation strength
sorted_modulation = sorted(modulation_strength.items(), key=lambda x: -x[1])

# Display results
for fname, val in sorted_modulation:
    print(f"{fname}: mean |interaction with RPE| = {val:.4f}")
#%%
import matplotlib.pyplot as plt

# Assuming `modulation_strength` is already computed
features = list(modulation_strength.keys())
strengths = [modulation_strength[f] for f in features]

# Sort by strength
sorted_indices = np.argsort(strengths)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_strengths = [strengths[i] for i in sorted_indices]

# Plot
plt.figure(figsize=(8, 5))
plt.barh(sorted_features, sorted_strengths)
plt.xlabel("Mean |SHAP Interaction with RPE|")
plt.title("Feature Modulation of RPE Contribution")
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()
#%%
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Compute interaction values
explainer = shap.TreeExplainer(model)
interaction_values = explainer.shap_interaction_values(X_clean)

# Compute mean absolute interaction across all samples
mean_interactions = np.abs(interaction_values).mean(axis=0)

# Create heatmap
feature_names = X_clean.columns
plt.figure(figsize=(10, 8))
sns.heatmap(mean_interactions, xticklabels=feature_names, yticklabels=feature_names, cmap='jet')
plt.title("Mean Absolute SHAP Interaction Values")
plt.tight_layout()
plt.show()


#%%
from sklearn.model_selection import GroupKFold

groups = np.array(session_ids)[mask]
gkf = GroupKFold(n_splits=5)
y_pred_groupcv = cross_val_predict(model, X_clean, y_clean, cv=gkf, groups=groups)

r2_group = r2_score(y_clean, y_pred_groupcv)
r_group, p_group = pearsonr(y_clean, y_pred_groupcv)

print(f"Session-wise CV R² = {r2_group:.3f}")
print(f"Session-wise CV Pearson r = {r_group:.3f}, p = {p_group:.2e}")
#%%
example_session = 3  # or pick by max/min HI variance

session_mask = (np.array(session_ids)[mask] == example_session)
hi_true = y_clean[session_mask]
hi_pred = y_pred[session_mask]

plt.figure(figsize=(10, 4))
plt.plot(hi_true, label='True HI', alpha=0.8)
plt.plot(hi_pred, label='Predicted HI', alpha=0.8)
plt.title(f'Session {example_session}: True vs. Predicted HI')
plt.xlabel('Trial')
plt.ylabel('Z-scored Hebbian Index')
plt.legend()
plt.tight_layout()
plt.show()


#%%
import seaborn as sns
import matplotlib.pyplot as plt

# One heatmap per ΔTHR τ value
for tau_thr_val in sorted(results_df['tau_thr'].unique()):
    pivot = results_df[results_df['tau_thr'] == tau_thr_val].pivot(
        index='tau_hit', columns='tau_rpe', values='r'
    )
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'Pearson r'})
    plt.title(f'Pearson r for τ_ΔTHR = {tau_thr_val}')
    plt.xlabel('τ_RPE')
    plt.ylabel('τ_ΔHIT')
    plt.tight_layout()
    plt.show()
#%%
top_results = results_df.sort_values("r", ascending=False)

plt.figure(figsize=(10, 4))
sns.barplot(
    data=top_results,
    x=[f"RPE={r.tau_rpe}, dHIT={r.tau_hit}, dTHR={r.tau_thr}" for _, r in top_results.iterrows()],
    y="r"
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Pearson r")
plt.title("Model Performance Across Tau Combinations")
plt.tight_layout()
plt.show()
#%%
mean_r_by_rpe = results_df.groupby("tau_rpe")["r"].mean()

plt.figure()
sns.lineplot(x=mean_r_by_rpe.index, y=mean_r_by_rpe.values, marker="o")
plt.xlabel("τ_RPE")
plt.ylabel("Mean Pearson r")
plt.title("Effect of τ_RPE on Model Performance")
plt.tight_layout()
plt.show()
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

# Load your data
data_dict = np.load(
    r"C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\Data\Figures 2025\Hebbian index fits\hebbian_fit_inputs.npy",
    allow_pickle=True
).item()

HIb = data_dict["HIb"]
THR = data_dict["THR"]
RT = data_dict["RT"]
RPE = data_dict["RPE"]
HIT = data_dict["HIT"]

# === Helper: Exponential kernel smoothing ===
def exp_kernel_smooth(x, tau):
    kernel = np.exp(-np.arange(0, 10*tau) / tau)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')

# Use fixed taus
tau_rpe, tau_hit, tau_thr = 10, 10, 10

X_list, y_list = [], []

for i in range(len(HIb)):
    n = len(HIb[i])
    if n == 0 or np.nanstd(HIb[i]) == 0:
        continue

    rt = np.asarray(RT[i])[:n]
    rpe = np.nan_to_num(RPE[i])[:n]
    hit = np.asarray(HIT[i])[:n]
    thr = np.asarray(THR[i])[:n]
    if len(thr) <= 2 or thr[2] == 0:
        continue
    thr[np.isnan(thr)] = thr[~np.isnan(thr)][-1]  # fill end NaNs with last value
    thr_norm = (thr - thr[2]) / thr[2]

    delta_hit = np.concatenate([[0], np.diff(hit)])[:n]
    delta_thr = np.concatenate([[0], np.diff(thr_norm)])[:n]

    rpe_smooth = exp_kernel_smooth(rpe, tau_rpe)[:n]
    dhit_smooth = exp_kernel_smooth(delta_hit, tau_hit)[:n]
    dthr_smooth = exp_kernel_smooth(delta_thr, tau_thr)[:n]

    df = pd.DataFrame({
        'RT': rt,
        'RPE': rpe_smooth,
        'Delta_HIT': dhit_smooth,
        'Delta_THR': dthr_smooth,
        'HIT': hit,
        'THR': thr_norm
    })

    hi = np.asarray(HIb[i])
    hi_zscore = (hi - np.nanmean(hi)) / np.nanstd(hi)
    X_list.append(df)
    y_list.append(hi_zscore[:n])

X = pd.concat(X_list, ignore_index=True)
y = np.concatenate(y_list)

X = X.fillna(0)

# === Cross-validated predictions ===
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Model with interactions
model_tree = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
y_pred_tree = cross_val_predict(model_tree, X, y, cv=kf)
r2_tree = r2_score(y, y_pred_tree)

# Additive model (no interactions)
model_linear = LinearRegression()
y_pred_linear = cross_val_predict(model_linear, X, y, cv=kf)
r2_linear = r2_score(y, y_pred_linear)

# === Plot ===
plt.figure(figsize=(10, 5))

# Tree-based
plt.subplot(1, 2, 1)
sns.regplot(x=y, y=y_pred_tree, scatter_kws=dict(alpha=0.6))
plt.xlabel("True HI (z-scored)")
plt.ylabel("Predicted HI")
plt.title(f"Model with Interactions (R² = {r2_tree:.2f})")

# Linear
plt.subplot(1, 2, 2)
sns.regplot(x=y, y=y_pred_linear, scatter_kws=dict(alpha=0.6))
plt.xlabel("True HI (z-scored)")
plt.ylabel("Predicted HI")
plt.title(f"Additive Model Only (R² = {r2_linear:.2f})")

plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = X_clean.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# === Assume you already have X_clean and y_clean loaded ===
# X_clean = ... (DataFrame)
# y_clean = ... (1D array)

# === Define model ===
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)

# === Run recursive feature elimination with CV ===
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
    scoring='r2',
    min_features_to_select=1
)

rfecv.fit(X_clean, y_clean)

# === Plot cross-validated R² score vs number of features ===
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], marker='o')
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validated R²")
plt.title("Feature Elimination Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print selected features ===
selected_features = X_clean.columns[rfecv.support_].tolist()
print(f"Optimal number of features: {rfecv.n_features_}")
print("Selected features:", selected_features)
feature_ranks = pd.Series(rfecv.ranking_, index=X_clean.columns).sort_values()
print(feature_ranks)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Step 1: Fit the full model
model_full = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
model_full.fit(X_clean, y_clean)

# Step 2: Get top 3 features
importances = pd.Series(model_full.feature_importances_, index=X_clean.columns)
top3 = importances.sort_values(ascending=False).head(3).index.tolist()
print("Top 3 features:", top3)

# Step 3: Refit model using only top 3 features
X_top3 = X_clean[top3]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_pred_top3 = cross_val_predict(model_full, X_top3, y_clean, cv=kf)

# Step 4: Evaluate
r2 = r2_score(y_clean, y_pred_top3)
r, p = pearsonr(y_clean, y_pred_top3)
print(f"R² (top 3 features) = {r2:.3f}, Pearson r = {r:.3f}, p = {p:.2e}")
#%%
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Fit model
model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)

# Train the model for SHAP
model.fit(X_clean, y_clean)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_clean)

# SHAP summary plot
shap.summary_plot(shap_values, X_clean)

# Plot: predicted vs. true
r2 = r2_score(y_clean, y_pred)
r, p = pearsonr(y_clean, y_pred)

plt.figure(figsize=(5, 5))
sns.regplot(x=y_clean, y=y_pred, line_kws={"color": "red"})
plt.xlabel("True HI (z-scored)")
plt.ylabel("Predicted HI")
plt.title(f"Cross-Validated Predictions (R² = {r2:.2f})")
plt.tight_layout()
plt.show()

print(f"Pearson r = {r:.3f}, p = {p:.2e}")
#%%
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure your model is already trained on X_clean and y_clean
# Example:
# model = GradientBoostingRegressor(...).fit(X_clean, y_clean)

# Use the model trained on top 3 features
explainer = shap.TreeExplainer(model_full)  # model_full is your refit model
interaction_values = explainer.shap_interaction_values(X_top3)

# Compute mean absolute interaction values
mean_interactions = np.abs(interaction_values).mean(axis=0)

# Plot the heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

interaction_df = pd.DataFrame(mean_interactions, index=top3, columns=top3)

plt.figure(figsize=(6, 5))
sns.heatmap(interaction_df, cmap="viridis", annot=True, fmt=".2e")
plt.title("SHAP Interaction Matrix (Top 3 Features)")
plt.tight_layout()
plt.show()

