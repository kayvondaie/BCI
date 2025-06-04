import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import pearsonr
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re
from pysr import PySRRegressor

# === Load and unpack the .npy file ===
npy_path = r"C:\Users\kayvon.daie\OneDrive - Allen Institute\Documents\Data\Figures 2025\Hebbian index fits\hebbian_fit_inputs.npy"
data_dict = np.load(npy_path, allow_pickle=True).item()

HIb = data_dict["HIb"]
THR = data_dict["THR"]
RT  = data_dict["RT"]
RPE = data_dict["RPE"]
HIT = data_dict["HIT"]

print(f"Loaded {len(HIb)} sessions")
print(f"Session 0 HIb shape: {np.shape(HIb[0])}")
print(f"Session 0 RT shape: {np.shape(RT[0])}")

# Fix THR NaNs
for i in range(len(THR)):
    THR[i][0] = THR[i][1]
    ind = np.where(np.isnan(THR[i]))[0]
    if len(ind) > 0:
        THR[i][ind] = THR[i][ind[0]-1]
    THR[i][-1] = THR[i][-2]

# Build full dataset (X_feat and y_feat)
X_list = []
y_list = []
for i in range(len(HIb)):
    n = len(HIb[i])
    if n == 0 or np.nanstd(HIb[i]) == 0:
        continue
    hi = np.asarray(HIb[i])
    rt = np.asarray(RT[i])[:n]
    rpe = np.asarray(RPE[i])[:n]
    hit = np.asarray(HIT[i])[:n]
    thr = np.asarray(THR[i])[:n]
    if len(thr) <= 2 or thr[2] == 0:
        continue
    thr_norm = (thr) / thr[2]
    delta_hit = np.concatenate([[0], np.diff(hit)])[:n]
    delta_thr = np.concatenate([[0], np.diff(thr_norm)])[:n]

    df = pd.DataFrame({
        'RPE': rpe,
        'Delta_HIT': delta_hit,
        'Delta_THR': delta_thr,
        'RT': rt,
        'HIT': hit,
        'THR': thr_norm
    })
    if not df.isnull().values.any():
        X_list.append(df)
        #y_list.append((hi - np.nanmean(hi)) / np.nanstd(hi))
        y_list.append((hi) / np.nanstd(hi))

X_feat = pd.concat(X_list, ignore_index=True)
y_feat = np.concatenate(y_list)

# === Run symbolic regression with PySR ===
features = ['RPE','Delta_HIT','Delta_THR','RT','HIT','THR']
X_sub = X_feat[features].to_numpy()
y_sub = y_feat

model = PySRRegressor(
    unary_operators=["sin","tanh"],
    binary_operators=["+", "-", "*", "/"],
    model_selection="best",
    niterations=50,
    population_size=20,
    select_k_features=6,
    verbosity=1,
    timeout_in_seconds=600,
    maxsize=10,
    constraints={op: {"max_count": 1} for op in features}
)

model.fit(X_sub, y_sub, variable_names=features)
print(model.equations_)

# === Cross-validate discovered equations ===
cv = KFold(n_splits=5, shuffle=True, random_state=0)
rows = []
sympy_locals = {"sig": lambda x: 1/(1+sp.exp(-x)), "exp": sp.exp}
y_pred_list = []

for _, row in model.equations_.iterrows():
    eq_str = row["equation"]
    try:
        expr = sp.sympify(eq_str, locals=sympy_locals)
        required_vars = [str(s) for s in expr.free_symbols]
        feature_subset = [feat for feat in features if feat in required_vars]
        if not feature_subset:
            raise ValueError("No valid input features for equation")

        f = sp.lambdify(
            feature_subset,
            expr,
            modules=[{"sig": lambda x: 1/(1+np.exp(-x)), "exp": np.exp}, "numpy"]
        )

        class SymbEstimator:
            def __init__(self, func): 
                self.f = func
            def fit(self, X, y): 
                return self
            def predict(self, X): 
                return self.f(*[X[:, i] for i in range(X.shape[1])])
            def get_params(self, deep=False): 
                return {"func": self.f}
            def set_params(self, func=None): 
                self.f = func
                return self

        X_input = X_feat[feature_subset].to_numpy()
        y_cv = cross_val_predict(SymbEstimator(f), X_input, y_feat, cv=cv)
        r_cv, p_cv = pearsonr(y_feat, y_cv)
        rows.append({
            "complexity": row.get("complexity", np.nan),
            "loss":       row.get("loss", np.nan),
            "r_cv":       r_cv,
            "p_cv":       p_cv,
            "equation":   eq_str
        })
        y_pred_list.append(y_cv)
    except Exception as e:
        print(f"Failed to evaluate equation: {eq_str}\nError: {e}")

if rows:
    cv_df = pd.DataFrame(rows).sort_values("r_cv", ascending=False)
    print(cv_df.head(10))

    top7 = cv_df.iloc[:7].copy()
    top7["y_pred"] = y_pred_list[:7]
    neg_log_p = -np.log10(top7["p_cv"])
    labels = top7["equation"].tolist()

    def truncate_nums(s, decimals=3):
        pattern = rf"(-?\d+\.\d{{{decimals}}})\d*"
        return re.sub(pattern, r"\1", s)

    labels_short = [truncate_nums(lbl) for lbl in labels]

    plt.figure(figsize=(10,4))
    plt.bar(np.arange(len(neg_log_p)), neg_log_p)
    plt.xticks(np.arange(len(labels_short)), labels_short, rotation=45, ha='right')
    plt.ylabel('-log10(p-value)')
    plt.title('Top 7 Symbolic Rules by Cross-Validated Significance')
    plt.tight_layout()
    plt.show()
else:
    print("No equations could be evaluated successfully.")

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Plot cross-validated predictions vs true values for selected model
selected_index = 4 # Change this index to select a different model from top7

if 'top7' in globals() and 'y_feat' in globals():
    selected_model = top7.iloc[selected_index]
    if 'y_pred' in selected_model:
        y_cv = selected_model['y_pred']
        # Recompute Pearson correlation and p-value
        r_plot, p_plot = pearsonr(y_feat, y_cv)

        plt.figure(figsize=(5, 5))
        pf.mean_bin_plot(y_feat,y_cv,5,1,1,'k')
        plt.xlabel("True HI")
        plt.ylabel("Predicted HI")
        plt.title(f"Model {selected_index + 1}: Predicted vs. True HI\nr = {r_plot:.3f}, p = {p_plot:.2e}")
        plt.tight_layout()
        plt.show()
    else:
        print("Selected model has no predictions stored (y_pred missing).")
else:
    print("Required variables 'top7' or 'y_feat' not found in the global scope.")

#%%
import matplotlib.pyplot as plt
import numpy as np

# Compute normalized HI and RT vectors across all sessions
y_all = []
x_all = []

for i in range(len(HIb)):
    hi = np.asarray(HIb[i])
    rt = np.asarray(RT[i])
    n = len(hi)
    if n == 0 or np.nanstd(hi) == 0:
        continue
    if len(rt) < n:
        continue

    hi_z = (hi - np.nanmean(hi)) / np.nanstd(hi)
    rt_valid = rt[:n]

    # Remove NaNs
    mask = ~np.isnan(hi_z) & ~np.isnan(rt_valid)
    y_all.append(hi_z[mask])
    x_all.append(rt_valid[mask])

# Concatenate across sessions
y_all = np.concatenate(y_all)
x_all = np.concatenate(x_all)

# Plot
plt.figure(figsize=(5, 5))
ind = np.where(x_all < 20)[0]
pf.mean_bin_plot(x_all[ind],y_all[ind],5,1,1,'k')
r,p = pearsonr(x_all[ind],y_all[ind])
print(p)
plt.xlabel("RT")
plt.ylabel("Z-scored HI")
plt.title("HI vs RT (all sessions)")
plt.tight_layout()
plt.show()
#%%
from scipy.stats import ttest_ind
X_list = []
y_list = []
rt_list = []
for i in range(len(HIb)):
    n = len(HIb[i])
    if n == 0 or np.nanstd(HIb[i]) == 0:
        continue
    hi = np.asarray(HIb[i])
    thr = np.asarray(THR[i])[:n]
    if len(thr) <= 2 or thr[2] == 0:
        continue
    thr_norm = (thr ) / thr[2]
    X_list.append(thr_norm)
    y_list.append((hi) / np.nanstd(hi))
    rt_list.append(RT[i])

x = np.concatenate(X_list)
y = np.concatenate(y_list)
z = np.concatenate(rt_list)
ind = np.where(np.isnan(x)==0)[0]
x = x[ind];y = y[ind]; z=z[ind]

pf.mean_bin_plot(y,x,10,1,1,'k');plt.xlabel('HI');plt.ylabel('THR')

b = 0
ind_p = np.where((y > 0) & (x > b))[0]
ind_n = np.where((y < 0) & (x > b))[0]
plt.show()
plt.bar((1,2),(np.nanmean(z[ind_n]),np.nanmean(z[ind_p])))
t_stat, p = ttest_ind(z[ind_n],z[ind_p]);print(p)

plt.show()
pf.mean_bin_plot(y[ind_n],z[ind_n],3,1,1,'k')
pf.mean_bin_plot(y[ind_p],z[ind_p],3,1,1,'r')
plt.xlabel('HI');plt.ylabel('RT')

#%%
plt.figure(figsize = (7,3))
plt.subplot(121)
b = 1.5
ind = np.where((z < 20))[0]
pf.mean_bin_plot((z[ind]),y[ind],9,1,1,'k')
r,p = pearsonr(x[ind],y[ind])
print(p)
plt.xlabel('Time to reward (s)')
plt.ylabel('Hebbian Index')

plt.subplot(122)
pf.mean_bin_plot(x,y,3,1,1,'k');plt.xlabel('THR');plt.ylabel('HI')
#%%
ind = np.where((z<20) & (x < 1.1));
pf.mean_bin_plot(y[ind],z[ind],7,1,1,'k')

ind = np.where((z<20) & (x > 1.1));
pf.mean_bin_plot(y[ind],z[ind],7,1,1,'r')
plt.xlabel("HI")
plt.ylabel("RT")
plt.plot([],[],'r',label = 'high thr')
plt.plot([],[],'k',label = 'low thr')
plt.legend()
#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Construct design matrix
df = pd.DataFrame({
    'HI': y,
    'RT_shifted': z,
    'THR_norm': x,
    'abs_HI': np.abs(y),
    'log_THR_norm': np.log(x + 1.01)  # offset to avoid log(0)
})

# Fit: HI ~ RT - 3
X1 = sm.add_constant(df[['RT_shifted']])
model1 = sm.OLS(df['HI'], X1).fit()
print("HI ~ RT:")
print(model1.summary())

# Fit: abs(HI) ~ THR_norm
X2 = sm.add_constant(df[['THR_norm', 'log_THR_norm']])
model2 = sm.OLS(df['abs_HI'], X2).fit()
print("\nabs(HI) ~ THR_norm:")
print(model2.summary())

import matplotlib.pyplot as plt

# HI vs. RT-3
plt.figure(figsize=(6, 4))
plt.scatter(df['RT_shifted'], df['HI'], alpha=0.3)
plt.plot(df['RT_shifted'], model1.predict(X1), color='red')
plt.xlabel("RT - 3")
plt.ylabel("HI")
plt.title("HI vs RT (shifted)")
plt.tight_layout()

# abs(HI) vs. THR_norm
plt.figure(figsize=(6, 4))
plt.scatter(df['THR_norm'], df['abs_HI'], alpha=0.3)
plt.plot(df['THR_norm'], model2.predict(X2), color='green')
plt.xlabel("THR_norm")
plt.ylabel("abs(HI)")
plt.title("abs(HI) vs THR_norm")
plt.tight_layout()
plt.show()








