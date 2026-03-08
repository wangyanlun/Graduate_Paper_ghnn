import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# --- Henon-Heiles energy ---
def energy_terms(x, y, px, py):
    K = 0.5 * (px * px + py * py)
    V = 0.5 * (x * x + y * y) + x * x * y - y * y * y / 3.0
    H = K + V
    return K, V, H


def ensure_numeric(df, cols, name):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[cols].isna().any().any():
        df = df.dropna(subset=cols).copy()
    if df.empty:
        raise ValueError(f"{name} is empty or all-NaN in required columns: {cols}")
    return df


os.makedirs('Results/HenonHeiles_SYMPNET_OOD', exist_ok=True)

# --- Load data ---
full_df = pd.read_hdf('Data/HenonHeiles/henonheiles_full.h5', key='trajectories')
pred_df = pd.read_hdf('NeuralNets/HenonHeiles_SYMPNET_OOD/sympnet_rollout_0_50.h5', key='preds')

# Auto-fix predicted column names if needed.
if {'x', 'y', 'px', 'py'}.issubset(pred_df.columns) and not {'x_pred', 'y_pred', 'px_pred', 'py_pred'}.issubset(pred_df.columns):
    pred_df = pred_df.rename(columns={
        'x': 'x_pred',
        'y': 'y_pred',
        'px': 'px_pred',
        'py': 'py_pred'
    })

need_true = ['traj', 't', 'x', 'y', 'px', 'py']
need_pred = ['traj', 't', 'x_pred', 'y_pred', 'px_pred', 'py_pred']
if not set(need_true).issubset(full_df.columns):
    raise ValueError(f"full_df is missing required columns: {need_true}")
if not set(need_pred).issubset(pred_df.columns):
    raise ValueError(f"pred_df is missing required columns: {need_pred}")

full_df = ensure_numeric(full_df, ['x', 'y', 'px', 'py', 't'], "full_df")
pred_df = ensure_numeric(pred_df, ['x_pred', 'y_pred', 'px_pred', 'py_pred', 't'], "pred_df")

c_true = "#1f3b73"
c_pred = "#f2a241"

# ====== 1) XY Trajectory (ONLY traj=0) ======
traj = 0
ref = full_df[full_df['traj'] == traj].sort_values('t')
pred = pred_df[pred_df['traj'] == traj].sort_values('t')

merged = pd.merge(
    ref[['t', 'x', 'y', 'px', 'py']],
    pred[['t', 'x_pred', 'y_pred', 'px_pred', 'py_pred']],
    on='t',
    how='inner'
)

if merged.empty:
    raise ValueError("traj=0 has no aligned rows between reference and prediction.")

t = merged['t'].values
train_mask = (t <= 10.0)

x_true = merged['x'].values
y_true = merged['y'].values
x_pred_vals = merged['x_pred'].values
y_pred_vals = merged['y_pred'].values

plt.figure(figsize=(8, 6))
plt.plot(x_true[train_mask], y_true[train_mask], lw=2.6, color=c_true, label='True (t<=10)')
plt.plot(x_true[~train_mask], y_true[~train_mask], lw=2.6, color=c_true, ls='--', label='True (t>10)')
plt.plot(x_pred_vals[train_mask], y_pred_vals[train_mask], lw=2.2, color=c_pred, label='SympNet (t<=10)')
plt.plot(x_pred_vals[~train_mask], y_pred_vals[~train_mask], lw=2.2, color=c_pred, ls='--', label='SympNet (t>10)')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory (Henon-Heiles) - Traj 0 (SympNet)")
plt.legend()
plt.savefig('Results/HenonHeiles_SYMPNET_OOD/trajectory_xy_traj0.png', dpi=200)
plt.close()

# ====== 2) Energy vs Time (MULTI-TRAJ, 3x2 subplots H/K/V) ======
n_examples = 6
example_trajs = np.linspace(0, full_df['traj'].max(), n_examples, dtype=int)

for traj in example_trajs:
    ref = full_df[full_df['traj'] == traj].sort_values('t')
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')

    merged = pd.merge(
        ref[['t', 'x', 'y', 'px', 'py']],
        pred[['t', 'x_pred', 'y_pred', 'px_pred', 'py_pred']],
        on='t',
        how='inner'
    )
    if merged.empty:
        continue

    t = merged['t'].values
    K_true, V_true, H_true = energy_terms(merged['x'].values, merged['y'].values, merged['px'].values, merged['py'].values)
    K_pred, V_pred, H_pred = energy_terms(
        merged['x_pred'].values, merged['y_pred'].values, merged['px_pred'].values, merged['py_pred'].values
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    axes[0, 0].plot(t, H_true, color=c_true, lw=2.4)
    axes[0, 0].axvline(10.0, color='k', ls=':', lw=2)
    axes[0, 0].set_ylabel("H (True)")
    axes[0, 0].set_title("True")

    axes[1, 0].plot(t, K_true, color=c_true, lw=2.0)
    axes[1, 0].axvline(10.0, color='k', ls=':', lw=2)
    axes[1, 0].set_ylabel("K (True)")

    axes[2, 0].plot(t, V_true, color=c_true, lw=2.0)
    axes[2, 0].axvline(10.0, color='k', ls=':', lw=2)
    axes[2, 0].set_ylabel("V (True)")
    axes[2, 0].set_xlabel("Time t")

    axes[0, 1].plot(t, H_true, color=c_true, lw=2.4, label='True H')
    axes[0, 1].plot(t, H_pred, color=c_pred, lw=2.0, ls='--', label='SympNet H')
    axes[0, 1].axvline(10.0, color='k', ls=':', lw=2)
    axes[0, 1].set_ylabel("H")
    axes[0, 1].set_title("True vs Pred")
    axes[0, 1].legend()

    axes[1, 1].plot(t, K_true, color=c_true, lw=2.0, label='True K')
    axes[1, 1].plot(t, K_pred, color=c_pred, lw=1.8, ls='--', label='SympNet K')
    axes[1, 1].axvline(10.0, color='k', ls=':', lw=2)
    axes[1, 1].set_ylabel("K")
    axes[1, 1].legend()

    axes[2, 1].plot(t, V_true, color=c_true, lw=2.0, label='True V')
    axes[2, 1].plot(t, V_pred, color=c_pred, lw=1.8, ls='--', label='SympNet V')
    axes[2, 1].axvline(10.0, color='k', ls=':', lw=2)
    axes[2, 1].set_xlabel("Time t")
    axes[2, 1].set_ylabel("V")
    axes[2, 1].legend()

    fig.suptitle(f"Energy vs Time (Henon-Heiles) - Traj {traj} (SympNet)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'Results/HenonHeiles_SYMPNET_OOD/energy_traj_{traj}.png', dpi=200)
    plt.close()

# ====== 3) MAE vs Time (all trajectories) ======
merged_all = pd.merge(
    full_df[['traj', 't', 'x', 'y', 'px', 'py']],
    pred_df[['traj', 't', 'x_pred', 'y_pred', 'px_pred', 'py_pred']],
    on=['traj', 't'],
    how='inner'
)

if merged_all.empty:
    raise ValueError("No matched rows between full_df and pred_df on ['traj', 't'].")

merged_all['mae_row'] = (
    np.abs(merged_all['x'] - merged_all['x_pred']) +
    np.abs(merged_all['y'] - merged_all['y_pred']) +
    np.abs(merged_all['px'] - merged_all['px_pred']) +
    np.abs(merged_all['py'] - merged_all['py_pred'])
) / 4.0

mae_by_time_df = merged_all.groupby('t', as_index=False)['mae_row'].mean().sort_values('t')

plt.figure(figsize=(10, 5))
plt.plot(mae_by_time_df['t'].values, mae_by_time_df['mae_row'].values, color=c_true, lw=2.2, label="SympNet MAE")
plt.axvline(10.0, color='k', ls=':', lw=2, label='Train/Test split (t=10)')
plt.xlabel("Time t")
plt.ylabel("Mean Abs Error (x,y,px,py)")
plt.title("SympNet MAE vs Time (Henon-Heiles)")
plt.legend()
plt.savefig('Results/HenonHeiles_SYMPNET_OOD/mae_vs_time.png', dpi=200)
plt.close()

print("XY plot: Results/HenonHeiles_SYMPNET_OOD/trajectory_xy_traj0.png")
print("Energy plots: Results/HenonHeiles_SYMPNET_OOD/energy_traj_[traj].png")
print("MAE plot: Results/HenonHeiles_SYMPNET_OOD/mae_vs_time.png")