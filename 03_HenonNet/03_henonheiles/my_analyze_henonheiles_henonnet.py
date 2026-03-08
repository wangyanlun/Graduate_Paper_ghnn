import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- H茅non-Heiles energy ---
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
        raise ValueError(f"{name} 鏁版嵁涓虹┖鎴栧叏涓?NaN锛歿cols}")
    return df

os.makedirs('Results/HenonHeiles_HENONNET_OOD', exist_ok=True)

# --- Load data ---
full_df = pd.read_hdf('Data/HenonHeiles/henonheiles_full.h5', key='trajectories')
pred_df = pd.read_hdf('NeuralNets/HenonHeiles_HENONNET_OOD/henonnet_rollout_0_50.h5', key='preds')

need_true = ['traj','t','x','y','px','py']
need_pred = ['traj','t','x_pred','y_pred','px_pred','py_pred']
if not set(need_true).issubset(full_df.columns):
    raise ValueError(f"full_df 缂哄け鍒? {need_true}")
if not set(need_pred).issubset(pred_df.columns):
    raise ValueError(f"pred_df 缂哄け鍒? {need_pred}")

full_df = ensure_numeric(full_df, ['x','y','px','py','t'], "full_df")
pred_df = ensure_numeric(pred_df, ['x_pred','y_pred','px_pred','py_pred','t'], "pred_df")

# 閰嶈壊
c_true = "#1f3b73"   # 娣辫摑
c_pred = "#f2a241"   # 浜

# ====== 1) XY Trajectory (ONLY traj=0) ======
traj = 0
ref = full_df[full_df['traj'] == traj].sort_values('t')
pred = pred_df[pred_df['traj'] == traj].sort_values('t')
merged = pd.merge(
    ref[['t','x','y','px','py']],
    pred[['t','x_pred','y_pred','px_pred','py_pred']],
    on='t', how='inner'
)
if merged.empty:
    raise ValueError("traj=0 棰勬祴涓庣湡瀹炴椂闂磋酱鏈榻愶紝鏃犳硶缁樺浘銆?)

t = merged['t'].values
train_mask = (t <= 10.0)
x_true = merged['x'].values
y_true = merged['y'].values
x_pred_vals = merged['x_pred'].values
y_pred_vals = merged['y_pred'].values

plt.figure(figsize=(8, 6))
plt.plot(x_true[train_mask], y_true[train_mask], lw=2.6, color=c_true, label='True (t<=10)')
plt.plot(x_true[~train_mask], y_true[~train_mask], lw=2.6, color=c_true, ls='--', label='True (t>10)')
plt.plot(x_pred_vals[train_mask], y_pred_vals[train_mask], lw=2.2, color=c_pred, label='HenonNet (t<=10)')
plt.plot(x_pred_vals[~train_mask], y_pred_vals[~train_mask], lw=2.2, color=c_pred, ls='--', label='HenonNet (t>10)')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory (H茅non-Heiles) - Traj 0 (HenonNet)")
plt.legend()
plt.savefig('Results/HenonHeiles_HENONNET_OOD/trajectory_xy_traj0.png', dpi=200)
plt.close()

# ====== 2) Energy vs Time (MULTI-TRAJ, 3x2 subplots) ======
n_examples = 6
example_trajs = np.linspace(0, full_df['traj'].max(), n_examples, dtype=int)

for traj in example_trajs:
    ref = full_df[full_df['traj'] == traj].sort_values('t')
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')

    merged = pd.merge(
        ref[['t','x','y','px','py']],
        pred[['t','x_pred','y_pred','px_pred','py_pred']],
        on='t', how='inner'
    )
    if merged.empty:
        continue

    t = merged['t'].values
    K_true, V_true, H_true = energy_terms(merged['x'].values, merged['y'].values, merged['px'].values, merged['py'].values)
    K_pred, V_pred, H_pred = energy_terms(merged['x_pred'].values, merged['y_pred'].values, merged['px_pred'].values, merged['py_pred'].values)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    axes[0,0].plot(t, H_true, color=c_true, lw=2.4)
    axes[0,0].axvline(10.0, color='k', ls=':', lw=2)
    axes[0,0].set_ylabel("H (True)")
    axes[0,0].set_title("True")

    axes[1,0].plot(t, K_true, color=c_true, lw=2.0)
    axes[1,0].axvline(10.0, color='k', ls=':', lw=2)
    axes[1,0].set_ylabel("K (True)")

    axes[2,0].plot(t, V_true, color=c_true, lw=2.0)
    axes[2,0].axvline(10.0, color='k', ls=':', lw=2)
    axes[2,0].set_ylabel("V (True)")
    axes[2,0].set_xlabel("Time t")

    axes[0,1].plot(t, H_true, color=c_true, lw=2.4, label='True H')
    axes[0,1].plot(t, H_pred, color=c_pred, lw=2.0, ls='--', label='HenonNet H')
    axes[0,1].axvline(10.0, color='k', ls=':', lw=2)
    axes[0,1].set_ylabel("H")
    axes[0,1].set_title("True vs Pred")
    axes[0,1].legend()

    axes[1,1].plot(t, K_true, color=c_true, lw=2.0, label='True K')
    axes[1,1].plot(t, K_pred, color=c_pred, lw=1.8, ls='--', label='HenonNet K')
    axes[1,1].axvline(10.0, color='k', ls=':', lw=2)
    axes[1,1].set_ylabel("K")
    axes[1,1].legend()

    axes[2,1].plot(t, V_true, color=c_true, lw=2.0, label='True V')
    axes[2,1].plot(t, V_pred, color=c_pred, lw=1.8, ls='--', label='HenonNet V')
    axes[2,1].axvline(10.0, color='k', ls=':', lw=2)
    axes[2,1].set_xlabel("Time t")
    axes[2,1].set_ylabel("V")
    axes[2,1].legend()

    fig.suptitle(f"Energy vs Time (H茅non-Heiles) - Traj {traj} (HenonNet)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'Results/HenonHeiles_HENONNET_OOD/energy_traj_{traj}.png', dpi=200)
    plt.close()

# ====== 3) MAE vs Time (all trajectories) ======
mae_by_time = []
t_unique = np.sort(full_df['t'].unique())
for t_this in t_unique:
    ref = full_df[full_df['t'] == t_this][['t','x','y','px','py']]
    pred = pred_df[pred_df['t'] == t_this][['t','x_pred','y_pred','px_pred','py_pred']]
    merged = pd.merge(ref, pred, on='t', how='inner')
    if merged.empty:
        continue

    true_x = merged['x'].values
    true_y = merged['y'].values
    true_px = merged['px'].values
    true_py = merged['py'].values
    pred_x = merged['x_pred'].values
    pred_y = merged['y_pred'].values
    pred_px = merged['px_pred'].values
    pred_py = merged['py_pred'].values

    mae = (np.mean(np.abs(true_x - pred_x)) +
           np.mean(np.abs(true_y - pred_y)) +
           np.mean(np.abs(true_px - pred_px)) +
           np.mean(np.abs(true_py - pred_py))) / 4.0
    mae_by_time.append((t_this, mae))

mae_by_time = np.array(mae_by_time)
plt.figure(figsize=(10, 5))
plt.plot(mae_by_time[:,0], mae_by_time[:,1], color=c_true, lw=2.2, label="HenonNet MAE")
plt.axvline(10.0, color='k', ls=':', lw=2, label='Train/Test split (t=10)')
plt.xlabel("Time t")
plt.ylabel("Mean Abs Error (x,y,px,py)")
plt.title("HenonNet MAE vs Time (H茅non-Heiles)")
plt.legend()
plt.savefig('Results/HenonHeiles_HENONNET_OOD/mae_vs_time.png', dpi=200)
plt.close()

print("XY plot: Results/HenonHeiles_HENONNET_OOD/trajectory_xy_traj0.png")
print("Energy plots: Results/HenonHeiles_HENONNET_OOD/energy_traj_[traj].png")
print("MAE plot: Results/HenonHeiles_HENONNET_OOD/mae_vs_time.png")

