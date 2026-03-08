import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Physical constants ---
g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0

def energy_terms(q1, q2, p1, p2):
    delta = q1 - q2
    a = (m1 + m2) * l1 * l1
    b = m2 * l1 * l2 * np.cos(delta)
    c = m2 * l2 * l2
    det = a * c - b * b
    inv11 = c / det
    inv12 = -b / det
    inv22 = a / det
    K = 0.5 * (inv11 * p1 * p1 + 2 * inv12 * p1 * p2 + inv22 * p2 * p2)
    V = -(m1 + m2) * g * l1 * np.cos(q1) - m2 * g * l2 * np.cos(q2)
    H = K + V
    return K, V, H

def xy2(q1, q2):
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q2)
    y2 = y1 - l2 * np.cos(q2)
    return x2, y2

os.makedirs('Results/DoublePendulum_MLP', exist_ok=True)

# --- Load data ---
full_df = pd.read_hdf('Data/DoublePendulum/doublependulum_full.h5', key='trajectories')
pred_df = pd.read_hdf('NeuralNets/DoublePendulum_MLP/mlp_predictions.h5', key='preds')

# ====== 1) XY Trajectory (ONLY traj=0) ======
traj = 0
ref = full_df[full_df['traj'] == traj].sort_values('t')
pred = pred_df[pred_df['traj'] == traj].sort_values('t')
t = ref['t'].values
train_mask = (t <= 5.0)

x2_true, y2_true = xy2(ref['q1'].values, ref['q2'].values)
x2_pred, y2_pred = xy2(pred['q1_pred'].values, pred['q2_pred'].values)

# 閰嶈壊
c_true = "#1f3b73"   # 娣辫摑
c_pred = "#f2a241"   # 浜

plt.figure(figsize=(8, 6))
# 鐪熷疄杞ㄨ抗
plt.plot(x2_true[train_mask], y2_true[train_mask], lw=2.6, color=c_true, label='True (t<=5)')
plt.plot(x2_true[~train_mask], y2_true[~train_mask], lw=2.6, color=c_true, ls='--', label='True (t>5)')
# 棰勬祴杞ㄨ抗
plt.plot(x2_pred[train_mask], y2_pred[train_mask], lw=2.2, color=c_pred, label='MLP (t<=5)')
plt.plot(x2_pred[~train_mask], y2_pred[~train_mask], lw=2.2, color=c_pred, ls='--', label='MLP (t>5)')

plt.xlabel("x2")
plt.ylabel("y2")
plt.title("End-Bob Trajectory (Double Pendulum) - Traj 0")
plt.legend()
plt.savefig('Results/DoublePendulum_MLP/xy_traj_0.png', dpi=200)
plt.close()

# ====== 2) Energy vs Time (MULTI-TRAJ, 3 subplots H/K/V) ======
n_examples = 6
example_trajs = np.linspace(0, full_df['traj'].max(), n_examples, dtype=int)

for traj in example_trajs:
    ref = full_df[full_df['traj'] == traj].sort_values('t')
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')
    t = ref['t'].values
    K_true, V_true, H_true = energy_terms(ref['q1'].values, ref['q2'].values, ref['p1'].values, ref['p2'].values)
    K_pred, V_pred, H_pred = energy_terms(pred['q1_pred'].values, pred['q2_pred'].values, pred['p1_pred'].values, pred['p2_pred'].values)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    # H
    axes[0].plot(t, H_true, color=c_true, lw=2.4, label='True H')
    axes[0].plot(t, H_pred, color=c_pred, lw=2.0, ls='--', label='MLP H')
    axes[0].axvline(5.0, color='k', ls=':', lw=2)
    axes[0].set_ylabel("H")
    axes[0].legend()
    # K
    axes[1].plot(t, K_true, color=c_true, lw=2.0, label='True K')
    axes[1].plot(t, K_pred, color=c_pred, lw=1.8, ls='--', label='MLP K')
    axes[1].axvline(5.0, color='k', ls=':', lw=2)
    axes[1].set_ylabel("K")
    axes[1].legend()
    # V
    axes[2].plot(t, V_true, color=c_true, lw=2.0, label='True V')
    axes[2].plot(t, V_pred, color=c_pred, lw=1.8, ls='--', label='MLP V')
    axes[2].axvline(5.0, color='k', ls=':', lw=2)
    axes[2].set_xlabel("Time t")
    axes[2].set_ylabel("V")
    axes[2].legend()

    fig.suptitle(f"Energy vs Time (Double Pendulum) - Traj {traj}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'Results/DoublePendulum_MLP/energy_traj_{traj}.png', dpi=200)
    plt.close()

# ====== 3) MAE vs Time (all trajectories) ======
mae_by_time = []
t_unique = np.sort(full_df['t'].unique())
for t_this in t_unique:
    mask = (full_df['t'] == t_this)
    true_q1 = full_df.loc[mask, 'q1'].values
    true_q2 = full_df.loc[mask, 'q2'].values
    true_p1 = full_df.loc[mask, 'p1'].values
    true_p2 = full_df.loc[mask, 'p2'].values

    pred_this = pred_df.loc[mask, :]
    pred_q1 = pred_this['q1_pred'].values
    pred_q2 = pred_this['q2_pred'].values
    pred_p1 = pred_this['p1_pred'].values
    pred_p2 = pred_this['p2_pred'].values

    mae = (np.mean(np.abs(true_q1 - pred_q1)) +
           np.mean(np.abs(true_q2 - pred_q2)) +
           np.mean(np.abs(true_p1 - pred_p1)) +
           np.mean(np.abs(true_p2 - pred_p2))) / 4.0
    mae_by_time.append((t_this, mae))

mae_by_time = np.array(mae_by_time)
plt.figure(figsize=(10, 5))
plt.plot(mae_by_time[:,0], mae_by_time[:,1], color=c_true, lw=2.2, label="MLP MAE")
plt.axvline(5.0, color='k', ls=':', lw=2, label='Train/Test split (t=5)')
plt.xlabel("Time t")
plt.ylabel("Mean Abs Error (q1,q2,p1,p2)")
plt.title("MLP MAE vs Time (Double Pendulum)")
plt.legend()
plt.savefig('Results/DoublePendulum_MLP/mae_vs_time.png', dpi=200)
plt.close()

print("XY plot: Results/DoublePendulum_MLP/xy_traj_0.png")
print("Energy plots: Results/DoublePendulum_MLP/energy_traj_[traj].png")
print("MAE plot: Results/DoublePendulum_MLP/mae_vs_time.png")
