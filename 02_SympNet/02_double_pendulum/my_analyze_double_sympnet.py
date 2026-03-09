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

def ensure_numeric(df, cols, name):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[cols].isna().any().any():
        df = df.dropna(subset=cols).copy()
    if df.empty:
        raise ValueError(f"{name} 数据为空或全为 NaN：{cols}")
    return df

os.makedirs('Results/DoublePendulum_SYMPNET', exist_ok=True)

# --- Load data ---
full_df = pd.read_hdf('Data/DoublePendulum_MLP/doublependulum_full.h5', key='trajectories')
pred_df = pd.read_hdf('NeuralNets/DoublePendulum_SYMPNET/sympnet_predictions.h5', key='preds')

# === 自动修正预测列名 ===
if {'q1','q2','p1','p2'}.issubset(pred_df.columns) and not {'q1_pred','q2_pred','p1_pred','p2_pred'}.issubset(pred_df.columns):
    pred_df = pred_df.rename(columns={
        'q1': 'q1_pred',
        'q2': 'q2_pred',
        'p1': 'p1_pred',
        'p2': 'p2_pred'
    })

# === 列检查 ===
need_true = ['traj','t','q1','q2','p1','p2']
need_pred = ['traj','t','q1_pred','q2_pred','p1_pred','p2_pred']
if not set(need_true).issubset(full_df.columns):
    raise ValueError(f"full_df 缺失列: {need_true}")
if not set(need_pred).issubset(pred_df.columns):
    raise ValueError(f"pred_df 缺失列: {need_pred}")

# === 强制数值化 ===
full_df = ensure_numeric(full_df, ['q1','q2','p1','p2','t'], "full_df")
pred_df = ensure_numeric(pred_df, ['q1_pred','q2_pred','p1_pred','p2_pred','t'], "pred_df")

# 配色
c_true = "#1f3b73"   # 深蓝
c_pred = "#f2a241"   # 亮橙

# ====== 1) XY Trajectory (ONLY traj=0) ======
traj = 0
ref = full_df[full_df['traj'] == traj].sort_values('t')
pred = pred_df[pred_df['traj'] == traj].sort_values('t')

# 按 t 对齐，防止错位
merged = pd.merge(
    ref[['t','q1','q2','p1','p2']],
    pred[['t','q1_pred','q2_pred','p1_pred','p2_pred']],
    on='t', how='inner'
)

if merged.empty:
    raise ValueError("traj=0 预测与真实时间轴未对齐，无法绘图。")

t = merged['t'].values
train_mask = (t <= 5.0)

x2_true, y2_true = xy2(merged['q1'].values, merged['q2'].values)
x2_pred, y2_pred = xy2(merged['q1_pred'].values, merged['q2_pred'].values)

plt.figure(figsize=(8, 6))
plt.plot(x2_true[train_mask], y2_true[train_mask], lw=2.6, color=c_true, label='True (t<=5)')
plt.plot(x2_true[~train_mask], y2_true[~train_mask], lw=2.6, color=c_true, ls='--', label='True (t>5)')
plt.plot(x2_pred[train_mask], y2_pred[train_mask], lw=2.2, color=c_pred, label='SympNet (t<=5)')
plt.plot(x2_pred[~train_mask], y2_pred[~train_mask], lw=2.2, color=c_pred, ls='--', label='SympNet (t>5)')
plt.xlabel("x2")
plt.ylabel("y2")
plt.title("End-Bob Trajectory (Double Pendulum) - Traj 0 (SympNet)")
plt.legend()
plt.savefig('Results/DoublePendulum_SYMPNET/xy_traj_0.png', dpi=200)
plt.close()

# ====== 2) Energy vs Time (MULTI-TRAJ, 3 subplots H/K/V) ======
n_examples = 6
example_trajs = np.linspace(0, full_df['traj'].max(), n_examples, dtype=int)

for traj in example_trajs:
    ref = full_df[full_df['traj'] == traj].sort_values('t')
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')

    merged = pd.merge(
        ref[['t','q1','q2','p1','p2']],
        pred[['t','q1_pred','q2_pred','p1_pred','p2_pred']],
        on='t', how='inner'
    )
    if merged.empty:
        continue

    t = merged['t'].values
    K_true, V_true, H_true = energy_terms(
        merged['q1'].values, merged['q2'].values, merged['p1'].values, merged['p2'].values
    )
    K_pred, V_pred, H_pred = energy_terms(
        merged['q1_pred'].values, merged['q2_pred'].values, merged['p1_pred'].values, merged['p2_pred'].values
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    # Left column: True only
    axes[0,0].plot(t, H_true, color=c_true, lw=2.4)
    axes[0,0].axvline(5.0, color='k', ls=':', lw=2)
    axes[0,0].set_ylabel("H (True)")
    axes[0,0].set_title("True")

    axes[1,0].plot(t, K_true, color=c_true, lw=2.0)
    axes[1,0].axvline(5.0, color='k', ls=':', lw=2)
    axes[1,0].set_ylabel("K (True)")

    axes[2,0].plot(t, V_true, color=c_true, lw=2.0)
    axes[2,0].axvline(5.0, color='k', ls=':', lw=2)
    axes[2,0].set_ylabel("V (True)")
    axes[2,0].set_xlabel("Time t")

    # Right column: Compare
    axes[0,1].plot(t, H_true, color=c_true, lw=2.4, label='True H')
    axes[0,1].plot(t, H_pred, color=c_pred, lw=2.0, ls='--', label='SympNet H')
    axes[0,1].axvline(5.0, color='k', ls=':', lw=2)
    axes[0,1].set_ylabel("H")
    axes[0,1].set_title("True vs Pred")
    axes[0,1].legend()

    axes[1,1].plot(t, K_true, color=c_true, lw=2.0, label='True K')
    axes[1,1].plot(t, K_pred, color=c_pred, lw=1.8, ls='--', label='SympNet K')
    axes[1,1].axvline(5.0, color='k', ls=':', lw=2)
    axes[1,1].set_ylabel("K")
    axes[1,1].legend()

    axes[2,1].plot(t, V_true, color=c_true, lw=2.0, label='True V')
    axes[2,1].plot(t, V_pred, color=c_pred, lw=1.8, ls='--', label='SympNet V')
    axes[2,1].axvline(5.0, color='k', ls=':', lw=2)
    axes[2,1].set_xlabel("Time t")
    axes[2,1].set_ylabel("V")
    axes[2,1].legend()

    fig.suptitle(f"Energy vs Time (Double Pendulum) - Traj {traj} (SympNet)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'Results/DoublePendulum_SYMPNET/energy_traj_{traj}.png', dpi=200)
    plt.close()

# ====== 3) MAE vs Time (all trajectories) ======
mae_by_time = []
t_unique = np.sort(full_df['t'].unique())
for t_this in t_unique:
    ref = full_df[full_df['t'] == t_this][['t','q1','q2','p1','p2']]
    pred = pred_df[pred_df['t'] == t_this][['t','q1_pred','q2_pred','p1_pred','p2_pred']]
    merged = pd.merge(ref, pred, on='t', how='inner')
    if merged.empty:
        continue

    true_q1 = merged['q1'].values
    true_q2 = merged['q2'].values
    true_p1 = merged['p1'].values
    true_p2 = merged['p2'].values
    pred_q1 = merged['q1_pred'].values
    pred_q2 = merged['q2_pred'].values
    pred_p1 = merged['p1_pred'].values
    pred_p2 = merged['p2_pred'].values

    mae = (np.mean(np.abs(true_q1 - pred_q1)) +
           np.mean(np.abs(true_q2 - pred_q2)) +
           np.mean(np.abs(true_p1 - pred_p1)) +
           np.mean(np.abs(true_p2 - pred_p2))) / 4.0
    mae_by_time.append((t_this, mae))

mae_by_time = np.array(mae_by_time)
plt.figure(figsize=(10, 5))
plt.plot(mae_by_time[:,0], mae_by_time[:,1], color=c_true, lw=2.2, label="SympNet MAE")
plt.axvline(5.0, color='k', ls=':', lw=2, label='Train/Test split (t=5)')
plt.xlabel("Time t")
plt.ylabel("Mean Abs Error (q1,q2,p1,p2)")
plt.title("SympNet MAE vs Time (Double Pendulum)")
plt.legend()
plt.savefig('Results/DoublePendulum_SYMPNET/mae_vs_time.png', dpi=200)
plt.close()

print("XY plot: Results/DoublePendulum_SYMPNET/xy_traj_0.png")
print("Energy plots: Results/DoublePendulum_SYMPNET/energy_traj_[traj].png")
print("MAE plot: Results/DoublePendulum_SYMPNET/mae_vs_time.png")