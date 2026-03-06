import numpy as np
import pandas as pd
import os

# --- Physical constants for the pendulum ---
g = 9.81       # gravity
l = 1.0        # length
mass = 1.0

# --- Generate pendulum Hamiltonian system ---
def pendulum_hamiltonian(q, p):
    return p**2 / (2*mass*l**2) - mass*g*l * np.cos(q)

def grad_q_H(q, p):
    return mass*g*l * np.sin(q)

def grad_p_H(q, p):
    return p / (mass*l**2)

# --- Störmer-Verlet integration (structure-preserving) ---
def stormer_verlet(q0, p0, dt, steps):
    traj_q = np.zeros(steps+1)
    traj_p = np.zeros(steps+1)
    traj_q[0] = q0
    traj_p[0] = p0
    q, p = q0, p0
    for i in range(steps):
        p_half = p - 0.5 * dt * grad_q_H(q, p)
        q_new = q + dt * grad_p_H(q, p_half)
        p_new = p_half - 0.5 * dt * grad_q_H(q_new, p_half)
        traj_q[i+1], traj_p[i+1] = q_new, p_new
        q, p = q_new, p_new
    return traj_q, traj_p

# --- Dataset parameters ---
num_trajs = 500
dt = 0.01
total_time = 7.0  # One max period for |q0|=pi is about 7s (should fit full cycle)
steps = int(total_time / dt)
tvec = np.arange(steps+1) * dt

# For each trajectory, the quarter period is approximately T1/4, but must estimate for each q0
def quarter_period(q0):
    # Small amplitude approx: T = 2*pi*sqrt(l/g)  (full period, but for large q0 need elliptic integral)
    # Here: We'll detect q crossing zero starting at any q0
    return None  # We'll determine dynamically per trajectory

np.random.seed(2026)
initial_q0s = np.random.uniform(-np.pi, np.pi, num_trajs)
initial_p0s = np.zeros(num_trajs)

results = []
for i in range(num_trajs):
    q0 = initial_q0s[i]
    p0 = initial_p0s[i]
    traj_q, traj_p = stormer_verlet(q0, p0, dt, steps)
    period_cross = None
    # ---- Find the first crossing from q0 to 0 (downward swing), only in correct direction ----
    # The downward swing is when q0 in (0, pi]: look for q crossing zero from above; else from below
    sign = -1 if q0 > 0 else 1
    # Find index where sign(traj_q) != sign(q0), the first time the pendulum crosses q=0 in downward direction
    for idx in range(1, len(traj_q)):
        if (qe := traj_q[idx]) * q0 < 0:
            # It crossed 0; check direction of crossing matches downward
            if (sign < 0 and qe < 0) or (sign > 0 and qe > 0):
                period_cross = idx
                break
    if period_cross is None:
        # Use 1/4 small amplitude period as fallback for extremely small amplitudes (should rarely occur)
        period_cross = int(np.pi/2 / np.sqrt(g/l) / dt)
    # Compose trajectory row-by-row
    for j in range(len(traj_q)):
        in_train = (j <= period_cross)
        results.append(dict(traj=i, t=tvec[j], q=traj_q[j], p=traj_p[j], train=in_train, q0=q0))

df = pd.DataFrame(results)

# --- Split into files ---
os.makedirs('Data/Pendulum_MLP', exist_ok=True)
df.to_hdf('Data/Pendulum_MLP/pendulum_full.h5', key='trajectories', mode='w')
# Save train and test masks
df[df['train']].to_hdf('Data/Pendulum_MLP/pendulum_train.h5', key='trajs', mode='w')
df[~df['train']].to_hdf('Data/Pendulum_MLP/pendulum_test.h5', key='trajs', mode='w')

print(f"Generated {num_trajs} pendulum trajectories.")
print(f"Train set: {df['train'].sum()} points; Test set: {(~df['train']).sum()} points.")

# Also save a summary for plotting region cut-off per traj
df_cross = df[df['train']].groupby('traj').tail(1)[['traj','t']]
df_cross.to_csv('Data/Pendulum_MLP/quarter_period.csv', index=False)