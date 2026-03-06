import numpy as np
import pandas as pd
import os

# --- Physical constants for double pendulum ---
g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0

# --- Dynamics (q1, q2, qd1, qd2) ---
def double_pendulum_derivs(state):
    q1, q2, qd1, qd2 = state
    delta = q1 - q2
    s, c = np.sin(delta), np.cos(delta)

    den1 = (m1 + m2) * l1 - m2 * l1 * c * c
    den2 = (l2 / l1) * den1

    qdd1 = (m2 * l1 * qd1 * qd1 * s * c +
            m2 * g * np.sin(q2) * c +
            m2 * l2 * qd2 * qd2 * s -
            (m1 + m2) * g * np.sin(q1)) / den1

    qdd2 = (-m2 * l2 * qd2 * qd2 * s * c +
            (m1 + m2) * (g * np.sin(q1) * c -
                         l1 * qd1 * qd1 * s -
                         g * np.sin(q2))) / den2

    return np.array([qd1, qd2, qdd1, qdd2], dtype=np.float64)

def rk4_step(state, dt):
    k1 = double_pendulum_derivs(state)
    k2 = double_pendulum_derivs(state + 0.5 * dt * k1)
    k3 = double_pendulum_derivs(state + 0.5 * dt * k2)
    k4 = double_pendulum_derivs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def qp_from_qv(q1, q2, qd1, qd2):
    delta = q1 - q2
    a = (m1 + m2) * l1 * l1
    b = m2 * l1 * l2 * np.cos(delta)
    c = m2 * l2 * l2
    p1 = a * qd1 + b * qd2
    p2 = b * qd1 + c * qd2
    return p1, p2

# --- Dataset parameters ---
num_trajs = 400
dt = 0.01
t_end = 10.0
steps = int(t_end / dt)
tvec = np.arange(steps + 1) * dt

np.random.seed(2026)
# Random initial angles, small velocities
q1_0 = np.random.uniform(-np.pi, np.pi, num_trajs)
q2_0 = np.random.uniform(-np.pi, np.pi, num_trajs)
qd1_0 = np.random.uniform(-0.5, 0.5, num_trajs)
qd2_0 = np.random.uniform(-0.5, 0.5, num_trajs)

results = []
for i in range(num_trajs):
    state = np.array([q1_0[i], q2_0[i], qd1_0[i], qd2_0[i]], dtype=np.float64)
    for j, t in enumerate(tvec):
        q1, q2, qd1, qd2 = state
        p1, p2 = qp_from_qv(q1, q2, qd1, qd2)
        in_train = (t <= 5.0)
        results.append(dict(
            traj=i, t=t, q1=q1, q2=q2, p1=p1, p2=p2, train=in_train,
            qd1=qd1, qd2=qd2
        ))
        if j < steps:
            state = rk4_step(state, dt)

df = pd.DataFrame(results)

os.makedirs('Data/DoublePendulum_MLP', exist_ok=True)
df.to_hdf('Data/DoublePendulum_MLP/doublependulum_full.h5', key='trajectories', mode='w')
df[df['train']].to_hdf('Data/DoublePendulum_MLP/doublependulum_train.h5', key='trajs', mode='w')
df[~df['train']].to_hdf('Data/DoublePendulum_MLP/doublependulum_test.h5', key='trajs', mode='w')

print(f"Generated {num_trajs} double pendulum trajectories.")
print(f"Train set: {df['train'].sum()} points; Test set: {(~df['train']).sum()} points.")