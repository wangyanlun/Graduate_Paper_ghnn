import os
import numpy as np
import pandas as pd

# ============================================================
# Henon–Heiles dataset generation (match pendulum style)
# dt = 0.01, E = 0.125, T = 50
# Train:  t in [0,10]  -> henonheiles_train.h5 (key='trajs')
# Rollout/GT: t in [0,50] -> henonheiles_full.h5 (key='trajectories')
# ============================================================

def H_henonheiles(x, y, px, py):
    return 0.5*(px**2 + py**2) + 0.5*(x**2 + y**2) + (x**2)*y - (y**3)/3.0

def dU_dx(x, y):
    # U(x,y)=1/2(x^2+y^2)+x^2 y - 1/3 y^3
    return x + 2.0*x*y

def dU_dy(x, y):
    return y + x**2 - y**2

def stormer_verlet(x0, y0, px0, py0, dt, steps):
    xs = np.zeros(steps+1)
    ys = np.zeros(steps+1)
    pxs = np.zeros(steps+1)
    pys = np.zeros(steps+1)

    x, y, px, py = x0, y0, px0, py0
    xs[0], ys[0], pxs[0], pys[0] = x, y, px, py

    for i in range(steps):
        # kick half step
        px_half = px - 0.5*dt*dU_dx(x, y)
        py_half = py - 0.5*dt*dU_dy(x, y)

        # drift full step (T=1/2(p^2) => dT/dp=p)
        x_new = x + dt*px_half
        y_new = y + dt*py_half

        # kick half step
        px_new = px_half - 0.5*dt*dU_dx(x_new, y_new)
        py_new = py_half - 0.5*dt*dU_dy(x_new, y_new)

        x, y, px, py = x_new, y_new, px_new, py_new
        xs[i+1], ys[i+1], pxs[i+1], pys[i+1] = x, y, px, py

    return xs, ys, pxs, pys

def sample_initial_conditions_energy_shell(num_trajs, E=0.125, seed=2026):
    """
    Enforce exact energy shell at t=0:
      y0=0, py0=0
      sample x0 uniformly in [-sqrt(2E), sqrt(2E)]
      set px0 = +/- sqrt(2E - x0^2)
    """
    rng = np.random.default_rng(seed)
    max_abs_x = np.sqrt(2.0*E)
    x0s = rng.uniform(-max_abs_x, max_abs_x, size=num_trajs)
    y0s = np.zeros(num_trajs)
    py0s = np.zeros(num_trajs)

    px0_mag = np.sqrt(np.clip(2.0*E - x0s**2, 0.0, None))
    signs = rng.choice([-1.0, 1.0], size=num_trajs)
    px0s = signs * px0_mag
    return x0s, y0s, px0s, py0s

def main():
    out_dir = os.path.join('Data', 'HenonHeiles')
    os.makedirs(out_dir, exist_ok=True)

    num_trajs = 500
    dt = 0.01
    T_total = 50.0
    steps = int(T_total / dt)
    tvec = np.arange(steps + 1) * dt

    E = 0.125
    seed = 2026
    train_t_max = 10.0

    x0s, y0s, px0s, py0s = sample_initial_conditions_energy_shell(num_trajs, E=E, seed=seed)

    rows_full = []
    for traj in range(num_trajs):
        xs, ys, pxs, pys = stormer_verlet(x0s[traj], y0s[traj], px0s[traj], py0s[traj], dt, steps)
        for k, t in enumerate(tvec):
            in_train = (t <= train_t_max)
            rows_full.append({
                'traj': traj,
                't': float(t),
                'x': float(xs[k]),
                'y': float(ys[k]),
                'px': float(pxs[k]),
                'py': float(pys[k]),
                'train': in_train,
            })

    full_df = pd.DataFrame(rows_full)
    state_cols = ['x', 'y', 'px', 'py']

    full_df.to_hdf(os.path.join(out_dir, 'henonheiles_full.h5'), key='trajectories', mode='w')
    full_df[full_df['train']][['traj', 't'] + state_cols].to_hdf(os.path.join(out_dir, 'henonheiles_train.h5'), key='trajs', mode='w')
    full_df[~full_df['train']][['traj', 't'] + state_cols].to_hdf(os.path.join(out_dir, 'henonheiles_test.h5'), key='trajs', mode='w')

    meta = {
        'system': 'Henon-Heiles',
        'dt': dt,
        'T_total': T_total,
        'E': E,
        'num_trajs': num_trajs,
        'seed': seed,
        'integrator': 'Stormer-Verlet',
        'train_split': 'time_cutoff',
        'train_t_max': train_t_max,
    }
    pd.Series(meta).to_csv(os.path.join(out_dir, 'meta.csv'))

    print(f"Generated {num_trajs} Henon-Heiles trajectories.")
    print(f"Train set: {full_df['train'].sum()} points; Test set: {(~full_df['train']).sum()} points.")

if __name__ == "__main__":
    main()