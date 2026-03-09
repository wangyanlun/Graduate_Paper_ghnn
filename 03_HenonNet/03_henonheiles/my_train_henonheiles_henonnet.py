import os
import numpy as np
import pandas as pd
import torch
torch.cuda.set_per_process_memory_fraction(0.22, device=0)  # 每个任务最多用 ~1.8GB
import torch.nn as nn
import torch.optim as optim

hidden_dim = 128
hidden_layers = 5
lr = 1e-3
batch_size = 512
max_epochs = 3000
step = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 2026

torch.manual_seed(seed)
np.random.seed(seed)

os.makedirs('NeuralNets/HenonHeiles_HENONNET_OOD', exist_ok=True)

train_df = pd.read_hdf('Data/HenonHeiles/henonheiles_train.h5', key='trajs')

X_train, y_train = [], []
for traj in train_df['traj'].unique():
    td = train_df[train_df['traj'] == traj].sort_values('t')
    X_train.append(td[['x','y','px','py']].values[:-1])
    y_train.append(td[['x','y','px','py']].values[1:])
X_train = torch.tensor(np.vstack(X_train), dtype=torch.float32).to(device)
y_train = torch.tensor(np.vstack(y_train), dtype=torch.float32).to(device)

class HenonMapModule(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, q, p, step=1.0):
        p_new = p - step * self.f(q)
        q_new = q + step * self.f(p_new)
        return q_new, p_new

class HenonNet(nn.Module):
    def __init__(self, dim, hidden_dim, layers):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([HenonMapModule(dim, hidden_dim) for _ in range(layers)])
    def forward(self, x, step=1.0):
        q, p = x[..., :self.dim], x[..., self.dim:]
        for m in self.layers:
            q, p = m(q, p, step)
        return torch.cat([q, p], dim=-1)

dim = 2
model = HenonNet(dim, hidden_dim, hidden_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

num_train = X_train.shape[0]
loss_history = []
for epoch in range(1, max_epochs+1):
    idx = np.random.permutation(num_train)
    Xb, yb = X_train[idx], y_train[idx]
    total_batches = int(np.ceil(num_train / batch_size))
    running = 0.0
    model.train()
    for i in range(total_batches):
        s, e = i*batch_size, min((i+1)*batch_size, num_train)
        pred = model(Xb[s:e], step=step)
        loss = loss_fn(pred, yb[s:e])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * (e-s)
    avg = running / num_train
    loss_history.append(avg)
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}, MSE: {avg:.6g}")

torch.save(model.state_dict(), 'NeuralNets/HenonHeiles_HENONNET_OOD/henonnet_model.pt')
np.savetxt('NeuralNets/HenonHeiles_HENONNET_OOD/loss.txt', loss_history)

# Rollout [0,50] from t=0
full_df = pd.read_hdf('Data/HenonHeiles/henonheiles_full.h5', key='trajectories')

pred_rows = []
model.eval()
with torch.no_grad():
    for traj in full_df['traj'].unique():
        td = full_df[full_df['traj'] == traj].sort_values('t')
        states_true = td[['x','y','px','py']].values
        tvals = td['t'].values

        pred_states = [states_true[0]]
        for _ in range(len(tvals) - 1):
            inp = torch.tensor(pred_states[-1], dtype=torch.float32).unsqueeze(0).to(device)
            pred_states.append(model(inp, step=step).cpu().numpy()[0])

        pred_states = np.asarray(pred_states)
        for k in range(len(tvals)):
            pred_rows.append({
                'traj': int(traj), 't': float(tvals[k]),
                'x_pred': float(pred_states[k,0]),
                'y_pred': float(pred_states[k,1]),
                'px_pred': float(pred_states[k,2]),
                'py_pred': float(pred_states[k,3]),
            })

pd.DataFrame(pred_rows).to_hdf(
    'NeuralNets/HenonHeiles_HENONNET_OOD/henonnet_rollout_0_50.h5',
    key='preds',
    mode='w'
)
print("Saved: NeuralNets/HenonHeiles_HENONNET_OOD/henonnet_rollout_0_50.h5")
