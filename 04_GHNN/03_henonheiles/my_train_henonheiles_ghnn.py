import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ghnn_model_henonheiles import GHNN

hidden_dim = 60
hidden_layers = 2
lr = 1e-3
batch_size = 512
max_epochs = 3000
step = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 2026

torch.manual_seed(seed)
np.random.seed(seed)

os.makedirs('NeuralNets/HenonHeiles_GHNN_OOD', exist_ok=True)

# Train [0,10]
train_df = pd.read_hdf('Data/HenonHeiles/henonheiles_train.h5', key='trajs')

X_train, y_train = [], []
for traj in train_df['traj'].unique():
    td = train_df[train_df['traj'] == traj].sort_values('t')
    X_train.append(td[['x','y','px','py']].values[:-1])
    y_train.append(td[['x','y','px','py']].values[1:])
X_train = torch.tensor(np.vstack(X_train), dtype=torch.float32).to(device)
y_train = torch.tensor(np.vstack(y_train), dtype=torch.float32).to(device)

model = GHNN(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

num_train = X_train.shape[0]
loss_history = []
for epoch in range(1, max_epochs + 1):
    idx = np.random.permutation(num_train)
    Xb, yb = X_train[idx], y_train[idx]
    total_batches = int(np.ceil(num_train / batch_size))
    running = 0.0
    for i in range(total_batches):
        s, e = i*batch_size, min((i+1)*batch_size, num_train)
        inp = Xb[s:e].clone().detach().requires_grad_(True)
        pred = model(inp, step=step)
        loss = loss_fn(pred, yb[s:e])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * (e-s)
    avg = running / num_train
    loss_history.append(avg)
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}, MSE: {avg:.6g}")

torch.save(model.state_dict(), 'NeuralNets/HenonHeiles_GHNN_OOD/ghnn_model.pt')
np.savetxt('NeuralNets/HenonHeiles_GHNN_OOD/loss.txt', loss_history)

# Rollout [0,50] from t=0
full_df = pd.read_hdf('Data/HenonHeiles/henonheiles_full.h5', key='trajectories')

pred_rows = []
model.eval()
for traj in full_df['traj'].unique():
    td = full_df[full_df['traj'] == traj].sort_values('t')
    states_true = td[['x','y','px','py']].values
    tvals = td['t'].values

    pred_states = [states_true[0]]
    for _ in range(len(tvals) - 1):
        inp = torch.tensor(pred_states[-1], dtype=torch.float32).unsqueeze(0).to(device)
        inp = inp.clone().detach().requires_grad_(True)
        pred_states.append(model(inp, step=step).detach().cpu().numpy()[0])

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
    'NeuralNets/HenonHeiles_GHNN_OOD/ghnn_rollout_0_50.h5',
    key='preds',
    mode='w'
)
print("Saved: NeuralNets/HenonHeiles_GHNN_OOD/ghnn_rollout_0_50.h5")
