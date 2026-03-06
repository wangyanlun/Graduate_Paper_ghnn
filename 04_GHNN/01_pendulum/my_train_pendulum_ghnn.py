import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from ghnn_model import GHNN

# Hyperparams
hidden_dim = 60
hidden_layers = 2
lr = 1e-3
batch_size = 256
max_epochs = 2000
step = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed=2026

os.makedirs('NeuralNets/Pendulum_GHNN', exist_ok=True)
train_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_train.h5', key='trajs')
X_train, y_train = [], []
for traj in train_df['traj'].unique():
    traj_data = train_df[train_df['traj']==traj].sort_values('t')
    X_traj = traj_data[['q','p']].values[:-1]
    y_traj = traj_data[['q','p']].values[1:]
    X_train.append(X_traj)
    y_train.append(y_traj)
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

model = GHNN(hidden_dim, hidden_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

torch.manual_seed(seed)
num_train = X_train.shape[0]
loss_history = []
for epoch in range(1, max_epochs+1):
    idx = np.random.permutation(num_train)
    Xb = X_train[idx]
    yb = y_train[idx]
    total_batches = int(np.ceil(num_train / batch_size))
    running_loss = 0
    for i in range(total_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, num_train)
        inp = Xb[start:end]
        inp = inp.clone().detach().requires_grad_(True)
        pred = model(inp, step=step)
        loss = loss_fn(pred, yb[start:end])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * (end-start)
    avg_loss = running_loss / num_train
    loss_history.append(avg_loss)
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}, MSE: {avg_loss:.6g}")

torch.save(model.state_dict(), 'NeuralNets/Pendulum_GHNN/ghnn_model.pt')
np.savetxt('NeuralNets/Pendulum_GHNN/loss.txt', loss_history)

# Save predictions for full trajectories
full_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_full.h5', key='trajectories')
all_pred = []
model.eval()
for traj in full_df['traj'].unique():
    traj_data = full_df[full_df['traj']==traj].sort_values('t')
    qps = traj_data[['q','p']].values
    pred_qp = [qps[0]]
    for i in range(qps.shape[0] - 1):
        # NOTE: requires_grad must be True for each step!
        inp = torch.tensor(pred_qp[-1], dtype=torch.float32).unsqueeze(0).to(device)
        inp = inp.clone().detach().requires_grad_(True)
        next_pred = model(inp, step=step).cpu().detach().numpy()[0]
        pred_qp.append(next_pred)
    pred_qp = np.array(pred_qp)
    for j in range(len(pred_qp)):
        all_pred.append({
            'traj': traj, 't': traj_data['t'].values[j],
            'q_pred': pred_qp[j,0], 'p_pred': pred_qp[j,1]
        })
pred_df = pd.DataFrame(all_pred)
pred_df.to_hdf('NeuralNets/Pendulum_GHNN/ghnn_predictions.h5', key='preds', mode='w')
print("GHNN training and prediction finished. Results saved.")