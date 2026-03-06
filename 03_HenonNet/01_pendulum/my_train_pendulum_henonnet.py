import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

hidden_layers = 2
hidden_dim = 60
input_dim = 2
output_dim = 2
lr = 1e-3
batch_size = 256
max_epochs = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 2026

os.makedirs('NeuralNets/Pendulum_HENONNET', exist_ok=True)
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

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# --- HénonNet definition (symplectic map) ---
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
        self.layers = nn.ModuleList([HenonMapModule(dim, hidden_dim) for _ in range(layers)])
        self.dim = dim
    def forward(self, x, step=1.0):
        q, p = x[..., 0:self.dim], x[..., self.dim:]
        for mod in self.layers:
            q, p = mod(q, p, step)
        x_out = torch.cat([q, p], dim=-1)
        return x_out

torch.manual_seed(seed)
model = HenonNet(input_dim//2, hidden_dim, hidden_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

num_train = X_train.shape[0]
loss_history = []
for epoch in range(1, max_epochs+1):
    idx = np.random.permutation(num_train)
    Xb = X_train[idx].to(device)
    yb = y_train[idx].to(device)
    total_batches = int(np.ceil(num_train / batch_size))
    running_loss = 0
    for i in range(total_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, num_train)
        inp = Xb[start:end]
        pred = model(inp)
        loss = loss_fn(pred, yb[start:end])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * (end-start)
    avg_loss = running_loss / num_train
    loss_history.append(avg_loss)
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}, MSE: {avg_loss:.6g}")

torch.save(model.state_dict(), 'NeuralNets/Pendulum_HENONNET/henonnet_model.pt')
np.savetxt('NeuralNets/Pendulum_HENONNET/loss.txt', loss_history)

# --- Prediction for full trajectories ---
full_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_full.h5', key='trajectories')
all_pred = []
model.eval()
with torch.no_grad():
    for traj in full_df['traj'].unique():
        traj_data = full_df[full_df['traj']==traj].sort_values('t')
        qps = traj_data[['q','p']].values
        pred_qp = [qps[0]]
        for i in range(qps.shape[0] - 1):
            inp = torch.tensor(pred_qp[-1], dtype=torch.float32).unsqueeze(0).to(device)
            next_pred = model(inp).cpu().numpy()[0]
            pred_qp.append(next_pred)
        pred_qp = np.array(pred_qp)
        for j in range(len(pred_qp)):
            all_pred.append({
                'traj': traj, 't': traj_data['t'].values[j],
                'q_pred': pred_qp[j,0], 'p_pred': pred_qp[j,1]
            })
pred_df = pd.DataFrame(all_pred)
pred_df.to_hdf('NeuralNets/Pendulum_HENONNET/henonnet_predictions.h5', key='preds', mode='w')
print("HenonNet training and prediction finished. Results saved.")