import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

# --------------------- Hyperparameters ------------------------
hidden_layers = 4         # See SympNet/HenonNet for speed comparison if needed
hidden_dim = 128
input_dim = 2             # [q, p]
output_dim = 2            # [q, p] next state
lr = 1e-3
batch_size = 256
max_epochs = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 2026

# --------------------- Data Loading ---------------------------

os.makedirs('NeuralNets/Pendulum_MLP', exist_ok=True)
train_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_train.h5', key='trajs')

# Each data point: (q_i, p_i) -> (q_{i+1}, p_{i+1}) within 1/4 period
X_train = []
y_train = []
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

# --------------------- Model Definition -----------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

torch.manual_seed(seed)
model = MLP(input_dim, hidden_dim, output_dim, hidden_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --------------------- Training Loop --------------------------
loss_history = []
num_train = X_train.shape[0]
for epoch in range(1, max_epochs + 1):
    idx = np.random.permutation(num_train)
    Xb = X_train[idx].to(device)
    yb = y_train[idx].to(device)
    total_batches = int(np.ceil(num_train / batch_size))
    running_loss = 0
    for i in range(total_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, num_train)
        pred = model(Xb[start:end])
        loss = loss_fn(pred, yb[start:end])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * (end-start)
    avg_loss = running_loss / num_train
    loss_history.append(avg_loss)
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}, MSE: {avg_loss:.6g}")

# Save model and loss history
torch.save(model.state_dict(), 'NeuralNets/Pendulum_MLP/mlp_model.pt')
np.savetxt('NeuralNets/Pendulum_MLP/loss.txt', loss_history)
print(f"Training complete. Final MSE loss: {loss_history[-1]:.6g}")

# Save normalization stats (needed for reproducibility, if scaling was applied)

# --------------------- Model Inference on Full Trajectories for Evaluation -----------------------
# On ALL trajectories (train + test) for fair comparison plots

full_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_full.h5', key='trajectories')
all_pred = []
model.eval()
with torch.no_grad():
    for traj in full_df['traj'].unique():
        traj_data = full_df[full_df['traj']==traj].sort_values('t')
        qps = traj_data[['q','p']].values
        pred_qp = [qps[0]]  # Initial (q, p)
        for i in range(qps.shape[0] - 1):
            input_tensor = torch.tensor(pred_qp[-1], dtype=torch.float32).unsqueeze(0).to(device)
            pred_next = model(input_tensor).cpu().numpy()[0]
            pred_qp.append(pred_next)
        pred_qp = np.array(pred_qp)
        # Store predictions in the same format as the reference
        for j in range(len(pred_qp)):
            all_pred.append({
                'traj': traj, 't': traj_data['t'].values[j],
                'q_pred': pred_qp[j,0], 'p_pred': pred_qp[j,1]
            })
pred_df = pd.DataFrame(all_pred)
pred_df.to_hdf('NeuralNets/Pendulum_MLP/mlp_predictions.h5', key='preds', mode='w')