import torch
import torch.nn as nn

# 单摆专用的分离式辛更新模块
class SeparablePendulumHamiltonianModule(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=60, hidden_layers=2):
        super().__init__()
        # 势能网络 U(q)
        layers_U = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers_U += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_U += [nn.Linear(hidden_dim, 1)]
        self.U_net = nn.Sequential(*layers_U)
        
        # 动能网络 T(p)
        layers_T = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers_T += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_T += [nn.Linear(hidden_dim, 1)]
        self.T_net = nn.Sequential(*layers_T)

    def forward(self, q, p, step):
        # 单摆严格交错辛欧拉法则更新
        U = self.U_net(q).squeeze(-1)
        grad_q = torch.autograd.grad(outputs=U.sum(), inputs=q, create_graph=True)[0]
        p_next = p - step * grad_q
        
        T_next = self.T_net(p_next).squeeze(-1)
        grad_p = torch.autograd.grad(outputs=T_next.sum(), inputs=p_next, create_graph=True)[0]
        q_next = q + step * grad_p
        
        return q_next, p_next

# 适用于单摆的GHNN模型
class GHNNPendulum(nn.Module):
    def __init__(self, hidden_dim=60, hidden_layers=2):
        super().__init__()
        self.module = SeparablePendulumHamiltonianModule(
            input_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers
        )

    def forward(self, x, step=0.01):
        # x: (batch, 2), x[..., 0] = q, x[..., 1] = p
        q = x[:, :1].clone().requires_grad_(True)
        p = x[:, 1:].clone().requires_grad_(True)
        q_next, p_next = self.module(q, p, step)
        return torch.cat([q_next, p_next], dim=1)