import torch
import torch.nn as nn

# 这是一个单层的“分离式辛更新模块”（对应原论文的一个 HamiltonModule）
class SeparableHamiltonianModule(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=25, hidden_layers=2):
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
        # 严格的交错辛欧拉更新
        U = self.U_net(q).squeeze(-1)
        grad_q = torch.autograd.grad(outputs=U.sum(), inputs=q, create_graph=True)[0]
        p_next = p - step * grad_q
        
        T_next = self.T_net(p_next).squeeze(-1)
        grad_p = torch.autograd.grad(outputs=T_next.sum(), inputs=p_next, create_graph=True)[0]
        q_next = q + step * grad_p
        
        return q_next, p_next

# 这才是完整版的 GHNN：通过堆叠逼近不可分离系统
class StackedGHNN(nn.Module):
    # num_modules=5 完全契合原论文的 l_hamilt=5
    def __init__(self, state_dim=4, hidden_dim=25, hidden_layers=2, num_modules=5):
        super().__init__()
        self.state_dim = state_dim
        self.q_dim = state_dim // 2
        # 堆叠 5 个独立的分离模块
        self.modules_list = nn.ModuleList([
            SeparableHamiltonianModule(self.q_dim, hidden_dim, hidden_layers) 
            for _ in range(num_modules)
        ])

    def forward(self, x, step=0.01):
        # 拆分 q 和 p (对于双摆，输入 x 是 4 维的: q1, q2, p1, p2)
        q = x[:, :self.q_dim].clone().requires_grad_(True)
        p = x[:, self.q_dim:].clone().requires_grad_(True)
        
        # 依次穿过 5 个辛更新层（复合映射）
        for mod in self.modules_list:
            q, p = mod(q, p, step)
            
        return torch.cat([q, p], dim=1)