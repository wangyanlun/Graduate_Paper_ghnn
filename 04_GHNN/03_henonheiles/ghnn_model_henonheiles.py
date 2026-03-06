import torch
import torch.nn as nn


# ============================================================
# Henon–Heiles specific GHNN (Separable Hamiltonian)
# State: z = [x, y, px, py]
# q = [x, y], p = [px, py]
# H(q,p) = U(q) + T(p)
# Update: Symplectic Euler (staggered)
#   p_{n+1} = p_n - h * dU/dq(q_n)
#   q_{n+1} = q_n + h * dT/dp(p_{n+1})
# ============================================================

def hh_energy_terms(z: torch.Tensor):
    """
    Compute true Henon–Heiles energies for a batch of states.
    z: [B,4] = [x,y,px,py]
    Returns: K, V, H each [B]
    """
    x, y, px, py = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    K = 0.5 * (px**2 + py**2)
    V = 0.5 * (x**2 + y**2) + (x**2) * y - (y**3) / 3.0
    H = K + V
    return K, V, H


class SeparableHamiltonianNetHH(nn.Module):
    """
    Two-network separable Hamiltonian:
      U(q): R^2 -> R
      T(p): R^2 -> R
    """
    def __init__(self, hidden_dim=60, hidden_layers=2, activation=nn.Tanh):
        super().__init__()

        # U(q): 2 -> 1
        layers_U = [nn.Linear(2, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers_U += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers_U += [nn.Linear(hidden_dim, 1)]
        self.U_net = nn.Sequential(*layers_U)

        # T(p): 2 -> 1
        layers_T = [nn.Linear(2, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers_T += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers_T += [nn.Linear(hidden_dim, 1)]
        self.T_net = nn.Sequential(*layers_T)

    def U(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B,2] -> [B]
        return self.U_net(q).squeeze(-1)

    def T(self, p: torch.Tensor) -> torch.Tensor:
        # p: [B,2] -> [B]
        return self.T_net(p).squeeze(-1)

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        # return U(q), T(p) both [B]
        return self.U(q), self.T(p)


class GHNN_HenonHeiles(nn.Module):
    """
    GHNN for Henon–Heiles in separable form with symplectic Euler update.
    Input/Output:
      x: [B,4] = [x,y,px,py]
      return: [B,4]
    """
    def __init__(self, hidden_dim=60, hidden_layers=2, activation=nn.Tanh):
        super().__init__()
        self.H = SeparableHamiltonianNetHH(hidden_dim=hidden_dim,
                                           hidden_layers=hidden_layers,
                                           activation=activation)

    def forward(self, x: torch.Tensor, step: float = 0.01) -> torch.Tensor:
        # split
        q = x[:, 0:2].clone().detach().requires_grad_(True)
        p = x[:, 2:4].clone().detach().requires_grad_(True)

        # 1) p_next = p - h * dU/dq(q)
        U = self.H.U(q)                      # [B]
        grad_q = torch.autograd.grad(U.sum(), q, create_graph=True)[0]  # [B,2]
        p_next = p - step * grad_q

        # 2) q_next = q + h * dT/dp(p_next)
        p_next = p_next.clone().detach().requires_grad_(True)
        T = self.H.T(p_next)                # [B]
        grad_p = torch.autograd.grad(T.sum(), p_next, create_graph=True)[0]  # [B,2]
        q_next = q + step * grad_p

        return torch.cat([q_next, p_next], dim=1)