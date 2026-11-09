import torch
import torch.nn as nn
import torch.nn.functional as F


class CMvCNet(nn.Module):
    """
    共性 D@Z + 互补 F@Q + 噪声 E；每层独立 theta / rho / R、S；可学习动量 (Z, Q, E)。
    F、D 由 SVD Procrustes 闭式解直接更新（全局最优，无需动量）。

    张量: Z (m x n), D (d_v x m), E (d_v x n), F (d_v x r), Q (r x n), features [v] (d_v x n).
    """

    def __init__(
        self,
        m: int,
        n_view: int,
        layers: int,
        rank: int,
        theta1: float,
        theta2: float,
        theta_q: float,
        Z_init,
        D_init,
        E_init,
        F_init,
        Q_init,
        device: torch.device,
    ):
        super(CMvCNet, self).__init__()
        self.layers = layers
        self.device = device
        self.n_view = n_view
        self.m = m
        self.rank = rank
        self.d_dims = [D_init[i].shape[0] for i in range(n_view)]

        # Store initial state for external use (not used in forward directly)
        self.Z_init = torch.from_numpy(Z_init).float().to(device)
        self.D_init = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_init]
        self.E_init = [torch.tensor(e, dtype=torch.float32, device=device) for e in E_init]
        self.F_init = [torch.tensor(f, dtype=torch.float32, device=device) for f in F_init]
        self.Q_init = [torch.tensor(q, dtype=torch.float32, device=device) for q in Q_init]

        self.theta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor([theta1], dtype=torch.float32, device=device)) for _ in range(layers)]
        )
        self.rho_v = nn.ParameterList(
            [nn.Parameter(torch.tensor([theta2], dtype=torch.float32, device=device)) for _ in range(layers)]
        )
        self.rho_q = nn.ParameterList(
            [nn.Parameter(torch.tensor([theta_q], dtype=torch.float32, device=device)) for _ in range(layers)]
        )
        self.R_layer = nn.ModuleList([nn.Linear(m, m, device=device) for _ in range(layers)])
        self.S_layer = nn.ModuleList([nn.Linear(m, m, device=device) for _ in range(layers)])
        # learned momentum: sigmoid to (0,1) scale
        def betas():
            return nn.ParameterList(
                [nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32)) for _ in range(layers)]
            )
        self.beta_H = betas()
        self.beta_Q = betas()
        self.beta_E = betas()

    @staticmethod
    def _row_l21_shrink(x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Row-wise L2,1 shrinkage. Each row: max(0, 1 - rho/||row||_2) * row."""
        row_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(1.0 - rho / row_norm, min=0.0, max=1.0)
        return x * scale

    @staticmethod
    def _stiefel_svd_procrustes(c: torch.Tensor, device, dtype) -> torch.Tensor:
        """Procrustes solution on Stiefel manifold: P = U V^T via SVD(C)."""
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(c.detach(), full_matrices=False)
            return (U @ Vh).to(device=device, dtype=dtype)

    @staticmethod
    def _soft_threshold_l1(u: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return F.relu(u - theta) - F.relu(-u - theta)

    def _m(self, b: int, beta: nn.ParameterList) -> torch.Tensor:
        return torch.sigmoid(beta[b])

    def forward(
        self,
        features: list,
        Z: torch.Tensor,
        D: list,
        E: list,
        F: list,
        Q: list,
    ) -> tuple:
        """
        Args:
            features: list[v] of (d_v × n)  — view features
            Z:        (m × n)               — current representation (detached)
            D:        list[v] of (d_v × m)  — current view dictionaries
            E:        list[v] of (d_v × n)  — current noise
            F:        list[v] of (d_v × r)  — current complementary bases
            Q:        list[v] of (r × n)    — current complementary coefficients
        Returns:
            Z_new, D_new, E_new, F_new, Q_new
        """
        # _old: previous iterate (for momentum);  _t: extrapolated point (momentum-shifted)
        # Inputs are already detached, no need to clone.
        Z_old = Z
        Z_t = Z
        D_cur = list(D)
        E_old = list(E)
        E_cur = list(E)
        F_cur = list(F)
        Q_old = list(Q)
        Q_cur = list(Q)

        Z_out = Z

        for b in range(self.layers):
            th1 = self.theta1[b]
            rho_e = self.rho_v[b]
            rho_qv = self.rho_q[b]
            mH = self._m(b, self.beta_H)
            mQ = self._m(b, self.beta_Q)
            mE = self._m(b, self.beta_E)

            # -------- RepresentModule: update Z --------
            z_acc = torch.zeros_like(Z)
            for v in range(self.n_view):
                in1 = self.R_layer[b](Z_t.t()).t()
                res_v = features[v] - torch.mm(F_cur[v], Q_cur[v]) - E_cur[v]
                in2 = torch.mm(D_cur[v].t(), res_v)
                in2 = self.S_layer[b](in2.t()).t()
                z_acc += in1 + in2
            Z_new = self._soft_threshold_l1(z_acc / self.n_view, th1)
            Z_extrap = Z_new + mH * (Z_new - Z_old)

            # -------- ComplementModule: update Q --------
            Q_new = []
            for v in range(self.n_view):
                r_tail = features[v] - torch.mm(D_cur[v], Z_new) - E_cur[v]
                qv = self._row_l21_shrink(torch.mm(F_cur[v].t(), r_tail), rho_qv)
                Q_new.append(qv + mQ * (qv - Q_old[v]))

            # -------- BasisModule: F_v (SVD Procrustes, globally optimal per step) --------
            F_new = []
            for v in range(self.n_view):
                m_fd = features[v] - torch.mm(D_cur[v], Z_new) - E_cur[v]
                c_f = torch.mm(m_fd, Q_new[v].t())
                F_new.append(self._stiefel_svd_procrustes(c_f, self.device, c_f.dtype))

            # -------- NoiseModule: E --------
            E_new = []
            for v in range(self.n_view):
                r_e = features[v] - torch.mm(D_cur[v], Z_new) - torch.mm(F_new[v], Q_new[v])
                ev = self._row_l21_shrink(r_e.t(), rho_e).t()
                E_new.append(ev + mE * (ev - E_old[v]))

            # -------- AnchorModule: D (SVD Procrustes, globally optimal per step) --------
            D_new = []
            for v in range(self.n_view):
                r_d = features[v] - torch.mm(F_new[v], Q_new[v]) - E_new[v]
                c_d = torch.mm(r_d, Z_new.t())
                D_new.append(self._stiefel_svd_procrustes(c_d, self.device, c_d.dtype))

            # Advance state for next layer
            Z_old = Z_new
            Z_t = Z_extrap
            Z_out = Z_new
            D_cur = D_new
            Q_old = Q_cur
            Q_cur = Q_new
            E_old = E_cur
            E_cur = E_new
            F_cur = F_new

        return Z_out, D_cur, E_cur, F_cur, Q_cur


# Backward compatibility alias
DBONet_with_E = CMvCNet
