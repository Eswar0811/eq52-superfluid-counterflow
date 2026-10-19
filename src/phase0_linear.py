"""Phase 0 — linear-conduction limit (guide §7).

Mixed formulation with the Gorter-Mellink law replaced by Fourier's law:
    R1 = theta_t + q_x - Q          R2 = k theta_x + q
x in [0,1], t in [0,0.5], k = 0.1, Q = 4x(1-x) (vanishes at the walls, so BC/IC/source are
compatible at the corners and the reference has no corner layer), theta(0,t)=theta(1,t)=0, theta(x,0)=sin(pi x).
Reference: P1 finite elements (400 elements) + Crank-Nicolson (dt = 1e-4).
Pass: relative L2 error vs FEM < 1%.
"""
import pinn  # noqa: F401  load torch (and its libomp) before numpy/scipy

import json
import sys

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from pinn import MLP, d, sobol, train

K, TMAX = 0.1, 0.5
Q = lambda x: 4 * x * (1 - x)


def fem_reference(ne=400, dt=1e-4):
    h = 1.0 / ne
    x = np.linspace(0, 1, ne + 1)
    n = ne - 1  # interior nodes
    M = diags([h / 6, 4 * h / 6, h / 6], [-1, 0, 1], shape=(n, n)).tocsc()
    S = diags([-1 / h, 2 / h, -1 / h], [-1, 0, 1], shape=(n, n)).tocsc() * K
    f = Q(x[1:-1]) * h
    A = splu((M + 0.5 * dt * S).tocsc())
    B = (M - 0.5 * dt * S).tocsc()
    u = np.sin(np.pi * x[1:-1])
    nt = int(round(TMAX / dt))
    snaps, times = [np.r_[0, u, 0]], [0.0]
    save_every = nt // 100
    for k in range(1, nt + 1):
        u = A.solve(B @ u + dt * f)
        if k % save_every == 0:
            snaps.append(np.r_[0, u, 0])
            times.append(k * dt)
    return x, np.array(times), np.array(snaps)


def main(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xf, tf, U = fem_reference()

    net_T = MLP(2, 1, 64, 4)
    net_q = MLP(2, 1, 64, 4)
    sob = sobol(4096, 2, seed)
    X = sob[:, :1].clone().requires_grad_(True)
    Tt = (sob[:, 1:] * TMAX).clone().requires_grad_(True)

    def theta(x, t):  # hard BC + IC
        return torch.sin(np.pi * x) + x * (1 - x) * t * net_T(torch.cat([x, t / TMAX], 1))

    def loss_fn():
        th = theta(X, Tt)
        q = net_q(torch.cat([X, Tt / TMAX], 1))
        r1 = d(th, Tt) + d(q, X) - Q(X)
        r2 = K * d(th, X) + q
        l1, l2 = (r1 ** 2).mean(), (r2 ** 2).mean() * 100  # r2 is O(k): rescale by 1/k^2
        return l1 + l2, dict(R1=l1.item(), R2=l2.item())

    train(list(net_T.parameters()) + list(net_q.parameters()), loss_fn,
          adam_iters=4000, lr=2e-3, lbfgs_iters=8000, log_every=2000, tag=f"phase0/s{seed}", verbose=False)

    XX, TT = np.meshgrid(xf, tf)
    with torch.no_grad():
        P = theta(torch.tensor(XX.reshape(-1, 1)), torch.tensor(TT.reshape(-1, 1))).numpy().reshape(XX.shape)
    rel = np.linalg.norm(P - U) / np.linalg.norm(U)
    # FEM self-convergence check (coarse vs fine) so the reference error is known
    xc, _, Uc = fem_reference(ne=200, dt=2e-4)
    fem_self = np.linalg.norm(np.interp(xf, xc, Uc[-1]) - U[-1]) / np.linalg.norm(U[-1])
    res = dict(phase=0, rel_L2_vs_FEM=rel, pass_criterion=0.01, passed=bool(rel < 0.01),
               fem_selfconv_rel=fem_self)
    print(json.dumps(res, indent=2))
    res["seed"] = seed
    json.dump(res, open(f"../results/phase0_seed{seed}.json", "w"), indent=2)
    np.savez(f"../results/phase0_fields_seed{seed}.npz", x=xf, t=tf, fem=U, pinn=P)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
