"""Phase 2 — forward transient He-II counterflow, manufactured solution (guide §3 Tier 2, §7).

Normalised system (rho*c absorbed into the time scale), n = 3:
    R1 = theta_t + q_x - Q(x,t)
    R2 = theta_x + F(theta) |q|^2 q
Manufactured solution, x in [0,1], t in [0,1]:
    theta*(x,t) = 0.2 + (1-x)(0.45 + 0.15 sin(pi t)) + 0.03 (1+t) sin(pi x)
theta*_x < 0 everywhere, so q* = (-theta*_x / F(theta*))^(1/3) is smooth and satisfies the
Gorter-Mellink law EXACTLY (no forcing in R2). Q = theta*_t + q*_x, evaluated by autograd.
T* spans 1.715 K .. 2.06 K.  Dirichlet theta at both ends + initial condition, enforced exactly.
Pass: relative L2 error < 1e-3 (theta and q).
"""
import pinn  # noqa: F401  (load torch first)

import json
import sys

import numpy as np
import torch

import he2
from pinn import MLP, d, sobol, train

PI = np.pi


def theta_star(x, t):
    return 0.2 + (1 - x) * (0.45 + 0.15 * torch.sin(PI * t)) + 0.03 * (1 + t) * torch.sin(PI * x)


def q_star(x, t):
    th = theta_star(x, t)
    thx = -(0.45 + 0.15 * torch.sin(PI * t)) + 0.03 * PI * (1 + t) * torch.cos(PI * x)
    return (-thx / he2.F_true(th)) ** (1.0 / 3.0)


def source(x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    return (d(theta_star(x, t), t) + d(q_star(x, t), x)).detach()


def main(seed=0, w1=1.0):
    torch.manual_seed(seed)
    P = sobol(4096, 2, seed)
    X = P[:, :1].clone().requires_grad_(True)
    Tt = P[:, 1:].clone().requires_grad_(True)
    Qc = source(X.detach(), Tt.detach())

    net_T, net_q = MLP(2, 1, 64, 5), MLP(2, 1, 64, 4)

    def theta(x, t):
        z0, z1 = torch.zeros_like(x), torch.ones_like(x)
        A = (theta_star(z0, t) * (1 - x) + theta_star(z1, t) * x
             + theta_star(x, 0 * t) - theta_star(z0, 0 * t) * (1 - x) - theta_star(z1, 0 * t) * x)
        return A + x * (1 - x) * t * net_T(torch.cat([x, t], 1))

    flux = lambda x, t: net_q(torch.cat([x, t], 1))

    def loss_fn():
        th, q = theta(X, Tt), flux(X, Tt)
        r1 = d(th, Tt) + d(q, X) - Qc
        r2 = d(th, X) + he2.F_true(th.clamp(0.0, 0.98)) * q ** 3
        l1, l2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return w1 * l1 + l2, dict(R1=l1.item(), R2=l2.item())

    train(list(net_T.parameters()) + list(net_q.parameters()), loss_fn, adam_iters=5000, lr=2e-3,
          lbfgs_iters=20000, log_every=5000, verbose=True, tag=f"phase2/s{seed}")

    xg, tg = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 101))
    xt, tt = torch.tensor(xg.reshape(-1, 1)), torch.tensor(tg.reshape(-1, 1))
    with torch.no_grad():
        thp, qp = theta(xt, tt).numpy(), flux(xt, tt).numpy()
        the, qe = theta_star(xt, tt).numpy(), q_star(xt, tt).numpy()
    res = dict(phase=2, seed=seed, w1=w1,
               relL2_theta=float(np.linalg.norm(thp - the) / np.linalg.norm(the)),
               relL2_q=float(np.linalg.norm(qp - qe) / np.linalg.norm(qe)),
               relL2_T=float(np.linalg.norm(he2.T_of_theta(thp) - he2.T_of_theta(the))
                             / np.linalg.norm(he2.T_of_theta(the))),
               maxabs_T_mK=float(1e3 * he2.DT_BAND * np.abs(thp - the).max()),
               pass_criterion=1e-3)
    res["passed"] = bool(max(res["relL2_theta"], res["relL2_q"]) < 1e-3)
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"../results/phase2_seed{seed}.json", "w"), indent=2)
    np.savez(f"../results/phase2_fields_seed{seed}.npz", x=xg, t=tg, theta_pinn=thp.reshape(xg.shape),
             theta_exact=the.reshape(xg.shape), q_pinn=qp.reshape(xg.shape), q_exact=qe.reshape(xg.shape))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
