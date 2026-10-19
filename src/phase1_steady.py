"""Phase 1 — forward steady 1D He-II counterflow, mixed formulation (guide §5, §7).

Channel x in [0,1] (normalised), heater flux q0 enters at x=0, bath at x=1 (theta = theta_b).
    R1 = dq/dx            (steady energy balance, Q = 0)
    R2 = dtheta/dx + F(theta) q^3        (Gorter-Mellink, n = 3; no fractional power anywhere)
Hard constraints: theta(1) = theta_b, q(0) = q0. Loss = w1*|R1|^2 + |R2|^2 with w1 = 100
(stiffens the q-theta trade-off valley; see results/phase1.json for the w1 = 1 comparison).
Reference: closed-form conductivity-integral solution (he2.steady_profile).
Pass: relative L2 error of the temperature rise theta - theta_b < 1e-4.
"""
import pinn  # noqa: F401  (load torch first)

import json
import sys

import numpy as np
import torch

import he2
from pinn import MLP, d, sobol, train

CASES = {  # name: (bath T [K], load as fraction of the load that brings the heater end to 2.12 K)
    "Tb1.70_high": (1.70, 1.0),
    "Tb1.90_high": (1.90, 1.0),
    "Tb1.70_low": (1.70, 0.1),
}
TOP = he2.theta_of_T(2.12)


def solve_case(Tb, frac, seed=0, verbose=True, tag="", w1=100.0, lbfgs_iters=30000):
    torch.manual_seed(seed)
    th_b = he2.theta_of_T(Tb)
    q0 = frac * he2.qhat_max(th_b, TOP)
    s = q0 ** 3  # output scale for the rise (removes the 1000x dynamic range across a decade of load)

    net_T, net_q = MLP(1, 1, 64, 4), MLP(1, 1, 32, 3)
    X = torch.cat([sobol(512, 1, seed), torch.tensor([[0.0], [1.0]])]).requires_grad_(True)

    theta = lambda x: th_b + (1 - x) * s * net_T(x)
    flux = lambda x: q0 * (1 + x * net_q(x))

    def loss_fn():
        th, q = theta(X), flux(X)
        Fth = he2.F_true(th.clamp(0.0, 0.98))
        r1 = d(q, X) / q0
        r2 = (d(th, X) + Fth * q ** 3) / s
        l1, l2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return w1 * l1 + l2, dict(R1=l1.item(), R2=l2.item())

    train(list(net_T.parameters()) + list(net_q.parameters()), loss_fn, adam_iters=3000, lr=2e-3,
          lbfgs_iters=lbfgs_iters, log_every=5000, verbose=verbose, tag=tag)

    xe = np.linspace(0, 1, 2001)
    ex = he2.steady_profile(xe, th_b, q0)
    with torch.no_grad():
        xt = torch.tensor(xe[:, None])
        pr, qp = theta(xt).numpy()[:, 0], flux(xt).numpy()[:, 0]
    rise_err = np.linalg.norm(pr - ex) / np.linalg.norm(ex - th_b)
    T_err = np.linalg.norm(he2.T_of_theta(pr) - he2.T_of_theta(ex)) / np.linalg.norm(he2.T_of_theta(ex))
    q_err = np.linalg.norm(qp - q0) / (np.sqrt(len(xe)) * q0)
    return dict(Tb=Tb, qhat=q0, q_W_m2=q0 * he2.q_ref(), T_heater_K=float(he2.T_of_theta(ex[0])),
                dT_mK=float(1e3 * he2.DT_BAND * (ex[0] - th_b)), relL2_rise=rise_err, relL2_T=T_err,
                relL2_q=q_err, passed=bool(rise_err < 1e-4)), (xe, ex, pr)


def main(seed):
    out, fields = {}, {}
    for name, (Tb, frac) in CASES.items():
        r, (xe, ex, pr) = solve_case(Tb, frac, seed=seed, tag=f"{name}/s{seed}", verbose=False)
        out[name] = r
        fields[name + "_exact"], fields[name + "_pinn"] = ex, pr
        print(name, f"seed={seed}", json.dumps(r), flush=True)
    fields["x"] = xe
    json.dump(dict(phase=1, seed=seed, pass_criterion=1e-4, cases=out),
              open(f"../results/phase1_seed{seed}.json", "w"), indent=2)
    np.savez(f"../results/phase1_fields_seed{seed}.npz", **fields)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
