"""Inverse recovery of the Gorter-Mellink function f(T) (and exponent n) from sparse steady
channel thermometry — shared engine for Phases 3, 4, 5 (guide §4, §6, §7).

Synthetic experiment: 1D channel, heater flux qhat at x=0, bath at x=1 with known theta_b.
Sensors at x_i = i/m, i = 0..m-1. Noise is Gaussian with std = p * (T_lambda - 1.6 K)
(1% -> 5.8 mK, the mK-level thermometry noise the guide warns about).

Model (steady energy balance makes q uniform = known heater flux, so the mixed constitutive
residual is a polynomial in q and contains no fractional power):
    theta(x; theta_b, qhat) = theta_b + (1 - x) qhat^3 N_T(x, theta_b, log qhat)
    F_psi(theta)  = exp(N_F(2 theta - 1))      log-parameterised: positivity + steep rise near T_lambda (guide §9)
    R             = [theta_x + F_psi(theta) qhat^n] / qhat^3       (per-profile normalised)
    n             = fixed | trainable scalar | n0 + 0.5 tanh(N_n(theta))  (guide §6)
Loss = data + w_res * R^2 + w_curv * ((log F)'')^2   (curvature penalty, guide §9).
Collocation: n_col Sobol points per profile + n_hot points clustered at the heater end x in [0, 0.15],
where each profile is hottest (densest collocation near T_lambda, guide §9).

Usage:  python inverse.py <config_name> <seed>
"""
import pinn  # noqa: F401  (load torch first)

import json
import os
import sys

import numpy as np
import torch

import he2
from pinn import MLP, d, sobol, train

N_TRUE = 3.4                       # measured exponent (Lebrun & Tavian; Sato et al. 2006)
BATHS = [1.70, 1.80, 1.90, 2.00]   # K
TOP = he2.theta_of_T(2.12)         # hottest allowed heater-end temperature
BAND = (he2.theta_of_T(1.7), he2.theta_of_T(2.1))   # evaluation band, guide §7 Phase 3


def n_fun_true(th):
    """Illustrative temperature-dependent exponent: 3.25 at 1.7 K rising linearly to 3.55 at 2.1 K."""
    T = he2.T_of_theta(th)
    return 3.25 + 0.30 * (T - 1.7) / 0.4


def _qmax(th_b, n_true):
    if callable(n_true):  # bisection on load using the ODE solution
        lo, hi = 0.0, 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            try:
                ok = he2.steady_profile_ode([0.0], th_b, mid, n_true)[0] <= TOP
            except Exception:
                ok = False
            lo, hi = (mid, hi) if ok else (lo, mid)
        return lo
    return he2.qhat_max(th_b, TOP, n_true)


def design(name, n_true):
    """List of (theta_b, qhat) profiles."""
    if name == "full":  # 4 baths x 5 loads, each bath spanning one decade of load
        out = []
        for Tb in BATHS:
            th_b = he2.theta_of_T(Tb)
            qm = _qmax(th_b, n_true)
            out += [(th_b, qm * f) for f in np.logspace(-1, 0, DEFAULTS["n_loads"])]
        return out
    if name == "single_load_all_baths":  # one common load (the largest the 2.0 K bath tolerates)
        q = _qmax(he2.theta_of_T(BATHS[-1]), n_true)
        return [(he2.theta_of_T(Tb), q) for Tb in BATHS]
    if name == "single_profile_high":
        th_b = he2.theta_of_T(1.70)
        return [(th_b, _qmax(th_b, n_true))]
    if name == "single_profile_low":
        th_b = he2.theta_of_T(1.70)
        return [(th_b, 0.3 * _qmax(th_b, n_true))]
    raise ValueError(name)


def profile(x, th_b, q, n_true):
    if callable(n_true):
        return he2.steady_profile_ode(x, th_b, q, n_true)
    return he2.steady_profile(x, th_b, q, n_true)


CONFIGS = {
    # Phase 3: exponent fixed at its true value, recover f(T)
    "p3_full": dict(design="full", n_mode="fixed", n_fixed=N_TRUE),
    "p3_single_load_all_baths": dict(design="single_load_all_baths", n_mode="fixed", n_fixed=N_TRUE),
    "p3_single_profile_high": dict(design="single_profile_high", n_mode="fixed", n_fixed=N_TRUE),
    "p3_single_profile_low": dict(design="single_profile_low", n_mode="fixed", n_fixed=N_TRUE),
    # Phase 4: exponent trainable (init 3.0, truth 3.4)
    "p4_full": dict(design="full", n_mode="scalar"),
    "p4_single_load_all_baths": dict(design="single_load_all_baths", n_mode="scalar"),
    "p4_single_profile_high": dict(design="single_profile_high", n_mode="scalar"),
    "p4_assume_n3": dict(design="full", n_mode="fixed", n_fixed=3.0),     # idealised-exponent bias
    "p4_nfun_full": dict(design="full", n_mode="function", n_true="function"),
}
for p_ in (1, 3, 5):   # Phase 5: noise x sensor sparsity, exponent trainable
    for m_ in (4, 8, 16):
        CONFIGS[f"p5_noise{p_}_m{m_}"] = dict(design="full", n_mode="scalar", noise=p_ / 100, m=m_)

DEFAULTS = dict(m=8, noise=0.0, n_true=N_TRUE, n_fixed=None, w_res=10.0, w_curv=None,
                adam_iters=5000, lbfgs_iters=12000, n_col=64, n_hot=32, n_loads=5, T_top=2.15, wd_mode="sigma")
NOISY_W_RES = 300.0   # physics weight for noisy runs; tuned on seed 0 of (1%, m=8) only


def run(name, seed, verbose=True, tag=""):
    global TOP
    explicit_w = "w_res" in OVERRIDES
    cfg = {**DEFAULTS, **CONFIGS[name]}
    TOP = he2.theta_of_T(cfg["T_top"])
    n_true = n_fun_true if cfg["n_true"] == "function" else cfg["n_true"]
    if cfg["noise"] > 0 and "w_res" not in CONFIGS[name] and not explicit_w:
        cfg["w_res"] = NOISY_W_RES
    if cfg["w_curv"] is None:
        cfg["w_curv"] = 1e-6 if cfg["noise"] == 0 else 1e-4
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- synthetic data -------------------------------------------------------------------
    prof = design(cfg["design"], n_true)
    m, sig = cfg["m"], cfg["noise"]
    xs = np.arange(m) / m
    rows = []
    for k, (th_b, q) in enumerate(prof):
        clean = profile(xs, th_b, q, n_true)
        obs = clean + sig * rng.standard_normal(m)
        rows += [(x, th_b, q, o, k) for x, o in zip(xs, obs)]
    D = torch.tensor(np.array(rows))
    xd, tbd, qd, od = (D[:, i:i + 1] for i in range(4))
    # data weight: noise-free -> per-profile rise scale; noisy -> the actual noise sigma (a proper likelihood)
    wd = torch.sqrt(qd ** 6 + sig ** 2) if (sig == 0 or cfg["wd_mode"] == "rise") else torch.full_like(qd, sig)

    nc = cfg["n_col"] + cfg["n_hot"]
    x_one = torch.cat([sobol(cfg["n_col"], 1, seed), 0.15 * sobol(cfg["n_hot"], 1, seed + 1) ** 2])
    xc = x_one.repeat(len(prof), 1)
    tbc = torch.tensor([p[0] for p in prof]).repeat_interleave(nc)[:, None]
    qc = torch.tensor([p[1] for p in prof]).repeat_interleave(nc)[:, None]
    xc.requires_grad_(True)
    thg = torch.linspace(0.05, 0.95, 181)[:, None].requires_grad_(True)  # curvature-penalty grid

    # ---- model ----------------------------------------------------------------------------
    net_T = MLP(3, 1, 64, 4)
    net_F = MLP(1, 1, 32, 3)
    with torch.no_grad():
        net_F.net[-1].bias.fill_(float(np.log(2.0)))   # F ~ 2: an uninformative O(1) start
    params = list(net_T.parameters()) + list(net_F.parameters())
    if cfg["n_mode"] == "scalar":
        n_par = torch.nn.Parameter(torch.tensor(3.0))
        params.append(n_par)
        n_of = lambda th: n_par
    elif cfg["n_mode"] == "function":
        net_n = MLP(1, 1, 16, 2)
        n0 = torch.nn.Parameter(torch.tensor(3.0))
        params += list(net_n.parameters()) + [n0]
        n_of = lambda th: n0 + 0.5 * torch.tanh(net_n(2 * th - 1))
    else:
        n_of = lambda th: torch.tensor(cfg["n_fixed"])

    def theta(x, tb, q):
        z = torch.cat([2 * x - 1, 2 * tb - 1, torch.log10(q)], 1)
        return tb + (1 - x) * q ** 3 * net_T(z)

    logF_of = lambda th: net_F(2 * th.clamp(0.0, 0.98) - 1)
    F_of = lambda th: torch.exp(logF_of(th))

    def loss_fn():
        ld = (((theta(xd, tbd, qd) - od) / wd) ** 2).mean()
        th = theta(xc, tbc, qc)
        r = (d(th, xc) + F_of(th) * qc ** n_of(th)) / qc ** 3
        lr_ = (r ** 2).mean()
        lc = (d(d(logF_of(thg), thg), thg) ** 2).mean()
        loss = ld + cfg["w_res"] * lr_ + cfg["w_curv"] * lc
        parts = dict(data=ld.item(), res=lr_.item(), curv=lc.item())
        if cfg["n_mode"] != "fixed":
            parts["n"] = float(n_of(torch.tensor([[0.57]])).mean())
        return loss, parts

    train(params, loss_fn, adam_iters=cfg["adam_iters"], lr=2e-3, lr_final=1e-4,
          lbfgs_iters=cfg["lbfgs_iters"], log_every=4000, verbose=verbose, tag=f"{name}/s{seed}")

    # ---- evaluation -----------------------------------------------------------------------
    thb = np.linspace(*BAND, 401)
    with torch.no_grad():
        Fp = F_of(torch.tensor(thb[:, None])).numpy()[:, 0]
        n_p = n_of(torch.tensor(thb[:, None]))
        n_p = np.broadcast_to(n_p.numpy().reshape(-1), thb.shape).copy()
    Ft = he2.F_true(thb)
    rel = np.abs(Fp - Ft) / Ft
    covered = [min(profile([0.0], tb, q, n_true)[0], 1.0) for tb, q in prof]
    res = dict(config=name, seed=seed, props=os.environ.get("HE2_PROPS", "correlation"), **{k: v for k, v in cfg.items() if not callable(v)},
               n_profiles=len(prof), T_coverage_K=[float(he2.T_of_theta(min(p[0] for p in prof))),
                                                    float(he2.T_of_theta(max(covered)))],
               F_maxrel_band=float(rel.max()), F_rmsrel_band=float(np.sqrt((rel ** 2).mean())),
               F_pass_10pct=bool(rel.max() < 0.10))
    if cfg["n_mode"] == "scalar":
        res["n_recovered"] = float(n_p[0])
    if cfg["n_mode"] == "function":
        nt = n_fun_true(thb)
        res["n_fun_maxabs_err"] = float(np.abs(n_p - nt).max())
        res["n_fun_rms_err"] = float(np.sqrt(((n_p - nt) ** 2).mean()))
    res["n_true"] = "3.25->3.55 linear in T" if callable(n_true) else n_true
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"../results/inv_{name}{tag}_seed{seed}.json", "w"), indent=2)
    np.savez(f"../results/inv_{name}{tag}_seed{seed}.npz", T=he2.T_of_theta(thb), F_pred=Fp, F_true=Ft, n_pred=n_p,
             data=D.numpy())
    return res


import os
OVERRIDES = {}
if __name__ == "__main__":
    # optional 3rd arg: JSON overrides of DEFAULTS, e.g. '{"lbfgs_iters": 40000}'; 4th: output tag
    if len(sys.argv) > 3:
        OVERRIDES = json.loads(sys.argv[3])
        DEFAULTS.update(OVERRIDES)
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, tag=sys.argv[4] if len(sys.argv) > 4 else ("_nist" if os.environ.get("HE2_PROPS") == "nist" else ""))
