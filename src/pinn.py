"""Minimal PINN utilities (PyTorch, float64)."""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("PINN_THREADS", "4")))


class MLP(nn.Module):
    def __init__(self, din, dout, width=64, depth=4):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def d(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def train(params, loss_fn, adam_iters=5000, lr=1e-3, lr_final=1e-4, lbfgs_iters=5000,
          log_every=1000, verbose=True, tag=""):
    """Adam (exponential decay lr -> lr_final) then L-BFGS. loss_fn() -> (loss, dict of floats)."""
    params = list(params)
    hist = []
    t0 = time.time()

    def log(stage, it, loss, parts):
        rec = dict(stage=stage, it=it, loss=loss.item(), **parts, time=time.time() - t0)
        hist.append(rec)
        if verbose:
            p = " ".join(f"{k}={v:.3e}" for k, v in parts.items())
            print(f"[{tag}] {stage} {it:6d} loss={rec['loss']:.3e} {p} ({rec['time']:.0f}s)", flush=True)

    if adam_iters:
        opt = torch.optim.Adam(params, lr=lr)
        gamma = (lr_final / lr) ** (1.0 / adam_iters)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        for it in range(adam_iters):
            opt.zero_grad()
            loss, parts = loss_fn()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at Adam iter {it}")
            loss.backward()
            opt.step()
            sched.step()
            if it % log_every == 0:
                log("adam", it, loss, parts)

    if lbfgs_iters:
        # SciPy L-BFGS-B: torch.optim.LBFGS (strong-Wolfe) stalls around 1e-7 on these problems,
        # SciPy's implementation reaches ~1e-11 on the same test ODE in the same budget.
        from scipy.optimize import minimize
        sizes = [p.numel() for p in params]
        state = dict(n=0)

        def setp(w):
            with torch.no_grad():
                i = 0
                for p, s in zip(params, sizes):
                    p.copy_(torch.from_numpy(w[i:i + s]).view_as(p))
                    i += s

        def fg(w):
            setp(w)
            for p in params:
                p.grad = None
            loss, parts = loss_fn()
            state["n"] += 1
            if not torch.isfinite(loss):
                return 1e30, np.zeros_like(w)
            loss.backward()
            if state["n"] % log_every == 0:
                log("lbfgs", state["n"], loss, parts)
            g = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            return loss.item(), torch.cat([x.reshape(-1) for x in g]).numpy().copy()

        w0 = torch.cat([p.detach().reshape(-1) for p in params]).numpy().copy()
        res = minimize(fg, w0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=lbfgs_iters, maxfun=int(1.25 * lbfgs_iters), maxcor=50,
                                    ftol=0.0, gtol=0.0))
        setp(res.x)

    loss, parts = loss_fn()
    log("final", 0, loss, parts)
    return hist


def sobol(n, dim, seed=0):
    """Scrambled Sobol points in [0,1]^dim (scipy; torch's sampler clashes with scipy's OpenMP)."""
    from scipy.stats import qmc
    return torch.tensor(qmc.Sobol(dim, scramble=True, seed=seed).random(n))
