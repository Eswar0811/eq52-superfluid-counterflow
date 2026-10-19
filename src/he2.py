"""He-II counterflow physics: Gorter-Mellink function, normalisation, exact steady solutions.

Normalisation (guide §4):
    theta = (T - T_LO) / DT_BAND        T in [1.6 K, T_lambda]  ->  theta in [0, 1]
    xhat  = x / L
    qhat  = q / q_ref
Steady 1D Gorter-Mellink law  dT/dx = -f(T) q^n  becomes
    dtheta/dxhat = -F(theta) * qhat^n,     F = L f(T) q_ref^n / DT_BAND
q_ref is chosen so that F = 1 at the peak of f^{-1} (T ~ 1.93 K).

f(T) shape: the standard correlation  f^{-1}(T) = g_lambda [t^5.7 (1 - t^5.7)]^3,  t = T/T_lambda
(Van Sciver, Helium Cryogenics). G_LAMBDA below is a NOMINAL magnitude (peak f^{-1} ~ 1.3e13
W^3 m^-5 K^-1) used only to map normalised results back to physical units. Replace with
NIST TN 1334 / HEPAK values for Phase 6.
"""
import numpy as np
from scipy.integrate import solve_ivp

T_LAMBDA = 2.1768          # K
T_LO = 1.6                 # K, bottom of operating band
DT_BAND = T_LAMBDA - T_LO  # K
EXP_T = 5.7
G_LAMBDA = 8.5e14          # W^3 m^-5 K^-1 (nominal, see docstring)
PHI_MAX = 0.25 ** 3        # max over t of [u(1-u)]^3, u = t^5.7
FINV_MAX = G_LAMBDA * PHI_MAX


def theta_of_T(T):
    return (T - T_LO) / DT_BAND


def T_of_theta(th):
    return T_LO + DT_BAND * th


def F_true(theta):
    """Normalised Gorter-Mellink f (works for numpy arrays and torch tensors). F >= 1."""
    t = (T_LO + DT_BAND * theta) / T_LAMBDA
    u = t ** EXP_T
    return PHI_MAX / (u * (1 - u)) ** 3


def finv_phys(T):
    t = np.asarray(T) / T_LAMBDA
    u = t ** EXP_T
    return G_LAMBDA * (u * (1 - u)) ** 3


def q_ref(L=1.0):
    """Physical flux scale [W/m^2] for an n=3 channel of length L [m]."""
    return (DT_BAND * FINV_MAX / L) ** (1.0 / 3.0)


# --- closed-form conductivity integral  G(theta) = int_0^theta dtheta' / F(theta') -------------
_P = np.array([3, 4, 5, 6]) * EXP_T          # u^3 - 3u^4 + 3u^5 - u^6  in powers of t
_C = np.array([1.0, -3.0, 3.0, -1.0])


def _prim(t):
    t = np.asarray(t, dtype=float)[..., None]
    return np.sum(_C * t ** (_P + 1) / (_P + 1), axis=-1)


def G(theta):
    t = (T_LO + DT_BAND * np.asarray(theta, dtype=float)) / T_LAMBDA
    t0 = T_LO / T_LAMBDA
    return (T_LAMBDA / DT_BAND) / PHI_MAX * (_prim(t) - _prim(t0))


def steady_profile(x, theta_b, qhat, n=3.0):
    """Exact steady solution, constant n:  G(theta(x)) = G(theta_b) + qhat^n (1 - x).

    Inverts the monotone G by vectorised bisection to machine precision.
    """
    x = np.asarray(x, dtype=float)
    target = G(theta_b) + qhat ** n * (1.0 - x)
    lo = np.full_like(x, theta_b)
    hi = np.full_like(x, 1.0 - 1e-12)
    if np.any(G(hi) < target):
        raise ValueError("profile exceeds T_lambda")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = G(mid) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def steady_profile_ode(x, theta_b, qhat, n_fun):
    """Steady solution for temperature-dependent exponent n(theta): integrate from bath (x=1) to x=0."""
    rhs = lambda s, th: [-F_true(th[0]) * qhat ** n_fun(th[0])]
    sol = solve_ivp(rhs, (1.0, 0.0), [theta_b], rtol=1e-12, atol=1e-14, dense_output=True,
                    method="DOP853")
    return sol.sol(np.asarray(x, dtype=float))[0]


def qhat_max(theta_b, theta_top, n=3.0):
    """Largest load whose heater-end temperature stays at or below theta_top."""
    return float((G(theta_top) - G(theta_b)) ** (1.0 / n))


# --- NIST TN 1334 property backend (Phase 6) ---------------------------------------------------
def set_backend(name):
    """'correlation' (default, analytic Van Sciver-type shape) or 'nist' (NIST TN 1334 tables, numpy only)."""
    global F_true, G, FINV_MAX
    if name == "correlation":
        F_true, G, FINV_MAX = _F_corr, _G_corr, G_LAMBDA * PHI_MAX
        return
    from scipy.interpolate import CubicSpline
    import nist_props as npp
    th = np.linspace(0.0, 0.99, 6001)
    finv = npp.finv_nist(T_of_theta(th))
    fmax = float(finv.max())
    Fg = fmax / finv                                    # normalised: min F = 1
    logF = CubicSpline(th, np.log(Fg))
    Gs = CubicSpline(th, 1.0 / Fg).antiderivative()     # exact integral of the spline of 1/F
    F_true = lambda theta: np.exp(logF(np.clip(theta, 0.0, 0.99)))
    G = lambda theta: Gs(np.clip(theta, 0.0, 0.99)) - Gs(0.0)
    FINV_MAX = fmax


_F_corr, _G_corr = F_true, G
import os as _os
if _os.environ.get("HE2_PROPS") == "nist":
    set_backend("nist")
