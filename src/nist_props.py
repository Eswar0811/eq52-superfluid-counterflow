"""He-II counterflow conductivity function from NIST TN 1334 (Arp, McCarty & Friend, 1998).

Source: data/nist_tn1334/tn1334.pdf (nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1334.pdf).
 - Saturated-liquid density rho and entropy S: Appendix A (pdftotext rows, 1.60-2.15 K, 0.05 K steps).
 - Superfluid fraction  rho_s/rho = C1 x ln(x) + C2 x^2 + C3 x^3,  x = 1 - T/T_lambda   (Eq 13-14;
   natural log verified: rho_s/rho(1.9 K) = 0.572).   T_lambda(SVP) = 2.1768 K.
 - Mutual friction A = 383.7 + 10.649 rho_n [m s/kg] (Eq 16), replaced by 558.9 rho/rho_s (Eq 18) when it
   exceeds A_lim = 1413 + 12.4 (rho - 160) (Eq 17; checked against the page image).
 - Conductivity function  k = f^-1 = S (rho_s S T)^3 / (A rho_n)   [W^3 m^-5 K^-1]   (Eq 19-20, Q^3 = k dT/dx).
NIST quotes ~10% agreement with the cited data, and warns other data may differ by larger factors.
"""
import re

import numpy as np
from scipy.interpolate import CubicSpline

T_LAMBDA = 2.1768
C1, C2, C3 = -2.084035690, 1.760235312, -1.469764627


def _parse_table(path="../data/nist_tn1334/tn1334.txt"):
    lines = open(path).read().splitlines()
    rows = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*(1\.[6-9]\d0|2\.[01]\d0|2\.150)\s+(\d{3}\.\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d\.\d+)\s+", ln)
        if m and i + 1 < len(lines) and lines[i + 1].strip().startswith(m.group(1)):
            T = float(m.group(1))
            if 1.6 <= T <= 2.15 and abs(round(T / 0.05) * 0.05 - T) < 1e-9:
                rows[T] = (float(m.group(2)), float(m.group(6)))   # rho [kg/m3], S [J/g-K]
    return rows


def table():
    r = _parse_table()
    T = np.array(sorted(r))
    return T, np.array([r[t][0] for t in T]), np.array([r[t][1] for t in T])


_T, _RHO, _S = table()
_rho_spl = CubicSpline(_T, _RHO)
_S_spl = CubicSpline(_T, _S * 1e3)          # J/(kg K)


def rho_s_frac(T):
    x = 1.0 - np.asarray(T, dtype=float) / T_LAMBDA
    return C1 * x * np.log(x) + C2 * x ** 2 + C3 * x ** 3


def finv_nist(T):
    """k(T) = f^-1(T) in W^3 m^-5 K^-1 for T in [1.6, 2.17] K."""
    T = np.asarray(T, dtype=float)
    rho, S = _rho_spl(np.minimum(T, 2.15)), _S_spl(np.minimum(T, 2.15))
    fs = rho_s_frac(T)
    rho_s, rho_n = rho * fs, rho * (1 - fs)
    A = 383.7 + 10.649 * rho_n
    A = np.where(A > 1413 + 12.4 * (rho - 160), 558.9 / fs, A)
    return S * (rho_s * S * T) ** 3 / (A * rho_n)


if __name__ == "__main__":
    print("parsed rows:", len(_T), "T =", _T[0], "..", _T[-1])
    for t in [1.6, 1.7, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.17]:
        print(f"T={t:.2f}  rho_s/rho={rho_s_frac(t):.4f}  f^-1={finv_nist(t):.3e}")
    Tg = np.linspace(1.6, 2.17, 5701)
    print("peak f^-1 = %.3e at T = %.3f K" % (finv_nist(Tg).max(), Tg[finv_nist(Tg).argmax()]))
