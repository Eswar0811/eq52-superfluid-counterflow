"""Aggregate all results into figures + results/summary.json (run after all jobs finish)."""
import pinn  # noqa: F401

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "../results"
ld = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{pat}"))]
S = {}

# ---- Phase 0 / 1 / 2 ---------------------------------------------------------------------------
p0 = ld("phase0_seed?.json")
S["phase0"] = dict(n=len(p0), relL2=[r["rel_L2_vs_FEM"] for r in p0], target=0.01)
p1 = ld("phase1_seed?.json")
S["phase1"] = {c: [r["cases"][c]["relL2_rise"] for r in p1] for c in p1[0]["cases"]}
S["phase1"]["target"] = 1e-4
p2 = ld("phase2_seed?.json")
S["phase2"] = dict(n=len(p2), relL2_theta=[r["relL2_theta"] for r in p2], relL2_q=[r["relL2_q"] for r in p2], target=1e-3)


def agg(pat):
    rs = ld(pat)
    if not rs:
        return None
    m = np.array([r["F_maxrel_band"] for r in rs])
    out = dict(n=len(rs), Fmax_mean=m.mean(), Fmax_std=m.std(), Fmax_all=m.tolist(),
               Frms_mean=float(np.mean([r["F_rmsrel_band"] for r in rs])),
               pass10=int(sum(r["F_pass_10pct"] for r in rs)))
    if "n_recovered" in rs[0]:
        n = np.array([r["n_recovered"] for r in rs])
        out.update(n_mean=n.mean(), n_std=n.std(), n_all=n.tolist())
    if "n_fun_rms_err" in rs[0]:
        out.update(nfun_rms=float(np.mean([r["n_fun_rms_err"] for r in rs])),
                   nfun_max=float(np.mean([r["n_fun_maxabs_err"] for r in rs])))
    return out


cfgs = ["p3_full", "p3_single_load_all_baths", "p3_single_profile_high", "p3_single_profile_low",
        "p4_full", "p4_single_load_all_baths", "p4_single_profile_high", "p4_assume_n3", "p4_nfun_full"]
S["inverse_correlation"] = {c: agg(f"inv_{c}_seed?.json") for c in cfgs}
S["inverse_nist"] = {c: agg(f"inv_{c}_nist_seed?.json") for c in cfgs}
S["phase5_correlation"] = {f"noise{p}_m{m}": agg(f"inv_p5_noise{p}_m{m}_seed?.json") for p in (1, 3, 5) for m in (4, 8, 16)}
S["phase5_nist"] = {f"noise{p}_m8": agg(f"inv_p5_noise{p}_m8_nist_seed?.json") for p in (1, 3)}
json.dump(S, open(f"{R}/summary.json", "w"), indent=2, default=float)

# ---- Fig 1: Phase 3 recovery + ablation -------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for c, col, lab in [("p3_full", "C0", "full design (20 profiles)"), ("p3_single_load_all_baths", "C1", "one load, 4 baths"),
                    ("p3_single_profile_high", "C3", "one high-load profile"), ("p3_single_profile_low", "C2", "one low-load profile")]:
    z = np.load(f"{R}/inv_{c}_seed0.npz")
    if c == "p3_full":
        ax[0].plot(z["T"], z["F_true"], "k", lw=2, label="truth")
    ax[0].plot(z["T"], z["F_pred"], color=col, lw=1.3, ls="-" if c == "p3_full" else "--", label=lab)
    ax[1].plot(z["T"], 100 * (z["F_pred"] / z["F_true"] - 1), color=col, label=lab)
ax[0].set_yscale("log"); ax[0].set_xlabel("T [K]"); ax[0].set_ylabel("F(T)  (normalised)"); ax[0].legend(fontsize=7)
ax[1].axhspan(-10, 10, color="gray", alpha=.2); ax[1].set_ylim(-100, 100); ax[1].set_xlabel("T [K]")
ax[1].set_ylabel("error in F [%]"); ax[1].set_title("grey band = 10% target (seed 0, n fixed at truth)", fontsize=8)
plt.tight_layout(); plt.savefig("../figures/phase3_recovery_ablation.png", dpi=160); plt.close()

# ---- Fig 2: Phase 4 exponent recovery ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))
labs = [("p4_full", "full design"), ("p4_single_load_all_baths", "one load\n4 baths"),
        ("p4_single_profile_high", "one profile\nhigh load")]
for i, (c, lab) in enumerate(labs):
    a = S["inverse_correlation"][c]
    ax.scatter([i] * a["n"], a["n_all"], color="C0", s=25, label="correlation truth" if i == 0 else None)
    b = S["inverse_nist"][c]
    ax.scatter([i + .18] * b["n"], b["n_all"], color="C1", s=25, marker="s", label="NIST truth" if i == 0 else None)
ax.axhline(3.4, color="k", lw=1, label="n used to generate data (3.4)"); ax.axhline(3.0, color="r", ls=":", label="idealised n=3")
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in labs]); ax.set_ylabel("recovered n"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("../figures/phase4_exponent.png", dpi=160); plt.close()

# ---- Fig 3: Phase 5 noise x sensors -----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for m, mk in zip((4, 8, 16), "osd"):
    for k, (key, ylab) in enumerate([("Fmax_mean", "max error in F"), ("Frms_mean", "rms error in F")]):
        ax[k].errorbar([1, 3, 5], [S["phase5_correlation"][f"noise{p}_m{m}"][key] for p in (1, 3, 5)],
                       yerr=[S["phase5_correlation"][f"noise{p}_m{m}"]["Fmax_std"] if key == "Fmax_mean" else 0 for p in (1, 3, 5)],
                       marker=mk, label=f"{m} sensors"); ax[k].set_ylabel(ylab); ax[k].set_xlabel("noise [% of band]")
ax[0].axhline(.10, color="gray", ls="--"); ax[0].legend(fontsize=7); ax[0].set_title("dashed = 10% target", fontsize=8)
plt.tight_layout(); plt.savefig("../figures/phase5_noise_sensors.png", dpi=160); plt.close()
print(json.dumps({k: v for k, v in S.items() if k in ("phase0", "phase1", "phase2")}, indent=1, default=float))
