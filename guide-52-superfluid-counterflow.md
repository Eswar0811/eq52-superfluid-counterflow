# Implementation Guide — Eq 52
## Superfluid Helium Counterflow: Inverting a Cubic Flux Law

**Ratings:** Novelty 5/5 · Complexity 3/5 · PINN impact over classical 4/5

---

## 1. The claim, stated precisely

In superfluid helium (He-II, below the lambda point `T_λ = 2.1768 K`), heat is **not conducted**. It is carried by counterflow of the normal and superfluid components, limited by mutual friction between them. The resulting transport law — Gorter–Mellink — is **cubic in heat flux**:

```
∇T = -f(T) |q|² q
```

Equivalently, solving for flux:

```
q = -( |∇T| / f(T) )^(1/3) · ∇T/|∇T|
```

so the **effective conductivity depends on the heat load itself**:

```
k_eff = ( f(T) |∇T|² )^(-1/3)
```

This spans orders of magnitude across operating conditions. No Fourier-based solver represents it natively, and — critically — **no PINN work on superfluid heat transport surfaced in a dedicated literature search.** This is the cleanest open position identified across all 60 equations in the parent survey.

### Full system

```
ρc ∂T/∂t + ∇·q = Q(x,t)
∇T = -f(T) |q|² q
```

with `f(T)` the inverse Gorter–Mellink conductivity function, strongly peaked in temperature between 1.6 K and `T_λ`.

**The inverse problem (the contribution):** recover `f(T)` as a continuous function from temperature measurements, rather than interpolating tabulated values.

---

## 2. Why this is the easiest of the three top candidates

Complexity is 3/5, not 4 or 5, because:

- It is a **single PDE**, not a coupled multiphysics system
- The domain is typically **1D or axisymmetric** (a channel, a duct, a cooling tube)
- The degenerate limit is clean and checkable
- Analytical solutions exist for steady 1D counterflow in a uniform channel

The hard part is purely numerical: the `1/3` power and a singularity at zero gradient. That is a contained, solvable problem (§5).

**If you want a result in one paper cycle, start here, not with Eq 40.**

---

## 3. Data strategy

### Tier 1 — Analytical steady solutions (mandatory)
For a 1D channel of length `L` with uniform heat flux `q₀` and fixed end temperature, the steady state integrates directly:

```
dT/dx = -f(T) q₀³   →   ∫ dT/f(T) = -q₀³ x + C
```

With a simple analytic form for `f(T)` this gives a closed-form `T(x)`. **This is your Phase 1 ground truth and it costs nothing to generate.**

### Tier 2 — Manufactured solutions
Pick `T(x,t)`, substitute, derive the forcing. Necessary for the transient case where no analytic solution exists.

### Tier 3 — Published He-II property data
The Gorter–Mellink function `f(T)` is extensively tabulated in the cryogenics literature. Primary sources to obtain and verify yourself:

- **Van Sciver, *Helium Cryogenics*** (Springer) — the standard reference; contains `f(T)` tabulations and counterflow correlations
- **NIST cryogenic material property databases** — helium thermophysical properties
- **HEPAK** — commercial helium property software, widely used in the cryogenics community
- Superconducting magnet cooling literature (LHC, ITER, fusion magnet) — contains real channel temperature measurements under known heat loads

**Honest note:** I have not verified current access to any of these. Confirm availability before building the plan around Tier 3. Tiers 1 and 2 are sufficient for the core methodological result.

### Tier 4 — Your own experiment (if a cryostat is accessible)
A He-II channel with resistive heater and calibrated thermometry at several stations. Vary heat load across a decade. **The varying-power axis is essential** — it is what makes the cubic exponent identifiable (see §6).

---

## 4. Architecture

```
T_θ(x, t)  → MLP, 4–5 layers × 64, tanh
f_ψ(T)     → MLP, 3 layers × 32, softplus output (positivity mandatory)
```

Optionally a separate flux network `q_χ(x,t)` with the constitutive law enforced as a residual (a **mixed formulation**). This is strongly recommended — see §5.

**Hard constraints:**
- `f > 0` via softplus — a negative `f` violates the second law
- Restrict `f` input to the physical range `[1.6 K, T_λ]`; clamp or penalise excursions
- Normalise `T` to `[0,1]` over the operating band. The raw Kelvin values are a narrow band around 2, which is terrible conditioning.

---

## 5. The core numerical problem: the 1/3 power and zero-gradient singularity

Writing the PDE in temperature-only form gives

```
ρc ∂T/∂t = ∇·[ ( |∇T| / f(T) )^(1/3) · ∇T/|∇T| ]
```

The factor `|∇T|^(1/3 - 1) = |∇T|^(-2/3)` **diverges as `∇T → 0`**. This occurs at symmetry planes, insulated boundaries, and anywhere the solution flattens. Autodiff will produce NaN.

### Fix 1 — Mixed formulation (recommended)
Introduce `q` as a separate network output and impose the constitutive law as a residual instead of substituting it:

```
R₁ = ρc ∂T/∂t + ∇·q - Q
R₂ = ∇T + f(T) |q|² q
```

**No fractional power appears anywhere.** `R₂` is a smooth polynomial in `q`. This removes the singularity entirely and is the single most important implementation decision in this guide.

Cost: more network outputs, one extra residual. Worth it.

### Fix 2 — Regularised magnitude (if you insist on temperature-only)
```
|∇T|_ε = sqrt(|∇T|² + ε²),  ε ≈ 1e-6 · (scale of ∇T)
```
Report `ε` and show results are insensitive to it across a decade. Reviewers will check.

**Use Fix 1.** Fix 2 is documented only so you recognise the failure mode if you see NaNs.

---

## 6. Identifiability — the varying-power requirement

`f(T)` and the flux exponent are **confounded** at a single heat load. A different exponent with a rescaled `f` fits identically. To separate them you need data across **multiple heat loads** spanning at least a decade in `q`.

### Important correction on the exponent

The idealised Gorter–Mellink law gives flux cubed (exponent `n = 3`). **The experimentally measured exponent is not 3.** The cryogenics literature reports a best fit closer to **`n ≈ 3.4`**, and the exponent is itself **temperature-dependent** — there is published work specifically on measuring the temperature dependence of the Gorter–Mellink exponent in a He-II channel (Sato et al., *AIP Conf. Proc.* **823**, 387, 2006).

This makes the problem *better*, not worse:

- Do **not** write "recover n = 3" as a success criterion. You would be validating against the wrong number.
- The real open question is whether `n(T)` is recoverable as a function, alongside `f(T)`, from counterflow data. That is a sharper and more defensible contribution than confirming a textbook constant.
- Be aware this is a harder identifiability problem than a constant exponent: `f(T)` and `n(T)` are both temperature-dependent and will fight each other. Begin with constant `n` trainable, then relax.

| To recover | You need |
|---|---|
| `f(T)` shape only (exponent fixed at literature value) | One heat load, several temperatures |
| `f(T)` **and** a constant exponent `n` | Multiple heat loads across ≥1 decade |
| `f(T)` **and** a temperature-dependent `n(T)` | Multiple heat loads × multiple bath temperatures (a 2D sweep) |

A strong secondary result: **treat `n` as trainable** and compare the recovered value against the reported ≈3.4 rather than the idealised 3. Agreement with the *measured* exponent, recovered without being told, is a genuine validation. Agreement with 3.0 would actually indicate a problem.

---

## 7. Phase-by-phase plan

### Phase 0 — Linear limit sanity check (2 days)
Replace the constitutive law with `q = -k∇T`. Solve ordinary conduction. Must match FEM to <1%.

### Phase 1 — Forward steady 1D counterflow (3–5 days)
Known `f(T)`, uniform heat flux, compare to the analytic integral from §3.
- **Success:** relative L2 error < 1e-4
- This alone is, as far as the search showed, the first PINN solution of He-II counterflow

### Phase 2 — Forward transient (1 week)
Manufactured solution, time-dependent. Validates the mixed formulation under transients.
- **Success:** < 1e-3 relative L2

### Phase 3 — Inverse recovery of f(T) (2 weeks, core result)
Synthetic data from known `f`, multiple heat loads, sparse sensors.
- **Success:** `f(T)` recovered within 10% across `[1.7 K, 2.1 K]`
- **Ablation:** single heat load must give a degraded or degenerate recovery — this proves the varying-power requirement

### Phase 4 — Trainable exponent (1–2 weeks)
Let `n` be free. **Compare against the measured ≈3.4, not the idealised 3** (see §6).
- **Success:** recovered `n` consistent with the value reported for your bath temperature, across seeds
- Stretch: recover `n(T)` as a function, which requires a 2D sweep of heat load × bath temperature
- High-impact figure, moderate effort. Do not skip.

### Phase 5 — Noise and sensor sparsity (1 week)
1%, 3%, 5% noise; 4, 8, 16 sensors. Thermometry in He-II is genuinely noisy at the mK level, so this is not a formality.

### Phase 6 — Real data application (3–4 weeks)
Published channel data or your own cryostat. Compare recovered `f(T)` against the tabulated standard. **Agreement with tabulation is the validation**; disagreement is either a finding or a bug, and you must be able to tell which.

---

## 8. Testing protocol

| Test | Pass criterion |
|---|---|
| Linear-conduction limit | <1% vs FEM |
| Steady 1D vs analytic integral | <1e-4 relative L2 |
| Manufactured transient | <1e-3 relative L2 |
| Inverse `f(T)` recovery | <10% over operating band |
| Single-load ablation | **Must degrade** |
| Trainable exponent | Consistent with measured ≈3.4 at that bath temperature, not 3.0 |
| Seed robustness | 5+ seeds, mean±std reported |
| `ε`-insensitivity (if Fix 2 used) | Stable across decade of `ε` |

---

## 9. Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| NaN within a few hundred epochs | Zero-gradient singularity | Switch to mixed formulation (§5) |
| `f` recovered with correct shape, wrong magnitude | Exponent/scale confounding | Multiple heat loads |
| Training stalls near `T_λ` | `f(T)` varies very steeply there | Denser collocation in `T` near `T_λ`; consider log-parameterising `f` |
| Flux field non-smooth | `R₂` underweighted | Raise constitutive residual weight; it is as important as the energy equation |
| Good fit, unphysical `f` | Insufficient positivity/smoothness | Softplus output plus curvature penalty on `f` |

---

## 10. Why this is a strong paper even if modest in scope

The honest framing: this is not a methodological advance in PINNs. It is a **first application to a transport regime that the machine learning literature has entirely missed**, where the constitutive law is nonlinear in flux in a way that nothing in the Fourier-based corpus resembles.

Pair that with:
1. A clean degenerate limit
2. An analytic benchmark
3. Data-driven recovery of a textbook physical exponent
4. A real engineering stake (magnet quench margin)

That is a complete, defensible paper. It will not be called incremental, because there is nothing to be incremental *to*.

---

## 11. What to verify before committing

- [ ] Search: `"Gorter-Mellink" + neural network`; `superfluid + "physics-informed"`; `"He-II" + machine learning heat`
- [ ] Confirm access to a tabulated `f(T)` source (Van Sciver or HEPAK)
- [ ] Check the superconducting-magnet literature for published channel temperature profiles with known heat loads
- [ ] Verify whether anyone in the cryogenics community has already done data-driven `f(T)` fitting by classical regression — if so, that is your baseline, not a competitor
