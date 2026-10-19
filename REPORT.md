# Eq 52 — PINN for He-II Counterflow: Results Against the Implementation Guide

All numbers below are in `results/summary.json`; code in `src/`; logs in `logs/`; figures in `figures/`.
Every phase was run with **5 seeds** (0–4) and is reported as mean ± std unless stated.

## 1. Scorecard vs the guide's §8 pass criteria

| Test | Criterion | Result | Verdict |
|---|---|---|---|
| Phase 0 · linear limit vs FEM | < 1% | 0.0084–0.019% (5 seeds) | **Pass** |
| Phase 1 · steady 1D vs analytic integral | < 1e-4 rel. L2 | 14 of 15 runs pass (typically 1e-6 – 1e-9); **seed 4, hot case (1.70→2.12 K): 2.6e-3** | **Partial — one documented miss** |
| Phase 2 · manufactured transient | < 1e-3 | θ: 3.2–4.6e-6, q: 1.0–3.0e-5 (5 seeds) | **Pass** |
| Phase 3 · recover f(T), n at true value | < 10% max over 1.7–2.1 K | 9.1–9.4% (5/5 pass) | **Pass, thin margin** |
| Phase 3 · single-load ablation | **must degrade** | all three designs degrade (below) | **Pass** |
| Phase 4 · trainable exponent | consistent with ≈3.4, not 3.0 | n = 3.403 ± 0.001 — **but see §3: circular** | **Code check only** |
| Phase 5 · noise × sparsity | characterise | no noisy cell meets 10% max error (below) | **Does not meet criterion** |
| Phase 6 · real data | agreement with tabulation | **Not done** — no measured channel data obtained (§5) | **Open** |
| Seed robustness | 5+ seeds | yes | Done |
| ε-insensitivity | only if Fix 2 used | N/A — mixed formulation (Fix 1) used throughout | N/A |

## 2. Method actually run

Mixed formulation (guide §5 Fix 1): networks for θ and flux q, residuals R₁ = θ_t + q_x − Q and R₂ = θ_x + F(θ)|q|^(n−1)q. No fractional power appears, and no NaNs occurred in any run. Temperatures are normalised to [0,1] over 1.6 K–T_λ. In the inverse problem, F = exp(MLP(θ)) is log-parameterised (positivity by construction). Optimiser is Adam then SciPy L-BFGS-B, in float64.

**Property function.** Phases 1–5 used an analytic Van Sciver-type shape f⁻¹ ∝ [t^5.7(1−t^5.7)]³ with a **nominal** magnitude (peak 1.3e13 W³m⁻⁵K⁻¹). That shape is not from a verified source. Results were therefore repeated with f(T) built from **NIST TN 1334** (§4).

## 3. Phase 4 — the n = 3.404 result must not be read as validation

The synthetic data were generated with n = 3.4. Recovering 3.403 shows the optimiser and parameterisation work. It is **not** data-driven confirmation of the Gorter–Mellink exponent, and a reviewer would be right to call it circular. The result that carries information is the **comparison across designs**:

| Design (n trainable, init 3.0) | f(T) max error | recovered n |
|---|---|---|
| Full: 4 baths × 5 loads over a decade | 9.4% (5/5 pass) | 3.403 ± 0.001 |
| One load, 4 baths | 63 ± 8% (0/5) | 1.92 ± 0.33 |
| One high-load profile | 31 ± 3% (0/5) | 2.73 ± 0.22 |
| Full design, n **fixed at 3.0** | 52% (0/5) | — |

Without load variation, f and n are confounded and both come out wrong and scattered. Fixing the idealised n = 3 biases f(T) by about 52% even with the best data. That supports the guide's §6 argument and the "do not validate against 3.0" instruction.

**Ablation caveat.** With n *fixed* at truth (Phase 3), one profile that sweeps 1.7 → 2.15 K still gets 20% error, degraded but not degenerate: f(T) is identifiable from one profile if the exponent is known. The "one low-load profile" case (96% error) is trivial because it only samples 1.70–1.71 K. The identifiability claim therefore holds for **f and n jointly**, not for f alone.

**n(T) as a function (stretch):** recovered to 0.014 rms error (0.033 max) around a synthetic 3.25→3.55 profile, but f(T) passes 10% on only 3/5 seeds. Treat as preliminary.

## 4. Phase 6 — NIST TN 1334 (what was and was not done)

NIST TN 1334 (7.4 MB; I had estimated 0.5 MB) was downloaded to `data/nist_tn1334/`. f⁻¹(T) was built from its Eq 13–20 plus the Appendix A saturated-liquid density and entropy (12 rows, 1.60–2.15 K). Two points were checked against page images rather than the garbled OCR: Eq 17 is A_lim = 1413 + 12.4(ρ − 160), and Eq 13 uses the natural log (ρ_s/ρ = 0.572 at 1.9 K). The result peaks at **1.95 K with f⁻¹ = 1.8e13 W³m⁻⁵K⁻¹**. NIST states about 10% agreement with the cited data and warns other data may differ by larger factors. **The reference itself is uncertain to roughly the size of our 10% target.**

Shape difference from the correlation, after normalising: 12% max / 8% rms over 1.7–2.1 K. This is a modest shape change, not a radically different function.

Rerunning with NIST as ground truth gave essentially the same outcomes as the correlation:

| Config | correlation truth | NIST truth |
|---|---|---|
| Phase 3 full | 9.2% (5/5) | 9.4% (5/5) |
| Phase 3 one load, 4 baths | 11.5% (0/5) | 11.2% (0/5) |
| Phase 4 full: n | 3.403 ± 0.001 | 3.404 ± 0.001 |
| Phase 4 one load, 4 baths: n | 1.92 ± 0.33 | 1.98 ± 0.25 |
| Phase 5, 1% noise, 8 sensors | 12% (2/5) | 12% (2/5) |

This shows the recovery is not tuned to one function shape. **It is still synthetic temperature data generated from NIST properties — not measured data, so it is not the guide's real-data validation.**

## 5. What Phase 6 still needs

The guide's Phase 6 needs published or measured channel temperature profiles with known heat loads. None was obtained. Candidates from the literature search: Dimotakis (Caltech thesis, 1972, supercritical counterflow temperature profiles and A(T)); Sato et al., AIP Conf. Proc. 823, 387 (2006, exponent m(T)). Using these means digitising figures or contacting authors. Also, NIST's law is strictly n = 3 (Eq 19), while the measured exponent is ≈3.4, so comparing recovered f(T) to the NIST tabulation requires a consistent treatment of units for the n ≠ 3 case.

## 6. Phase 5 — noise and sparsity (honest account)

Noise is a percentage of the 0.58 K band (1% ≈ 5.8 mK). Max / rms error in f, mean over 5 seeds:

| Noise | 4 sensors | 8 sensors | 16 sensors |
|---|---|---|---|
| 1% | 17% / 6.0% | 12% / 4.5% | 13% / 4.6% |
| 3% | 22% / 8.4% | 23% / 8.0% | 17% / 6.7% |
| 5% | 27% / 11% | 33% / 12% | 29% / 9.0% |

Recovered n: 3.40–3.58 across all cells. **No noisy cell meets the 10% max-error criterion** (best: 2/5 seeds at 1% noise, 8 sensors). RMS error degrades gracefully, and the worst errors sit at the hot band edge, which has the fewest data. More sensors helped weakly and not monotonically.

**Process failure, disclosed.** My first Phase 5 sweep weighted the data by signal size rather than by the noise σ. This under-weighted the data and gave 57–71% errors, which got *worse* with more sensors, a sign of a setup fault. It is archived in `results/p5_v1_rise_weighting/`. The corrected run uses σ-weighting with physics-residual weight 300. **That weight was tuned on seed 0 of the (1%, 8-sensor) cell only**, so the other cells are out-of-sample, but 300 was then applied unchanged to every cell. Trends should be called indicative.

## 7. Things that were tuned, and what that means

- **Heater-end limit raised from 2.12 K to 2.15 K** (T_λ = 2.1768 K). At 2.12 K, the error in the top 2.08–2.10 K sub-band was 23%, because only one sensor per top-load profile reached it. This is an experimental-design choice: a real experiment must drive the channel near T_λ to constrain f there. The guide's 10% target is met only with that coverage. Tuning runs are archived in `results/tuning/`.
- Longer optimisation (40k iterations), heavier physics weight, and more loads did **not** help the 2.12 K case, which points to a data-coverage limit, not an optimiser one.
- **Phase 3's 9.1–9.4% sits just under the 10% line.** The tiny seed spread (±0.1%) reflects noise-free data with only initialisation and collocation varying. It indicates a data-set-determined floor, not a lucky result.

## 8. Phase 1 near-miss (kept in the report on purpose)

Seed 4, hot case (1.70 K bath, heater end at 2.12 K): relative L2 error 2.6e-3 against a 1e-4 target; the other 14 runs pass. Seed 3 on the same case passed at 2.1e-5. This is a forward-solve convergence failure in the stiff region near T_λ, **distinct from** the inverse-problem coverage issue, so the inverse-side fix does not apply to it. It was not reseeded or retuned. I recommend a longer L-BFGS budget or a staged load ramp if it needs closing.

## 9. Literature check (guide §11)

| Item | Finding |
|---|---|
| "Gorter-Mellink" + neural network | No relevant work found (a keyword-index artifact only) |
| Superfluid + physics-informed | None found; only general PINN heat-transfer papers |
| Classical regression baseline | Closest: a 2021 tube model that tuned m ∈ [3,4] against rise times (not a profile regression of f(T)); Dimotakis & Broadwell (1973) measured A(T) directly. These are the baselines to cite. |
| Tabulated f(T) | NIST TN 1334 — accessed, with the caveats of §4 |
| Published channel profiles | Not yet obtained (§5) |

The searches were web searches and are **not exhaustive**. "No PINN work found" should be reconfirmed on arXiv and Scopus before any novelty claim.

## 10. Bottom line

The mixed-formulation PINN solves forward steady and transient He-II counterflow to high accuracy, with one documented near-miss. It recovers f(T) to about 9% over 1.7–2.1 K from noise-free multi-load data (on both the analytic and the NIST-derived function), and the single-load ablations fail as the identifiability argument requires. It does **not** yet meet the 10% criterion under measurement noise, and the real-data phase has not been done. Those two items are what stand between this and the "complete, defensible paper" of the guide's §10.
