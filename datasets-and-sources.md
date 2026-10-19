# Datasets and Sources

**Verification status:** every link below was returned by a live search on 11 October 2026. I have confirmed the pages exist and describe what I say they describe. I have **not** downloaded the files, run the code, or verified that paywalled items are accessible from your institution. Check access before building a plan around any Tier 2 item.

**Read this first:** there is no packaged benchmark dataset for any of these three problems. That is a consequence of their novelty — the same fact that makes them publishable makes them data-poor. For all three, **synthetic data you generate yourself is the primary source**, and real data enters only in the application phase. Plan accordingly; do not wait for a dataset that does not exist.

---

## Eq 52 — Superfluid Counterflow

The best-supplied of the three. Helium property data is mature, standardised, and partly free.

### Primary: NIST Technical Note 1334 (FREE, full PDF)

**https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1334.pdf**

Arp, McCarty & Friend, *Thermophysical Properties of Helium-4 from 0.8 to 1500 K*. The standard reference, covering the superfluid range. Its contents list includes **"Mutual Friction, Conductivity, and Viscosity in Superfluid"** and "Superfluid Density Fraction" — exactly the `f(T)` function you need. Tabulated, free, citable, and government-published.

**This is your primary data source. Start here.**

### HEPAK (commercial)

**https://htess.com/hepak/**

Property code descended from NIST TN 1334, explicitly listing *"Mutual friction and conductivity function in helium II"* among its outputs. It is the de facto standard in the cryogenics community — papers modelling He-II heat transfer routinely state that properties were extracted from HEPAK. Commercial licence; check whether your institution has one before budgeting for it. TN 1334 above covers the same ground for free if not.

### CoolProp (free, open source) — with a caveat

**https://coolprop.org/fluid_properties/fluids/Helium.html**

Free and scriptable, referenced to the same NIST TN 1334. **Caveat:** CoolProp is built for normal-fluid thermodynamics and its helium implementation is not a superfluid-transport code. Use it for density, specific heat and the lambda-point location; do **not** expect the Gorter–Mellink conductivity function from it. Verify what it actually returns below `T_λ` before relying on it.

### The exponent correction (important)

**https://arxiv.org/pdf/1501.07156** — Lebrun & Tavian, *Cooling with Superfluid Helium* (CERN Accelerator School, free PDF)

Contains the Gorter–Mellink treatment with tabulated conductivity-integral function `X(T)`, and states that **the best experimental fit for the exponent is 3.4, not 3**, with apparent conductivity peaking near 1.9 K. Also gives the engineering-relevant flux regime (above ~1 kW/m²).

Supporting: Sato et al., *Temperature dependence of the Gorter–Mellink exponent m measured in a channel containing He II*, AIP Conf. Proc. **823**, 387 (2006).

**This corrects an error in an earlier version of the Eq 52 guide.** Do not validate against n = 3.

### Supporting physics

- **https://arxiv.org/pdf/1310.6094** — *Effective thermal conductivity of helium II: from Landau to Gorter–Mellink regimes*. Free. Covers the regime transition, useful for setting your validity range.
- **https://arxiv.org/pdf/2206.05419** — Transient He-II heat transfer from a planar heater, two-fluid model, properties from HEPAK. Free. A worked transient setup you can mirror.
- **https://arxiv.org/pdf/1809.10337** — TUCAN nEDM experiment. Free. Contains a figure comparing Gorter–Mellink conductivity functions from **different models**, plus Kapitza conductance for helium/copper. Useful both for `f(T)` model spread and as a link to Eq 48.

- Van Sciver, *Helium Cryogenics*, 2nd ed., Springer (2012) — the textbook. Library access.

---

## Eq 48 — Interfacial Resistance Field

Best real-data availability of the three, because the NDE community has already built open thermography datasets.

### Pulsed thermography on composites with known defects (FREE, open access)

**https://www.sciencedirect.com/science/article/pii/S2352340920312075**

Erazo-Aux et al., *Thermal imaging dataset from composite material academic samples inspected by pulsed thermography*, Data in Brief (2020). CC-BY open access.

Contents: **12 sequences of ~2000 images each at 512×512**, CFRP and GFRP plates with **Teflon inserts representing delaminations between plies**. Each plate has **25 inserts with length/depth ratios between 1.7 and 75**.

**This is the closest thing to a ready-made benchmark for your problem.** The inserts are planted defects at known positions and depths — i.e. a known `R_K(x)` field with ground truth. The length/depth ratio spread is exactly what you need for the resolution-limit curve in Phase 3. Published as a Data in Brief article, so the dataset is the product and is meant to be reused.

### SDNET2021 — annotated IRT with delamination ground truth (free)

**https://doi.org/10.31356/data019** (dataset DOI)
**https://commons.und.edu/theses/4170/** (describing thesis)
**https://doi.org/10.3390/app14062455** (evaluation paper, open access)

Bridge-deck IRT with ~1.7M delaminated and ~2.9M sound annotated pixels, plus co-registered impact-echo and GPR. Annotated autonomously against ground truth.

**Caveat:** concrete bridge decks, not a two-layer interface with a Kapitza condition. The geometry does not match your formulation. Use it as a **realism check on noise levels and defect contrast**, and as evidence that the NDE community will care — not as your primary training data.

### Thick-composite step-heating, simulated + experimental (open access)

**https://www.sciencedirect.com/science/article/pii/S1359836825010285**

Li et al., *Infrared-based damage detection in thick composites via transfer learning*, Composites Part B (2025). CC licence, open access, with a data-availability statement. GFRP and epoxy plates with artificial subsurface defects, detection demonstrated at **15–30 mm depth**. Pairs simulated FE data with experiment — the same synthetic-to-real strategy your guide recommends.

**This is also your closest competitor.** Read it before writing your novelty claim: it does U-Net segmentation of defects, not physics-informed recovery of a resistance field. That distinction is your contribution and you must state it explicitly.

### TIM degradation (for the aging link to Eq 40)

**https://pubs.acs.org/doi/10.1021/acsami.1c05685** — CNT array TIM degradation through thermal aging. Reports **interface resistance evolution over 500 thermal cycles**, with one configuration rising continuously from 29.8 to 42.8 K·mm²/W (a 50% increase). Paywall likely; the figures are the valuable part.

This is a real `R_K` vs cycle-count dataset in published-figure form. Digitising those curves is legitimate and standard practice — cite it.

### Additional

**https://data.mendeley.com/datasets/9f26y33jsx/1** — Thermal imaging of 3D-printed polymer composites, two IR cameras, labelled defect types. Free Mendeley Data. Process-monitoring rather than interfacial resistance, so lower relevance, but freely downloadable if you want extra IR data for pretraining.

---

## Eq 40 — Aging Memory Kernel

**The weakest data position of the three, and you should know this before committing.** No dataset anywhere records a thermal memory kernel as a function of material age. What exists is *property-versus-aging-coordinate* data: conductivity measured at several damage states. That constrains the aging envelope `g(t)` but says nothing directly about the memory shape `h(s)`.

**Implication:** Phases 0–5 of the Eq 40 guide must run entirely on manufactured solutions. The experimental application in Phase 6 requires you to generate the data yourself.

### Nuclear graphite irradiation degradation (FREE, US government)

**https://www.osti.gov/servlets/purl/2332722** — High-temperature annealing of irradiated nuclear graphite (AGC experiment). Free full text. Each irradiation capsule contains **400+ specimens characterised before and after irradiation**. Reports thermal diffusivity recovery on annealing.

**https://www.osti.gov/servlets/purl/1374593** — Effect of reactor radiation on thermal conductivity of TREAT fuel. Free. Normalised conductivity degradation curves vs dose and temperature.

**https://arxiv.org/pdf/1301.5619** — Fermilab beam absorber design. Free. Contains a **thermal conductivity reduction factor as a function of temperature and damage**, compiled from the literature and binned by damage level. The most directly usable `g` surrogate found.

Why graphite is a good proxy: irradiation dose is a clean, monotonic, *measured* aging coordinate, and conductivity degradation is documented across many grades and doses. The literature also notes **self-annealing**, meaning the damage state depends on the full irradiation *and* temperature history — which is precisely the non-stationary behaviour your kernel is meant to capture. That is a strong motivating argument for the introduction.

**Limitation to state honestly:** these are steady-state property measurements, not transient responses. They give you `g`, not `K(t,τ)`.

### TIM aging

- **https://pubs.acs.org/doi/10.1021/acsami.1c05685** (as above) — 500-cycle interface resistance evolution
- **https://www.researchgate.net/publication/259092689_Degradation_of_thermal_interface_materials_for_high-temperature_power_electronics_applications** — elastomer films, phase-change materials and graphite sheets cycled between −50 and +150 °C at 20 bar, with acoustic microscopy showing delamination and cracking

Note the finding that **some materials showed little change or slight improvement** over cycling. Do not assume monotonic degradation; your `g(t)` should be free to be non-monotonic, and you should say so.

### The experiment you will probably have to run

Transient plane source (Hot Disk / TPS) on an insulation or polymer specimen, identical protocol repeated weekly over 8–12 weeks of accelerated thermal or UV aging. Transient measurement is essential — steady-state guarded hot plate gives conductivity only and cannot see the kernel.

Cost is modest if a TPS instrument is available. This is the single piece of original data that would make the Eq 40 paper experimental rather than purely synthetic.

---

## Tooling

| Need | Tool | Link |
|---|---|---|
| PINN framework | DeepXDE | https://github.com/lululxvi/deepxde |
| PINN framework | NVIDIA PhysicsNeMo (formerly Modulus) | https://github.com/NVIDIA/physicsnemo |
| FEM ground truth | FEniCSx | https://fenicsproject.org |
| Symbolic manufactured solutions | SymPy | https://www.sympy.org |
| Granular DEM (if Eq 55 revisited) | LIGGGHTS / YADE | https://yade-dem.org |

---

## Summary

| Guide | Primary data | Real data available? | Risk |
|---|---|---|---|
| **52** | NIST TN 1334 `f(T)` tables + analytic 1D solutions | Yes — free, standardised | Low |
| **48** | Erazo-Aux pulsed thermography (planted Teflon defects) | Yes — free, with ground truth | Low |
| **40** | Manufactured solutions only | No — proxies only | **High** |

If data availability drives your choice, **52 and 48 are safe and 40 is not**. Eq 40 remains the strongest theoretical contribution, but it is a synthetic-data paper unless you run your own aging campaign. Decide which of those you are willing to be before committing a year to it.
