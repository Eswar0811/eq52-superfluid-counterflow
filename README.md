# Eq. 52: PINN for Superfluid Helium Counterflow

Research code and results for recovering the temperature-dependent Gorter–Mellink
transport function and exponent from synthetic He-II channel temperature profiles.
The governing constitutive relation is

```text
∇T = -f(T) |q|^(n-1) q
```

The project uses a mixed PINN formulation and compares recovery across load
designs, noise levels, sensor densities, and two property-function shapes.

## Results and limitations

The detailed assessment is in [REPORT.md](REPORT.md). In brief, the noiseless
full-load design recovers `f(T)` to about 9.4% maximum error over the reported
band, and the trainable exponent is recovered in synthetic tests. This is not
real-data validation: Phase 6 remains open. Noisy cases do not meet the 10%
maximum-error target, and the report documents a Phase 1 near-miss and the
limitations of the NIST comparison.

## Repository contents

- `src/`: PINN, forward solvers, inverse solver, NIST properties, and report/figure generation.
- `results/`: saved JSON metrics and NPZ arrays used by the report and figures.
- `figures/`: report figures.
- `data/nist_tn1334/`: NIST Technical Note 1334 source and extracted text used by `src/nist_props.py`.
- `REPORT.md`: results, caveats, and interpretation.
- `guide-52-superfluid-counterflow.md`: implementation guide.
- `datasets-and-sources.md`: source and dataset notes.

Local virtual environments and execution logs are intentionally not tracked.

## Setup

Tested with Python 3.14.4 and the versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Running

Run an individual phase from `src/` (outputs are written under `results/`):

```bash
cd src
python phase0_linear.py 0
python phase1_steady.py 0
python phase2_transient.py 0
```

Run inverse configurations from the repository root; each `config:seed` pair
is a separate job:

```bash
python run_batch.py 2 p3_full:0 p3_full:1
```

`run_batch.py` skips completed jobs and writes logs under `logs/`. To use NIST
properties for inverse runs, set `HE2_PROPS=nist` in the environment before
starting the batch. After running the desired experiments, generate the
summary and figures from `src/`:

```bash
cd src
python make_report.py
```

Full training runs can take substantial time. Existing result files and
figures are included so the reported analysis can be inspected without
rerunning them.

## Source data

The included NIST Technical Note 1334 is by Arp, McCarty, and Friend,
*Thermophysical Properties of Helium-4 from 0.8 to 1500 K* (NIST TN 1334).
The source is also available from
[NIST](https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1334.pdf).
See [datasets-and-sources.md](datasets-and-sources.md) for additional citations
and caveats.
