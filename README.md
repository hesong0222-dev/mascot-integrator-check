# mascot-integrator-check

A small, self-contained check of how BEAST 2 MASCOT (v3.0.0) integrates the MASCOT ODE. It goes with the GitHub
issue at https://github.com/CompEvol/Mascot/issues (title: "Log-likelihood second-derivative term in Euler2ndOrder
appears not to be reset between ODE steps").

`bench/mascot_vs_beast.py` re-implements, in Python, the integrator of `Euler2ndOrder3.java` / `Euler2ndOrderCPU.cpp`
(second-order Taylor steps, step size from `epsilon`, renormalisation) and evaluates it on all 401 logged states of
the Taming the BEAST H3N2 tutorial run. It compares four variants against BEAST's logged `Mascot` density and against
a converged solution of the same MASCOT equations (`approx/mascot.py`, SciPy DOP853, rtol 1e-10):

| variant | `epsilon` | log-slot second derivative |
|---|---|---|
| `emul_eps1e-3` | 1e-3 (BEAST default) | carried over between steps, as in the source |
| `emul_eps1e-3_nocarry` | 1e-3 | recomputed each step |
| `emul_eps1e-9` | 1e-9 | carried |
| `emul_eps1e-9_nocarry` | 1e-9 | recomputed |

This code was written for a methods paper on exact structured-coalescent likelihoods, in which MASCOT is a
comparison point; nothing here depends on that paper.

## Reproduce (about 5 minutes on a laptop, one core)

Needs Python 3.11 or later (NumPy 2.5 / SciPy 1.18).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python bench/mascot_vs_beast.py     # writes bench/mascot_vs_beast.csv
.venv/bin/python check_results.py                                     # summary table + diff to the reference CSV
```

`python bench/mascot_vs_beast.py 10` uses every 10th logged state (41 states, about 30 s); `check_results.py` then
summarises and compares those 41 rows only.

Expected output of `check_results.py` (identical to `results/mascot_vs_beast_reference.csv`, produced on macOS arm64,
Python 3.13, NumPy 2.5.3 / SciPy 1.18.1 with OpenBLAS):

```
401 logged states
comparison (log units)                                     max abs   mean abs
converged MASCOT ODE (DOP853) - BEAST logged              8.60e-01   1.32e-01
emulated integrator (eps 1e-3, carry) - BEAST logged      2.13e-14   4.00e-15
carry effect: with - without carry (eps 1e-3)             7.19e-01   1.13e-01
step size only: no carry eps 1e-3 - converged             1.75e-01   2.55e-02
with carry, eps 1e-9 - converged                          1.61e-02   4.16e-03
no carry, eps 1e-9 - converged                            1.75e-05   3.40e-06
BEAST logged value above the converged value in 293/401 states
```

Checked on 2026-10-09: a fresh copy of this folder reproduces the reference CSV exactly (full run, 283 s, NumPy
2.5.3/SciPy 1.18.1 with OpenBLAS), and with the plain PyPI wheels from `requirements.txt` (Apple Accelerate
backend) the 41-state run agrees with it to 2e-14. On other platforms the last digits may differ slightly.

## Files

| path | content |
|---|---|
| `bench/mascot_vs_beast.py` | integrator emulation and the 401-state comparison |
| `approx/mascot.py` | MASCOT equations solved with SciPy DOP853 (converged reference) |
| `data/convert.py` | NEXUS/Newick reader, converts a BEAST tree to a list of sampling/coalescence events |
| `data/raw/H3N2.trees`, `data/raw/H3N2.log` | tutorial run (see below) |
| `results/mascot_vs_beast_reference.csv` | output of the script, one row per logged state |
| `check_results.py` | summary and comparison with the reference CSV |

Conventions (checked against the MASCOT source): `M[a][b]` = `b_migrationConstant.a_to_b` (backward-in-time rate
a→b), coalescent rate 1/Ne per pair, no root or conditioning term.

## Limits

- One dataset, 3 demes, `Constant` dynamics. The Java/C++ code was read, not built or run; the claim is that the
  Python emulation reproduces BEAST's logged values to 1e-13.
- Fixed tree and fixed parameters only. Whether this changes BEAST posteriors was not tested.

## Data attribution

`data/raw/H3N2.trees` and `data/raw/H3N2.log` are `precooked_runs/H3N2.trees` and `precooked_runs/H3N2.log` from
the Taming the BEAST MASCOT tutorial (Nicola F. Müller and contributors),
https://github.com/Taming-the-BEAST/Mascot-Tutorial, commit 6d5b9b8e19ee78fcacc3f3632903672d759523cd, licensed
under Creative Commons Attribution 4.0 International (`data/raw/LICENSE-CC-BY-4.0.txt`). The files are unchanged.

| file | sha256 |
|---|---|
| H3N2.trees | af0eb00f6390868fbacc4100428f5a7b873059bab96548d378091efb8f850f40 |
| H3N2.log | 3a06c406ca2e2b2fe87aab95080fbfe40343a8e259c8199150555e0f95330603 |

MASCOT: Müller NF, Rasmussen D, Stadler T. 2018. MASCOT: parameter and state inference under the marginal structured
coalescent approximation. Bioinformatics 34(22):3843–3848. https://doi.org/10.1093/bioinformatics/bty406

## License

Code: MIT (`LICENSE`). Tutorial data: CC BY 4.0, as above.
