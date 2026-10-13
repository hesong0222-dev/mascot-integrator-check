"""Explain the gap between our MASCOT (approx/mascot.py) and BEAST2-MASCOT's logged density (H3N2 tutorial).

Re-implements the integrator of MASCOT v3.0.0 (github.com/nicfel/Mascot, tag v3.0.0, commit 7943662):
src/mascot/ode/Euler2ndOrder3.java (3 states, Java path) == jni/Euler2ndOrderCPU.cpp (native path):
  second-order Taylor step  p += h p' + h^2/2 p''   (also for the log-likelihood slot),
  h = min(cbrt(6 eps / max|p'''~|), remaining, maxStep),  eps = 1e-3, maxStep = inf (defaults; the
  tutorial XML sets neither), p'''~ = (C_i - T_ia) p'' + migration(p'')  (an approximation),
  h *= 0.9 while any new p leaves [0,1] or moves by > 0.2;  every lineage renormalised after each step.

Usage: python bench/mascot_vs_beast.py [stride]   -> bench/mascot_vs_beast.csv
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
from approx.mascot import loglik as mascot_ode  # noqa: E402
from convert import parse_newick, read_nexus, to_events  # noqa: E402

DEMES = ["Hong_Kong", "New_York", "New_Zealand"]


def derivs(p, M, lam):
    """p: (n,d). Returns p', p'', p'''~ and log', log'' exactly as Euler2ndOrder3.java does."""
    S = p.sum(0)
    T = lam[None, :] * (S[None, :] - p)          # tCR
    C = (p * T).sum(1, keepdims=True)            # sumCoal
    coal = C - T
    out = M.sum(1)
    mig = lambda x: x @ M - x * out[None, :]     # sum_b x_b m_ba - x_a sum_b m_ab
    pd = p * coal + mig(p)
    ld = -0.5 * C.sum()
    Sd = pd.sum(0)
    pC = (lam[None, :] * (pd * (S[None, :] - 2 * p) + p * Sd[None, :])).sum(1, keepdims=True)
    pdd = coal * pd + p * (pC - lam[None, :] * (Sd[None, :] - pd)) + mig(pd)
    ldd = -0.5 * pC.sum()
    pddd = coal * pdd + mig(pdd)
    return pd, pdd, pddd, ld, ldd


def integrate_beast(p, dur, M, lam, eps=1e-3, max_step=math.inf, log_order=2, stats=None, carry=True):
    """carry=True reproduces MASCOT's handling of the log slot of pDotDot: it is zeroed once per interval
    (calculateValues), then each step does pDotDot[log] -= sum(pCoalRate); pDotDot[log] /= 2, so the value
    used is ldd_n = ldd_{n-1}/2 + true_ldd_n instead of true_ldd_n."""
    logp = 0.0
    it = 0
    ldd_prev = 0.0
    while dur > 0:
        it += 1
        pd, pdd, pddd, ld, ldd = derivs(p, M, lam)
        if carry:
            ldd = ldd_prev / 2 + ldd
            ldd_prev = ldd
        mx = np.abs(pddd).max()
        h = min(np.cbrt(eps * 6 / mx) if mx > 0 else math.inf, min(dur, max_step))
        for x, xd, xdd in zip(p.ravel(), pd.ravel(), pdd.ravel()):  # sequential shrink, as in Java
            nv = x + xd * h + xdd * h * h * 0.5
            while nv > 1 or nv < 0 or abs(nv - x) > 0.2:
                h *= 0.9
                nv = x + xd * h + xdd * h * h * 0.5
        p = p + pd * h + pdd * h * h * 0.5
        logp += ld * h + (ldd * h * h * 0.5 if log_order == 2 else 0.0)
        p = p / p.sum(1, keepdims=True)
        dur -= h
        if it > 10000:
            return p, -math.inf
    if stats is not None:
        stats.append(it)
    return p, logp


def loglik_beast(tree, eps=1e-3, max_step=math.inf, log_order=2, stats=None, carry=True):
    M = np.array(tree["M"], float)
    np.fill_diagonal(M, 0.0)
    lam = 1.0 / np.array(tree["Ne"], float)
    d = M.shape[0]
    ids, p, logL, t = [], np.zeros((0, d)), 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and ids:
            p, lp = integrate_beast(p, ev["t"] - t, M, lam, eps, max_step, log_order, stats, carry)
            logL += lp
        t = ev["t"]
        if ev["type"] == "sample":
            row = np.zeros(d)
            row[ev["deme"]] = 1.0
            p = np.vstack([p, row])
            ids.append(ev["id"])
        else:
            i, j = ids.index(ev["a"]), ids.index(ev["b"])
            w = lam * p[i] * p[j]
            s = w.sum()
            logL += math.log(s)
            keep = [x for x in range(len(ids)) if x not in (i, j)]
            p = np.vstack([p[keep], w / s])
            ids = [ids[x] for x in keep] + [ev["new"]]
    return logL


def load_log(path):
    rows = [l for l in open(path) if not l.startswith("#")]
    hdr = rows[0].rstrip("\n").split("\t")
    return [dict(zip(hdr, r.rstrip("\n").split("\t"))) for r in rows[1:]]


def params(row):
    Ne = [float(row[f"NeConstant.{a}"]) for a in DEMES]
    M = [[0.0] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            if a != b:
                M[a][b] = float(row[f"b_migrationConstant.{DEMES[a]}_to_{DEMES[b]}"])
    return M, Ne


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    raw = os.path.join(ROOT, "data", "raw")
    trans, trees = read_nexus(os.path.join(raw, "H3N2.trees"))
    trees = dict(trees)
    log = load_log(os.path.join(raw, "H3N2.log"))
    out = []
    for row in log[::stride]:
        name = f"STATE_{row['Sample']}"
        ev, err, _ = to_events(parse_newick(trees[name]), trans)
        M, Ne = params(row)
        tree = {"M": M, "Ne": Ne, "events": ev}
        beast = float(row["Mascot"])
        ode = mascot_ode(tree)
        steps = []
        e2 = loglik_beast(tree, stats=steps)                 # faithful emulation (carry bug included)
        e2nc = loglik_beast(tree, carry=False)               # same scheme, exact 2nd derivative of log slot
        e2fine = loglik_beast(tree, eps=1e-9)                # faithful emulation, tiny eps
        e2ncfine = loglik_beast(tree, eps=1e-9, carry=False)  # corrected scheme, tiny eps -> should equal ours
        out.append({"sample": row["Sample"], "beast_logged": beast, "ours_dop853": ode,
                    "emul_eps1e-3": e2, "emul_eps1e-3_nocarry": e2nc, "emul_eps1e-9": e2fine, "emul_eps1e-9_nocarry": e2ncfine,
                    "n_steps_eps1e-3": sum(steps), "tip_date_err": err})
        print(f"{name:14s} beast {beast:12.6f}  ours-beast {ode - beast:+.2e}  emul-beast {e2 - beast:+.2e}"
              f"  emul_nocarry-beast {e2nc - beast:+.2e}  emul(1e-9)-ours {e2fine - ode:+.1e}  emul_nc(1e-9)-ours {e2ncfine - ode:+.1e}  steps {sum(steps)}",
              flush=True)
    with open(os.path.join(HERE, "mascot_vs_beast.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
