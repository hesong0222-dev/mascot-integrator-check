"""MASCOT approximation (Mueller, Rasmussen, Stadler 2018, Bioinformatics 34(22):3843).

Lineage-state probabilities p_i(a) evolve independently (pairwise-independence closure):
  dp_i(a)/dt = sum_b (m_ba p_i(b) - m_ab p_i(a))
               + p_i(a) sum_b lam_b p_i(b) sum_{k!=i} p_k(b)
               - p_i(a) lam_a sum_{k!=i} p_k(a)
  dlog P(T)/dt = - sum_a lam_a sum_{i<j} p_i(a) p_j(a)
Coalescence of i,j: P(T) *= sum_a lam_a p_i(a) p_j(a);  p_parent(a) proportional to lam_a p_i(a) p_j(a).
Sampling: new lineage with p = delta(deme).  lam_a = 1/Ne[a];  m_ab = M[a][b] (backward a->b).
"""
import json
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp


def _rhs(t, y, n, d, Mt, out, lam):
    p = y[:-1].reshape(n, d)
    S = p.sum(axis=0)  # sum over all lineages, per deme
    others = S[None, :] - p  # sum_{k != i} p_k(a)
    mig = p @ Mt.T - p * out[None, :]  # sum_b m_ba p_i(b) - m_ab p_i(a);  Mt = M^T
    rate_ia = lam[None, :] * others  # lam_a sum_{k!=i} p_k(a)
    lam_i = (p * rate_ia).sum(axis=1, keepdims=True)
    dp = mig + p * lam_i - p * rate_ia
    # sum_{i<j} p_i p_j = (S^2 - sum_i p_i^2)/2
    dlog = -0.5 * (lam * (S * S - (p * p).sum(axis=0))).sum()
    return np.concatenate([dp.ravel(), [dlog]])


def loglik(tree, rtol=1e-10, atol=1e-12):
    if isinstance(tree, str):
        with open(tree) as f:
            tree = json.load(f)
    M = np.array(tree["M"], dtype=float)
    np.fill_diagonal(M, 0.0)
    d = M.shape[0]
    lam = 1.0 / np.array(tree["Ne"], dtype=float)
    out = M.sum(axis=1)
    Mt = M.T  # (p @ Mt.T)[i,a] = sum_b p_i(b) M[b,a]
    ids, p = [], np.zeros((0, d))
    logL, t = 0.0, None
    for ev in tree["events"]:
        if t is not None and ev["t"] > t and len(ids) > 0:
            n = len(ids)
            y0 = np.concatenate([p.ravel(), [0.0]])
            sol = solve_ivp(_rhs, (t, ev["t"]), y0, method="DOP853", rtol=rtol, atol=atol,
                            args=(n, d, Mt, out, lam))
            if not sol.success:  # do not continue from an intermediate time
                raise RuntimeError(f"MASCOT ODE failed on ({t}, {ev['t']}): {sol.message}")
            y = sol.y[:, -1]
            p = np.maximum(y[:-1].reshape(n, d), 0.0)
            p /= p.sum(axis=1, keepdims=True)
            logL += y[-1]
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
            if s <= 0:
                return -math.inf
            logL += math.log(s)
            keep = [x for x in range(len(ids)) if x not in (i, j)]
            p = np.vstack([p[keep], w / s])
            ids = [ids[x] for x in keep] + [ev["new"]]
    return logL


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"{path}\t{loglik(path):.10f}")
