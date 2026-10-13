"""Summarise bench/mascot_vs_beast.csv (or a given CSV) and compare it with results/mascot_vs_beast_reference.csv.

Usage: python check_results.py [csv]
"""
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summary(rows):
    def col(a, b):
        return [float(r[a]) - float(r[b]) for r in rows]

    lines = [
        ("converged MASCOT ODE (DOP853) - BEAST logged", col("ours_dop853", "beast_logged")),
        ("emulated integrator (eps 1e-3, carry) - BEAST logged", col("emul_eps1e-3", "beast_logged")),
        ("carry effect: with - without carry (eps 1e-3)", col("emul_eps1e-3", "emul_eps1e-3_nocarry")),
        ("step size only: no carry eps 1e-3 - converged", col("emul_eps1e-3_nocarry", "ours_dop853")),
        ("with carry, eps 1e-9 - converged", col("emul_eps1e-9", "ours_dop853")),
        ("no carry, eps 1e-9 - converged", col("emul_eps1e-9_nocarry", "ours_dop853")),
    ]
    print(f"{len(rows)} logged states")
    print(f"{'comparison (log units)':55s} {'max abs':>10s} {'mean abs':>10s}")
    for name, d in lines:
        a = [abs(x) for x in d]
        print(f"{name:55s} {max(a):10.2e} {statistics.mean(a):10.2e}")
    hi = sum(float(r["beast_logged"]) > float(r["ours_dop853"]) for r in rows)
    print(f"BEAST logged value above the converged value in {hi}/{len(rows)} states")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bench", "mascot_vs_beast.csv")
    ref = load(os.path.join(HERE, "results", "mascot_vs_beast_reference.csv"))
    rows = load(path)
    summary(rows)
    refd = {r["sample"]: r for r in ref}
    worst = 0.0
    for r in rows:
        for k, v in r.items():
            if k in ("sample", "n_steps_eps1e-3"):
                continue
            worst = max(worst, abs(float(v) - float(refd[r["sample"]][k])))
    print(f"max abs difference to the reference CSV over {len(rows)} rows: {worst:.1e}")


if __name__ == "__main__":
    main()
