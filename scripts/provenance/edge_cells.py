"""Print 'regime p ratio' for every cell whose best-F1 lambda is the smallest
grid value, i.e. the optimum may lie below the grid."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
lam_min = min(float(r["lambda"]) for r in rows)
best = {}
for r in rows:
    k = (r["regime"], int(r["p"]), int(r["n_samples"]) // int(r["p"]))
    if k not in best or float(r["support_f1"]) > float(best[k]["support_f1"]):
        best[k] = r
for (reg, p, ratio), r in sorted(best.items()):
    if float(r["lambda"]) == lam_min:
        print(reg, p, ratio)
