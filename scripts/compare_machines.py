"""compare_machines.py — cross-platform table from returned run packages."""

import csv
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def key(r):
    return (r["regime"], int(r["p"]), int(r["n_samples"]), round(float(r["lambda"]), 4))


def best_f1(rs):
    best = {}
    for r in rs:
        k = (r["regime"], r["p"], r["n_samples"])
        if fl(r, "support_f1") == fl(r, "support_f1") and (
                k not in best or fl(r, "support_f1") > fl(best[k], "support_f1")):
            best[k] = r
    return list(best.values())


def first_win(rs, col):
    by_p = {}
    for r in rs:
        by_p.setdefault(int(r["p"]), []).append(r)
    out = []
    for p, sub in sorted(by_p.items()):
        # a point counts only if the GPU run converged (older CSVs lack the column)
        valid = lambda r: "not_converged_torch" not in r or fl(r, "not_converged_torch") == 0
        won = [int(r["s_max"]) for r in sorted(sub, key=lambda r: int(r["s_max"]))
               if fl(r, col) > 1 and valid(r)]
        out.append(f"{p}:{won[0] if won else '-'}")
    return " ".join(out)


def machine(name, d, files):
    sysj = json.load(open(files["system"])) if os.path.exists(files["system"]) else {}
    s = sysj.get("system", sysj.get("system", {})) or {}
    torch_info = sysj.get("torch") or {}
    gpu = torch_info.get("gpu") if isinstance(torch_info, dict) else None
    out = {"machine": name, "cpu": s.get("cpu_model"), "blas": s.get("blas_library"),
           "gpu": gpu or (s.get("nvidia_smi") or "").split(",")[0] or "none"}

    one, multi = rows(files["rq1_1t"]), rows(files["rq1_multi"])
    if multi:
        b = best_f1(multi)
        sp = [fl(r, "algorithmic_speedup") for r in b]
        out["bestF1_speedup_median"] = round(statistics.median(sp), 2)
        out["blas_threads"] = multi[0].get("blas_threads")
    if one and multi:
        m = {key(r): r for r in multi}
        ret = {}
        for r in best_f1(one):
            o = m.get(key(r))
            if o:
                ret.setdefault(int(r["p"]), []).append(fl(o, "algorithmic_speedup") / fl(r, "algorithmic_speedup"))
        out["retention_by_p"] = " ".join(f"{p}:{100 * statistics.median(v):.0f}%" for p, v in sorted(ret.items()))
    if files.get("primary_multi") and multi and files["primary_multi"] != files["rq1_multi"]:
        prim = {key(r): r for r in rows(files["primary_multi"])}
        shared = [r for r in multi if key(r) in prim]
        same = sum(r["detected_sizes_json"] == prim[key(r)]["detected_sizes_json"] for r in shared)
        fro = [abs(fl(r, "S_fro") - fl(prim[key(r)], "S_fro")) / fl(prim[key(r)], "S_fro")
               for r in shared if "S_fro" in r]
        out["structure_identical_to_primary"] = f"{same}/{len(shared)}"
        out["S_fro_max_rel_diff"] = f"{max(fro):.1e}" if fro else "n/a"

    x64, x32 = rows(files["xover64"]), rows(files["xover32"])
    col = "gpu_speedup" if (x64 and "gpu_speedup" in x64[0]) else "ratio_torch_over_numpy"
    if x64:
        out["crossover_fp64_smax"] = first_win(x64, col)
    if x32:
        out["crossover_fp32_smax"] = first_win(x32, "gpu_speedup")
    base = rows(files["fp32_base"])
    if base:
        ok = [r for r in base if fl(r, "not_converged_torch32") == 0 and int(r["s_max"]) >= 1000]
        if ok:
            v = [fl(r, "gpu_speedup_vs_cpu32") for r in ok]
            out["fp32_like_for_like_smax1000"] = f"{min(v):.1f}-{max(v):.1f}x"
        out["fp32_gpu_not_converged"] = sum(fl(r, "not_converged_torch32") > 0 for r in base)
    return out


def main():
    primary = {"system": f"{RES}/environment/system.json",
               "rq1_1t": f"{RES}/rq1_primary_1t.csv", "rq1_multi": f"{RES}/rq1_primary_8t.csv",
               "xover64": f"{RES}/gpu_crossover.csv", "xover32": f"{RES}/gpu_crossover_fp32.csv",
               "fp32_base": f"{RES}/fp32_cpu_baseline.csv"}
    table = [machine("primary", RES, primary)]
    for d in sorted(glob.glob(f"{RES}/machines/*/")):
        name = os.path.basename(os.path.normpath(d))
        multi = sorted(glob.glob(f"{d}rq1_*t.csv"))
        multi = [m for m in multi if not m.endswith("rq1_1t.csv")]
        files = {"system": f"{d}system.json", "rq1_1t": f"{d}rq1_1t.csv",
                 "rq1_multi": multi[0] if multi else None,
                 "primary_multi": primary["rq1_multi"],
                 "xover64": f"{d}gpu_crossover.csv", "xover32": f"{d}gpu_crossover_fp32.csv",
                 "fp32_base": f"{d}fp32_cpu_baseline.csv"}
        table.append(machine(name, d, files))

    cols = sorted({k for t in table for k in t}, key=lambda k: list(table[0]).index(k)
                  if k in table[0] else 99)
    with open(f"{RES}/cross_platform.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(f"\n== {t['machine']}")
        for k in cols[1:]:
            print(f"  {k:32s} {t.get(k, '')}")
    print(f"\nwrote {RES}/cross_platform.csv ({len(table)} machines)")


if __name__ == "__main__":
    main()
