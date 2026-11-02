"""Fetch the riboflavin data (Buhlmann et al. 2014) from the archived CRAN hdi
package and write data/riboflavin/riboflavin.npz (X, y, genes, samples).

Needs the rdata package, kept out of the project lock:
    python3 -m venv /tmp/rd && /tmp/rd/bin/pip install rdata
    /tmp/rd/bin/python scripts/fetch_riboflavin.py
"""

import hashlib
import io
import os
import tarfile
import urllib.request

import numpy as np
import rdata

URL = "https://cran.r-project.org/src/contrib/Archive/hdi/hdi_0.1-9.tar.gz"
SHA256 = "42f400cca50569320da4a4e7fdde1fa55c3c8f247a4219cf2d9d51543efe654c"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "riboflavin", "riboflavin.npz")


def main():
    raw = urllib.request.urlopen(URL, timeout=120).read()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "archive differs from the recorded hash"
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        blob = t.extractfile("hdi/data/riboflavin.RData").read()

    # data.frame holding a matrix column: keep the raw list, not a pandas frame
    keep = lambda obj, attrs: obj
    conv = rdata.conversion.SimpleConverter(
        constructor_dict={**rdata.conversion.DEFAULT_CLASS_MAP, "data.frame": keep, "AsIs": keep})
    d = conv.convert(rdata.parser.parse_data(blob))["riboflavin"]
    y, x = np.asarray(d["y"], float), d["x"]
    samples, genes = (x.coords[dim].values.astype(str) for dim in x.dims)
    X = np.asarray(x.values, float)
    assert X.shape == (71, 4088) and y.shape == (71,) and np.isfinite(X).all()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, genes=genes, samples=samples)
    print(f"X {X.shape}, range {X.min():.3f}..{X.max():.3f}, "
          f"constant genes {(X.std(0) == 0).sum()}, "
          f"duplicate genes {len(genes) - len(set(genes))}, "
          f"duplicate samples {len(X) - len({r.tobytes() for r in X})}")
    print("samples:", " ".join(samples[:12]), "...")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
