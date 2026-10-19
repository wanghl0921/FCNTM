# -*- coding: utf-8 -*-
"""Store the baseline fits that the occlusion protocol depends on.

The fourth review had to re-fit NMF to check the zero-input fallback, and its
refit did not reproduce the delivered NLL bit-for-bit (max difference ~0.0007)
because the fit state was never saved.  The BTM fits are written by
``rerun_btm_inference_sampling_fix.py``; this script stores the NMF factors
(plus the prior the adapter falls back to) so a later audit can re-score
without re-fitting.

    python save_baseline_fits.py

Outputs
    outputs/rerun/nmf_fit_<corpus>_seed<k>.npz
    outputs/rerun/baseline_fit_env.json     library versions
"""

from __future__ import annotations

import json

import numpy as np

import occlusion_robustness as O


def main() -> int:
    import sklearn
    import scipy

    env = {"python": __import__("sys").version.split()[0],
           "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit-learn": sklearn.__version__}
    (O.OUT_DIR / "rerun" / "baseline_fit_env.json").write_text(
        json.dumps(env, indent=1, ensure_ascii=False), encoding="utf-8")
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        for seed in (11, 23, 37):
            adapter = O.NMFAdapter(train, 30, seed)
            components = np.asarray(adapter.model.components_, dtype=float)
            path = O.OUT_DIR / "rerun" / f"nmf_fit_{corpus}_seed{seed}.npz"
            np.savez_compressed(path, components=components,
                                beta=np.asarray(adapter.beta, dtype=float),
                                topic_mass=np.asarray(adapter.topic_mass),
                                prior=np.asarray(adapter.prior),
                                seed=int(seed),
                                n_iter=int(adapter.model.n_iter_))
            print(f"  saved {path.name} (n_iter={adapter.model.n_iter_}, "
                  f"prior sums to {float(np.sum(adapter.prior)):.6f})",
                  flush=True)
    print("saved", O.OUT_DIR / "rerun" / "baseline_fit_env.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
