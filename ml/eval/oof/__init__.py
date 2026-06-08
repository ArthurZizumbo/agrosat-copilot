"""Per-pixel softmax / OOF dump for the segmentation ensembles (US-031).

This package regenerates, for each of the six segmentation ``best.pt`` re-scored
in US-030, the per-pixel POST-softmax probability map and the OOF prediction over
the PASTIS fold-5 held-out split, persisting them to ``ml/eval/oof/*.parquet``
(18-class contiguous convention identical to US-030). The artifacts feed the
Voting / Bagging / Stacking / Blending ensembles (US-040) and E-a / E-b.

Modules:
    dump_oof: the ``dump_oof`` orchestrator + ``python -m ml.eval.oof.dump_oof``.
    parquet_io: float16 + zstd (de)serialization of the dense softmax rows.
"""

from __future__ import annotations
