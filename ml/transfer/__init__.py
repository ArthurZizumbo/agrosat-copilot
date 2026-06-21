"""Cross-region transfer pipelines (EPIC 12).

This package hosts the transnational few-shot transfer experiments that
quantify the domain gap of the AgroSatCopilot crop classifier across
European regions. The flagship module is :mod:`ml.transfer.eurocropsml_fewshot`
(US-076): it reuses the tabular XGBoost *recipe* of the AlphaEarth baseline
(:mod:`ml.train.baseline`) over the pre-coded EuroCropsML k-shot splits to
produce a real F1-macro-vs-k curve for Latvia/Portugal -> Estonia transfer.

Nothing in this package touches the critical-path segmentation, the agent or
the serving layers; it is an isolated CPU-friendly experiment.
"""

from __future__ import annotations
