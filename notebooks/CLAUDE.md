# Notebooks Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo en notebooks de los Avances del curso.

**Rol**: Notebooks Jupyter secuenciales ejecutables con papermill que documentan los Avances del curso (Avances 1-5 son notebooks entregables; Avances 0, 6, 7 son PDFs).

## Skills References

- [agrosat-ml-features](../.claude/skills/agrosat-ml-features/SKILL.md) — Polars, índices espectrales
- [agrosat-ml-baseline](../.claude/skills/agrosat-ml-baseline/SKILL.md) — Baseline XGBoost
- [agrosat-ml-segmentation](../.claude/skills/agrosat-ml-segmentation/SKILL.md) — 6 modelos EPIC 5
- [agrosat-ml-ensemble](../.claude/skills/agrosat-ml-ensemble/SKILL.md) — Ensambles EPIC 6
- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — Plots interpretados

## Estructura Canónica (mapeada a Avances del curso)

| Notebook | Avance | Fecha entrega | Rúbrica |
|----------|--------|---------------|---------|
| `01_avance0_propuesta.ipynb` | Avance 0 (PDF derivado) | 26-abr-2026 | Propuesta |
| `02a_eda_sentinel2.ipynb` | Avance 1 | 3-may-2026 | Univariado |
| `02b_eda_alphaearth.ipynb` | Avance 1 | 3-may-2026 | AlphaEarth caracterización |
| `02c_eda_bivariado_temporal.ipynb` | Avance 1 | 3-may-2026 | Bivariado + temporal |
| `03_feature_engineering.ipynb` | Avance 2 | 17-may-2026 | FE 30+30+30+10 pts |
| `04_baseline_xgboost_alphaearth.ipynb` | Avance 3 | 20-may-2026 | Baseline |
| `05_alt_models.ipynb` | Avance 4 | 24-may-2026 | 6 arquitecturas |
| `06_final_gemma4_ensembles.ipynb` | Avance 5 | 31-may-2026 | Gemma 4 + ensambles |
| `07_agent_eval.ipynb` | Avance 6 (PDF) | 7-jun-2026 | Conclusiones |

## Critical Rules

- **ALWAYS**: Ejecutable end-to-end con `papermill notebook.ipynb output.ipynb -p param value`
- **ALWAYS**: Imports y configs al inicio (`%load_ext autoreload`, `%autoreload 2`)
- **ALWAYS**: Polars para DataFrames (no pandas salvo conversión final a `.to_pandas()` para libs que lo requieran)
- **ALWAYS**: Reutilizar funciones de `ml/` — el notebook llama, no implementa lógica
- **ALWAYS**: Cada notebook tiene sección "Conclusiones" mapeada 1:1 a rúbrica del Avance
- **ALWAYS**: Plots exportados a `paper/figures/` con alta resolución si van al paper
- **ALWAYS**: `make notebooks-strip` antes de commitear (nbstripout en CI, sin pre-commit)
- **NEVER**: Implementar lógica nueva en notebook — refactorizar a `ml/` y llamar
- **NEVER**: Hardcodear paths absolutos — usar `pathlib` y env vars
- **NEVER**: Commitear notebook con outputs — `make notebooks-strip` o CI bloquea
- **NEVER**: Notebook que requiere intervención manual entre celdas

## Plantilla de Encabezado

```python
"""
# Avance N — Título del Avance
**Curso**: Proyecto Integrador MNA — Tec de Monterrey
**Fecha entrega**: YYYY-MM-DD
**EPIC**: EX (US-XXX a US-YYY)
**Rúbrica**: docs/general/Rubricas Integrador.html sección X
**Autores**: Arthur Zizumbo, Aaron Bocanegra, Isaac Ávila
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ml.features.spectral_indices import compute_index
from ml.utils.spatial_cv import spatial_block_split
from ml.utils.mlflow_utils import track_experiment

ROOT = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
DATA = ROOT / "data"
```

## QA Checklist Notebooks

- [ ] Ejecuta end-to-end con papermill sin intervención
- [ ] Polars en lugar de pandas
- [ ] Lógica reutilizable refactorizada a `ml/`
- [ ] Sección "Conclusiones" mapeada a rúbrica
- [ ] Plots con alta resolución exportados a `paper/figures/` si aplican
- [ ] `make notebooks-strip` aplicado
- [ ] Encabezado con metadatos (Avance, EPIC, fecha, rúbrica)
- [ ] Tests de papermill en CI
