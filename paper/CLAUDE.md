# Paper Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo en el Paper Track opcional (semanas 10-11 post-presentación).

**Rol**: Redacción del paper en formato IEEE con LaTeX, figuras reproducibles, evaluación formal contra GEO-Bench-2, publicación del benchmark AgroMind-IT/ES en Zenodo con DOI.

## Skills References

- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — GEO-Bench-2, AgroMind, GeoAnalystBench
- [agrosat-llm-finetuning](../.claude/skills/agrosat-llm-finetuning/SKILL.md) — Detalles Gemma 4 / Qwen3-VL para sección Methods
- [agrosat-dvc-mlflow](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — Reproducibilidad

## Critical Rules

- **ALWAYS**: Paper Track es **opcional** y NO compromete entregables del curso (Avance 7 y Presentación final el 21-jun-2026)
- **ALWAYS**: Solo se inicia DESPUÉS de la presentación final, semanas 10-11 (22-jun a 3-jul-2026)
- **ALWAYS**: Figuras reproducibles desde scripts en `paper/figures/scripts/`
- **ALWAYS**: Bibliografía en `paper/bib/refs.bib` con entradas BibTeX completas
- **ALWAYS**: Citar a los dos papers del profesor (TSViT y Phenology) en Related Work
- **ALWAYS**: Publicar AgroMind-IT/ES en Zenodo con DOI antes del submission
- **ALWAYS**: Atribución a Google DeepMind (AlphaEarth, Gemma 4), Alibaba (Qwen), Meta (DINOv3)
- **NEVER**: Sacrificar Avances del curso por trabajo en el paper

## Estructura

```
paper/
├── main.tex                    # IEEE format
├── sections/
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_related_work.tex     # TSViT + Phenology + Google Earth AI + AgriFM
│   ├── 04_methods.tex          # AlphaEarth + Gemma 4 LoRA + ADK + Spatial-RAG
│   ├── 05_experiments.tex      # GEO-Bench-2 + AgroMind-IT/ES
│   ├── 06_results.tex
│   ├── 07_discussion.tex
│   └── 08_conclusions.tex
├── figures/
│   ├── scripts/                # Python scripts reproducibles
│   │   ├── fig1_architecture.py
│   │   ├── fig2_lineage.py
│   │   └── fig3_benchmarks.py
│   └── *.pdf, *.png            # Outputs
├── bib/
│   └── refs.bib
└── README.md                   # Cómo compilar
```

## QA Checklist Paper

- [ ] Avance 7 entregado + Presentación final exitosa ANTES de iniciar paper
- [ ] Bibliografía completa con DOIs
- [ ] Papers del profesor citados en Related Work
- [ ] Figuras reproducibles desde scripts
- [ ] Resultados de GEO-Bench-2 + AgroMind-IT/ES reportados
- [ ] AgroMind-IT/ES publicado en Zenodo con DOI
- [ ] Validación nativa italiana del benchmark (Scuola Sant'Anna vía sponsor)
- [ ] Atribuciones legales completas
- [ ] Sponsor (Dr. Camacho) listado como coautor o en Acknowledgments según acuerdo
