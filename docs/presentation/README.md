# Presentación AgroSatCopilot (Reveal.js, ES/EN)

Presentación de ~64 láminas para la defensa final, bilingüe (español/inglés) con
**switch de idioma** en vivo, lista para **GitHub Pages**.

## Ver localmente

```bash
cd docs/presentation
python -m http.server 8765
# abrir http://127.0.0.1:8765/index.html
```

- Switch ES/EN: botón flotante arriba a la derecha (o `?lang=es` / `?lang=en` en la URL).
- Navegación Reveal.js: flechas, `Esc` (vista general), `S` (notas del ponente), `F` (pantalla completa).
- La elección de idioma se recuerda (localStorage).

## Desplegar en GitHub Pages

La presentación es estática (HTML + CSS + JS + PNG, Reveal.js por CDN). Para publicarla:

1. **Settings → Pages** del repo.
2. Source: *Deploy from a branch*; carpeta `/docs` (o mover `docs/presentation/` a la
   raíz de una rama `gh-pages`).
3. La URL será `https://<usuario>.github.io/<repo>/presentation/`.

El archivo `.nojekyll` evita que GitHub Pages procese el sitio con Jekyll (necesario para
que sirva los assets tal cual).

## Estructura

```
docs/presentation/
  index.html        # la presentación completa (64 secciones bilingues)
  css/theme.css     # tema agro-satelital (paleta crema/verde/tierra)
  js/i18n.js        # switch de idioma ES/EN persistente
  assets/figs/      # 17 figuras reales del proyecto + 4 diagramas conceptuales
  .nojekyll
```

## Contenido

1. **Negocio** — problema, impacto económico (ROI >1500%), métrica macro-F1.
2. **Datos y EDA** — CRISP-ML(Q), PASTIS-R, la señal temporal manda, desbalance, AlphaEarth.
3. **Modelado** — baselines → 6 arquitecturas de segmentación → 4 ensambles, benchmark unificado.
4. **Modelo final Voting-3** — campeón, curva de cardinalidad, 4 modos de producción, F1 por clase.
5. **Copiloto y app** — Be My Eyes, 10 FunctionTools, Spatial-RAG, doble backend LLM, frontend, seguridad.
6. **Transfer, MLOps, H100, futuro** — multi-región honesto, DE4 Baja Sajonia, harness, aprendizajes, agradecimientos.

Todas las cifras provienen de artefactos reales del proyecto (cero placeholders). Las figuras
de resultados son matplotlib reales de `reports/`; los diagramas conceptuales (portada,
arquitectura, pipeline, mapa de transfer) se generaron con calidad-paper.
