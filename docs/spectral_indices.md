# Catálogo canónico de los 17 índices espectrales (AgroSatCopilot)

Esta es la tabla de referencia académica del catálogo implementado en
[`ml/features/spectral_indices.py`](../ml/features/spectral_indices.py).
Cada índice tiene una **única versión canónica fijada en este documento** y
no debe revisitarse sin abrir un ADR.

## Convenciones

- **Bandas Sentinel-2**: orden canónico `PASTIS_S2_BANDS` =
  (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`).
  Mapeo a nomenclatura spyndex en `_BAND_TO_SPYNDEX` del módulo:
  `B02→B`, `B03→G`, `B04→R`, `B05→RE1`, `B06→RE2`, `B07→RE3`, `B08→N`,
  `B8A→RE4`, `B11→S1`, `B12→S2`.
- **Reflectancia**: el caller es responsable de escalar DN a [0, 1]
  (dividir por 10000) y aplicar máscara SCL antes de invocar `compute_index`.
  Sin máscara, el EDA del Avance 1 mostró saturación NDVI a 1.0 en 75 % de
  parcelas.
- **Backends**: `spyndex` para 14/17 (11 literal + 3 alias) y fórmula custom
  para los 3 restantes (LAI/FAPAR/CCCI) con DOI propio.

## Tabla académica

| # | Nombre | Fórmula | Autor + año | DOI / Referencia | Uso agronómico | Rango esperado | Bandas S2 | Backend |
|---|--------|---------|-------------|------------------|----------------|----------------|-----------|---------|
| 1 | NDVI | (N − R)/(N + R) | Rouse et al. 1974 | NASA SP-351 (1974) | Vigor vegetativo general; satura a partir de LAI ~3 | [−1, 1] | B04, B08 | spyndex `NDVI` |
| 2 | NDWI | (G − N)/(G + N) | McFeeters 1996 | 10.1080/01431169608948714 | Cuerpos de agua y agua foliar | [−1, 1] | B03, B08 | spyndex `NDWI` |
| 3 | NDMI | (N − S1)/(N + S1) | Gao 1996 | 10.1016/S0034-4257(96)00067-3 | Humedad canopy / estrés hídrico | [−1, 1] | B08, B11 | spyndex `NDMI` |
| 4 | EVI | g·(N − R)/(N + C1·R − C2·B + L), g=2.5, C1=6, C2=7.5, L=1 | Huete et al. 2002 | 10.1016/S0034-4257(02)00096-2 | Vigor canopy denso; corrige aerosoles | [−1, 1] | B02, B04, B08 | spyndex `EVI` |
| 5 | SAVI | (1+L)·(N − R)/(N + R + L), L=0.5 | Huete 1988 | 10.1016/0034-4257(88)90106-X | Vigor con suelo expuesto | [−1.5, 1.5] | B04, B08 | spyndex `SAVI` |
| 6 | MSAVI2 | 0.5·(2N+1 − √((2N+1)² − 8(N−R))) | Qi et al. 1994 | 10.1016/0034-4257(94)90134-1 | SAVI auto-calibrado | [−1, 1] | B04, B08 | spyndex `MSAVI` (alias) |
| 7 | NBR | (N − S2)/(N + S2) | Key & Benson 2006 | USGS Tech. Rep. RMRS-GTR-164 | Severidad de fuego | [−1, 1] | B08, B12 | spyndex `NBR` |
| 8 | MCARI | ((RE1 − R) − 0.2·(RE1 − G))·(RE1/R) | Daughtry et al. 2000 | 10.1016/S0034-4257(00)00113-9 | Clorofila en hoja | [−2, 2] | B03, B04, B05 | spyndex `MCARI` |
| 9 | CCCI | NDRE / NDVI | Barnes et al. 2000 | Proc. 5th Int. Conf. Precision Agric. | Clorofila corregida por canopy (N status) | [−2, 2] | B04, B05, B08 | **custom** `_ccci_barnes_2000` |
| 10 | LAI | −ln(1 − (NDVI − 0.05)/0.95) / 0.5 | Boegh et al. 2002 | 10.1016/S0034-4257(01)00342-X | Índice de área foliar | [0, 8] | B04, B08 | **custom** `_lai_boegh_2002` |
| 11 | FAPAR | 1.24·NDVI − 0.168 | Myneni & Williams 1994 | 10.1016/0034-4257(94)90016-7 | Fracción PAR absorbida (fotosíntesis) | [−0.2, 1.1] | B04, B08 | **custom** `_fapar_myneni_1997` |
| 12 | PSRI | (R − B)/RE2 | Merzlyak et al. 1999 | 10.1034/j.1399-3054.1999.106119.x | Senescencia / pigmentos | [−1, 2] | B02, B04, B06 | spyndex `PSRI` |
| 13 | NDCI | (RE1 − R)/(RE1 + R) | Mishra & Mishra 2012 | 10.1016/j.rse.2011.10.016 | Clorofila-a en aguas continentales | [−1, 1] | B04, B05 | spyndex `NDCI` |
| 14 | GCVI | N/G − 1 | Gitelson et al. 2003 | 10.1029/2002GL016450 | Clorofila verde en canopy | [0, 20] | B03, B08 | spyndex `CIG` (alias) |
| 15 | RENDVI | (RE2 − RE1)/(RE2 + RE1) | Gitelson & Merzlyak 1994 | 10.1016/S0176-1617(11)81633-0 | NDVI red-edge; estrés temprano | [−1, 1] | B05, B06 | spyndex `RENDVI` |
| 16 | NDRE | (N − RE1)/(N + RE1) | Barnes et al. 2000 | Proc. 5th Int. Conf. Precision Agric. | Cultivos densos donde NDVI satura | [−1, 1] | B05, B08 | spyndex `NDREI` (alias) |
| 17 | TSAVI | sla·(N − sla·R − slb)/(sla·N + R − sla·slb), sla=1, slb=0 | Baret & Guyot 1991 | 10.1016/0034-4257(91)90009-U | SAVI calibrado con línea de suelo | [−1.5, 1.5] | B04, B08 | spyndex `TSAVI` |

## Alias documentados (3)

| Nombre del proyecto | Alias spyndex 0.10 | Justificación |
|---------------------|--------------------|---------------|
| MSAVI2 | `MSAVI` | La fórmula spyndex de `MSAVI` coincide literal con MSAVI2 de Qi 1994; el nombre histórico `MSAVI1` no se popularizó |
| NDRE | `NDREI` | Mismo cálculo (Normalized Difference Red-Edge Index); spyndex usa el sufijo `I` |
| GCVI | `CIG` | spyndex documenta este índice como **Chlorophyll Index Green** (Gitelson) |

## Fórmulas custom auditadas (3)

Para los 3 índices sin equivalente en spyndex 0.10 fijamos una versión
canónica única con DOI. Estos índices tienen ≥3 variantes en la literatura;
la elección se justifica abajo y no se revisita sin ADR.

### LAI (Boegh et al. 2002)

```
LAI = -ln(1 - (NDVI - 0.05) / 0.95) / 0.5
```

Forma logarítmica derivada de la ley de Beer-Lambert aplicada al NDVI con
constantes calibradas sobre cultivos europeos (trigo, cebada, remolacha).
Alternativas descartadas: Su 2002 (válida solo bosques tropicales), Asner
2003 (requiere LAI in-situ para calibrar α/β regionales).

**Referencia**: Boegh, E., Soegaard, H., Broge, N., Hasager, C.B.,
Jensen, N.O., Schelde, K., Thomsen, A. (2002). *Airborne multispectral
data for quantifying leaf area index, nitrogen concentration, and
photosynthetic efficiency in agriculture*. Remote Sensing of Environment
81(2-3), 179-193. DOI [10.1016/S0034-4257(01)00342-X](https://doi.org/10.1016/S0034-4257(01)00342-X).

### FAPAR (Myneni & Williams 1994)

```
FAPAR = 1.24 · NDVI − 0.168
```

Ajuste lineal sobre bosques templados, frecuentemente citado como
"Myneni 1997". Alternativas descartadas: Sellers 1985 (no lineal,
requiere LUT), MODIS-FAPAR (depende de inversión 3D del radiative
transfer no replicable a nivel de pixel S2 aislado).

**Referencia**: Myneni, R.B., Williams, D.L. (1994). *On the
relationship between FAPAR and NDVI*. Remote Sensing of Environment
49(3), 200-211. DOI [10.1016/0034-4257(94)90016-7](https://doi.org/10.1016/0034-4257(94)90016-7).

### CCCI (Barnes et al. 2000)

```
CCCI = NDRE / NDVI
```

Ratio normalizado del Canopy Chlorophyll Content Index. NDRE captura
clorofila en hoja (sensible vía red-edge) y NDVI normaliza por densidad
de canopy, eliminando confusión "más verde porque más denso" vs "más
verde porque más clorofila".

**Referencia**: Barnes, E.M., Clarke, T.R., Richards, S.E., Colaizzi, P.D.,
Haberland, J., Kostrzewski, M., Waller, P., Choi, C., Riley, E.,
Thompson, T., Lascano, R.J., Li, H., Moran, M.S. (2000). *Coincident
detection of crop water stress, nitrogen status and canopy density using
ground-based multispectral data*. Proceedings of the 5th International
Conference on Precision Agriculture, Bloomington MN.

## Hallazgos del EDA Avance 1 que validan el catálogo

`notebooks/eda/Avance1.Equipo17.ipynb` §3 (bivariado) y §5 (conclusiones):

- **Redundancia detectada** dentro de `{NDVI, NDRE, NDWI, SAVI}` con Pearson
  0.95-0.97. La selección final (US-017) retendrá `{NDVI, NDMI, EVI}` por
  diversidad espectral (NDMI usa SWIR B11, dimensión independiente).
- **NDVI satura a 1.0** en 75 % de parcelas sin máscara SCL. El módulo
  documenta el contrato del caller en su docstring.
- **DN vs reflectancia**: las bandas S2 distribuidas como DN (0-10000)
  deben dividirse por 10000 antes de invocar `compute_index`. El módulo no
  hace esta conversión para no acoplarse a una fuente de datos específica.

## Atribución

Spyndex (MIT) — Montero, D., Aybar, C., Mahecha, M.D. et al. (2023).
*A standardized catalogue of spectral indices to advance the use of
remote sensing in Earth system research*. Scientific Data 10, 197.
DOI [10.1038/s41597-023-02096-0](https://doi.org/10.1038/s41597-023-02096-0).
Repositorio: [awesome-spectral-indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices).
