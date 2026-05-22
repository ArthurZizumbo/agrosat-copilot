# Criterio de validacion cruzada espacial del baseline (EPIC 4)

**US-021 — Curvas de aprendizaje y diagnostico de sub/sobreajuste · Avance 3.**

Este documento justifica por que el baseline tabular (RF/XGB de US-019) y las
curvas de aprendizaje/validacion de US-021 usan **validacion cruzada espacial**
en lugar de una validacion cruzada aleatoria, y describe como
`ml.features.spatial_split.build_spatial_kfold` construye los 5 folds.

## 1. Por que no validacion cruzada aleatoria

Las parcelas de PASTIS-R no son observaciones independientes: parcelas
geograficamente cercanas comparten suelo, clima, calendario de siembra y, en el
caso de PASTIS-R, **el mismo patch satelital de 128 x 128 px**. Un `KFold` o un
`train_test_split` aleatorio reparte parcelas vecinas entre train y test, de
modo que el modelo "ve" en entrenamiento informacion casi identica a la que
luego se le evalua. Esto es **leakage espacial**: el score de validacion sube de
forma artificial y no mide la capacidad real de generalizar a regiones nuevas.

El efecto esta documentado en la literatura de teledeteccion y ecologia:

- **Lyons et al. 2018** — *A comparison of resampling methods for remote sensing
  classification and accuracy assessment* (Remote Sensing of Environment 208,
  145-153, DOI 10.1016/j.rse.2018.02.026). Muestra que el remuestreo aleatorio
  infla la exactitud reportada en clasificacion de cobertura del suelo frente a
  un remuestreo que respeta la estructura espacial.
- **Roberts et al. 2017** — *Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure* (Ecography 40, 913-929).
  Recomienda particiones por bloques espaciales con separacion (buffer) entre
  folds cuando las observaciones presentan autocorrelacion espacial.

Conclusion operativa: el baseline reporta su F1-macro y las curvas de
aprendizaje reportan su accuracy **bajo CV espacial**. Un numero mas bajo pero
honesto es preferible a uno alto e inflado por leakage — y es el numero que el
EPIC 5/6 debe superar para justificar arquitecturas mas complejas.

## 2. Como se construyen los 5 folds espaciales

`build_spatial_kfold` (US-016, `ml/features/spatial_split.py`) produce K folds
disjuntos en seis pasos:

1. **Centroide por parcela.** Se calcula el centroide de cada parcela en
   EPSG:4326 (proyectando a EPSG:3857 para la operacion geometrica).
2. **Tessellation H3.** Cada centroide se asigna a una celda H3 de resolucion 5
   (~252 km^2 por celda). Las celdas H3 son hexagonos de tamano uniforme que
   agregan parcelas en bloques geograficos contiguos.
3. **Clustering KMeans.** Los centroides de las celdas H3 unicas se agrupan con
   `KMeans(n_clusters=k)`. Cada cluster define un fold; las celdas vecinas caen
   en el mismo cluster, asi que cada fold es una **region compacta y no
   contigua con los demas**.
4. **Herencia de fold.** Cada parcela hereda el fold de su celda H3.
5. **Buffer anti-leakage de 1 km.** Las parcelas a menos de `buffer_km = 1.0` km
   de la frontera entre dos folds se **excluyen** del test del fold y se
   devuelven al pool de train. Asi ninguna parcela de test tiene una vecina
   inmediata en el train de ese fold — se corta la autocorrelacion residual en
   la frontera.
6. **Particion train/val.** Dentro del train de cada fold se reserva una
   fraccion como validacion interna con una semilla determinista.

El resultado son 5 `FoldAssignment` con `train_ids`, `val_ids` y `test_ids`
disjuntos por construccion. `ml.train.baseline._build_cv_splits` los traduce a
una **lista materializada** de tuplas `(train_idx, test_idx)` de indices
posicionales, y la cachea en `data/test_fixtures/baseline_spatial_folds_*.parquet`
(la construccion es O(N^2) por el buffer; cachear evita recomputar).

```
        Fold 0        Fold 1        Fold 2        Fold 3        Fold 4
     +---------+   +---------+   +---------+   +---------+   +---------+
     | region  |   | region  |   | region  |   | region  |   | region  |
     | KMeans  |   | KMeans  |   | KMeans  |   | KMeans  |   | KMeans  |
     | cluster |   | cluster |   | cluster |   | cluster |   | cluster |
     +---------+   +---------+   +---------+   +---------+   +---------+
       |  ^  buffer 1 km  ^  buffer 1 km  ^  buffer 1 km  ^  buffer 1 km
       |  +--- parcelas a < 1 km de la frontera: excluidas del test ---+
       v
   En cada iteracion, 1 fold es test y los otros 4 (menos el buffer) son train.
```

## 3. Uso en las curvas de aprendizaje y validacion (US-021)

`plot_learning_curve` y `plot_validation_curve` reciben esa lista materializada
como parametro `cv`. La materializacion es **critica**: `learning_curve` reusa
el `cv` una vez por cada `train_size`; si se pasara un generador, se agotaria
tras el primer tamano y los demas quedarian sin folds (decision D2, riesgo R2 de
`docs/us-planning/us-021.md`). `_materialize_cv_splits` garantiza una `list`
reutilizable y verifica que ningun fold quede sin muestras.

Las curvas usan los mismos 5 folds espaciales que el baseline, de modo que el
diagnostico de sub/sobreajuste (`diagnose_fit`) se hace sobre un score de
validacion sin leakage — coherente con el F1-macro reportado por US-019.
