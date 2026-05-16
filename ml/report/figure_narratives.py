"""Narrativas interpretativas por figura para dashboard y PDF.

Cada figura del EDA recibe un párrafo que explica:
    - Qué muestra la figura (lectura visual).
    - Por qué se hizo ese análisis.
    - Cómo se llegó a la conclusión (método y datos).
    - Qué implica para los siguientes Avances del proyecto.

El texto se escribe en lenguaje claro para que un sponsor académico no
técnico pueda seguir el razonamiento sin saltar a la fórmula.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureNarrative:
    """Narrativa asociada a una figura PNG.

    Attributes:
        filename: Nombre del archivo PNG (sin path), tal como existe en
            ``paper/figures/<dir>/``.
        title: Título legible para mostrar sobre la figura.
        narrative: Párrafo interpretativo de lectura accesible.
        method: Cómo se construyó la figura (datos y procesamiento).
    """

    filename: str
    title: str
    narrative: str
    method: str


# ---------------------------------------------------------------------------
# US-010 — Sentinel-2 Univariado
# ---------------------------------------------------------------------------

SENTINEL2_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="band_distributions.png",
        title="Distribuciones de cada banda Sentinel-2",
        narrative=(
            "Cada panel muestra cómo se reparten los valores de "
            "reflectancia en una de las 10 bandas del satélite, después "
            "de filtrar los píxeles con nubes. Las bandas del visible "
            "(azul, verde, rojo) tienen colas largas hacia valores "
            "altos por nubes residuales, mientras que las del infrarrojo "
            "tienen valores más altos y consistentes — eso es la huella "
            "de la vegetación sana. Ninguna banda se comporta como "
            "una campana de Gauss; todas están sesgadas hacia valores "
            "bajos, lo cual es normal en imágenes satelitales."
        ),
        method=(
            "Muestreo estratificado de píxeles en las 3 regiones italianas "
            "(Pianura Padana, Toscana y Apulia) más PASTIS-R Francia como "
            "control. Histogramas por banda con bins fijos después de "
            "aplicar la máscara SCL del satélite."
        ),
    ),
    FigureNarrative(
        filename="band_stats_heatmap.png",
        title="Resumen de estadísticas por banda y región",
        narrative=(
            "Esta tabla en forma de mapa de calor compara las cuatro "
            "zonas en términos de media, desviación, percentiles y sesgo. "
            "Apulia es la zona más homogénea (menor desviación en el "
            "infrarrojo), mientras que la Pianura Padana es la más "
            "heterogénea por su mosaico parcelario denso. La banda B12 "
            "(infrarrojo de onda corta) concentra los valores más "
            "extremos por su sensibilidad al estrés hídrico de los "
            "cultivos."
        ),
        method=(
            "Agregación por (región, banda) sobre píxeles válidos. "
            "Normalización min-max por columna para que todas las "
            "métricas sean comparables en el mismo color."
        ),
    ),
    FigureNarrative(
        filename="class_cardinality.png",
        title="Cuántos píxeles hay por cada cultivo en PASTIS-R",
        narrative=(
            "Trigo blando, maíz y prados dominan el dataset; legumbres, "
            "viñedos y frutales son minoritarios. Esta asimetría obliga "
            "a hacer muestreo estratificado por clase y región durante "
            "los splits de entrenamiento, y a usar una función de "
            "pérdida con pesos inversos (o focal loss) durante la "
            "fase de segmentación, para que las clases raras no se "
            "‘pierdan’ durante el entrenamiento."
        ),
        method=(
            "Conteo directo de píxeles por clase usando los labels "
            "densos del metadata.geojson de PASTIS-R, anotados por "
            "agricultores."
        ),
    ),
    FigureNarrative(
        filename="outliers_by_band.png",
        title="Outliers detectados por dos métodos complementarios",
        narrative=(
            "Aplicamos dos detectores: IQR (mira cada banda por separado) "
            "e Isolation Forest (mira las 10 bandas juntas). IQR marca "
            "entre 4,6 % y 15,2 % por banda — más en el visible por "
            "nubes. Isolation Forest marca un 5 % uniforme. El valor "
            "promedio de los puntos atípicos es muy alto "
            "(4.700–6.300), lo que confirma que son superficies muy "
            "brillantes (nubes que escaparon a la máscara del satélite). "
            "Recomendación: aplicar winsorización al percentil 99,5 en "
            "lugar de eliminarlos."
        ),
        method=(
            "IQR robusto por banda con los límites Q1−1.5·IQR y "
            "Q3+1.5·IQR. Isolation Forest con 100 estimadores sobre "
            "las 10 bandas en conjunto."
        ),
    ),
    FigureNarrative(
        filename="scl_missingness_by_roi_season.png",
        title="Cuántos píxeles se pierden por nubes (por región y estación)",
        narrative=(
            "Pianura Padana pierde el 53 % de los píxeles en otoño "
            "por nubes; Apulia mantiene el 88 % utilizable en verano. "
            "El verano es la mejor estación para todas las zonas; el "
            "otoño queda como un gap conocido especialmente en la "
            "Pianura. Esta tabla guía la priorización de adquisiciones "
            "para el pipeline Dagster que construiremos en el Avance 2."
        ),
        method=(
            "Conteo de píxeles con la bandera SCL válida (clases 4, 5, "
            "6 = vegetación, no-vegetación, agua) sobre el total "
            "muestreado por (región, estación del año)."
        ),
    ),
    FigureNarrative(
        filename="stretch_2_98_grid.png",
        title="Visualización RGB de las regiones italianas",
        narrative=(
            "Composiciones RGB en color verdadero (B04 rojo, B03 verde, "
            "B02 azul) con un ajuste de contraste entre los percentiles "
            "2 y 98 que descarta los extremos saturados. Permite "
            "verificar visualmente que las regiones cubren mosaicos "
            "agrícolas reales y no superficies degeneradas. Funciona "
            "como sanity check antes de entrar al feature engineering."
        ),
        method=(
            "Reflectancia escalada por banda al rango "
            "[percentil 2, percentil 98] y luego normalizada a [0, 1] "
            "para visualización."
        ),
    ),
)


# ---------------------------------------------------------------------------
# US-011 — AlphaEarth Foundations
# ---------------------------------------------------------------------------

ALPHAEARTH_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="sec1_alphaearth_vs_ndvi.png",
        title="AlphaEarth aporta más información que el NDVI clásico",
        narrative=(
            "Panel comparativo de tres vistas: RGB Sentinel-2 real, "
            "NDVI escalar y un ‘pseudo-RGB’ construido con las 3 "
            "dimensiones AlphaEarth más informativas. El pseudo-RGB "
            "resalta la variación estructural (parcelas, caminos) que "
            "el NDVI aplana en un solo número. Eso justifica usar "
            "AlphaEarth como pool de features principal en el baseline "
            "del Avance 3."
        ),
        method=(
            "Selección de las 3 dimensiones con mayor importancia "
            "Random Forest sobre Italia. Mapeo de cada dimensión a un "
            "canal R/G/B con normalización min-max."
        ),
    ),
    FigureNarrative(
        filename="sec1_corr_italia_dw.png",
        title="Correlación entre las 64 dimensiones AlphaEarth (Italia)",
        narrative=(
            "Mapa de calor que muestra cuánto se parecen entre sí las "
            "64 dimensiones AlphaEarth sobre los 6.000 píxeles "
            "italianos etiquetados con Dynamic World. Hay varios pares "
            "con |r| > 0,7 que indican redundancia. Aplicar PCA antes "
            "de XGBoost reduce el ruido sin perder capacidad "
            "discriminativa."
        ),
        method=(
            "Correlación Pearson sobre los embeddings AlphaEarth v2.1 "
            "estándar; matriz simétrica con diagonal = 1."
        ),
    ),
    FigureNarrative(
        filename="sec1_qq_italia_dw.png",
        title="¿Las dimensiones AlphaEarth son aproximadamente normales? (Italia)",
        narrative=(
            "Los QQ-plots comparan cada dimensión con la normal "
            "estándar. La mayoría se desvía en las colas (colas "
            "pesadas), pero el centro es aproximadamente normal. Eso "
            "justifica usar PCA directamente sobre los embeddings, sin "
            "necesidad de transformación logarítmica previa."
        ),
        method=(
            "QQ-plot de cada dim_XX contra distribución normal "
            "estándar. Submuestra de 1.000 píxeles para mantener "
            "los gráficos legibles."
        ),
    ),
    FigureNarrative(
        filename="sec1_tsne_italia_dw.png",
        title="Proyección t-SNE 2D coloreada por clase Dynamic World (Italia)",
        narrative=(
            "Proyección 2D que comprime las 64 dimensiones en un plano. "
            "Se ven clusters claros: agricultura separada de zonas "
            "construidas, agua y bosque. Un Random Forest simple sobre "
            "estos 64 números logra OOB 0,888 sobre 8 clases — los "
            "embeddings ya capturan la información necesaria sin "
            "feature engineering manual."
        ),
        method=(
            "t-SNE con perplexity = 30 sobre una submuestra de 5.000 "
            "píxeles. Coloreado por la clase Dynamic World "
            "(8 categorías dominantes)."
        ),
    ),
    FigureNarrative(
        filename="sec1_umap_italia_dw.png",
        title="Proyección UMAP 2D coloreada por clase Dynamic World (Italia)",
        narrative=(
            "Variante de t-SNE con UMAP. UMAP preserva mejor la "
            "estructura global y permite ver las relaciones entre "
            "clusters. Sirve como sanity check de robustez: si UMAP y "
            "t-SNE coinciden, la estructura es real (no es artefacto "
            "de un solo método)."
        ),
        method=("UMAP con n_neighbors = 15 y min_dist = 0,1 sobre la misma submuestra que t-SNE."),
    ),
    FigureNarrative(
        filename="sec2_corr_francia_pastis.png",
        title="Correlación entre las 64 dimensiones AlphaEarth (Francia)",
        narrative=(
            "Misma matriz pero sobre los 5.000 píxeles franceses con "
            "etiquetas reales de agricultores. Comparada con Italia, la "
            "estructura de correlaciones cambia: confirma que "
            "AlphaEarth aprendió representaciones diferentes según la "
            "región."
        ),
        method=(
            "Mismo procedimiento que sec1_corr_italia_dw pero sobre los "
            "10 patches PASTIS-R con labels densos."
        ),
    ),
    FigureNarrative(
        filename="sec2_tsne_francia_pastis.png",
        title="Proyección t-SNE 2D coloreada por cultivo PASTIS-R (Francia)",
        narrative=(
            "Los 13 cultivos etiquetados forman clusters separables. "
            "Random Forest alcanza OOB 0,831 — menor que en Italia "
            "(0,888) porque PASTIS tiene 13 clases agronómicas finas "
            "frente a 8 categorías de Dynamic World. La separación es "
            "visible: cereales de invierno en una zona, cultivos de "
            "verano en otra, prados en una tercera."
        ),
        method=("t-SNE con perplexity = 30 sobre 5.000 píxeles franceses estratificados."),
    ),
    FigureNarrative(
        filename="sec2_tsne_francia_phenology.png",
        title="t-SNE coloreado por familia fenológica (Francia)",
        narrative=(
            "Variante del t-SNE anterior que agrupa los cultivos por "
            "familia fenológica: cereales de invierno, cereales de "
            "verano, oleaginosas y prados. La separación mejora "
            "respecto a las 13 clases finas, lo cual sugiere que un "
            "baseline jerárquico (predecir primero la familia y luego "
            "el cultivo) podría ser más efectivo."
        ),
        method=(
            "Se aplica el mapping PASTIS_R_GROUPINGS antes del "
            "coloreado. Misma proyección t-SNE que sec2_tsne_francia_pastis."
        ),
    ),
    FigureNarrative(
        filename="sec2_umap_francia_pastis.png",
        title="Proyección UMAP 2D coloreada por cultivo PASTIS-R (Francia)",
        narrative=(
            "UMAP sobre los mismos 5.000 píxeles franceses. Los "
            "cultivos PASTIS forman clusters más compactos que en "
            "t-SNE; UMAP captura mejor la estructura agronómica de "
            "transición entre familias (cereales → oleaginosas → "
            "prados)."
        ),
        method=("UMAP con n_neighbors = 15 y min_dist = 0,1 sobre 5.000 píxeles franceses."),
    ),
    FigureNarrative(
        filename="sec3_cross_region_consistency.png",
        title="¿Las dimensiones útiles en Italia y Francia coinciden?",
        narrative=(
            "Mapa de calor que muestra cuánta similitud hay entre las "
            "top-10 dimensiones más útiles en cada país. Solo dim_40 "
            "coincide en el top de ambos; el resto del top-10 cambia "
            "totalmente. Conclusión: AlphaEarth se especializa por "
            "región. Entrenar un solo clasificador global rinde menos "
            "que entrenar uno por región."
        ),
        method=(
            "Importancia de features con Random Forest en cada región. "
            "Intersección de top-K = 10. Visualización como matriz de "
            "overlap."
        ),
    ),
)


# ---------------------------------------------------------------------------
# US-012 — Bivariado, Multivariado y Temporal
# ---------------------------------------------------------------------------

BIVARIATE_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="sec3_corr_pearson_bands_bands.png",
        title="Correlación Pearson entre bandas Sentinel-2",
        narrative=(
            "Mapa de calor de correlaciones entre las 10 bandas: "
            "19 pares únicos banda–banda superan |r| > 0,85. El "
            "cuarteto {B07, B8A, B02, B03} forma un bloque casi "
            "perfecto (B07 vs B8A = 0,997). Aplicar PCA al subconjunto "
            "de bandas elimina la redundancia sin perder información "
            "antes del baseline."
        ),
        method=(
            "Pearson r sobre píxeles válidos tras la máscara de "
            "calidad (rango físico [0; 1,5] tras escalar a "
            "reflectancia)."
        ),
    ),
    FigureNarrative(
        filename="sec3_corr_pearson_bands_indices.png",
        title="Correlación Pearson entre bandas e índices espectrales",
        narrative=(
            "Hay un techo en r ≈ 0,80 (B04 vs NDWI). La no-linealidad "
            "de la fórmula (a−b)/(a+b) que usan los índices impide "
            "que crucen correlaciones perfectas con las bandas crudas. "
            "Esto justifica mantener bandas e índices como fuentes "
            "complementarias en el feature engineering."
        ),
        method=(
            "Mismo procedimiento Pearson, cruzando las 10 bandas con los 17 índices candidatos."
        ),
    ),
    FigureNarrative(
        filename="sec3_corr_pearson_indices_indices.png",
        title="Correlación Pearson entre índices espectrales",
        narrative=(
            "Del cuarteto {NDVI, NDRE, NDWI, SAVI} basta uno como "
            "representante: NDVI vs NDRE = 0,974, NDVI vs SAVI = "
            "0,949, NDVI vs NDWI = −0,959. NDMI usa la banda B11 "
            "(SWIR) y aporta una dimensión propia que el resto no "
            "captura. Recomendación: {NDVI, NDMI, EVI} cubre la señal "
            "espectral con redundancias mínimas."
        ),
        method=(
            "Pearson sobre las 6 columnas de índices. Las "
            "recomendaciones se cruzan con el VIF para confirmarlas."
        ),
    ),
    FigureNarrative(
        filename="sec3_corr_spearman_bands_bands.png",
        title="Correlación Spearman entre bandas (robusta a outliers)",
        narrative=(
            "Variante de Pearson que usa rangos en lugar de valores "
            "absolutos, por lo que es robusta a outliers. Los pares "
            "fuertes se mantienen, lo cual confirma que la redundancia "
            "banda–banda no es un artefacto de valores extremos. Útil "
            "como cross-check antes de descartar features."
        ),
        method=("Coeficiente de rangos Spearman sobre los mismos pares banda–banda."),
    ),
    FigureNarrative(
        filename="sec3_corr_spearman_bands_indices.png",
        title="Correlación Spearman banda × índice",
        narrative=(
            "Confirma el techo agronómico de ~0,80 en rangos, no solo "
            "en valores absolutos. Los pares líderes (B04 vs NDWI, "
            "B12 vs NDVI) siguen capturando estructura física: la "
            "clorofila absorbe el rojo y el SWIR responde al estrés "
            "hídrico de los cultivos."
        ),
        method=("Spearman sobre las 10 bandas contra los 17 índices (correlaciones por rangos)."),
    ),
    FigureNarrative(
        filename="sec3_corr_spearman_indices_indices.png",
        title="Correlación Spearman entre índices (robusta a outliers)",
        narrative=(
            "Misma conclusión que la matriz Pearson de índices, pero "
            "robusta a outliers: el cuarteto "
            "{NDVI, NDRE, NDWI, SAVI} es altamente redundante incluso "
            "bajo ranking. EVI y NDMI conservan diversidad."
        ),
        method="Spearman sobre los 6 índices candidatos.",
    ),
    FigureNarrative(
        filename="sec4_vif_barplot.png",
        title="VIF por feature: cuánto se pueden predecir entre sí",
        narrative=(
            "El VIF (Variance Inflation Factor) mide cuánto se puede "
            "predecir una feature a partir del resto. EVI tiene el "
            "VIF más bajo (11,3), seguido de NDMI (15,7), pero ambos "
            "superan el umbral convencional de 10. Ninguna feature es "
            "ortogonal al resto. Recomendación: PCA sobre el bloque "
            "de bandas o subconjunto reducido (NDVI + NDMI + 1–2 "
            "SWIR + 64 AlphaEarth)."
        ),
        method=(
            "VIF = 1 / (1 − R²ⱼ), donde R²ⱼ es el R² de regresar la "
            "feature j contra el resto. Umbrales: 5 (atención), "
            "10 (descartar)."
        ),
    ),
    FigureNarrative(
        filename="sec5_pairplot_top5_by_class.png",
        title="Pairplot de las 5 features más informativas por cultivo",
        narrative=(
            "Diagramas de dispersión entre las 5 features con mayor "
            "importancia Random Forest, coloreados por clase PASTIS. "
            "Permite ver visualmente cuán separable es cada clase en "
            "el espacio bidimensional de las features líderes y "
            "validar que están capturando estructura agronómica, no "
            "ruido aleatorio."
        ),
        method=(
            "Selección por feature importance Random Forest sobre "
            "embeddings e índices. Submuestra estratificada de 2.000 "
            "píxeles."
        ),
    ),
    FigureNarrative(
        filename="sec6_peak_ndvi_by_class.png",
        title="Mes del pico NDVI por clase de cultivo",
        narrative=(
            "Boxplot que muestra en qué mes alcanza cada clase su "
            "máximo NDVI. Soft winter wheat pica en abril–mayo; Corn "
            "y Beet en agosto–septiembre; Meadow en "
            "septiembre–octubre. La feature ‘mes del pico’ es por "
            "sí sola fuertemente discriminativa, así que se incorpora "
            "como feature explícita al baseline."
        ),
        method=(
            "Por parcela, se toma el argmax temporal del NDVI dentro "
            "de los 14 meses de cobertura PASTIS-R. Luego se grafica "
            "la distribución del mes resultante por clase."
        ),
    ),
    FigureNarrative(
        filename="sec7_acf_grid_by_class.png",
        title="¿Hay autocorrelación temporal en las series NDVI?",
        narrative=(
            "Solo el 6,5 % del total de pares (parcela, lag ≥ 1) "
            "supera el umbral de significancia de Bartlett ±0,524 "
            "(para n ≈ 14 observaciones mensuales). El lag más "
            "informativo es lag = 1, con un 22 % de pares "
            "significativos. La autocorrelación temporal es débil con "
            "esta cobertura, así que los modelos seq2seq pueden no "
            "aportar frente a features agregadas."
        ),
        method=("ACF por parcela hasta lag = 12. Umbral de significancia ±1,96/√n."),
    ),
    FigureNarrative(
        filename="sec8_dtw_centroids.png",
        title="Agrupamiento DTW: ¿qué cultivos comparten perfiles temporales?",
        narrative=(
            "El agrupamiento DTW con k = 6 alcanza una pureza de "
            "0,380 sobre 12 clases mayoritarias (vs 0,06 que daría "
            "una asignación aleatoria con 16 clases). El cluster más "
            "nítido reúne cereales de invierno (Soft winter wheat + "
            "Winter barley). El resto queda mezclado porque PASTIS-R "
            "cubre solo 14 meses y muchas clases tienen perfiles "
            "temporales similares en ese horizonte."
        ),
        method=(
            "K-means con métrica DTW (tslearn) sobre series NDVI por "
            "parcela. El k se selecciona por elbow y por "
            "interpretabilidad."
        ),
    ),
    FigureNarrative(
        filename="sec9_era5_ndvi_dual_axis.png",
        title="Lluvia ERA5 vs NDVI (eje dual)",
        narrative=(
            "Cruce temporal entre la precipitación ERA5-Land y el "
            "NDVI medio sobre las regiones italianas. Permite "
            "identificar el lag entre las lluvias y la respuesta de "
            "la vegetación (típicamente 2–4 semanas) y guiar la "
            "construcción de features climatológicas para el baseline."
        ),
        method=(
            "Agregación diaria de precipitación ERA5 y mensual de NDVI "
            "(mediana) por región. Plot con eje dual de doble escala "
            "en Y."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Avance 1 Equipo 17 — PASTIS-R consolidado
# ---------------------------------------------------------------------------

PASTIS_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="cell_013_3_2_distribuciones_y_analisis_de_sesgo.png",
        title="Distribuciones y sesgo por banda (PASTIS-R)",
        narrative=(
            "Las 10 bandas presentan un sesgo significativo "
            "(|skew| > 0,5) y todas requieren transformación no lineal "
            "antes del modelado. Las bandas del visible tienen colas "
            "largas hacia valores altos por nubes residuales. Justifica "
            "el preprocesamiento obligatorio: Z-score con "
            "NORM_S2_patch.json más transformación logarítmica."
        ),
        method=(
            "Histogramas por banda sobre 2.468 patches PASTIS-R × 43 "
            "timesteps. Skewness calculado con scipy.stats.skew."
        ),
    ),
    FigureNarrative(
        filename="cell_015_4_analisis_temporal.png",
        title="Evolución temporal de los índices de vegetación",
        narrative=(
            "Series temporales de NDVI y EVI promediadas por clase de "
            "cultivo. El NDVI tiene un CV de 53,9 % (variabilidad "
            "significativa) y el EVI alcanza 74,9 % (más sensible a "
            "cambios). Los ciclos fenológicos son claros y "
            "consistentes — el dataset es apto para modelos "
            "secuenciales (LSTM, U-TAE, TSViT)."
        ),
        method=(
            "Cálculo de NDVI = (B08 − B04) / (B08 + B04) y EVI por "
            "timestep, promediado por clase. CV = std / mean a lo "
            "largo del eje temporal."
        ),
    ),
    FigureNarrative(
        filename="cell_017_4_2_deteccion_de_gaps_temporales.png",
        title="Detección de gaps temporales",
        narrative=(
            "Solo el 2,3 % de los timesteps tienen problemas moderados "
            "(nubes); el 97,7 % es utilizable. PASTIS-R tiene una "
            "cobertura suficientemente densa (entre 38 y 61 timesteps "
            "por patch, media 48,1) para modelos secuenciales sin "
            "necesidad de interpolación agresiva."
        ),
        method=(
            "Conteo de timesteps por patch con bandera válida vs "
            "bandera de nube. Visualización como heatmap "
            "parche × timestep."
        ),
    ),
    FigureNarrative(
        filename="cell_019_5_analisis_de_segmentacion_y_categorizacion.png",
        title="Análisis de segmentación de parcelas",
        narrative=(
            "Hay en promedio 116 parcelas por patch, con una cobertura "
            "del 80,58 % del área. La fragmentación es alta: los "
            "tamaños varían de 1 a 1.388 píxeles. Esto obliga a usar "
            "arquitecturas multi-escala en segmentación (U-Net con "
            "atención, SegFormer) que capturen las parcelas pequeñas "
            "sin perderlas en el downsampling."
        ),
        method=(
            "Por patch, conteo de objetos segmentados y distribución "
            "de áreas. Visualización como histogramas más ejemplos de "
            "patches concretos."
        ),
    ),
    FigureNarrative(
        filename="cell_021_5_2_categorizacion_por_tamano_de_parcela.png",
        title="Categorización por tamaño de parcela",
        narrative=(
            "El 56 % de las parcelas son pequeñas (<100 píxeles), el "
            "42,2 % son medianas (100–500 px) y solo el 1,7 % son "
            "grandes (>500 px). El predominio de parcelas pequeñas "
            "exige features multi-escala y una loss ponderada por "
            "área inversa, para evitar que las parcelas grandes "
            "dominen el entrenamiento."
        ),
        method=(
            "Discretización del área en 3 bins. Conteo por bin sobre "
            "todas las parcelas del dataset (n ≈ 286.000)."
        ),
    ),
    FigureNarrative(
        filename="cell_025_6_analisis_bivariante_multivariante.png",
        title="Correlación entre bandas espectrales (PASTIS-R)",
        narrative=(
            "Matriz de correlación 10×10 entre las bandas Sentinel-2 "
            "sobre PASTIS-R. Replica el hallazgo del análisis bivariado "
            "con el dataset francés completo: redundancia fuerte en el "
            "bloque {B07, B8A} y entre bandas del visible. Soporta "
            "la decisión de aplicar PCA antes de los modelos densos."
        ),
        method=(
            "Pearson r sobre 2.468 patches × 128×128 píxeles, "
            "agregado a nivel patch para evitar autocorrelación "
            "espacial."
        ),
    ),
    FigureNarrative(
        filename="cell_027_6_2_pca_para_reduccion_dimensional.png",
        title="PCA para reducción dimensional",
        narrative=(
            "Con solo 2 componentes principales se captura el 95 % "
            "de la varianza (reducción del 80 %); con 4 componentes "
            "se alcanza el 99 %. Esta compresión confirma que las 10 "
            "bandas son altamente redundantes. Para el baseline "
            "tabular, PCA a 2 componentes es una opción rápida; para "
            "segmentación densa conviene mantener las 10 bandas con "
            "transformación logarítmica."
        ),
        method=(
            "PCA sobre bandas estandarizadas con NORM_S2_patch.json. "
            "Scree plot más curva de varianza acumulada."
        ),
    ),
    FigureNarrative(
        filename="cell_029_7_deteccion_de_valores_atipicos.png",
        title="Detección de valores atípicos (PASTIS-R)",
        narrative=(
            "Entre el 5 % y el 15 % de outliers por banda, pero son "
            "valores reales, no errores. Recomendación: winsorización "
            "a percentiles 1–99 antes del entrenamiento, para "
            "preservar el rango físico sin que los outliers dominen "
            "el gradiente del modelo."
        ),
        method=(
            "IQR más visualización como boxplots por banda. "
            "Caracterización de los puntos atípicos con su valor "
            "medio para identificar nubes residuales."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Índice global: notebook_id -> narrativas
# ---------------------------------------------------------------------------

NARRATIVES_BY_NOTEBOOK: dict[str, tuple[FigureNarrative, ...]] = {
    "sentinel2": SENTINEL2_NARRATIVES,
    "alphaearth": ALPHAEARTH_NARRATIVES,
    "bivariate-temporal": BIVARIATE_NARRATIVES,
    "pastis-consolidado": PASTIS_NARRATIVES,
    "globales": (),
}


def get_narrative(notebook_id: str, filename: str) -> FigureNarrative | None:
    """Busca la narrativa asociada a una figura.

    Args:
        notebook_id: ID de la ficha (e.g. ``"sentinel2"``).
        filename: Nombre del PNG sin path (e.g. ``"band_distributions.png"``).

    Returns:
        ``FigureNarrative`` si la figura tiene narrativa asignada,
        ``None`` en caso contrario.
    """
    narratives = NARRATIVES_BY_NOTEBOOK.get(notebook_id, ())
    for narrative in narratives:
        if narrative.filename == filename:
            return narrative
    return None
