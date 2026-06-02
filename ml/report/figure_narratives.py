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
# BreizhCrops — Validación cross-region (Rußwurm & Körner, sucesor moderno)
# ---------------------------------------------------------------------------

BREIZHCROPS_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="breizhcrops_class_distribution.png",
        title="Distribución de cultivos en Bretaña (frh04)",
        narrative=(
            "Cada barra cuenta cuántas parcelas hay de cada cultivo en la "
            "región analizada. El reparto es muy desigual: pastizales "
            "temporales, maíz y pastizales permanentes acaparan casi el "
            "80 % de las 122.708 parcelas, mientras que girasol y frutos "
            "secos apenas aparecen (2 y 11 parcelas). Es el mismo patrón "
            "de desbalance que vimos en PASTIS-R, lo que confirma que no "
            "es una rareza de un solo dataset sino una característica real "
            "del paisaje agrícola europeo: hay que estratificar por clase "
            "o ponderar antes de entrenar."
        ),
        method=(
            "Conteo de parcelas por clase sobre el índice tabular completo "
            "de la región frh04 (año 2017, nivel L2A), usando el mapeo "
            "oficial de 9 clases agronómicas distribuido con el dataset."
        ),
    ),
    FigureNarrative(
        filename="breizhcrops_ndvi_by_class.png",
        title="Firma fenológica del NDVI por cultivo",
        narrative=(
            "Cada curva muestra cómo evoluciona el vigor vegetal (NDVI) a "
            "lo largo del año para un cultivo distinto. Las curvas no se "
            "superponen: el maíz alcanza su pico de verdor más tarde y más "
            "bajo (0,81) que el trigo o los huertos (más de 0,94), y cada "
            "cultivo dibuja una campana en un momento distinto. Esa "
            "diferencia en cuándo y cuánto crece la planta es exactamente "
            "la señal que un modelo temporal puede explotar, y justifica "
            "invertir en features de fenología en vez de promedios anuales."
        ),
        method=(
            "NDVI = (B08 - B04) / (B08 + B04) calculado por paso temporal "
            "sobre una muestra estratificada de parcelas; agregación de la "
            "mediana por cultivo y día del año para trazar la curva media."
        ),
    ),
    FigureNarrative(
        filename="breizhcrops_vs_pastis_ndvi.png",
        title="Comparación de vigor vegetal: BreizhCrops vs PASTIS-R",
        narrative=(
            "Esta figura enfrenta las distribuciones de NDVI de los dos "
            "datasets franceses. Se solapan ampliamente: la mediana es "
            "0,613 en BreizhCrops y 0,431 en PASTIS-R, con los extremos "
            "(percentiles 5 y 95) casi idénticos. El desplazamiento es "
            "esperable —Bretaña es más húmeda y verde que las zonas de "
            "PASTIS— pero no hay un cambio de dominio radical. La lectura "
            "para el proyecto es positiva: combinar ambos datasets no "
            "expone al modelo a una distribución de entrada irreconocible, "
            "lo que reduce el riesgo de domain shift."
        ),
        method=(
            "Muestreo equivalente de píxeles/observaciones NDVI de ambos "
            "datasets y comparación de percentiles (5, mediana, 95) más "
            "histogramas superpuestos. PASTIS-R se carga con el loader "
            "existente del proyecto para garantizar el mismo cálculo."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Métodos derivados de la literatura — cuatro artículos de referencia
# ---------------------------------------------------------------------------

PAPER_METHODS_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="boundary_interior_histograms.png",
        title="Distribución espectral en interior, frontera y exterior de parcela",
        narrative=(
            "Tres histogramas comparan los valores de la banda infrarroja "
            "cercana entre los píxeles interiores de una parcela, los de su "
            "frontera y los del fondo. El análisis replica la Figura 2 de "
            "Tarasiou et al. (2021): los bordes de parcela son la zona "
            "espectralmente más sensible, donde se mezclan firmas de "
            "cultivos vecinos. En el patch analizado la desviación estándar "
            "de la frontera es 0,91 veces la del interior — el artículo "
            "describe la mayor variabilidad de borde como una tendencia "
            "sobre el conjunto completo, no como una regla que se cumpla "
            "patch a patch. Aun así, los límites de parcela quedan "
            "señalados como el punto débil de la segmentación de cultivos."
        ),
        method=(
            "Clasificación de cada píxel del patch PASTIS-R en interior, "
            "frontera o exterior según una vecindad de 3×3 sobre el mapa "
            "semántico. Histogramas de la banda B08 (infrarrojo cercano) "
            "promediada en el tiempo, con la media marcada por grupo."
        ),
    ),
    FigureNarrative(
        filename="temporal_gap_distribution.png",
        title="Revisita irregular de Sentinel-2 entre adquisiciones",
        narrative=(
            "Cada barra mide los días que transcurren entre dos imágenes "
            "válidas consecutivas de una parcela. Los intervalos no son "
            "constantes: el patch tiene 43 imágenes al año con un hueco "
            "medio de 9,3 días, pero el mayor llega a 55 días y el menor a "
            "5. Esa irregularidad — consecuencia del preprocesamiento "
            "mínimo que defienden Rußwurm y Körner (2018), que trata la "
            "nube como ruido temporal — justifica interpolar la serie a "
            "una rejilla diaria antes de calcular las características "
            "temporales por transformada de Fourier."
        ),
        method=(
            "Conversión de las fechas de adquisición Sentinel-2 a días "
            "calendario y cálculo de las diferencias entre observaciones "
            "consecutivas. La línea roja marca el hueco medio."
        ),
    ),
    FigureNarrative(
        filename="confusion_symmetry_scatter.png",
        title="Confusiones simétricas vs asimétricas entre cultivos",
        narrative=(
            "Cada punto es un par de cultivos que un clasificador confunde. "
            "El eje horizontal mide la componente simétrica (ambos cultivos "
            "se confunden mutuamente en proporciones parecidas, señal de "
            "similitud espectral o fenológica real) y el vertical la "
            "componente asimétrica (la confusión va sobre todo en un "
            "sentido, señal de desbalance de clases o errores de "
            "anotación). De los 31 pares confundidos, solo 3 quedan "
            "dominados por la componente simétrica y 28 por la asimétrica. "
            "Distinguir ambos tipos, como proponen Rußwurm y Körner (2018), "
            "orienta dónde invertir: características más finas para las "
            "confusiones reales y rebalanceo o limpieza de etiquetas para "
            "las espurias."
        ),
        method=(
            "Matriz de confusión de un Random Forest entrenado en cuatro "
            "folds espaciales y evaluado en uno distinto. Para cada par de "
            "clases se calcula la componente simétrica como el mínimo de "
            "los dos sentidos de confusión y la asimétrica como su "
            "diferencia absoluta."
        ),
    ),
    FigureNarrative(
        filename="phenology_calendar_distribution.png",
        title="Distribución de parcelas por etapa fenológica",
        narrative=(
            "El gráfico cuenta cuántas parcelas alcanzan su pico de "
            "vegetación en cada una de las 4 etapas de crecimiento "
            "(dormancia, crecimiento, pico y senescencia). El concepto "
            "viene del Phenology-Aware Transformer (2025), que codifica las "
            "etapas de crecimiento como un calendario indexado por día del "
            "año. Aquí se usa como característica exploratoria: la etapa "
            "dominante separa familias de cultivos con claridad — praderas "
            "y trigo de invierno concentran su pico en la etapa de máximo "
            "verdor, el maíz y los girasoles en senescencia, y la vid y los "
            "frutales en dormancia."
        ),
        method=(
            "Discretización del día del año del pico de vegetación "
            "(`peak_doy`) de las 2.433 parcelas con dato en 4 etapas "
            "fenológicas. Conteo de parcelas por etapa."
        ),
    ),
    FigureNarrative(
        filename="cloud_gap_drift.png",
        title="Robustez de las características temporales ante huecos de nubes",
        narrative=(
            "Cada curva muestra cuánto se desplaza una característica "
            "temporal cuando se ocultan fracciones crecientes de imágenes "
            "de la parcela, simulando que las nubes las tapan. Una "
            "pendiente suave indica un proceso robusto; una abrupta delata "
            "características frágiles. La deriva media al ocultar el 60 % de "
            "las observaciones es de 0,73, un valor moderado que confirma "
            "que el proceso de ingeniería de características tolera la "
            "pérdida de datos por cobertura nubosa. La técnica adapta el "
            "enmascaramiento espaciotemporal de Qin et al. (2025) como "
            "herramienta de diagnóstico."
        ),
        method=(
            "Eliminación aleatoria del 20 %, 40 % y 60 % de las "
            "observaciones de una serie temporal de parcela, recálculo de "
            "las características temporales y medición de la deriva "
            "absoluta respecto al valor original."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Avance 2 — Ingeniería de Características sobre Sentinel-2 (03a)
# ---------------------------------------------------------------------------

FE_SENTINEL2_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="cell_011_3_generacion_de_nuevas_caracteristicas_indices_espectrales.png",
        title="Distribuciones de los 8 índices espectrales generados",
        narrative=(
            "Cada panel muestra cómo se reparten los valores de uno de los "
            "ocho índices espectrales construidos a partir de las 10 bandas "
            "de Sentinel-2: NDVI y NDRE (vigor de la vegetación), NDWI "
            "(contenido de agua), EVI (vegetación densa con corrección "
            "atmosférica), SAVI (ajustado por suelo desnudo), MSI (estrés "
            "hídrico), BSI (suelo desnudo) y NBR (zonas quemadas). La línea "
            "roja punteada marca la media de cada índice. El NDVI se "
            "distribuye alrededor de 0,42 y NDWI alrededor de −0,45, valores "
            "coherentes con paisajes agrícolas activos. El EVI presenta una "
            "cola muy larga porque su fórmula con denominador combinado "
            "amplifica los píxeles ruidosos; por eso conviene transformarlo "
            "antes de modelar. Estos índices condensan combinaciones de "
            "bandas en métricas con significado agronómico, de modo que el "
            "modelo recibe conocimiento del dominio en lugar de aprenderlo "
            "desde cero."
        ),
        method=(
            "Histogramas de 80 bins por índice. Cada índice se calcula sobre "
            "las bandas Sentinel-2 ya filtradas con la máscara de calidad "
            "(clip al percentil 99,5). Los valores no finitos producto de "
            "divisiones por cero se imputan con la mediana de la columna "
            "antes de graficar."
        ),
    ),
    FigureNarrative(
        filename="cell_014_4_discretizacion_binning.png",
        title="Discretización del NDVI por dominio y del EVI por k-means",
        narrative=(
            "Los dos paneles comparan dos estrategias de binning. A la "
            "izquierda, el NDVI se discretiza con umbrales físicos conocidos "
            "en cuatro categorías (agua o sombra, suelo desnudo, vegetación "
            "escasa y vegetación densa): las bins quedan alineadas con clases "
            "del mundo real, no con cortes arbitrarios. A la derecha, el EVI "
            "se discretiza por agrupamiento k-means en cinco bins, dejando "
            "que el propio algoritmo encuentre los cortes naturales de la "
            "distribución. La discretización permite a los modelos capturar "
            "relaciones no lineales y, de paso, habilita la prueba "
            "chi-cuadrado, que exige variables discretas. En todos los casos "
            "se conserva la columna continua original: el binning agrega "
            "información, no la sustituye."
        ),
        method=(
            "Binning por umbrales de dominio del NDVI con cortes en −0,1, "
            "0,2 y 0,5. Binning k-means del EVI con cinco bins mediante "
            "KBinsDiscretizer. Diagramas de barras con el conteo de píxeles "
            "por bin."
        ),
    ),
    FigureNarrative(
        filename="cell_017_5_transformaciones_para_correccion_de_sesgo.png",
        title="Comparación de transformaciones de corrección de sesgo (banda B08)",
        narrative=(
            "La banda B08 (infrarrojo cercano) es la más informativa para "
            "vegetación y también la más sesgada, así que sirve de ejemplo "
            "para comparar cuatro tratamientos. El primer panel muestra la "
            "distribución original, con una cola larga hacia valores altos. "
            "Los tres siguientes aplican log1p, raíz cuadrada y "
            "Yeo-Johnson; cada título reporta el sesgo resultante. "
            "Yeo-Johnson y log1p son los que dejan la distribución más "
            "simétrica, cercana a una campana. Corregir el sesgo es "
            "importante porque las distribuciones asimétricas penalizan a "
            "los modelos lineales y a los basados en distancias."
        ),
        method=(
            "Histogramas de 80 bins de la banda B08. Se aplican log1p, raíz "
            "cuadrada y PowerTransformer Yeo-Johnson sobre los mismos "
            "valores; el coeficiente de sesgo se calcula con scipy.stats."
        ),
    ),
    FigureNarrative(
        filename="cell_018_5_transformaciones_para_correccion_de_sesgo_1.png",
        title="Distribuciones de las 10 bandas tras la transformación automática",
        narrative=(
            "Esta rejilla muestra las 10 bandas Sentinel-2 después de "
            "aplicar la transformación elegida automáticamente para cada "
            "una. La regla es simple: si el mínimo de la banda es positivo "
            "se usa log1p, y si puede haber valores negativos se usa "
            "Yeo-Johnson. Las bandas del visible (B02, B03, B04) recibieron "
            "Yeo-Johnson por contener negativos producto de artefactos de "
            "corrección atmosférica; el resto recibió log1p. El sesgo, que "
            "antes superaba 2,0 en todas las bandas, baja a valores cercanos "
            "a cero tras la transformación, dejando las distribuciones "
            "aptas para el escalado posterior."
        ),
        method=(
            "Histogramas de 70 bins por banda. Selección automática del "
            "método por el signo del mínimo de cada banda; el sesgo previo y "
            "posterior se registra para verificar la corrección."
        ),
    ),
    FigureNarrative(
        filename="cell_021_6_escalamiento.png",
        title="Comparación de escaladores Min-Max, Z-score y Robust (banda B08)",
        narrative=(
            "Los cuatro paneles muestran la banda B08 ya transformada y "
            "luego escalada con tres técnicas distintas. El escalado Min-Max "
            "comprime el rango a [0, 1] (útil para redes neuronales); el "
            "Z-score centra en cero con desviación uno (requisito para PCA y "
            "modelos lineales); el escalador robusto usa mediana e IQR, por "
            "lo que es menos sensible a valores extremos. Las bandas "
            "Sentinel-2 tienen magnitudes muy distintas entre el visible y "
            "el infrarrojo, así que sin escalado los modelos basados en "
            "distancias o gradiente darían más peso implícito a las bandas "
            "de mayor magnitud. El escalado se aplica sobre las features ya "
            "transformadas para que opere sobre distribuciones más "
            "simétricas."
        ),
        method=(
            "Histogramas de 70 bins de la banda B08. Se aplican "
            "MinMaxScaler, StandardScaler y RobustScaler de scikit-learn "
            "sobre las features transformadas; el Z-score queda como "
            "escalador por defecto del conjunto."
        ),
    ),
    FigureNarrative(
        filename="cell_027_8_seleccion_de_caracteristicas_fase_2.png",
        title="Varianza por feature continua y umbral de descarte",
        narrative=(
            "El gráfico de barras ordena las 18 features continuas "
            "(10 bandas transformadas más 8 índices) por su varianza. La "
            "línea negra punteada marca el umbral de 0,01: las features por "
            "debajo se descartarían por aportar poca capacidad "
            "discriminativa. En este caso todas las barras quedan en verde "
            "porque ninguna feature continua cae por debajo del umbral: la "
            "fase de generación produjo features informativas, sin columnas "
            "casi constantes. El filtro de varianza es el primer paso de la "
            "selección, antes de revisar redundancia entre features."
        ),
        method=(
            "VarianceThreshold de scikit-learn con umbral 0,01 sobre las "
            "features continuas. Diagrama de barras con la varianza de cada "
            "feature; el color distingue conservadas (verde) de eliminadas "
            "(rojo)."
        ),
    ),
    FigureNarrative(
        filename="cell_029_8_2_correlacion_de_pearson_y_spearman.png",
        title="Matrices de correlación de Pearson y Spearman entre features",
        narrative=(
            "Los dos mapas de calor muestran cuánto se parecen entre sí las "
            "features continuas conservadas: a la izquierda con correlación "
            "de Pearson (asume relación lineal) y a la derecha con Spearman "
            "(basada en rangos, robusta a valores extremos). Los bloques "
            "rojos intensos señalan pares muy redundantes. A partir de esta "
            "lectura se eliminan cuatro features con correlación absoluta "
            "mayor a 0,90: las bandas B07, B08 y B12 transformadas y el "
            "índice SAVI, que es prácticamente idéntico al NDVI "
            "(correlación de 1,00). Reducir la redundancia disminuye la "
            "colinealidad sin sacrificar información."
        ),
        method=(
            "Correlación de Pearson y Spearman sobre las features continuas "
            "conservadas tras el filtro de varianza. Mapas de calor "
            "triangulares con escala divergente centrada en cero."
        ),
    ),
    FigureNarrative(
        filename="cell_032_8_3_chi_cuadrado.png",
        title="Ranking chi-cuadrado de las features discretizadas",
        narrative=(
            "El gráfico de barras horizontales ordena las features "
            "discretizadas por su estadístico chi-cuadrado, que mide la "
            "dependencia entre cada bin y la clase de cultivo. Las bins de "
            "las bandas del infrarrojo de onda corta (B12, B11) y del "
            "infrarrojo cercano (B08, B8A) encabezan el ranking, todas "
            "estadísticamente significativas. En el extremo opuesto, las "
            "bins k-means del EVI quedan en último lugar y no resultan "
            "significativas. La lectura es clara: las bandas crudas "
            "discretizadas por cuantiles aportan más señal para distinguir "
            "cultivos que el EVI discretizado por agrupamiento."
        ),
        method=(
            "Prueba chi-cuadrado de scikit-learn entre las 12 features "
            "discretizadas y la clase de cultivo. Diagrama de barras "
            "horizontales con el estadístico; el color marca las "
            "significativas (p < 0,05)."
        ),
    ),
    FigureNarrative(
        filename="cell_034_8_4_anova_f_score.png",
        title="Ranking ANOVA F-score de las features continuas",
        narrative=(
            "El gráfico ordena las features continuas por su F-score de "
            "ANOVA, que compara la varianza entre clases de cultivo con la "
            "varianza dentro de cada clase: un valor alto indica que la "
            "media de la feature difiere mucho de un cultivo a otro. Los "
            "índices NBR, BSI, MSI, NDVI y NDRE encabezan el ranking, todos "
            "significativos, lo que confirma que los índices espectrales "
            "construidos con conocimiento del dominio separan cultivos mejor "
            "que la mayoría de las bandas crudas transformadas. Este "
            "ranking guía la versión reducida del conjunto de features."
        ),
        method=(
            "Prueba ANOVA F (f_classif de scikit-learn) entre las features "
            "continuas tras el filtro de correlación y la clase de cultivo. "
            "Diagrama de barras horizontales con el F-score; el color marca "
            "las significativas (p < 0,01)."
        ),
    ),
    FigureNarrative(
        filename="cell_036_8_5_analisis_de_componentes_principales_pca.png",
        title="Varianza explicada por componente del PCA",
        narrative=(
            "Los dos paneles resumen el análisis de componentes principales "
            "sobre las 10 bandas escaladas. A la izquierda, la varianza que "
            "aporta cada componente: la primera concentra el 62,2 % y la "
            "segunda el 17,1 %. A la derecha, la varianza acumulada: con "
            "solo 6 componentes se alcanza el 96,8 %, y las dos primeras ya "
            "explican el 79,3 %. Esta fuerte compresión confirma que las 10 "
            "bandas son altamente redundantes, algo coherente con las "
            "matrices de correlación. El PCA elimina esa redundancia "
            "residual conservando casi toda la información."
        ),
        method=(
            "PCA de scikit-learn sobre las 10 bandas escaladas con Z-score. "
            "Diagrama de barras de varianza por componente más curva de "
            "varianza acumulada, con el corte del 95 % señalado."
        ),
    ),
    FigureNarrative(
        filename="cell_037_8_5_analisis_de_componentes_principales_pca_1.png",
        title="Biplot del PCA: primeras dos componentes por clase de cultivo",
        narrative=(
            "El biplot proyecta cada píxel sobre las dos primeras "
            "componentes principales, coloreado por clase de cultivo. Las "
            "flechas rojas son los vectores de carga: indican en qué "
            "dirección crece cada banda original dentro del nuevo espacio. "
            "Bandas que apuntan en la misma dirección están correlacionadas; "
            "la longitud de la flecha refleja su peso en esas dos "
            "componentes. El gráfico permite verificar visualmente que la "
            "proyección PCA conserva la estructura de los datos: la nube de "
            "píxeles muestra gradientes de color, señal de que las "
            "componentes capturan diferencias agronómicas reales."
        ),
        method=(
            "PCA con el número de componentes que alcanza el 95 % de "
            "varianza. Diagrama de dispersión de las componentes 1 y 2 "
            "coloreado por clase, con los vectores de carga de cada banda "
            "superpuestos."
        ),
    ),
    FigureNarrative(
        filename="cell_039_8_6_analisis_factorial_fa.png",
        title="Matriz de cargas del análisis factorial",
        narrative=(
            "El mapa de calor muestra cómo se relacionan las 10 bandas con "
            "cinco factores latentes. A diferencia del PCA, que maximiza la "
            "varianza total, el análisis factorial busca causas subyacentes "
            "compartidas que expliquen la correlación observada entre "
            "bandas. El primer factor carga fuerte sobre casi todas las "
            "bandas (es un factor general de brillo), mientras que el "
            "segundo separa el infrarrojo (B06, B07, B08, B8A) del resto. "
            "Esta lectura es interpretable en términos físicos: los factores "
            "agrupan bandas que responden a un mismo fenómeno, como el vigor "
            "vegetativo o el contenido de agua."
        ),
        method=(
            "FactorAnalysis de scikit-learn con 5 factores sobre las 10 "
            "bandas escaladas. Mapa de calor de la matriz de cargas, con "
            "valores anotados y escala divergente centrada en cero."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Avance 2 — Selección y extracción sobre PASTIS-R espectro-temporal (03b)
# ---------------------------------------------------------------------------

FE_SPECTRAL_TEMPORAL_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="cell_008_3_balance_de_clases_y_muestras_por_categoria.png",
        title="Balance de clases en el subconjunto estratificado de PASTIS-R",
        narrative=(
            "Cada barra cuenta cuántas parcelas hay de cada clase de cultivo "
            "en el subconjunto analizado de 11.999 parcelas. El reparto es "
            "muy desigual: la clase mayoritaria reúne 4.251 parcelas y la "
            "más pequeña apenas 68, con una mediana de 262. La línea "
            "punteada gris marca el umbral mínimo de 5 muestras por debajo "
            "del cual los rankings supervisados se vuelven poco fiables; "
            "ninguna clase cae por debajo de él. Dejar este desbalance "
            "visible antes de operar es importante porque condiciona la "
            "interpretación del F1-macro y de las importancias de Random "
            "Forest y XGBoost en las secciones siguientes."
        ),
        method=(
            "Conteo de parcelas por clase sobre el subconjunto estratificado "
            "por bloque espacial. Diagrama de barras con el número de "
            "muestras; el color distingue las clases por encima y por debajo "
            "del umbral de 5."
        ),
    ),
    FigureNarrative(
        filename="cell_012_4_filtros_estadisticos_umbral_de_varianza_y_correlacion_de_p.png",
        title="Matriz de correlación de Pearson entre las primeras features",
        narrative=(
            "El mapa de calor muestra la correlación de Pearson entre las "
            "primeras 30 features espectro-temporales, en el orden original "
            "del conjunto. Los bloques rojos intensos revelan grupos de "
            "features muy redundantes, típicamente estadísticos del mismo "
            "índice (por ejemplo, la media y la mediana del NDVI casi "
            "coinciden cuando la distribución es simétrica). Esta redundancia "
            "es el motivo por el que el filtro de correlación elimina 52 "
            "columnas: conservarlas no mejora el modelo y aumenta la "
            "colinealidad."
        ),
        method=(
            "Correlación de Pearson sobre el conjunto tras el filtro de "
            "varianza. Mapa de calor de las primeras 30 features en orden "
            "original, con escala divergente centrada en cero."
        ),
    ),
    FigureNarrative(
        filename="cell_013_4_filtros_estadisticos_umbral_de_varianza_y_correlacion_de_p_1.png",
        title="Matriz de correlación reordenada por agrupamiento jerárquico",
        narrative=(
            "Es la misma información de correlación que el mapa anterior, "
            "pero con las features reordenadas mediante agrupamiento "
            "jerárquico. El reordenamiento hace que los grupos de features "
            "fuertemente correlacionadas aparezcan como bloques rojos "
            "cuadrados sobre la diagonal: cada bloque es un cluster de "
            "redundancia. Justamente esos bloques son los que el filtro de "
            "correlación colapsa, conservando una sola feature representante "
            "por grupo. Visualizar la estructura de redundancia de esta "
            "forma confirma que el recorte de columnas no es arbitrario."
        ),
        method=(
            "Distancia 1 − |r| entre features y enlace jerárquico por el "
            "método de promedio (scipy). Las features se reordenan según las "
            "hojas del dendrograma; mapa de calor de las primeras 40."
        ),
    ),
    FigureNarrative(
        filename="cell_018_6_extractores_no_supervisados_pca_factor_analysis_y_umap_2d.png",
        title="Varianza acumulada del PCA sobre las features filtradas",
        narrative=(
            "La curva muestra cuánta varianza se acumula a medida que se "
            "añaden componentes principales. La línea roja marca el objetivo "
            "del 95 % y la verde el número de componentes que lo alcanza: "
            "42 componentes capturan el 95,3 % de la varianza, reduciendo el "
            "conjunto de 82 features filtradas a 42. La compresión es fuerte "
            "porque los 17 índices espectrales comparten un subespacio "
            "común. Aun así, el espacio reducido sigue siendo interpretable: "
            "la primera componente carga sobre indicadores de amplitud "
            "vegetativa (LAI, NDVI alto, NDCI, GCVI)."
        ),
        method=(
            "PCA de scikit-learn con objetivo de varianza 0,95 sobre las "
            "features filtradas y estandarizadas. Curva de varianza "
            "acumulada por número de componentes."
        ),
    ),
    FigureNarrative(
        filename="cell_021_6_extractores_no_supervisados_pca_factor_analysis_y_umap_2d_1.png",
        title="Proyección UMAP 2D coloreada por clase de cultivo",
        narrative=(
            "La proyección UMAP comprime las features espectro-temporales en "
            "un plano de dos dimensiones, coloreando cada parcela por su "
            "clase de cultivo. Es una herramienta estrictamente visual: las "
            "dos dimensiones resultantes no entran en el pipeline de "
            "modelado, solo sirven para inspeccionar si las clases forman "
            "grupos separables. Cuando se observan clusters de un mismo "
            "color, es señal de que las features capturan estructura "
            "agronómica real y no ruido. La superposición parcial entre "
            "clases es esperable, porque varios cultivos comparten perfiles "
            "temporales parecidos."
        ),
        method=(
            "UMAP 2D sobre una submuestra de 500 parcelas de las features "
            "filtradas y estandarizadas. Diagrama de dispersión coloreado "
            "por clase de cultivo."
        ),
    ),
    FigureNarrative(
        filename="cell_025_7_importancia_supervisada_random_forest_y_xgboost.png",
        title="Importancia supervisada agregada por familia agronómica",
        narrative=(
            "El gráfico de barras horizontales suma la importancia de todas "
            "las columnas de cada familia de índice espectral, comparando "
            "Random Forest (azul) y XGBoost (naranja). Agregar por familia "
            "da una lectura mucho más rápida del peso agronómico real que "
            "revisar feature por feature. La familia NDVI lidera con 0,1769 "
            "puntos de importancia acumulada en Random Forest, seguida del "
            "EVI; los dos modelos coinciden en señalar el NDVI y el EVI como "
            "las familias más relevantes. Esa coincidencia entre dos "
            "algoritmos independientes es la señal más robusta de "
            "relevancia."
        ),
        method=(
            "Importancia de features de Random Forest y XGBoost (100 árboles "
            "cada uno) sobre las features filtradas. Las importancias se "
            "suman por familia agronómica; diagrama de barras horizontales "
            "agrupado por modelo."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Avance 2 — Fusión multisensor y embeddings AlphaEarth (03c)
# ---------------------------------------------------------------------------

FE_ALPHAEARTH_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="cell_007_2_caracterizacion_diagnostica_del_bloque_alphaearth.png",
        title="Distribución de las 64 dimensiones del embedding AlphaEarth",
        narrative=(
            "La rejilla de 64 histogramas muestra, una por una, cómo se "
            "reparten los valores de cada dimensión del embedding AlphaEarth "
            "a nivel de parcela. La línea roja vertical marca el cero. La "
            "mayoría de las dimensiones están centradas cerca de cero y son "
            "aproximadamente simétricas, un comportamiento esperado en un "
            "modelo de base bien entrenado. El diagnóstico confirma además "
            "que no hay dimensiones muertas (todas tienen varianza útil): "
            "las 64 dimensiones aportan información, así que ninguna se "
            "descarta de entrada."
        ),
        method=(
            "Histogramas de 30 bins por dimensión sobre las 85.951 parcelas "
            "PASTIS-R vectorizadas. Las dimensiones muertas se identifican "
            "como aquellas con más del 95 % de valores cercanos a cero."
        ),
    ),
    FigureNarrative(
        filename="cell_008_2_caracterizacion_diagnostica_del_bloque_alphaearth_1.png",
        title="Correlación cruzada entre las 64 dimensiones de AlphaEarth",
        narrative=(
            "El mapa de calor de 64 por 64 muestra cuánto se parecen entre "
            "sí las dimensiones del embedding. El predominio de tonos "
            "neutros indica baja correlación: la correlación absoluta media "
            "es de 0,26 y la máxima de 0,82. El 22,0 % de los pares de "
            "dimensiones es casi ortogonal (|r| < 0,1) y solo el 11,8 % "
            "está acoplado (|r| > 0,5). El modelo de base aprendió una "
            "representación compacta y poco redundante; por eso conviene "
            "usar el embedding crudo sin aplicar un PCA agresivo antes de "
            "modelar."
        ),
        method=(
            "Matriz de correlación de Pearson entre las 64 dimensiones sobre "
            "las 85.951 parcelas. Mapa de calor con escala divergente "
            "centrada en cero."
        ),
    ),
    FigureNarrative(
        filename="cell_010_3_estabilidad_inter_anual_del_embedding_2022_2025_italia.png",
        title="Estabilidad inter-anual del embedding AlphaEarth (2022-2025)",
        narrative=(
            "El diagrama de cajas muestra la similitud coseno entre el "
            "vector AlphaEarth de un mismo punto en años consecutivos, sobre "
            "500 píxeles italianos comunes a los cuatro años. Las cajas se "
            "sitúan muy cerca de 1,0 (la media ronda 0,953 en los tres "
            "pares de años), bien por encima de la línea verde de alta "
            "estabilidad (0,9). Esto significa que la misma parcela produce "
            "vectores casi idénticos en años distintos: el embedding es "
            "estable en el tiempo, lo que habilita entrenar con un año y "
            "predecir en años cercanos, y hace fiables las comparaciones "
            "temporales."
        ),
        method=(
            "Similitud coseno por píxel entre dimensiones AlphaEarth de años "
            "consecutivos (2022-2023, 2023-2024, 2024-2025) sobre 500 "
            "píxeles comunes. Diagrama de cajas con la media marcada y "
            "líneas de referencia de alta y baja estabilidad."
        ),
    ),
    FigureNarrative(
        filename="cell_017_6_frame_parcel_level_final_y_umap_por_clase_y_grupo_agronomi.png",
        title="UMAP del embedding por clase de cultivo y por grupo agronómico",
        narrative=(
            "Los dos paneles proyectan el embedding AlphaEarth en un plano "
            "de dos dimensiones con UMAP. El panel izquierdo colorea cada "
            "parcela por su clase PASTIS fina (18 clases) y el derecho por "
            "su grupo agronómico (cereales, oleaginosas y legumbres, "
            "cultivos perennes de ciclo largo, cultivos de raíz y cultivos "
            "especiales). La vista por grupo muestra clusters más limpios "
            "que la vista por clase fina, lo que sugiere que el embedding "
            "captura sobre todo la jerarquía botánica de alto nivel: separa "
            "bien familias de cultivos, aunque las distinciones finas dentro "
            "de cada familia quedan más mezcladas."
        ),
        method=(
            "UMAP 2D sobre una submuestra de 2.000 parcelas del embedding "
            "AlphaEarth. Dos diagramas de dispersión con la misma "
            "proyección: uno coloreado por clase PASTIS, otro por grupo "
            "agronómico."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Índice global: notebook_id -> narrativas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Avance 4 — Segmentacion semantica densa
# ---------------------------------------------------------------------------

SEGMENTATION_NARRATIVES: tuple[FigureNarrative, ...] = (
    FigureNarrative(
        filename="samples_tsvit-pheno.png",
        title="Predicciones del modelo ganador (TSViT-pheno)",
        narrative=(
            "Cada fila compara la imagen RGB del patch, la etiqueta real del "
            "agricultor y la prediccion del modelo. TSViT-pheno reconstruye las "
            "parcelas grandes con limites netos; los errores se concentran en "
            "parcelas pequenas y en los bordes, justo donde la varianza "
            "espectral es mayor. Es la lectura cualitativa que acompana al "
            "mIoU de 0,625."
        ),
        method=(
            "Inferencia sobre patches del split de validacion espacial. Se "
            "muestrean cuatro ejemplos representativos; colores por clase de "
            "cultivo segun la paleta PASTIS-R."
        ),
    ),
    FigureNarrative(
        filename="per_class_iou_tsvit.png",
        title="IoU por clase - donde se pierde el mIoU macro",
        narrative=(
            "El IoU por clase revela que el modelo segmenta bien los cultivos "
            "mayoritarios (cereales de invierno, maiz, praderas) pero cae en "
            "las clases minoritarias (legumbres, vinedos, frutales). Como el "
            "mIoU macro promedia todas las clases por igual, esas pocas clases "
            "con IoU bajo arrastran la metrica global. El camino para cerrar la "
            "brecha con el target 0,70 es loss ponderada por frecuencia."
        ),
        method=(
            "IoU calculado por clase sobre el split de validacion. Barras "
            "ordenadas de mayor a menor para visualizar el efecto del "
            "desbalance ~31x."
        ),
    ),
    FigureNarrative(
        filename="confusion_tsvit.png",
        title="Matriz de confusion - que cultivos se confunden",
        narrative=(
            "La diagonal concentra la mayor parte de la masa: el modelo acierta "
            "la clase dominante. Las confusiones fuera de la diagonal siguen el "
            "patron esperado del EDA: cultivos con calendarios fenologicos "
            "parecidos (potato, beet, corn) se mezclan entre si, mientras que "
            "cereales de invierno y praderas quedan bien separados."
        ),
        method=(
            "Matriz de confusion normalizada por fila sobre el split de "
            "validacion. Cada celda es la fraccion de pixeles de la clase real "
            "predichos como la clase columna."
        ),
    ),
    FigureNarrative(
        filename="curves_tsvit-pheno.png",
        title="Curvas de entrenamiento - convergencia estable",
        narrative=(
            "Las curvas de perdida y mIoU de entrenamiento y validacion "
            "convergen sin divergencia, senal de que el modelo no sobreajusta "
            "en las 30 epocas presupuestadas. La epoca del mejor checkpoint de "
            "validacion queda marcada; ese es el modelo que se reporta y se "
            "promueve al ensamble."
        ),
        method=(
            "Registro de perdida y mIoU por epoca en MLflow. El mejor epoch se "
            "selecciona por mIoU de validacion (early stopping implicito)."
        ),
    ),
    FigureNarrative(
        filename="optuna_convergence.png",
        title="Convergencia de Optuna - ajuste fino eficiente",
        narrative=(
            "La curva muestra el mejor valor objetivo acumulado a lo largo de "
            "los trials de Optuna. El estudio converge pronto porque parte de "
            "un warm-start desde el mejor checkpoint, evitando gastar ventana "
            "H100 en reexplorar el espacio de hiperparametros desde cero."
        ),
        method=(
            "Optuna con storage PostgreSQL sobre los dos mejores candidatos "
            "(TSViT-pheno y U-TAE), 30 trials warm-start desde checkpoint."
        ),
    ),
)


NARRATIVES_BY_NOTEBOOK: dict[str, tuple[FigureNarrative, ...]] = {
    "sentinel2": SENTINEL2_NARRATIVES,
    "alphaearth": ALPHAEARTH_NARRATIVES,
    "bivariate-temporal": BIVARIATE_NARRATIVES,
    "pastis-consolidado": PASTIS_NARRATIVES,
    "breizhcrops": BREIZHCROPS_NARRATIVES,
    "paper-methods": PAPER_METHODS_NARRATIVES,
    "globales": (),
    "fe-sentinel2": FE_SENTINEL2_NARRATIVES,
    "fe-pastis-spectral-temporal": FE_SPECTRAL_TEMPORAL_NARRATIVES,
    "fe-alphaearth-fusion": FE_ALPHAEARTH_NARRATIVES,
    "segmentation-avance4": SEGMENTATION_NARRATIVES,
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
