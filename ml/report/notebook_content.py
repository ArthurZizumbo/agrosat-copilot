"""Contenido estructurado de los notebooks de EDA del Avance 1.

Provee la metadata (título, índice de secciones, conclusiones interpretadas
y directorio de figuras) que consumen tanto el dashboard Streamlit como
el template Jinja2 del reporte PDF. El texto vive aquí en lugar de embebido
en la plantilla para que se renderice idéntico en ambos canales y para que
sea testeable sin levantar weasyprint ni streamlit.

Las conclusiones son resúmenes interpretados (no genéricos) extraídos del
markdown final de cada notebook, citando números reales del análisis y
redactados para lectura del sponsor académico (no técnica densa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class KPI:
    """Indicador clave de la ficha (renderizado igual en dashboard, PDF y notebook).

    Attributes:
        label: Nombre corto del indicador.
        value: Valor formateado (string para preservar separadores locales).
        delta: Subtítulo o contexto del indicador.
    """

    label: str
    value: str
    delta: str


@dataclass(frozen=True)
class NotebookCard:
    """Ficha homogénea por notebook del Avance 1.

    Attributes:
        notebook_id: Identificador corto en kebab-case (e.g. ``"sentinel2"``).
        notebook_path: Path relativo al ``.ipynb`` fuente.
        title: Título legible del notebook.
        subtitle: Bajada de una línea con el alcance.
        sections: Lista ordenada de nombres de secciones del notebook.
        figures_dir: Subdirectorio bajo ``paper/figures/`` con las figuras.
        figure_glob: Glob para filtrar PNGs en ``figures_dir``.
        kpis: Indicadores principales de la ficha, mismos para los 3 canales.
        conclusions: Bloques de conclusión en orden de presentación.
    """

    notebook_id: str
    notebook_path: str
    title: str
    subtitle: str
    sections: tuple[str, ...]
    figures_dir: str
    figure_glob: str = "*.png"
    kpis: tuple[KPI, ...] = field(default_factory=tuple)
    conclusions: tuple[tuple[str, str], ...] = field(default_factory=tuple)


SENTINEL2_CARD = NotebookCard(
    notebook_id="sentinel2",
    notebook_path="notebooks/eda/02a_eda_sentinel2.ipynb",
    title="EDA Univariado — Sentinel-2",
    subtitle=(
        "Análisis univariado de las 10 bandas ópticas de Sentinel-2 sobre "
        "PASTIS-R (Francia, con etiquetas reales de agricultores) y tres "
        "regiones italianas (Pianura Padana, Toscana y Apulia) muestreadas "
        "mediante Google Earth Engine. Buscamos entender qué tan limpios "
        "están los datos, cómo se distribuyen los valores y cuánto afecta "
        "la nubosidad por región y estación del año."
    ),
    sections=(
        "1. Setup y configuración",
        "2. Carga de PASTIS-R y muestreo GEE sobre Italia",
        "3. Análisis de píxeles inválidos (nubes y sombras)",
        "4. Estadísticas descriptivas por banda espectral",
        "5. Detección de outliers (IQR + Isolation Forest)",
        "6. Cardinalidad de clases (cultivos PASTIS)",
        "7. Distribuciones, normalidad y transformaciones recomendadas",
        "8. Evolución temporal del NDVI por cultivo",
        "9. Estiramiento de contraste (Stretch 2-98) y visualización RGB",
        "10. Conclusiones del análisis",
    ),
    figures_dir="us-010",
    kpis=(
        KPI("Bandas analizadas", "10", "Sentinel-2 L2A"),
        KPI("Regiones italianas", "3", "Pianura, Toscana, Apulia"),
        KPI("Píxeles muestreados", "180k", "+ PASTIS-R Francia"),
        KPI("Mejor estación", "Verano", "12–21 % de nubes"),
    ),
    conclusions=(
        (
            "Las 10 bandas aportan información — ninguna se descarta",
            "Las bandas del visible (azul, verde, rojo) muestran valores "
            "promedio bajos (1.200–1.440) mientras que las del infrarrojo "
            "cercano y borde rojo (B07, B08, B8A) llegan a ser 2 a 3 veces "
            "más altas (3.100–3.400). La vegetación sana refleja mucho "
            "infrarrojo y poco rojo: ese contraste es la huella que permite "
            "separar cultivos. Conclusión práctica: mantener las 10 bandas "
            "como features candidatas para el baseline.",
        ),
        (
            "Los datos vienen sucios y hay que limpiarlos antes de modelar",
            "El 0,47 % de los píxeles tiene valor -10.000 (marcador de "
            "‘sin dato’ del satélite) y se filtra sin problema. Tras filtrar, "
            "persisten valores levemente negativos (entre -300 y -994) que "
            "son artefactos de la corrección atmosférica sobre agua y "
            "sombras — válidos, no errores. Ninguna banda sigue una "
            "distribución normal: todas están sesgadas hacia valores bajos, "
            "comportamiento típico de sensores satelitales.",
        ),
        (
            "La nubosidad domina y varía mucho por zona y estación",
            "Pianura Padana pierde el 53 % de los píxeles en otoño por "
            "nubes (33 % en verano). Toscana central queda en 41 % otoño "
            "y 21 % verano. Apulia es la zona más limpia todo el año "
            "(12 % en verano, 32 % en otoño). El verano es la mejor "
            "estación para las tres zonas. Esto guía cuándo priorizar "
            "adquisiciones para el pipeline del Avance 2.",
        ),
        (
            "Los outliers son nubes residuales, no errores de medición",
            "El método IQR univariado marca entre 4,6 % y 15,2 % de "
            "outliers por banda (más en visible por nubes). Isolation "
            "Forest multivariado marca un 5 % uniforme. El valor promedio "
            "de los puntos atípicos es 4.700–6.300, confirmando que son "
            "superficies muy brillantes (nubes que escaparon a la máscara "
            "del satélite). Recomendación: aplicar winsorización al "
            "percentil 99,5 en lugar de eliminarlos.",
        ),
        (
            "Lo que sigue para el Avance 2",
            "Replicar la máscara de calidad antes de cualquier feature "
            "engineering. Aplicar transformación logarítmica o "
            "Yeo-Johnson para corregir el sesgo. Priorizar verano para "
            "ventanas de adquisición y aceptar pérdidas mayores en otoño "
            "en Pianura Padana. Mantener las 10 bandas como features "
            "candidatas — no hay redundancia a este nivel univariado.",
        ),
    ),
)


ALPHAEARTH_CARD = NotebookCard(
    notebook_id="alphaearth",
    notebook_path="notebooks/eda/02b_eda_alphaearth.ipynb",
    title="EDA AlphaEarth Foundations (64 dimensiones)",
    subtitle=(
        "AlphaEarth es un modelo de Google que comprime cada píxel "
        "satelital de 10 m de la Tierra en un vector de 64 números. Aquí "
        "evaluamos si esos 64 números sirven para distinguir tipos de "
        "cultivos en dos países muy diferentes: Italia (3 regiones piloto "
        "con etiquetas Dynamic World, 6.000 píxeles) y Francia (10 patches "
        "PASTIS-R con 13 cultivos reales, 5.000 píxeles)."
    ),
    sections=(
        "Sección 1 — Italia × Dynamic World (separabilidad de clases)",
        "Sección 2 — Francia × PASTIS-R (separabilidad agronómica)",
        "Sección 3 — Comparativa cross-region (estabilidad y consistencia)",
        "Conclusiones — qué aprendimos de los embeddings",
    ),
    figures_dir="us-011",
    kpis=(
        KPI("Dimensiones", "64", "Embeddings por píxel"),
        KPI("OOB Italia", "0,888", "RF sobre 8 clases"),
        KPI("OOB Francia", "0,831", "RF sobre 13 cultivos"),
        KPI("Estabilidad entre años", "0,97", "Similitud coseno mediana"),
    ),
    conclusions=(
        (
            "Los embeddings separan cultivos sin feature engineering manual",
            "En las proyecciones t-SNE y UMAP se ven clusters claros: "
            "agricultura separada de zonas construidas, agua y bosque. "
            "No fue necesario calcular NDVI, EVI ni NDWI — los 64 números "
            "ya capturan esa información. Un Random Forest simple sobre "
            "los embeddings logra OOB score 0,888 en Italia (8 clases) y "
            "0,831 en Francia (13 cultivos PASTIS-R).",
        ),
        (
            "AlphaEarth carga más señal que el NDVI clásico",
            "El panel comparativo muestra el RGB real de Sentinel-2 al "
            "lado del ‘pseudo-RGB’ construido con las 3 dimensiones "
            "AlphaEarth más informativas. El pseudo-RGB resalta variación "
            "estructural (parcelas, caminos) que un solo índice escalar "
            "como NDVI aplana. Justifica usar AlphaEarth como pool de "
            "features principal en el baseline.",
        ),
        (
            "Los embeddings se especializan por región",
            "De las 10 dimensiones más útiles para clasificar, solo "
            "dim_40 coincide entre Italia y Francia. Italia se apoya en "
            "dim_30, dim_36, dim_16, dim_07 y dim_63; Francia en dim_10, "
            "dim_55, dim_05, dim_32 y dim_60. AlphaEarth aprendió "
            "representaciones que dependen del contexto regional. "
            "Consecuencia: entrenar un clasificador por región rinde más "
            "que entrenar uno solo global.",
        ),
        (
            "Estabilidad temporal alta entre años consecutivos",
            "La similitud coseno entre embeddings de las mismas "
            "coordenadas en años consecutivos es muy alta: 2022 vs 2023 "
            "mediana 0,967 — 2023 vs 2024 mediana 0,975 — 2024 vs 2025 "
            "mediana 0,976. Para parcelas con uso del suelo constante "
            "podemos entrenar con un solo año y predecir en años "
            "cercanos sin re-entrenar el modelo.",
        ),
        (
            "Existe redundancia entre dimensiones",
            "La matriz de correlación 64×64 detecta varios pares con "
            "|r| > 0,7. Aplicar PCA antes de XGBoost reduce el ruido sin "
            "perder capacidad discriminativa. No es bloqueante, pero "
            "ayuda al baseline a converger más rápido.",
        ),
        (
            "Lo que sigue para el Avance 2",
            "Probar el baseline XGBoost por región en lugar de uno "
            "global. Aplicar PCA o seleccionar las top-10 dimensiones "
            "por región antes de la fusión multisensor. Mantener las 64 "
            "dimensiones como pool principal, complementadas con NDVI, "
            "NDMI y EVI explícitos para interpretabilidad.",
        ),
    ),
)


BIVARIATE_CARD = NotebookCard(
    notebook_id="bivariate-temporal",
    notebook_path="notebooks/eda/02c_eda_bivariado_temporal.ipynb",
    title="EDA Bivariado, Multivariado y Temporal",
    subtitle=(
        "Aquí buscamos relaciones entre variables: correlaciones entre las "
        "bandas Sentinel-2 y los índices espectrales, redundancia "
        "(multicolinealidad VIF), fenología (mes del pico de vegetación), "
        "autocorrelación temporal y agrupamiento DTW sobre series "
        "temporales de PASTIS-R."
    ),
    sections=(
        "1. Setup y carga de timesteps PASTIS-R",
        "2. Máscara de calidad: filtrado de timesteps con bandas fuera de rango",
        "3. Correlaciones Pearson y Spearman (bandas–bandas, bandas–índices, índices–índices)",
        "4. VIF para detección de multicolinealidad",
        "5. Pairplot Top-5 features por clase",
        "6. Pico NDVI por clase (mes y valor)",
        "7. Autocorrelación (ACF) por clase",
        "8. Agrupamiento DTW de series temporales",
        "9. Cruces con ERA5 (precipitación y NDVI)",
        "Conclusiones",
    ),
    figures_dir="us-012",
    kpis=(
        KPI("Timesteps filtrados", "6,3 %", "545 de 8.600"),
        KPI("Pares |r| > 0,85", "19 + 6", "Banda–banda + índice–índice"),
        KPI("VIF mínimo", "11,3", "EVI (umbral 10)"),
        KPI("Pureza DTW k = 6", "0,380", "vs 0,06 aleatorio"),
    ),
    conclusions=(
        (
            "La calidad del dato es el primer cuello de botella",
            "De 8.600 timesteps cargados, 545 (6,3 %) tienen alguna "
            "banda fuera del rango físico [0; 1,5] tras escalar a "
            "reflectancia. Son artefactos BOA (DN negativo) o píxeles "
            "nubosos. Filtrar los timesteps en lugar de recortarlos "
            "(clip) preserva la variabilidad real del NDVI: ahora el "
            "valor del pico muestra una distribución continua en lugar "
            "de saturarse en 1,0 en el 75 % de las parcelas.",
        ),
        (
            "Las bandas de Sentinel-2 son fuertemente redundantes entre sí",
            "19 pares únicos banda–banda y 6 pares únicos índice–índice "
            "superan |r| > 0,85 (Pearson). Ningún par banda–índice cruza "
            "ese umbral (techo r ≈ 0,80 en B04 vs NDWI). El cuarteto "
            "{B07, B8A, B02, B03} forma un bloque casi perfecto "
            "(B07 vs B8A = 0,997).",
        ),
        (
            "Los índices principales también son redundantes",
            "NDVI vs NDRE = 0,974 — NDVI vs SAVI = 0,949 — NDRE vs SAVI = "
            "0,945 — NDVI vs NDWI = -0,959. Del cuarteto {NDVI, NDRE, "
            "NDWI, SAVI} basta uno como representante. NDMI usa la banda "
            "B11 (SWIR) y aporta una dimensión propia que el resto no "
            "captura.",
        ),
        (
            "El VIF marca todas las features para descarte",
            "EVI tiene el VIF más bajo (11,3), seguido de NDMI (15,7), "
            "pero ambos superan el umbral convencional de 10. Ninguna "
            "feature aislada es ortogonal al resto. Para el baseline "
            "conviene aplicar PCA sobre el bloque de bandas o trabajar "
            "con un subconjunto reducido (NDVI + NDMI + un par de bandas "
            "SWIR) aceptando colinealidad residual.",
        ),
        (
            "El mes del pico NDVI separa familias de cultivos",
            "Soft winter wheat tiene su pico en abril–mayo; Corn y Beet "
            "en agosto–septiembre; Meadow en septiembre–octubre; "
            "Fruits/vegetables/flowers más distribuido. La feature "
            "‘mes del pico’ es por sí sola fuertemente discriminativa "
            "y se incorpora explícita al baseline.",
        ),
        (
            "El agrupamiento DTW con k=6 alcanza pureza 0,380",
            "Sobre 12 clases mayoritarias (frente al 0,06 que daría una "
            "asignación aleatoria con 16 clases). El cluster más nítido "
            "agrupa cereales de invierno (Soft winter wheat + Winter "
            "barley) con baja presencia de Meadow. El resto de clusters "
            "queda mezclado porque PASTIS-R cubre solo 14 meses y "
            "muchas clases tienen perfiles temporales similares en ese "
            "horizonte.",
        ),
        (
            "La autocorrelación temporal es débil con esta cobertura",
            "Solo el 6,5 % del total de pares (parcela, lag ≥ 1) supera "
            "el umbral de Bartlett ±0,524 (para n ≈ 14 observaciones "
            "mensuales). El lag más informativo es lag=1, con un 22 % "
            "de pares significativos.",
        ),
        (
            "Lo que sigue para el Avance 2",
            "Replicar la máscara de calidad antes del feature "
            "engineering. Reducir el bloque {NDVI, NDRE, NDWI, SAVI} a "
            "un solo representante más NDMI más EVI. Incorporar "
            "‘mes del pico’ como feature explícita en el baseline. "
            "Usar DTW con k=6 como feature sintética para cereales de "
            "invierno; no aporta a clases de verano-otoño.",
        ),
    ),
)


PASTIS_CARD = NotebookCard(
    notebook_id="pastis-consolidado",
    notebook_path="notebooks/eda/Avance1.Equipo17.ipynb",
    title="EDA PASTIS-R Consolidado (Avance 1 Equipo 17)",
    subtitle=(
        "Análisis exploratorio completo del dataset PASTIS-R: estructura "
        "de los tensores, calidad de los datos, segmentación de parcelas, "
        "PCA, outliers y preprocesamiento recomendado para los modelos de "
        "segmentación que entrenaremos en la fase de modelado."
    ),
    sections=(
        "1. Carga y exploración inicial",
        "2. Análisis de estructura del dataset",
        "3. Análisis univariante (estadísticas y distribuciones por banda)",
        "4. Análisis temporal (índices de vegetación + gaps temporales)",
        "5. Análisis de segmentación y categorización (parcelas y geografía)",
        "6. Análisis bivariante y multivariante (correlaciones y PCA)",
        "7. Detección de valores atípicos",
        "8. Preprocesamiento recomendado (Z-score, log, winsorización)",
        "9. Conclusiones del EDA",
    ),
    figures_dir="avance1",
    kpis=(
        KPI("Parches", "2.468", "128 × 128 píxeles"),
        KPI("Timesteps medios", "48,1", "Rango 38–61"),
        KPI("Calidad de datos", "97,7 %", "Timesteps limpios"),
        KPI("PCA 2 componentes", "95 %", "Varianza capturada"),
    ),
    conclusions=(
        (
            "Estructura del dataset",
            "PASTIS-R contiene 2.468 parches de 128×128 píxeles. La "
            "longitud temporal varía entre 38 y 61 timesteps (media "
            "48,1). Tiene 10 bandas Sentinel-2 con un rango de valores "
            "de -1.338 a 13.756.",
        ),
        (
            "Calidad de los datos",
            "El 97,7 % de los timesteps son utilizables (42 de 43); solo "
            "un 2,3 % presenta problemas moderados por nubes. Hay "
            "valores negativos (normal en la corrección atmosférica). "
            "La normalización Z-score es crítica antes del modelado.",
        ),
        (
            "Características espectrales",
            "Las bandas del infrarrojo cercano y el borde rojo muestran "
            "los valores más altos, lo que indica vegetación activa. "
            "Las 10 bandas requieren transformación no lineal. El PCA "
            "es muy efectivo: 2 componentes capturan el 95 % de la "
            "varianza (reducción del 80 %) y 4 componentes el 99 %.",
        ),
        (
            "Patrones temporales",
            "Se observan ciclos fenológicos claros en los índices de "
            "vegetación. El NDVI presenta un CV de 53,9 % (variabilidad "
            "temporal significativa). El EVI alcanza 74,9 % (mayor "
            "sensibilidad a cambios). Las series temporales son completas "
            "y aptas para modelos secuenciales (LSTM, Transformer, "
            "U-TAE, TSViT).",
        ),
        (
            "Segmentación y categorización",
            "Hay un promedio de 116 parcelas por patch, con tamaños "
            "muy variables (de 1 a 1.388 píxeles, alta fragmentación). "
            "La cobertura promedio es del 80,58 % del patch. "
            "Distribución por tamaño: 56 % pequeñas (<100 px), 42,2 % "
            "medianas (100–500 px) y solo 1,7 % grandes (>500 px). La "
            "distribución geográfica es balanceada: Este-Norte 25 %, "
            "Este-Sur 26 %, Oeste-Norte 23 %, Oeste-Sur 26 %.",
        ),
        (
            "Distribuciones y outliers",
            "Las 10 bandas presentan sesgo significativo (|skew| > 0,5) "
            "y todas requieren transformación no lineal. Los outliers "
            "representan entre el 5 % y el 15 % por banda (valores "
            "reales, no errores). Recomendación: winsorización en "
            "lugar de eliminación.",
        ),
        (
            "Respuestas a las preguntas clave del Avance 1",
            "(1) Valores faltantes: sí, el 2,3 % de timesteps afectado "
            "por nubes. (2) Estadísticas: rango -1.338 a 13.756 con alta "
            "variabilidad entre bandas. (3) Valores atípicos: sí, "
            "5–15 % por banda. (4) Distribuciones sesgadas: sí, "
            "10 de 10 bandas requieren transformación. "
            "(5) Tendencias temporales: sí, CV de 53,9 % en NDVI y "
            "74,9 % en EVI. (6) Correlación: sí, reducción a 2 "
            "componentes viable (95 % de varianza). "
            "(7) Normalización: crítica — usar estadísticas oficiales "
            "NORM_S2_patch.json. (8) Categorización: implementada por "
            "tamaño (3 clases) y ubicación (4 cuadrantes).",
        ),
        (
            "Recomendaciones para el modelado",
            "(1) Preprocesamiento obligatorio: Z-score con "
            "NORM_S2_patch.json + transformación logarítmica + "
            "winsorización al rango 1–99. (2) Arquitectura sugerida: "
            "encoder temporal (LSTM, GRU o Transformer) más encoder "
            "espacial U-Net con atención y multi-escala para parcelas "
            "pequeñas. (3) Reducción dimensional: PCA a 2 componentes "
            "como opción rápida o mantener las 10 bandas con "
            "transformación logarítmica. (4) Validación: "
            "cross-validation respetando los folds del metadata; "
            "métricas IoU, F1-score y mAP para segmentación.",
        ),
    ),
)


GLOBAL_CARD = NotebookCard(
    notebook_id="globales",
    notebook_path="(síntesis cruzada)",
    title="Conclusiones Globales del Avance 1",
    subtitle=(
        "Síntesis cruzada de los cuatro notebooks de EDA, traducida en "
        "decisiones concretas para el Avance 2 (Feature Engineering) y "
        "el baseline del Avance 3 (XGBoost sobre AlphaEarth + índices "
        "espectrales)."
    ),
    sections=(
        "1. Hallazgos transversales entre los 4 notebooks",
        "2. Decisiones para Feature Engineering (Avance 2)",
        "3. Decisiones para Modelado (Avances 3–5)",
        "4. Limitaciones conocidas y mitigaciones",
        "5. Lo que sigue",
    ),
    figures_dir="",
    kpis=(
        KPI("Notebooks integrados", "4", "S2 + AE + Bivariado + PASTIS"),
        KPI("Decisión arquitectural", "Por región", "Italia y Francia separados"),
        KPI("Índices recomendados", "3", "NDVI + NDMI + EVI"),
        KPI("Baseline F1-macro", "≥ 0,60", "Objetivo XGBoost"),
    ),
    conclusions=(
        (
            "AlphaEarth es la mejor base de features que tenemos",
            "Un Random Forest crudo sobre los 64 embeddings alcanza "
            "OOB 0,831 en Francia y 0,888 en Italia sin feature "
            "engineering. Eso deja el piso del baseline muy por encima "
            "del mínimo 0,60 exigido por la rúbrica del Avance 3. El "
            "reto no es alcanzar la métrica, sino demostrar que los "
            "índices clásicos, DINOv3 y los ensambles aportan valor por "
            "encima de ese piso.",
        ),
        (
            "La especialización por región es la decisión clave",
            "AlphaEarth aprendió dimensiones distintas en Italia y "
            "Francia: solo dim_40 coincide en el top-10. Un único "
            "clasificador global pierde frente a clasificadores por "
            "región. Esta decisión propaga a la fase de modelado "
            "(entrenar dos ramas o dos modelos paralelos) y a la fase "
            "de ensamble (ensambles por región en lugar de globales).",
        ),
        (
            "La máscara de calidad va antes que cualquier otra cosa",
            "Los cuatro notebooks coinciden: filtrar timesteps y "
            "píxeles inválidos (SCL nubes + bandas fuera del rango "
            "[0; 1,5]) antes de calcular cualquier feature. Sin esa "
            "máscara, el 75 % de las parcelas saturan el NDVI a 1,0 y "
            "los outliers de banda se confunden con nubes residuales. "
            "Es el primer paso del pipeline de feature engineering.",
        ),
        (
            "La redundancia entre features es el segundo cuello de botella",
            "El VIF marca todas las features tabulares para descarte "
            "(EVI, con 11,3, es la mejor y aún supera el umbral 10). "
            "PCA antes de XGBoost o un subconjunto reducido (NDVI + "
            "NDMI + 1–2 SWIR + 64 AlphaEarth) son la vía para un "
            "baseline interpretable. AlphaEarth no necesita PCA global, "
            "pero sí por región.",
        ),
        (
            "Decisiones concretas para el Avance 2 (Feature Engineering)",
            "(1) Aplicar máscara SCL + rango [0; 1,5] antes de "
            "cualquier feature. (2) Z-score con NORM_S2_patch.json para "
            "PASTIS-R y estadísticas propias para Italia. "
            "(3) Transformación logarítmica o Yeo-Johnson sobre las 10 "
            "bandas. (4) Winsorización 1–99 sobre outliers reales. "
            "(5) Reducir los 17 índices del catálogo inicial a "
            "{NDVI, NDMI, EVI}. (6) Mes del pico NDVI como feature "
            "explícita. (7) DTW con k=6 como feature sintética para "
            "cereales de invierno. (8) AlphaEarth de 64 dimensiones "
            "como pool principal por región. (9) PCA opcional para "
            "visualización, no para modelado denso.",
        ),
        (
            "Decisiones para el Modelado (Avances 3–5)",
            "(1) Baseline XGBoost por región (Italia y Francia "
            "separados) sobre AlphaEarth + 3 índices + DTW cluster + "
            "mes del pico. Target F1-macro ≥ 0,60. "
            "(2) Segmentación densa con encoder temporal (U-TAE, "
            "TSViT) sobre PASTIS-R con los folds del metadata oficial. "
            "(3) Loss focal o ponderada para clases minoritarias "
            "(legumbres, viñedos, frutales). (4) Ensambles por región: "
            "voting top-3, stacking con Gemma 4 y blending con Optuna.",
        ),
        (
            "Limitaciones conocidas",
            "(1) PASTIS-R cubre solo 14 meses, por lo que las clases "
            "con perfiles temporales similares (potato, beet, corn, "
            "rapeseed) son difíciles de separar con DTW. "
            "(2) Italia no tiene etiquetas reales — solo Dynamic World — "
            "así que la validación final depende de cooperativas locales "
            "o de EuroCrops/HCAT3. (3) El verano es la única ventana "
            "limpia para Pianura Padana, así que la cobertura de otoño "
            "queda como gap conocido. (4) La autocorrelación temporal "
            "es débil (n=14), por lo que los modelos seq2seq pueden no "
            "aportar frente a features agregadas.",
        ),
        (
            "Lo que sigue (Avance 2)",
            "Migrar el pipeline EDA a un asset Dagster reproducible: "
            "ingesta GEE → máscara de calidad → normalización → "
            "features tabulares → partición espacial. Versionar el "
            "resultado en DVC (gs://agrosat-dvc-remote). Etiquetar con "
            "`data_version` y `code_version` en MLflow al cerrar el "
            "Avance 2.",
        ),
    ),
)


CARDS: tuple[NotebookCard, ...] = (
    SENTINEL2_CARD,
    ALPHAEARTH_CARD,
    BIVARIATE_CARD,
    PASTIS_CARD,
    GLOBAL_CARD,
)


def list_figures(card: NotebookCard, figures_root: Path) -> list[Path]:
    """Devuelve los PNG asociados a la ficha, ordenados alfabéticamente.

    Args:
        card: Ficha del notebook.
        figures_root: Raíz que contiene los subdirectorios por figura
            (e.g. ``paper/figures/``).

    Returns:
        Lista de paths PNG. Vacía si la ficha no tiene figures_dir o si el
        directorio no existe.
    """
    if not card.figures_dir:
        return []
    subdir = figures_root / card.figures_dir
    if not subdir.is_dir():
        return []
    return sorted(p for p in subdir.glob(card.figure_glob) if p.is_file())
