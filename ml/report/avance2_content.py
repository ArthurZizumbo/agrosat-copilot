"""Structured content of the feature engineering notebooks.

Provides the metadata (title, section table of contents, interpreted
conclusions and figures directory) consumed by the Avance 2 integrative
notebook. The text lives here instead of embedded in the builder so it is
testable without launching the notebook and to keep a single editorial source.

Reuses the ``KPI`` and ``NotebookCard`` dataclasses defined in
``ml.report.notebook_content`` (same pattern as Avance 1). The conclusions are
interpreted summaries with real numbers extracted from the outputs of the
three source notebooks (03a Sentinel-2, 03b spectro-temporal PASTIS-R, 03c
AlphaEarth fusion), written in accessible language.
"""

from __future__ import annotations

from pathlib import Path

from ml.report.notebook_content import KPI, NotebookCard

REPO_ROOT = Path(__file__).resolve().parents[2]


SENTINEL2_FE_CARD = NotebookCard(
    notebook_id="fe-sentinel2",
    notebook_path="notebooks/feature_engineering/03a_fe_sentinel2.ipynb",
    title="Ingenieria de Caracteristicas sobre Sentinel-2",
    subtitle=(
        "Transforma las 10 bandas opticas crudas de Sentinel-2 en un "
        "conjunto de caracteristicas listo para modelar, a nivel de pixel. "
        "Recorre las dos fases del pipeline: construccion de nuevas "
        "caracteristicas (indices espectrales, discretizacion y codificacion "
        "de variables categoricas) y seleccion/extraccion (filtros de "
        "varianza y correlacion, pruebas univariadas, PCA y analisis "
        "factorial). El insumo son 2.433 escenas PASTIS-R de Francia mas tres "
        "regiones italianas muestreadas con Google Earth Engine."
    ),
    sections=(
        "1. Configuracion del entorno",
        "2. Carga de datos y mascara de calidad",
        "3. Generacion de nuevas caracteristicas: indices espectrales",
        "4. Discretizacion / binning",
        "5. Transformaciones para correccion de sesgo",
        "6. Escalamiento",
        "7. Codificacion de variables categoricas",
        "8. Seleccion de caracteristicas (varianza, correlacion, chi-cuadrado, ANOVA F, PCA, FA)",
        "9. Consolidacion del conjunto de caracteristicas y exportacion",
        "10. Resumen y proximos pasos",
    ),
    figures_dir="feature-engineering/sentinel2",
    kpis=(
        KPI("Bandas crudas", "10", "Sentinel-2 L2A (B02-B12)"),
        KPI("Indices espectrales", "8", "NDVI, NDWI, EVI, NDRE, SAVI, MSI, BSI, NBR"),
        KPI("Conjunto completo", "70", "Caracteristicas generadas"),
        KPI("PCA al 95 %", "6 de 10", "Componentes (96,8 % de varianza)"),
    ),
    conclusions=(
        (
            "La mascara de calidad es el primer paso obligatorio",
            "Antes de calcular cualquier caracteristica se filtran los pixeles "
            "sin dato y se recortan los valores extremos al percentil 99,5 por "
            "banda. De 44.062.720 filas crudas, ese filtro descarta el 0,47 % "
            "(quedan 43.856.626) y limita la influencia de las nubes residuales "
            "y las superficies muy brillantes. Si no se aplica primero, ese "
            "ruido contamina todos los indices y transformaciones que se "
            "derivan despues.",
        ),
        (
            "Ocho indices espectrales condensan la informacion de las 10 bandas",
            "A partir de las bandas se construyen ocho indices con significado "
            "agronomico: NDVI y NDRE (vigor de la vegetacion), NDWI (contenido "
            "de agua), EVI (vegetacion densa con correccion atmosferica), SAVI "
            "(ajustado por suelo), MSI (estres hidrico), BSI (suelo desnudo) y "
            "NBR (zonas quemadas). Estos indices reducen el espacio de busqueda "
            "del modelo aportando conocimiento del dominio en lugar de dejar "
            "que lo aprenda desde cero.",
        ),
        (
            "Discretizacion: tres estrategias complementarias",
            "Las caracteristicas continuas se discretizan en bins con tres "
            "logicas distintas. El binning por cuantiles sobre las 10 bandas "
            "genera bins equipoblados (necesario para la prueba chi-cuadrado). "
            "El NDVI se discretiza por umbrales fisicos conocidos en cuatro "
            "categorias (agua, suelo, vegetacion escasa, vegetacion densa) y el "
            "EVI por agrupamiento k-means. Cada estrategia conserva la columna "
            "continua original: la discretizacion agrega informacion, no la "
            "destruye.",
        ),
        (
            "Codificacion de variables categoricas: ordinal y one-hot",
            "La estacion del ano se codifica de forma ordinal porque tiene un "
            "orden fenologico real (primavera antes que verano antes que "
            "otono). El tipo de cultivo, en cambio, se codifica con one-hot en "
            "20 columnas: son categorias sin orden inherente y el one-hot evita "
            "que el modelo infiera distancias inexistentes entre cultivos. La "
            "transformacion logaritmica y Yeo-Johnson corrigen el sesgo de las "
            "bandas (todas estan sesgadas hacia valores bajos) antes de "
            "escalarlas con z-score, min-max y escalador robusto.",
        ),
        (
            "La seleccion de caracteristicas reduce la redundancia a la mitad",
            "El filtro de correlacion elimina cuatro caracteristicas con "
            "|r| > 0,90 (las bandas B07, B08, B12 transformadas y el indice "
            "SAVI). La prueba ANOVA F ordena por relevancia: el NBR, el BSI, el "
            "MSI y el NDVI encabezan el ranking. El analisis de componentes "
            "principales es muy efectivo: con solo 6 componentes (de 10) se "
            "captura el 96,8 % de la varianza, y las dos primeras ya explican "
            "el 79,3 %. El conjunto final pasa de 70 caracteristicas a una "
            "version seleccionada de 34 y una version PCA de 26.",
        ),
        (
            "Lo que sigue",
            "Quedan exportados tres conjuntos de caracteristicas en disco "
            "(completo de 70, seleccionado de 34 y PCA de 26) listos para "
            "alimentar el modelo base tabular. El siguiente paso es entrenar "
            "ese modelo con validacion cruzada estratificada por clase de "
            "cultivo y medir la importancia de cada caracteristica para "
            "confirmar cuales aportan senal real.",
        ),
    ),
)


PASTIS_FE_CARD = NotebookCard(
    notebook_id="fe-pastis-spectral-temporal",
    notebook_path="notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb",
    title="Seleccion, Extraccion y Normalizacion sobre PASTIS-R",
    subtitle=(
        "Aplica seleccion y extraccion de caracteristicas sobre el banco "
        "espectro-temporal de PASTIS-R a nivel de parcela: 185 caracteristicas "
        "que combinan estadisticas de 17 indices espectrales, componentes de "
        "la transformada de Fourier y metricas de fenologia. El analisis usa "
        "filtros estadisticos, pruebas univariadas, extractores no "
        "supervisados, importancia supervisada y una tabla de consenso "
        "multi-criterio, todo validado sobre los cinco bloques espaciales "
        "oficiales del dataset."
    ),
    sections=(
        "1. Configuracion y carga del subconjunto",
        "2. Vista del dataset por familia agronomica",
        "3. Balance de clases y muestras por categoria",
        "4. Filtros estadisticos: umbral de varianza y correlacion de Pearson",
        "5. Pruebas univariadas: chi-cuadrado y ANOVA F",
        "6. Extractores no supervisados: PCA, analisis factorial y UMAP 2D",
        "7. Importancia supervisada: Random Forest y XGBoost",
        "8. Tabla de consenso multi-criterio",
        "9. Comparativa antes/despues y decisiones de normalizacion",
        "10. Persistencia del conjunto filtrado",
        "11. Generalizacion cross-region sobre BreizhCrops",
        "12. Conclusiones",
    ),
    figures_dir="feature-engineering/spectral-temporal",
    kpis=(
        KPI("Caracteristicas iniciales", "185", "Estadisticas + Fourier + fenologia"),
        KPI("Tras los filtros", "82", "Reduccion del 55,7 %"),
        KPI("PCA al 95 %", "42", "Componentes interpretables"),
        KPI("Parcelas analizadas", "11.999", "18 clases de cultivo"),
    ),
    conclusions=(
        (
            "Los filtros estadisticos eliminan mas de la mitad de las caracteristicas",
            "El conjunto parte de 185 caracteristicas. El umbral de varianza "
            "(0,01) descarta 51 columnas casi constantes y deja 134. El filtro "
            "de correlacion de Pearson (|r| > 0,95) elimina otras 52 columnas "
            "redundantes y deja 82. En total la matriz se reduce un 55,7 % sin "
            "perder informacion util: las caracteristicas eliminadas eran "
            "copias estadisticas de otras o tenian varianza despreciable.",
        ),
        (
            "El analisis de componentes principales comprime el espacio sin perder estructura",
            "Sobre las 185 caracteristicas estandarizadas, el PCA necesita 42 "
            "componentes para capturar el 95 % de la varianza: una compresion "
            "fuerte porque los 17 indices comparten un subespacio espectral "
            "comun. Lo importante es que el espacio reducido sigue siendo "
            "interpretable: la primera componente (21,4 % de la varianza) carga "
            "sobre indicadores de amplitud vegetativa (LAI, NDVI alto, NDCI, "
            "GCVI) y las tres primeras explican el 34,5 % acumulado con "
            "significado agronomico nombrable.",
        ),
        (
            "Las familias espectrales tienen pesos muy distintos",
            "La importancia agregada por familia agronomica (Random Forest y "
            "XGBoost) muestra que el NDVI lidera con 0,1769 puntos de "
            "importancia acumulada y retiene el 76,5 % de sus columnas tras los "
            "filtros. En el extremo opuesto, las familias TSAVI y FAPAR pierden "
            "el 100 % de sus columnas: su informacion estaba duplicada en otros "
            "indices o tenia varianza casi nula. Esa lectura por familia es mas "
            "accionable que revisar caracteristica por caracteristica.",
        ),
        (
            "La comparativa antes/despues confirma que el filtrado no degrada el modelo",
            "Un Random Forest de 80 arboles sin ajuste de hiperparametros, "
            "evaluado sobre los cinco bloques espaciales oficiales, alcanza un "
            "F1-macro de 0,4702 con la matriz cruda y 0,4624 con la matriz "
            "filtrada (una diferencia de 0,0078 puntos, dentro del ruido): la "
            "version filtrada reemplaza a la cruda con la mitad de columnas. La "
            "version PCA baja a 0,3451 y se reserva como representacion "
            "compacta, no como entrada principal del modelo.",
        ),
        (
            "Normalizacion segun la familia de modelo",
            "Sobre las 82 caracteristicas retenidas, la regla de enrutamiento "
            "asigna escaladores de forma automatica: 27 reciben escalado "
            "estandar (centradas y simetricas), 49 reciben Yeo-Johnson (las "
            "sesgadas con |skew| > 1,0) y 6 reciben log1p (LAI y proxies de "
            "biomasa positivas). Yeo-Johnson se elige en lugar de Box-Cox "
            "porque el NDVI puede ser negativo en agua y sombras. Para redes "
            "neuronales el escalado estandar se cambia por min-max a [0, 1]. El "
            "pipeline resultante se guarda con joblib para reutilizarse sin "
            "recalcular el preprocesamiento.",
        ),
        (
            "Lo que sigue",
            "El conjunto filtrado de 82 caracteristicas sobre 11.999 parcelas "
            "queda persistido junto con su lista de caracteristicas "
            "seleccionadas y el bloque PCA de 42 componentes. Las 20 "
            "caracteristicas con consenso alto entre los cinco criterios "
            "deberian recibir atencion prioritaria al revisar la importancia "
            "del modelo base. La comparativa cross-region sobre BreizhCrops "
            "verifica ademas que los extractores temporales no estan "
            "sobreajustados a una sola region de Francia.",
        ),
    ),
)


ALPHAEARTH_FE_CARD = NotebookCard(
    notebook_id="fe-alphaearth-fusion",
    notebook_path="notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb",
    title="Fusion Multisensor y Embeddings AlphaEarth",
    subtitle=(
        "Caracteriza el embedding AlphaEarth de 64 dimensiones a nivel de "
        "parcela, evalua su estabilidad entre anos, lo compara con las "
        "caracteristicas espectro-temporales manuales y compone una matriz de "
        "fusion multisensor que suma relieve (SRTM), clima mensual (ERA5) y "
        "geometria de la parcela. Cierra con la particion en bloques "
        "espaciales oficiales de PASTIS-R y un escalador ajustado solo sobre "
        "el conjunto de entrenamiento para evitar fuga de informacion."
    ),
    sections=(
        "1. Carga del banco AlphaEarth con etiquetas reales",
        "2. Caracterizacion diagnostica del bloque AlphaEarth",
        "3. Estabilidad inter-anual del embedding (2022-2025)",
        "4. Comparativa simetrica: AlphaEarth 64-dim vs espectro-temporal 185-dim",
        "5. Composicion de la matriz de fusion",
        "6. Frame a nivel de parcela y UMAP por clase y grupo agronomico",
        "7. Bloques espaciales oficiales de PASTIS-R",
        "8. Escalador ajustado solo sobre el conjunto de entrenamiento",
        "9. Seleccion y normalizacion sobre el bloque AlphaEarth",
        "10. Persistencia de los artefactos",
        "11. Conclusiones",
    ),
    figures_dir="feature-engineering/alphaearth",
    kpis=(
        KPI("Dimensiones del embedding", "64", "AlphaEarth por parcela"),
        KPI("Parcelas con etiqueta", "85.951", "PASTIS-R, 18 clases"),
        KPI("Estabilidad entre anos", "0,953", "Similitud coseno media"),
        KPI("Cobertura de la fusion", "95,6 %", "5 bloques multisensor"),
    ),
    conclusions=(
        (
            "El embedding AlphaEarth ya viene casi ortogonal",
            "La correlacion cruzada entre las 64 dimensiones tiene una media "
            "|r| de 0,2607 (mediana 0,2327, maximo 0,8228). El 22,0 % de los "
            "pares de dimensiones es casi ortogonal (|r| < 0,1) y solo el "
            "11,8 % esta acoplado (|r| > 0,5). El modelo de base aprendio una "
            "representacion compacta y poco redundante, lo que justifica usar "
            "el embedding crudo sin aplicar un PCA agresivo antes de modelar.",
        ),
        (
            "El embedding es muy estable entre anos consecutivos",
            "Sobre 500 pixeles comunes a los cuatro anos disponibles "
            "(2022-2025), la similitud coseno de un ano al siguiente tiene una "
            "media de 0,9529. La misma parcela produce vectores casi identicos "
            "en anos distintos: se puede entrenar con un ano y predecir en anos "
            "cercanos, y las comparaciones temporales son confiables.",
        ),
        (
            "AlphaEarth y las caracteristicas manuales rinden casi igual",
            "Un Random Forest de 80 arboles validado sobre los cinco bloques "
            "espaciales oficiales da un F1-macro de 0,5202 con el embedding "
            "AlphaEarth de 64 dimensiones y de 0,5394 con las 185 "
            "caracteristicas espectro-temporales manuales sobre las mismas "
            "85.951 parcelas. La diferencia de 0,0192 puntos esta dentro del "
            "ruido: las dos vistas aportan informacion comparable, asi que "
            "conviene fusionarlas en lugar de elegir una.",
        ),
        (
            "La fusion multisensor: el contexto geografico aporta senal marginal",
            "Sumar relieve (SRTM), clima mensual (ERA5) y geometria de la "
            "parcela al embedding produce una matriz de fusion de 103 columnas "
            "con un F1-macro de 0,5174, frente a 0,5202 con AlphaEarth solo: "
            "una ganancia practicamente nula. El contexto geografico-climatico "
            "aporta poco por encima de la firma semantica del embedding. El "
            "bloque Sentinel-1 de radar, aun pendiente, es el candidato con "
            "mayor potencial para cerrar el contrato de fusion completo.",
        ),
        (
            "Bloques espaciales sin fuga de informacion y escalador ajustado solo en entrenamiento",
            "Se usan los cinco bloques espaciales oficiales de PASTIS-R "
            "distribuidos en la metadata del dataset, con cero parcelas "
            "duplicadas en los conjuntos de prueba: cada tile Sentinel-2 cae en "
            "un solo bloque, asi que no hay fuga de informacion entre "
            "particiones. El escalador estandar se ajusta unicamente sobre las "
            "54.978 parcelas de entrenamiento del primer bloque y se guarda con "
            "joblib, verificando explicitamente que no se solapen los conjuntos "
            "de entrenamiento, validacion y prueba.",
        ),
        (
            "Lo que sigue",
            "Quedan persistidos los artefactos a nivel de parcela: el "
            "embedding AlphaEarth, la matriz de fusion multisensor, el "
            "escalador y un manifiesto en JSON. El siguiente paso es "
            "incorporar el bloque Sentinel-1 de radar para completar la matriz "
            "de fusion y entrenar el modelo base sobre la vista combinada de "
            "AlphaEarth mas caracteristicas espectro-temporales, donde se "
            "espera superar el rendimiento de cada vista por separado.",
        ),
    ),
)


GLOBAL_FE_CARD = NotebookCard(
    notebook_id="fe-conclusiones",
    notebook_path="(sintesis cruzada)",
    title="Conclusiones — Preparacion de los Datos (CRISP-ML(Q))",
    subtitle=(
        "Sintesis cruzada de la fase de ingenieria de caracteristicas sobre "
        "las tres fuentes de datos complementarias. Traduce los hallazgos en "
        "decisiones concretas para la fase de modelado base, siguiendo la "
        "etapa de preparacion de datos del ciclo CRISP-ML(Q)."
    ),
    sections=(
        "1. Construccion de caracteristicas",
        "2. Discretizacion y codificacion de variables",
        "3. Transformaciones y normalizacion",
        "4. Seleccion y extraccion de caracteristicas",
        "5. Hallazgos transversales",
        "6. Lo que sigue",
    ),
    figures_dir="",
    kpis=(
        KPI("Fuentes integradas", "3", "Sentinel-2 + PASTIS-R + AlphaEarth"),
        KPI("Caracteristicas construidas", "70 + 185 + 64", "Pixel, parcela y embedding"),
        KPI("Reduccion por seleccion", "Hasta 55,7 %", "Filtros varianza + correlacion"),
        KPI("Validacion", "Bloques espaciales", "5 folds oficiales PASTIS-R"),
    ),
    conclusions=(
        (
            "Construccion de caracteristicas sobre tres fuentes complementarias",
            "La fase de ingenieria de caracteristicas se aborda desde tres "
            "angulos. Sobre la imagen cruda de Sentinel-2 se construyen 8 "
            "indices espectrales que condensan las 10 bandas opticas en "
            "descriptores con significado agronomico. Sobre las parcelas de "
            "PASTIS-R se parte de 185 caracteristicas espectro-temporales que "
            "combinan estadisticas de 17 indices, componentes de la "
            "transformada de Fourier y metricas de fenologia. Y como tercera "
            "vista, el embedding AlphaEarth aporta 64 dimensiones aprendidas "
            "por un modelo de base. Las tres fuentes se complementan: la "
            "imagen cruda permite indices interpretables, la vista temporal "
            "captura el ciclo de crecimiento y el embedding resume la "
            "estructura semantica del paisaje.",
        ),
        (
            "Discretizacion y codificacion de variables categoricas",
            "La discretizacion convierte caracteristicas continuas en bins con "
            "tres logicas: por cuantiles (bins equipoblados, necesarios para la "
            "prueba chi-cuadrado), por umbrales fisicos del dominio (el NDVI en "
            "categorias de agua, suelo y vegetacion) y por agrupamiento "
            "k-means. En todos los casos se conserva la columna continua "
            "original. Las variables categoricas se codifican segun su "
            "naturaleza: la estacion del ano de forma ordinal porque tiene un "
            "orden fenologico real, y el tipo de cultivo con one-hot en 20 "
            "columnas porque son categorias sin orden inherente. Tambien se "
            "aplica una agrupacion de los cultivos en familias agronomicas "
            "para mitigar el desbalance del problema multiclase.",
        ),
        (
            "Transformaciones y normalizacion adaptadas a cada distribucion",
            "Todas las bandas opticas estan sesgadas hacia valores bajos, un "
            "comportamiento tipico de los sensores satelitales. La correccion "
            "de sesgo usa una regla automatica: log1p cuando los valores son "
            "estrictamente positivos y Yeo-Johnson cuando puede haber valores "
            "negativos (el caso del NDVI sobre agua y sombras), evitando "
            "Box-Cox que no admite negativos. El escalamiento se enruta segun "
            "la familia de modelo: escalado estandar para modelos lineales y "
            "min-max para redes neuronales. Sobre PASTIS-R esa regla asigna 27 "
            "caracteristicas a escalado estandar, 49 a Yeo-Johnson y 6 a log1p. "
            "El pipeline completo se guarda con joblib para reutilizarlo sin "
            "recalcular el preprocesamiento.",
        ),
        (
            "Seleccion y extraccion: menos caracteristicas, misma senal",
            "La redundancia entre caracteristicas es el cuello de botella mas "
            "claro de esta fase. Sobre Sentinel-2 el filtro de correlacion "
            "elimina 4 caracteristicas con |r| > 0,90. Sobre PASTIS-R los "
            "filtros de varianza y correlacion reducen la matriz de 185 a 82 "
            "caracteristicas (un 55,7 % menos) y la comparativa antes/despues "
            "confirma que el F1-macro apenas cambia (de 0,4702 a 0,4624). El "
            "analisis de componentes principales comprime de forma agresiva "
            "pero conservando estructura interpretable: 6 componentes para "
            "Sentinel-2 (96,8 % de varianza), 42 para PASTIS-R (95 %) y 23 para "
            "el bloque AlphaEarth. Las pruebas chi-cuadrado, ANOVA F y la "
            "importancia de Random Forest y XGBoost coinciden en senalar las "
            "familias del NDVI y los indices de borde rojo como las mas "
            "relevantes.",
        ),
        (
            "Hallazgos transversales entre las tres fuentes",
            "Cruzar los tres notebooks deja cuatro lecciones. Primero: la "
            "mascara de calidad debe ir antes que cualquier otra cosa; sin ella "
            "el ruido de las nubes contamina todos los indices derivados. "
            "Segundo: el embedding AlphaEarth (F1-macro 0,5202) y las "
            "caracteristicas espectro-temporales manuales (0,5394) rinden casi "
            "igual, asi que conviene fusionarlos en lugar de elegir uno. "
            "Tercero: el embedding es muy estable entre anos (similitud coseno "
            "media de 0,9529), lo que habilita el transfer learning de un ano a "
            "otro. Cuarto: la validacion sobre bloques espaciales es "
            "imprescindible; la particion oficial de PASTIS-R en cinco bloques "
            "garantiza cero fuga de informacion entre conjuntos de "
            "entrenamiento y prueba, algo que una particion aleatoria no "
            "asegura.",
        ),
        (
            "Lo que sigue",
            "La fase de preparacion de datos deja listos los insumos para el "
            "modelado base. El paso inmediato es entrenar un modelo tabular "
            "sobre la vista combinada de AlphaEarth mas las caracteristicas "
            "espectro-temporales, con validacion cruzada sobre los cinco "
            "bloques espaciales oficiales y la importancia de caracteristicas "
            "como diagnostico. Para completar la matriz de fusion queda "
            "pendiente incorporar el bloque Sentinel-1 de radar. Los conjuntos "
            "de caracteristicas, el escalador y los manifiestos quedan "
            "versionados para que el modelado parta de un origen reproducible.",
        ),
    ),
)


FE_CARDS: tuple[NotebookCard, ...] = (
    SENTINEL2_FE_CARD,
    PASTIS_FE_CARD,
    ALPHAEARTH_FE_CARD,
    GLOBAL_FE_CARD,
)
