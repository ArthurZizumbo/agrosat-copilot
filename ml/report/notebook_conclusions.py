"""Individual per-member conclusions for each avance.

Each avance closes with the personal reading of the results obtained by the
three members of Equipo 17, written from the notebook content. They do not
describe the task split, but the interpretation of the findings from each role's
angle. The text lives here to be reused both in the directly-edited notebooks and
in the builders.
"""

from __future__ import annotations

from ml.report.notebook_cover import MemberConclusion

_ISAAC = "Carlos Isaac Ávila Gutiérrez"
_AARON = "Carlos Aaron Bocanegra Buitrón"
_ARTHUR = "Arthur Jafed Zizumbo Velasco"

_ROLE_ISAAC = "ML / Data Scientist"
_ROLE_AARON = "Full-Stack / Backend"
_ROLE_ARTHUR = "MLOps / Platform"


A1_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "Desde el modelado, la conclusión del análisis exploratorio es que el techo "
        "del problema ya es alto sin esfuerzo: un Random Forest crudo sobre los 64 "
        "embeddings de AlphaEarth alcanza OOB 0,83 en Francia y 0,89 en Italia sin "
        "una sola feature manual. Eso reordena la prioridad: el reto del baseline no "
        "será llegar a la métrica mínima, sino demostrar que los índices clásicos y "
        "la vista temporal aportan algo por encima de ese piso. La separabilidad de "
        "clases en las proyecciones t-SNE/UMAP y el pico de NDVI por cultivo me "
        "confirman que la información discriminante existe y es explotable.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es sobre la viabilidad del dato como insumo de un producto. El "
        "análisis exploratorio deja claro que la calidad no es gratis: la nubosidad "
        "llega a perder el 53 % de los píxeles en Pianura Padana en otoño y obliga a "
        "una máscara de calidad antes de cualquier cálculo. Para la plataforma eso "
        "significa que la ingesta debe versionar la máscara y priorizar ventanas de "
        "verano. La estabilidad inter-anual alta del embedding (similitud coseno "
        "~0,97) es una buena noticia para el backend: permite cachear y reutilizar "
        "features entre años sin recomputar todo el pipeline.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, el hallazgo accionable es que la validación tiene que ser "
        "espacial, no aleatoria: los bloques oficiales de PASTIS-R garantizan cero "
        "fuga entre train y test, y cualquier métrica que no respete eso estará "
        "inflada. El análisis exploratorio también fija el contrato de "
        "reproducibilidad del pipeline (máscara, normalización con estadísticas "
        "oficiales, partición por bloques) que después versionamos en DVC y "
        "trackeamos en MLflow. Saber desde el inicio que el desbalance es ~31x "
        "condiciona la estrategia de pesos y muestreo de todas las fases siguientes.",
    ),
)


A2_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusión central de la ingeniería de características es que las dos "
        "vistas se equivalen: AlphaEarth (F1 0,52) y las 185 features "
        "espectro-temporales manuales (F1 0,54) rinden prácticamente igual sobre el "
        "mismo CV espacial. Eso me dice que no hay que elegir una, sino fusionarlas, "
        "y que la redundancia es el verdadero cuello de botella: los filtros reducen "
        "la matriz hasta un 55,7 % sin perder señal. Para el baseline me quedo con "
        "un conjunto compacto (NDVI, NDMI, EVI + mes del pico) más el embedding.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es de coste e ingeniería del pipeline. Que el PCA capture el "
        "95 % de la varianza con pocas componentes y que los filtros tiren más de la "
        "mitad de las columnas significa que la matriz que viaja por el backend puede "
        "ser mucho más ligera sin perder información útil. La regla de normalización "
        "por familia de modelo (log1p / Yeo-Johnson / z-score) y el pipeline guardado "
        "con joblib son justo lo que necesito para que el preprocesamiento sea "
        "determinístico y reutilizable en producción, sin recalcular en cada request.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, lo importante es que la fusión multisensor quedó trazable y "
        "que el contexto geográfico-climático (SRTM + ERA5) aporta poco por encima "
        "del embedding, porque AlphaEarth ya lo codifica internamente. Eso simplifica "
        "el grafo de dependencias del pipeline: menos fuentes que versionar para casi "
        "la misma señal. El escalador ajustado solo sobre el conjunto de entrenamiento "
        "y la verificación explícita de no solapamiento entre folds son el seguro "
        "anti-leakage que después hace creíbles las métricas del baseline.",
    ),
)


A3_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusión del baseline es honesta: el techo tabular closed-set es bajo "
        "(F1 0,32 sobre 18 clases) y el reencuadre fenológico lo sube a 0,41 (+0,09). "
        "No es el número soñado, pero el resultado más valioso es cualitativo: el "
        "modelo aprende fenología, no geografía (descartar las columnas geom_* no "
        "degrada nada). Que pheno_text con Gemini y la firma espectral REP no aporten "
        "(deltas negativos) también es una conclusión útil: cierra ramas y enfoca el "
        "esfuerzo de la segmentación densa en los modelos que leen la serie temporal "
        "completa.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es que el baseline define el contrato de features para todo lo "
        "que venga después. Tener el conjunto ganador nombrado y persistido en un "
        "parquet, con su escalador, es lo que permite que el backend sirva "
        "predicciones reproducibles sin depender del notebook. Que las features "
        "geométricas sean un atajo regional (leakage) y se descarten me ahorra "
        "exponer metadatos que engañarían al modelo en producción. El baseline es el "
        "punto de comparación contra el que se justificará cualquier modelo más caro.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, el baseline deja la trazabilidad cerrada: 5 modelos "
        "reentrenados sobre el conjunto ganador con runs MLflow y tags de "
        "data_version + code_version, y el conjunto de features versionado. La "
        "confirmación empírica de que las columnas geométricas son un proxy de la "
        "región valida que la metodología de CV espacial estaba bien planteada. El "
        "baseline saneado es el techo cuantificado a batir por la segmentación densa, "
        "y los modelos temporales que queden por debajo pasan a ser base learners del "
        "ensamble final.",
    ),
)


A4_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusión de la comparativa confirma la hipótesis que arrastramos desde "
        "el análisis exploratorio: los modelos que leen la serie temporal completa "
        "dominan. TSViT-pheno encabeza con mIoU 0,625 y F1-macro 0,75, muy por encima "
        "de los densos 2D (U-Net 0,24, DeepLabv3+ 0,27). Donde más aprendí fue en las "
        "arquitecturas que me tocó mirar de cerca: U-TAE (0,47) demuestra que la "
        "atención temporal ya aporta muchísimo, y SegFormer (0,23) quedó penalizado "
        "por correr sobre 3 bandas RGB, lo que es en sí mismo una conclusión sobre "
        "cuánta información espectral se pierde al recortar la entrada.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es sobre el coste-beneficio de servir cada modelo. Los baselines "
        "densos que corrí (U-Net, AnySat) entrenan rápido y son ligeros, pero su mIoU "
        "se queda corto; los temporales cuestan más cómputo pero ganan por amplio "
        "margen. Para un producto eso define un trade-off claro: vale la pena el "
        "encoder temporal. La pixel-accuracy del ganador (0,876) es la métrica que un "
        "usuario final percibe como calidad visual del mapa, y es alta; el mIoU macro "
        "es más duro porque castiga las clases minoritarias, que es donde el backend "
        "deberá advertir incertidumbre.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, mi aporte fue doble: entrené las dos arquitecturas de los "
        "extremos del ranking —TSViT-pheno (el ganador, mIoU 0,625 / F1-macro 0,75) "
        "y DeepLabv3+ (el denso 2D, mIoU 0,27)—, lo que dejó medido de punta a punta "
        "cuánto vale leer la serie temporal completa frente a un solo compuesto. Para "
        "correr el TSViT, que no cabía en la laptop, repliqué el entorno completo en "
        "una máquina con GPU en la nube (una VM L4 de Google Cloud): disco persistente "
        "para el dataset, mismas dependencias que en local y la corrida lanzada por "
        "línea de comandos para que quedara registrada. Cada experimento se trazó en "
        "MLflow con etiquetas de versión de datos y de código, y la validación usó los "
        "bloques espaciales oficiales de PASTIS-R, nunca un reparto aleatorio, para "
        "garantizar que no hubiera fuga de información entre entrenamiento y prueba. El "
        "ajuste fino con Optuna sobre los dos mejores convergió rápido gracias al "
        "warm-start desde el checkpoint ya entrenado, y confirmó a TSViT-pheno como "
        "modelo individual final; el margen de la rama fenológica sobre TSViT base "
        "(0,6253 vs 0,6215) es pequeño pero consistente en las tres métricas. La "
        "distancia que aún falta hasta un mIoU de 0,70 está acotada y diagnosticada "
        "—la arrastran las clases minoritarias— y se cierra con una función de pérdida "
        "ponderada y más épocas, no cambiando de familia de modelo. TSViT-pheno y "
        "U-TAE pasan a la siguiente etapa, la de los ensambles.",
    ),
)
