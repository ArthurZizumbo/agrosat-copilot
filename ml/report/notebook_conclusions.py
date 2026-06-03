"""Conclusiones individuales por integrante para cada avance.

Cada avance cierra con la lectura propia de los resultados obtenidos por los
tres integrantes del Equipo 17, redactada a partir del contenido del notebook.
No describen el reparto de tareas, sino la interpretacion de los hallazgos
desde el angulo de cada rol. El texto vive aqui para reutilizarse tanto en los
notebooks editados directamente como en los builders.
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
        "Desde el modelado, la conclusion del EDA es que el techo del problema ya "
        "es alto sin esfuerzo: un Random Forest crudo sobre los 64 embeddings de "
        "AlphaEarth alcanza OOB 0,83 en Francia y 0,89 en Italia sin una sola "
        "feature manual. Eso reordena la prioridad: el reto del baseline no sera "
        "llegar a la metrica minima, sino demostrar que los indices clasicos y la "
        "vista temporal aportan algo por encima de ese piso. La separabilidad de "
        "clases en las proyecciones t-SNE/UMAP y el pico de NDVI por cultivo me "
        "confirman que la informacion discriminante existe y es explotable.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es sobre la viabilidad del dato como insumo de un producto. El "
        "EDA deja claro que la calidad no es gratis: la nubosidad llega a perder el "
        "53 % de los pixeles en Pianura Padana en otono y obliga a una mascara de "
        "calidad antes de cualquier calculo. Para la plataforma eso significa que la "
        "ingesta debe versionar la mascara y priorizar ventanas de verano. La "
        "estabilidad inter-anual alta del embedding (similitud coseno ~0,97) es una "
        "buena noticia para el backend: permite cachear y reutilizar features entre "
        "anos sin recomputar todo el pipeline.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, el hallazgo accionable es que la validacion tiene que ser "
        "espacial, no aleatoria: los bloques oficiales de PASTIS-R garantizan cero "
        "fuga entre train y test, y cualquier metrica que no respete eso estara "
        "inflada. El EDA tambien fija el contrato de reproducibilidad del pipeline "
        "(mascara, normalizacion con estadisticas oficiales, particion por bloques) "
        "que despues versionamos en DVC y trackeamos en MLflow. Saber desde el "
        "inicio que el desbalance es ~31x condiciona la estrategia de pesos y "
        "muestreo de todos los avances siguientes.",
    ),
)


A2_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusion central de la ingenieria de caracteristicas es que las dos "
        "vistas se equivalen: AlphaEarth (F1 0,52) y las 185 features "
        "espectro-temporales manuales (F1 0,54) rinden practicamente igual sobre el "
        "mismo CV espacial. Eso me dice que no hay que elegir una, sino fusionarlas, "
        "y que la redundancia es el verdadero cuello de botella: los filtros reducen "
        "la matriz hasta un 55,7 % sin perder senal. Para el baseline me quedo con "
        "un conjunto compacto (NDVI, NDMI, EVI + mes del pico) mas el embedding.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es de coste e ingenieria del pipeline. Que el PCA capture el "
        "95 % de la varianza con pocas componentes y que los filtros tiren mas de la "
        "mitad de las columnas significa que la matriz que viaja por el backend puede "
        "ser mucho mas ligera sin perder informacion util. La regla de normalizacion "
        "por familia de modelo (log1p / Yeo-Johnson / z-score) y el pipeline guardado "
        "con joblib son justo lo que necesito para que el preprocesamiento sea "
        "deterministico y reutilizable en produccion, sin recalcular en cada request.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, lo importante es que la fusion multisensor quedo trazable y "
        "que el contexto geografico-climatico (SRTM + ERA5) aporta poco por encima "
        "del embedding, porque AlphaEarth ya lo codifica internamente. Eso simplifica "
        "el grafo de dependencias del pipeline: menos fuentes que versionar para casi "
        "la misma senal. El escalador ajustado solo sobre el conjunto de entrenamiento "
        "y la verificacion explicita de no solapamiento entre folds son el seguro "
        "anti-leakage que despues hace creibles las metricas del baseline.",
    ),
)


A3_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusion del baseline es honesta: el techo tabular closed-set es bajo "
        "(F1 0,32 sobre 18 clases) y el reencuadre fenologico lo sube a 0,41 (+0,09). "
        "No es el numero sonado, pero el resultado mas valioso es cualitativo: el "
        "modelo aprende fenologia, no geografia (descartar las columnas geom_* no "
        "degrada nada). Que pheno_text con Gemini y la firma espectral REP no aporten "
        "(deltas negativos) tambien es una conclusion util: cierra ramas y enfoca el "
        "esfuerzo del Avance 4 en los modelos que leen la serie temporal completa.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es que el baseline define el contrato de features para todo lo "
        "que venga despues. Tener el conjunto ganador nombrado y persistido en un "
        "parquet, con su escalador, es lo que permite que el backend sirva "
        "predicciones reproducibles sin depender del notebook. Que las features "
        "geometricas sean un atajo regional (leakage) y se descarten me ahorra "
        "exponer metadatos que enganarian al modelo en produccion. El baseline es el "
        "punto de comparacion contra el que se justificara cualquier modelo mas caro.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, el Avance 3 deja la trazabilidad cerrada: 5 modelos "
        "reentrenados sobre el conjunto ganador con runs MLflow y tags de "
        "data_version + code_version, y el conjunto de features versionado. La "
        "confirmacion empirica de la hipotesis del sponsor (geom_* es proxy de "
        "region) valida que la metodologia de CV espacial estaba bien planteada. El "
        "baseline saneado es el techo cuantificado a batir por la segmentacion densa, "
        "y los modelos temporales que quedan por debajo aqui pasan a ser base "
        "learners del ensamble del EPIC 6.",
    ),
)


A4_CONCLUSIONS: tuple[MemberConclusion, ...] = (
    MemberConclusion(
        _ISAAC,
        _ROLE_ISAAC,
        "La conclusion de la comparativa confirma la hipotesis que arrastramos desde "
        "el EDA: los modelos que leen la serie temporal completa dominan. TSViT-pheno "
        "encabeza con mIoU 0,625 y F1-macro 0,75, muy por encima de los densos 2D "
        "(U-Net 0,24, DeepLabv3+ 0,27). Donde mas aprendi fue en las arquitecturas "
        "que me toco mirar de cerca: U-TAE (0,47) demuestra que la atencion temporal "
        "ya aporta muchisimo, y SegFormer (0,23) quedo penalizado por correr sobre "
        "3 bandas RGB, lo que es en si mismo una conclusion sobre cuanta informacion "
        "espectral se pierde al recortar la entrada.",
    ),
    MemberConclusion(
        _AARON,
        _ROLE_AARON,
        "Mi lectura es sobre el coste-beneficio de servir cada modelo. Los baselines "
        "densos que corri (U-Net, AnySat) entrenan rapido y son ligeros, pero su mIoU "
        "se queda corto; los temporales cuestan mas computo pero ganan por amplio "
        "margen. Para un producto eso define un trade-off claro: vale la pena el "
        "encoder temporal. La pixel-accuracy del ganador (0,876) es la metrica que un "
        "usuario final percibe como calidad visual del mapa, y es alta; el mIoU macro "
        "es mas duro porque castiga las clases minoritarias, que es donde el backend "
        "debera advertir incertidumbre.",
    ),
    MemberConclusion(
        _ARTHUR,
        _ROLE_ARTHUR,
        "Desde MLOps, la conclusion es que el ajuste fino con Optuna sobre el top-2 "
        "convergio rapido gracias al warm-start desde checkpoint, sin gastar la "
        "ventana de H100, y confirmo a TSViT-pheno como modelo individual final. El "
        "margen que aporta la rama fenologica sobre TSViT base (0,6253 vs 0,6215) es "
        "pequeno pero consistente en las tres metricas, coherente con el Avance 3. La "
        "brecha frente al target de rubrica 0,70 esta acotada y diagnosticada: la "
        "arrastran las clases minoritarias, y se cierra con loss ponderada y mas "
        "epocas, no cambiando de familia de modelo. TSViT-pheno y U-TAE pasan al "
        "ensamble del EPIC 6.",
    ),
)
