# Ablation de features — reencuadre fenologico (US-022b-C)

| feature_set          | model   |   n_features |   f1_macro |   f1_weighted |     miou |   delta_vs_full |
|:---------------------|:--------|-------------:|-----------:|--------------:|---------:|----------------:|
| full                 | xgb     |          185 |     0.5167 |        0.7709 |   0.3956 |        nan      |
| no_geom              | xgb     |          185 |     0.5167 |        0.7709 |   0.3956 |          0      |
| no_geom_no_era5_srtm | xgb     |          185 |     0.5167 |        0.7709 |   0.3956 |          0      |
| alphaearth_only      | xgb     |            0 |   nan      |      nan      | nan      |        nan      |
| phenology_only       | xgb     |           32 |     0.4178 |        0.6917 |   0.3007 |         -0.0989 |
