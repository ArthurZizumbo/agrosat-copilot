# Comparativa de escenarios - baseline de cultivos

| scenario                     | model   |   n_features |   f1_macro |   f1_weighted |   miou |   train_time_s |
|:-----------------------------|:--------|-------------:|-----------:|--------------:|-------:|---------------:|
| Vector combinado (187 feat)  | XGB     |          185 |     0.4098 |        0.692  | 0.3121 |        308.219 |
| Vector combinado (187 feat)  | LGBM    |          185 |     0.3901 |        0.6834 | 0.299  |        925.48  |
| Vector combinado (187 feat)  | RF      |          185 |     0.3646 |        0.6571 | 0.2695 |        209.683 |
| AlphaEarth 64-dim            | XGB     |           64 |     0.3547 |        0.6545 | 0.2595 |        163.118 |
| AlphaEarth 64-dim            | LGBM    |           64 |     0.3391 |        0.6443 | 0.2506 |        378.717 |
| AlphaEarth 64-dim            | RF      |           64 |     0.3221 |        0.6306 | 0.2316 |        135.916 |
| Sentinel-2 crudo (10 bandas) | XGB     |           10 |     0.257  |        0.5634 | 0.1765 |         94.531 |
| Sentinel-2 crudo (10 bandas) | LGBM    |           10 |     0.2561 |        0.5668 | 0.1776 |        131.303 |
| Sentinel-2 crudo (10 bandas) | RF      |           10 |     0.2044 |        0.4948 | 0.135  |         52.046 |
