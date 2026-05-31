# Comparativa de escenarios — baseline de cultivos

| scenario                     | model   |   n_features |   f1_macro |   f1_weighted |   miou |   train_time_s |
|:-----------------------------|:--------|-------------:|-----------:|--------------:|-------:|---------------:|
| Vector combinado (187 feat)  | XGB     |          185 |     0.4106 |        0.6922 | 0.3127 |        469.386 |
| Vector combinado (187 feat)  | RF      |          185 |     0.3646 |        0.6571 | 0.2695 |         83.866 |
| AlphaEarth 64-dim            | XGB     |           64 |     0.3523 |        0.6529 | 0.2576 |        275.105 |
| AlphaEarth 64-dim            | RF      |           64 |     0.3275 |        0.6344 | 0.236  |         56.823 |
| Sentinel-2 crudo (10 bandas) | XGB     |           10 |     0.257  |        0.5634 | 0.1765 |        181.733 |
| Sentinel-2 crudo (10 bandas) | RF      |           10 |     0.2044 |        0.4948 | 0.135  |         21.272 |
