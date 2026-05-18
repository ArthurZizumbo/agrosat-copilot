# FarSLIP synthetic fixtures (US-017 / US-016b)

Fixtures generados automaticamente para tests offline del FarSLIPExtractor y el
dataset builder. Cada crop es un array uint16 (4, 256, 256) generado con seed
fija (numpy.random.default_rng(42)). Total < 500 KB cuando se materializan via
``tests/ml/farslip/conftest.py`` o el helper ``build_farslip_pairs`` con
``parcel_records`` sinteticos.

Se commitea SOLO este README y un ``manifest.parquet`` minimo; los crops .tif
se generan on-demand bajo tmp_path en CI. Esto evita inflado del repo y
respeta el limite <500 KB del planning.
