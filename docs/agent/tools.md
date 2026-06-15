# Catalogo de FunctionTools del agente

> Documento generado automaticamente por `scripts/gen_tools_doc.py` a
> partir de `ml.agent.tools.TOOL_SPECS`. No editar a mano: cualquier
> cambio se sobrescribe al regenerar. Los esquemas provienen de los
> modelos Pydantic v2 (`model_json_schema()`) de `ml/agent/schemas.py`.

Total de tools: **9** (5 sincronas, 4 diferidas).

## Resumen

| Tool | Diferida | Descripcion |
|------|----------|-------------|
| `list_parcels` | no | List the parcels of the current session, optionally restricted to an AOI polygon. |
| `get_parcel_timeseries` | no | Return the NDVI/NDWI/EVI time series of a parcel over a date window. |
| `get_aoi_stats` | no | Aggregate crop statistics (area, dominant crop, class fractions) over an AOI for a year. |
| `search_stac` | si | Search Sentinel-2 scenes in a STAC catalogue by bbox, datetime range and cloud cover. |
| `get_tiles` | si | Build a TiTiler XYZ tile-template URL for a scene rendered as an index or RGB. |
| `classify_new_parcel` | no | Classify the crop of a new parcel polygon with the stacking ensemble. |
| `add_aoi` | si | Persist a named Area Of Interest polygon for the current session. |
| `compare_models` | si | Compare the crop predictions of several ensemble members for one parcel. |
| `explain_prediction` | no | Explain a parcel prediction with phenology, vigor and a natural-language description. |

## `list_parcels`

- **Modo**: sincrona.
- **Descripcion**: List the parcels of the current session, optionally restricted to an AOI polygon.
- **Modelo de entrada**: `ListParcelsInput`
- **Modelo de salida**: `ParcelList`

### Esquema de entrada

```json
{
  "$defs": {
    "GeoJSONGeometry": {
      "additionalProperties": false,
      "description": "A GeoJSON geometry as produced by the frontend draw tools.\n\nOnly the geometry object is modelled (not a full ``Feature``). ``type`` is\nconstrained to the OGC geometry primitives the agent accepts; ``coordinates``\nkeeps the raw nested list because its depth depends on ``type``.\n\nAttributes:\n    type: GeoJSON geometry type (e.g. ``\"Polygon\"``, ``\"MultiPolygon\"``).\n    coordinates: Raw GeoJSON coordinate array (nesting depends on ``type``).",
      "properties": {
        "type": {
          "title": "Type",
          "type": "string"
        },
        "coordinates": {
          "items": {},
          "title": "Coordinates",
          "type": "array"
        }
      },
      "required": [
        "type",
        "coordinates"
      ],
      "title": "GeoJSONGeometry",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Arguments for ``list_parcels``.\n\nAttributes:\n    session_id: Tenant session; every DB query filters by it.\n    aoi: Optional polygon to spatially restrict the listing.",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "aoi": {
      "anyOf": [
        {
          "$ref": "#/$defs/GeoJSONGeometry"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    }
  },
  "required": [
    "session_id"
  ],
  "title": "ListParcelsInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "$defs": {
    "ParcelRef": {
      "additionalProperties": false,
      "description": "Lightweight reference to a parcel returned by listing/search tools.\n\nAttributes:\n    parcel_id: Primary key of the parcel in the ``parcels`` table.\n    crop_class: Predicted crop class label, if known.\n    confidence: Classifier confidence in ``[0, 1]``, if known.",
      "properties": {
        "parcel_id": {
          "title": "Parcel Id",
          "type": "integer"
        },
        "crop_class": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Crop Class"
        },
        "confidence": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Confidence"
        }
      },
      "required": [
        "parcel_id"
      ],
      "title": "ParcelRef",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Result of ``list_parcels``.\n\nAttributes:\n    parcels: Parcels visible to the session (optionally within the AOI).\n    count: Number of parcels returned.",
  "properties": {
    "parcels": {
      "items": {
        "$ref": "#/$defs/ParcelRef"
      },
      "title": "Parcels",
      "type": "array"
    },
    "count": {
      "title": "Count",
      "type": "integer"
    }
  },
  "required": [
    "parcels",
    "count"
  ],
  "title": "ParcelList",
  "type": "object"
}
```

## `get_parcel_timeseries`

- **Modo**: sincrona.
- **Descripcion**: Return the NDVI/NDWI/EVI time series of a parcel over a date window.
- **Modelo de entrada**: `ParcelTimeseriesInput`
- **Modelo de salida**: `TimeSeries`

### Esquema de entrada

```json
{
  "additionalProperties": false,
  "description": "Arguments for ``get_parcel_timeseries``.\n\nAttributes:\n    session_id: Tenant session.\n    parcel_id: Parcel whose temporal index is requested.\n    start: Inclusive start date of the window.\n    end: Inclusive end date of the window.\n    index: Spectral index to extract.",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    },
    "start": {
      "format": "date",
      "title": "Start",
      "type": "string"
    },
    "end": {
      "format": "date",
      "title": "End",
      "type": "string"
    },
    "index": {
      "enum": [
        "ndvi",
        "ndwi",
        "evi"
      ],
      "title": "Index",
      "type": "string"
    }
  },
  "required": [
    "session_id",
    "parcel_id",
    "start",
    "end",
    "index"
  ],
  "title": "ParcelTimeseriesInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``get_parcel_timeseries``.\n\nAttributes:\n    parcel_id: Parcel the series belongs to.\n    index: Spectral index name echoed back.\n    dates: Observation dates (ascending), aligned with ``values``.\n    values: Index values aligned one-to-one with ``dates``.",
  "properties": {
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    },
    "index": {
      "title": "Index",
      "type": "string"
    },
    "dates": {
      "items": {
        "format": "date",
        "type": "string"
      },
      "title": "Dates",
      "type": "array"
    },
    "values": {
      "items": {
        "type": "number"
      },
      "title": "Values",
      "type": "array"
    }
  },
  "required": [
    "parcel_id",
    "index",
    "dates",
    "values"
  ],
  "title": "TimeSeries",
  "type": "object"
}
```

## `get_aoi_stats`

- **Modo**: sincrona.
- **Descripcion**: Aggregate crop statistics (area, dominant crop, class fractions) over an AOI for a year.
- **Modelo de entrada**: `AoiStatsInput`
- **Modelo de salida**: `AoiStats`

### Esquema de entrada

```json
{
  "$defs": {
    "GeoJSONGeometry": {
      "additionalProperties": false,
      "description": "A GeoJSON geometry as produced by the frontend draw tools.\n\nOnly the geometry object is modelled (not a full ``Feature``). ``type`` is\nconstrained to the OGC geometry primitives the agent accepts; ``coordinates``\nkeeps the raw nested list because its depth depends on ``type``.\n\nAttributes:\n    type: GeoJSON geometry type (e.g. ``\"Polygon\"``, ``\"MultiPolygon\"``).\n    coordinates: Raw GeoJSON coordinate array (nesting depends on ``type``).",
      "properties": {
        "type": {
          "title": "Type",
          "type": "string"
        },
        "coordinates": {
          "items": {},
          "title": "Coordinates",
          "type": "array"
        }
      },
      "required": [
        "type",
        "coordinates"
      ],
      "title": "GeoJSONGeometry",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Arguments for ``get_aoi_stats``.\n\nAttributes:\n    session_id: Tenant session.\n    aoi: Polygon over which crop statistics are aggregated.\n    year: Campaign year of the AlphaEarth annual embedding.",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "aoi": {
      "$ref": "#/$defs/GeoJSONGeometry"
    },
    "year": {
      "title": "Year",
      "type": "integer"
    }
  },
  "required": [
    "session_id",
    "aoi",
    "year"
  ],
  "title": "AoiStatsInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``get_aoi_stats``.\n\nAttributes:\n    area_ha: Total AOI area in hectares.\n    dominant_crop: Most frequent crop class inside the AOI.\n    crop_fractions: Per-class area fraction in ``[0, 1]`` summing to ~1.\n    n_parcels: Number of parcels intersecting the AOI.",
  "properties": {
    "area_ha": {
      "title": "Area Ha",
      "type": "number"
    },
    "dominant_crop": {
      "title": "Dominant Crop",
      "type": "string"
    },
    "crop_fractions": {
      "additionalProperties": {
        "type": "number"
      },
      "title": "Crop Fractions",
      "type": "object"
    },
    "n_parcels": {
      "title": "N Parcels",
      "type": "integer"
    }
  },
  "required": [
    "area_ha",
    "dominant_crop",
    "crop_fractions",
    "n_parcels"
  ],
  "title": "AoiStats",
  "type": "object"
}
```

## `search_stac`

- **Modo**: diferida (background, `Behavior.NON_BLOCKING`).
- **Descripcion**: Search Sentinel-2 scenes in a STAC catalogue by bbox, datetime range and cloud cover.
- **Modelo de entrada**: `SearchStacInput`
- **Modelo de salida**: `SceneList`

### Esquema de entrada

```json
{
  "$defs": {
    "BBox": {
      "additionalProperties": false,
      "description": "Axis-aligned bounding box in EPSG:4326 (lon/lat degrees).\n\nAttributes:\n    minx: Minimum longitude (west edge).\n    miny: Minimum latitude (south edge).\n    maxx: Maximum longitude (east edge).\n    maxy: Maximum latitude (north edge).",
      "properties": {
        "minx": {
          "title": "Minx",
          "type": "number"
        },
        "miny": {
          "title": "Miny",
          "type": "number"
        },
        "maxx": {
          "title": "Maxx",
          "type": "number"
        },
        "maxy": {
          "title": "Maxy",
          "type": "number"
        }
      },
      "required": [
        "minx",
        "miny",
        "maxx",
        "maxy"
      ],
      "title": "BBox",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Arguments for ``search_stac`` (pgstac scene search).\n\nAttributes:\n    bbox: Bounding box to search within.\n    datetime_range: RFC 3339 interval string (e.g. ``\"2019-01-01/2019-12-31\"``).\n    cloud_cover_max: Maximum acceptable cloud cover percentage.",
  "properties": {
    "bbox": {
      "$ref": "#/$defs/BBox"
    },
    "datetime_range": {
      "title": "Datetime Range",
      "type": "string"
    },
    "cloud_cover_max": {
      "default": 20.0,
      "title": "Cloud Cover Max",
      "type": "number"
    }
  },
  "required": [
    "bbox",
    "datetime_range"
  ],
  "title": "SearchStacInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``search_stac``.\n\nAttributes:\n    scenes: STAC item dictionaries matching the query.\n    count: Number of scenes returned.",
  "properties": {
    "scenes": {
      "items": {
        "additionalProperties": true,
        "type": "object"
      },
      "title": "Scenes",
      "type": "array"
    },
    "count": {
      "title": "Count",
      "type": "integer"
    }
  },
  "required": [
    "scenes",
    "count"
  ],
  "title": "SceneList",
  "type": "object"
}
```

## `get_tiles`

- **Modo**: diferida (background, `Behavior.NON_BLOCKING`).
- **Descripcion**: Build a TiTiler XYZ tile-template URL for a scene rendered as an index or RGB.
- **Modelo de entrada**: `GetTilesInput`
- **Modelo de salida**: `TileUrl`

### Esquema de entrada

```json
{
  "additionalProperties": false,
  "description": "Arguments for ``get_tiles`` (TiTiler tile-template URL).\n\nAttributes:\n    scene_id: STAC scene identifier to render.\n    index: Visual product to render (spectral index or natural color).",
  "properties": {
    "scene_id": {
      "title": "Scene Id",
      "type": "string"
    },
    "index": {
      "enum": [
        "ndvi",
        "ndwi",
        "evi",
        "rgb"
      ],
      "title": "Index",
      "type": "string"
    }
  },
  "required": [
    "scene_id",
    "index"
  ],
  "title": "GetTilesInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``get_tiles``.\n\nAttributes:\n    scene_id: Scene identifier echoed back.\n    index: Rendered product echoed back.\n    tile_url: XYZ tile template URL (contains ``{z}/{x}/{y}`` placeholders).",
  "properties": {
    "scene_id": {
      "title": "Scene Id",
      "type": "string"
    },
    "index": {
      "title": "Index",
      "type": "string"
    },
    "tile_url": {
      "title": "Tile Url",
      "type": "string"
    }
  },
  "required": [
    "scene_id",
    "index",
    "tile_url"
  ],
  "title": "TileUrl",
  "type": "object"
}
```

## `classify_new_parcel`

- **Modo**: sincrona.
- **Descripcion**: Classify the crop of a new parcel polygon with the stacking ensemble.
- **Modelo de entrada**: `ClassifyParcelInput`
- **Modelo de salida**: `ClassificationResult`

### Esquema de entrada

```json
{
  "$defs": {
    "GeoJSONGeometry": {
      "additionalProperties": false,
      "description": "A GeoJSON geometry as produced by the frontend draw tools.\n\nOnly the geometry object is modelled (not a full ``Feature``). ``type`` is\nconstrained to the OGC geometry primitives the agent accepts; ``coordinates``\nkeeps the raw nested list because its depth depends on ``type``.\n\nAttributes:\n    type: GeoJSON geometry type (e.g. ``\"Polygon\"``, ``\"MultiPolygon\"``).\n    coordinates: Raw GeoJSON coordinate array (nesting depends on ``type``).",
      "properties": {
        "type": {
          "title": "Type",
          "type": "string"
        },
        "coordinates": {
          "items": {},
          "title": "Coordinates",
          "type": "array"
        }
      },
      "required": [
        "type",
        "coordinates"
      ],
      "title": "GeoJSONGeometry",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Arguments for ``classify_new_parcel`` (StackingEnsemble inference).\n\nAttributes:\n    session_id: Tenant session.\n    aoi: Polygon of the new parcel to classify.\n    year: Campaign year of the AlphaEarth annual embedding (default 2019).",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "aoi": {
      "$ref": "#/$defs/GeoJSONGeometry"
    },
    "year": {
      "default": 2019,
      "title": "Year",
      "type": "integer"
    }
  },
  "required": [
    "session_id",
    "aoi"
  ],
  "title": "ClassifyParcelInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``classify_new_parcel``.\n\nAttributes:\n    crop_class: Argmax crop class predicted by the ensemble.\n    confidence: Probability of ``crop_class`` in ``[0, 1]``.\n    class_probabilities: Full posterior over crop classes.",
  "properties": {
    "crop_class": {
      "title": "Crop Class",
      "type": "string"
    },
    "confidence": {
      "title": "Confidence",
      "type": "number"
    },
    "class_probabilities": {
      "additionalProperties": {
        "type": "number"
      },
      "title": "Class Probabilities",
      "type": "object"
    }
  },
  "required": [
    "crop_class",
    "confidence",
    "class_probabilities"
  ],
  "title": "ClassificationResult",
  "type": "object"
}
```

## `add_aoi`

- **Modo**: diferida (background, `Behavior.NON_BLOCKING`).
- **Descripcion**: Persist a named Area Of Interest polygon for the current session.
- **Modelo de entrada**: `AddAoiInput`
- **Modelo de salida**: `AoiRef`

### Esquema de entrada

```json
{
  "$defs": {
    "GeoJSONGeometry": {
      "additionalProperties": false,
      "description": "A GeoJSON geometry as produced by the frontend draw tools.\n\nOnly the geometry object is modelled (not a full ``Feature``). ``type`` is\nconstrained to the OGC geometry primitives the agent accepts; ``coordinates``\nkeeps the raw nested list because its depth depends on ``type``.\n\nAttributes:\n    type: GeoJSON geometry type (e.g. ``\"Polygon\"``, ``\"MultiPolygon\"``).\n    coordinates: Raw GeoJSON coordinate array (nesting depends on ``type``).",
      "properties": {
        "type": {
          "title": "Type",
          "type": "string"
        },
        "coordinates": {
          "items": {},
          "title": "Coordinates",
          "type": "array"
        }
      },
      "required": [
        "type",
        "coordinates"
      ],
      "title": "GeoJSONGeometry",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Arguments for ``add_aoi`` (persist an AOI for the session).\n\nAttributes:\n    session_id: Tenant session that owns the AOI.\n    aoi: Polygon geometry to persist.\n    name: Human-readable label for the AOI.",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "aoi": {
      "$ref": "#/$defs/GeoJSONGeometry"
    },
    "name": {
      "title": "Name",
      "type": "string"
    }
  },
  "required": [
    "session_id",
    "aoi",
    "name"
  ],
  "title": "AddAoiInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Reference to a persisted Area Of Interest.\n\nDoubles as the output of ``add_aoi`` (the created AOI).\n\nAttributes:\n    aoi_id: Primary key of the AOI in the ``aois`` table.\n    label: Human-readable AOI label, if any.\n    area_ha: AOI area in hectares, if computed.",
  "properties": {
    "aoi_id": {
      "title": "Aoi Id",
      "type": "integer"
    },
    "label": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Label"
    },
    "area_ha": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Area Ha"
    }
  },
  "required": [
    "aoi_id"
  ],
  "title": "AoiRef",
  "type": "object"
}
```

## `compare_models`

- **Modo**: diferida (background, `Behavior.NON_BLOCKING`).
- **Descripcion**: Compare the crop predictions of several ensemble members for one parcel.
- **Modelo de entrada**: `CompareModelsInput`
- **Modelo de salida**: `ModelComparison`

### Esquema de entrada

```json
{
  "additionalProperties": false,
  "description": "Arguments for ``compare_models``.\n\nAttributes:\n    session_id: Tenant session.\n    parcel_id: Parcel whose predictions are compared across models.\n    models: Model member names to compare (e.g. ``[\"tsvit-pheno\", \"utae\"]``).",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    },
    "models": {
      "items": {
        "type": "string"
      },
      "title": "Models",
      "type": "array"
    }
  },
  "required": [
    "session_id",
    "parcel_id",
    "models"
  ],
  "title": "CompareModelsInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``compare_models``.\n\nAttributes:\n    parcel_id: Parcel the comparison refers to.\n    predictions: Mapping of model name -> predicted crop class.\n    agreement: Fraction of models agreeing with the majority in ``[0, 1]``.",
  "properties": {
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    },
    "predictions": {
      "additionalProperties": {
        "type": "string"
      },
      "title": "Predictions",
      "type": "object"
    },
    "agreement": {
      "title": "Agreement",
      "type": "number"
    }
  },
  "required": [
    "parcel_id",
    "predictions",
    "agreement"
  ],
  "title": "ModelComparison",
  "type": "object"
}
```

## `explain_prediction`

- **Modo**: sincrona.
- **Descripcion**: Explain a parcel prediction with phenology, vigor and a natural-language description.
- **Modelo de entrada**: `ExplainPredictionInput`
- **Modelo de salida**: `Explanation`

### Esquema de entrada

```json
{
  "additionalProperties": false,
  "description": "Arguments for ``explain_prediction``.\n\nAttributes:\n    session_id: Tenant session.\n    parcel_id: Parcel whose prediction is explained.",
  "properties": {
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    }
  },
  "required": [
    "session_id",
    "parcel_id"
  ],
  "title": "ExplainPredictionInput",
  "type": "object"
}
```

### Esquema de salida

```json
{
  "additionalProperties": false,
  "description": "Result of ``explain_prediction`` (entry point of the Be My Eyes pattern).\n\nAttributes:\n    parcel_id: Parcel the explanation refers to.\n    crop_class: Predicted crop class being explained.\n    confidence: Confidence of the prediction in ``[0, 1]``.\n    phenology_text: Structured phenology text block (SOG/peak/senescence).\n    vigor: Qualitative vigor assessment (e.g. ``\"high\"``, ``\"moderate\"``).\n    description: Natural-language explanation suitable for the final answer.",
  "properties": {
    "parcel_id": {
      "title": "Parcel Id",
      "type": "integer"
    },
    "crop_class": {
      "title": "Crop Class",
      "type": "string"
    },
    "confidence": {
      "title": "Confidence",
      "type": "number"
    },
    "phenology_text": {
      "title": "Phenology Text",
      "type": "string"
    },
    "vigor": {
      "title": "Vigor",
      "type": "string"
    },
    "description": {
      "title": "Description",
      "type": "string"
    }
  },
  "required": [
    "parcel_id",
    "crop_class",
    "confidence",
    "phenology_text",
    "vigor",
    "description"
  ],
  "title": "Explanation",
  "type": "object"
}
```
