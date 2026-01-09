# Plasma Modeling Data Curation - Overview

## What This Pipeline Does

This pipeline converts raw simulation outputs (STL geometry, VTP surface fields, YAML parameters) into a format optimized for machine learning model training. Think of it as a factory that takes your simulation data and produces standardized, compressed, ML-ready datasets.

```
Raw Simulation Data          →    ML-Ready Dataset
─────────────────────────────────────────────────────
body.stl (geometry)          →    Zarr archive with:
surface.vtp (field data)     →      - Non-dimensionalized fields
parameters.yaml (conditions) →      - Consistent data types
                                    - Compressed storage
                                    - Parallel-friendly format
```

## ETL Pipeline Concept

**ETL = Extract, Transform, Load**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   EXTRACT   │ ──▶ │  TRANSFORM  │ ──▶ │    LOAD     │
│             │     │             │     │             │
│ Read files: │     │ Process:    │     │ Write:      │
│ - STL       │     │ - Normalize │     │ - Zarr      │
│ - VTP       │     │ - Validate  │     │ - NumPy     │
│ - YAML      │     │ - Convert   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                   │
     ▼                    ▼                   ▼
  DataSource         Transformations       DataSink
  (data_sources.py)  (data_transformations.py)
```

## Pipeline Architecture

### 1. Data Source (`data_sources.py`)

Reads your simulation files and packages them into a structured container:

```python
# What it reads from each simulation folder:
s_001/
├── body.stl          →  stl_polydata (geometry mesh)
├── surface.vtp       →  surface_polydata (field values on mesh)
└── parameters.yaml   →  simulation_params (operating conditions)
```

### 2. Transformations (processing chain)

Data flows through a series of transformation steps, each doing one thing:

```
Raw Data
    │
    ▼
┌──────────────────────────────────────┐
│  STL Transformation                  │
│  Extract vertices, faces, areas      │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Surface Transformation              │
│  Extract field values from VTP       │
│  ↓                                   │
│  normalize_surface_normals()         │
│  non_dimensionalize_surface_fields() │
│  update_surface_data_to_float32()    │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Global Params Transformation        │
│  Extract saga_flow_rate, etc.        │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Zarr Transformation                 │
│  Compress & chunk for storage        │
└──────────────────────────────────────┘
    │
    ▼
ML-Ready Data
```

### 3. Data Sink (`data_sources.py`)

Writes the processed data to disk in Zarr format (or NumPy).

## Key Files Explained

| File | Purpose |
|------|---------|
| `run_etl.py` | Entry point - orchestrates the pipeline |
| `config/plasma_DryResist.yaml` | Configuration - defines what transformations to apply |
| `data_sources.py` | Read/write logic for simulation files |
| `data_transformations.py` | Transformation classes that process data |
| `plasma_surface_data_processors.py` | Surface field processing functions |
| `plasma_geometry_data_processors.py` | Geometry processing functions |
| `plasma_global_params_data_processors.py` | Parameter extraction functions |
| `schemas.py` | Data structure definitions |
| `constants.py` | Physics constants and normalization bounds |
| `paths.py` | File path patterns for different datasets |

## Configuration System (Hydra)

The pipeline uses Hydra for configuration management. The YAML config defines:

```yaml
# config/plasma_DryResist.yaml (simplified)

etl:
  source:
    input_dir: /path/to/simulations    # Where to read from
    
  transformations:
    surface_preprocessing:
      surface_processors:              # Chain of processing functions
        - normalize_surface_normals
        - non_dimensionalize_surface_fields
        - update_surface_data_to_float32
        
  sink:
    output_dir: /path/to/output        # Where to write
```

**Key benefit**: Change processing steps without modifying code - just edit the YAML.

## Non-Dimensionalization

Surface fields span many orders of magnitude. We normalize them for ML stability:

```
Original values:          Normalized values:
─────────────────────────────────────────────
density: 0.08 - 0.15      density: 0.5 - 1.0
pressure: 6 - 11783 Pa    pressure: 0.0005 - 1.0
rho_SAGA: 1e-26 - 3e-3    rho_SAGA: ~0 - 1.0
```

Formula: `x_normalized = x / max_value`

The max values are precomputed from your dataset and stored in `constants.py`.

## Output Format: Zarr

Zarr is a chunked, compressed array format designed for large datasets:

```
s_001.zarr/
├── .zattrs              # Metadata (filename, physics constants)
├── stl_coordinates/     # Geometry vertices
├── stl_faces/           # Triangle connectivity
├── stl_areas/           # Face areas
├── surface_fields/      # All 25 field variables (normalized)
├── surface_normals/     # Unit normal vectors
├── surface_areas/       # Cell areas
├── global_params_values/      # [saga_flow_rate, water_flow_rate, pressure]
└── global_params_reference/   # Reference values for de-normalization
```

**Why Zarr over HDF5/VTK?**
- Parallel read/write (multiple workers can access simultaneously)
- Cloud-friendly (works with S3, GCS)
- Lazy loading (load only what you need)
- Built-in compression

## Parallel Processing

The pipeline processes multiple simulations in parallel:

```
┌─────────────┐
│  Worker 1   │──▶ s_001 ──▶ s_001.zarr
├─────────────┤
│  Worker 2   │──▶ s_002 ──▶ s_002.zarr
├─────────────┤
│  Worker 3   │──▶ s_003 ──▶ s_003.zarr
├─────────────┤
│     ...     │
├─────────────┤
│  Worker N   │──▶ s_256 ──▶ s_256.zarr
└─────────────┘
```

Configure parallelism in config or command line:
```bash
python run_etl.py etl.processing.num_processes=12
```

## Adding New Variables

To add a new surface variable:

1. **Add to config** (`config/variables/surface/dryresist.yaml`):
   ```yaml
   surface_variables:
     new_variable_name: scalar
   ```

2. **Add normalization bounds** (`constants.py`):
   ```python
   NEW_VARIABLE_MIN: float = 0.0
   NEW_VARIABLE_MAX: float = 100.0
   ```

3. **Update bounds mapping** (`constants.py` in `get_normalization_bounds()`):
   ```python
   "new_variable_name": (b.NEW_VARIABLE_MIN, b.NEW_VARIABLE_MAX),
   ```

## Customizing the Pipeline

### Add a new processing step

1. Create function in `plasma_surface_data_processors.py`:
   ```python
   def my_custom_filter(data):
       # Your processing logic
       return data
   ```

2. Add to config:
   ```yaml
   surface_processors:
     - _target_: plasma_surface_data_processors.my_custom_filter
       _partial_: true
   ```

### Skip a processing step

Remove or comment out the line in the YAML config.

## Glossary

| Term | Definition |
|------|------------|
| **ETL** | Extract, Transform, Load - data processing pattern |
| **Zarr** | Chunked array storage format for large datasets |
| **Hydra** | Configuration framework for Python applications |
| **Non-dimensionalization** | Scaling values to a standard range (e.g., 0-1) |
| **PolyData** | PyVista mesh object (from VTP files) |
| **Cell data** | Values defined at mesh cell centers (vs. vertices) |
