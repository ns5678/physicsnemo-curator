# Plasma Modeling ETL Pipeline

Processes DryResist plasma simulation data (STL geometry + VTP surface fields + YAML parameters) into ML-ready Zarr/NumPy format.

## Data Structure

Expected input directory layout:
```
input_dir/
├── s_001/
│   ├── body.stl          # Geometry
│   ├── surface.vtp       # Surface fields (25 variables)
│   └── parameters.yaml   # Global params (saga_flow_rate, water_flow_rate, pressure)
├── s_002/
│   └── ...
```

## Run with Docker

> **Note:** Container runs as root for pip install. Fix output file permissions with `chown` after processing.

### Generic
```bash
docker run -it --gpus all \
    -v /path/to/dataset:/data \
    -v /path/to/physicsnemo-curator:/workspace \
    nvcr.io/nvidia/physicsnemo/physicsnemo:25.11

# Inside container:
cd /workspace && pip install -e .
cd examples/plasma_modeling
python run_etl.py \
    etl.source.input_dir=/data/train \
    etl.sink.output_dir=/data/processed_zarr
exit

# Fix permissions on host:
sudo chown -R $(id -u):$(id -g) /path/to/dataset/processed_zarr
```

### DryResist Dataset on Sheel's Workstation
```bash
docker run -it --gpus all \
    -v /mnt/work/snidhan/LAM-Plasma-Modeling/data/DryResist_DataSet-1_1-256_simulations:/data \
    -v /mnt/work/snidhan/LAM-Plasma-Modeling/physicsnemo-curator:/workspace \
    nvcr.io/nvidia/physicsnemo/physicsnemo:25.11

# Inside container:
cd /workspace && pip install -e .
cd examples/plasma_modeling
python run_etl.py \
    etl.source.input_dir=/data/organized_dataset/train \
    etl.sink.output_dir=/data/processed_zarr
exit

# Fix permissions on host:
sudo chown -R $(id -u):$(id -g) /mnt/work/snidhan/LAM-Plasma-Modeling/data/DryResist_DataSet-1_1-256_simulations/processed_zarr
```

## Output

Each sample produces `{sample_name}.zarr` (or `.npz`) containing:
- `stl_coordinates`, `stl_centers`, `stl_faces`, `stl_areas` — geometry
- `surface_mesh_centers`, `surface_normals`, `surface_areas`, `surface_fields` — surface data
- `global_params_values`, `global_params_reference` — conditioning parameters

## Configuration

Base config: `config/plasma_DryResist.yaml`

Key overrides:
- `etl.processing.num_processes` — parallel workers
- `etl.sink.overwrite_existing` — overwrite existing outputs (default: true)
- `serialization_format` — `zarr` (default) or `numpy`

## Adding Surface Processors

Edit `config/plasma_DryResist.yaml`:
```yaml
surface_preprocessing:
  surface_processors:
    - _target_: plasma_surface_data_processors.normalize_surface_normals
      _partial_: true
    - _target_: plasma_surface_data_processors.non_dimensionalize_surface_fields
      _partial_: true
    - _target_: plasma_surface_data_processors.update_surface_data_to_float32
      _partial_: true
```
