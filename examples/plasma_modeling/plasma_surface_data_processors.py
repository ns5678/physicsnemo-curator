# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from typing import Optional

import numpy as np
from plasma_utils import to_float32
from plasma_validation_utils import (
    check_field_statistics,
    check_surface_physics_bounds,
)
from schemas import PlasmaModelingExtractedDataInMemory

logger = logging.getLogger(__name__)


def default_surface_processing_for_plasma(
    data: PlasmaModelingExtractedDataInMemory,
    surface_variables: dict[str, str],
) -> PlasmaModelingExtractedDataInMemory:
    """Default surface processing for DryResist Plasma Modeling.

    Extracts surface fields from VTP cell data, computes mesh centers,
    normals, and areas.

    Args:
        data: Container with surface_polydata loaded from VTP file
        surface_variables: Dict mapping variable names to types (scalar/vector)

    Returns:
        Updated data with surface_fields, surface_mesh_centers,
        surface_normals, and surface_areas populated
    """
    polydata = data.surface_polydata

    # Extract surface fields from cell data
    field_arrays = []
    for var_name, var_type in surface_variables.items():
        if var_name not in polydata.cell_data:
            raise KeyError(
                f"Variable '{var_name}' not found in VTP cell_data. "
                f"Available: {list(polydata.cell_data.keys())}"
            )
        arr = polydata.cell_data[var_name]
        # Ensure 2D shape (n_cells, n_components)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        field_arrays.append(arr)

    data.surface_fields = np.concatenate(field_arrays, axis=-1)
    data.surface_variable_names = list(surface_variables.keys())

    # Extract mesh centers
    data.surface_mesh_centers = np.array(polydata.cell_centers().points)

    # Extract normals - check if 'Normals' exists in cell data
    if "Normals" in polydata.cell_data:
        data.surface_normals = np.array(polydata.cell_data["Normals"])
    else:
        # Compute normals using PyVista
        data.surface_normals = np.array(polydata.cell_normals)

    # Compute cell areas
    areas_mesh = polydata.compute_cell_sizes(length=False, area=True, volume=False)
    data.surface_areas = np.array(areas_mesh.cell_data["Area"])

    return data


def filter_invalid_surface_cells(
    data: PlasmaModelingExtractedDataInMemory,
    tolerance: float = 1e-6,
) -> PlasmaModelingExtractedDataInMemory:
    """
    Filter out invalid surface cells based on area and normal criteria.

    Removes cells where:
    - Area is <= tolerance (zero or negative area)
    - Normal vector has L2-norm <= tolerance (degenerate normal)

    Args:
        data: Plasma modeling data with surface information
        tolerance: Minimum valid value for area and normal magnitude (default: 1e-6)

    Returns:
        Data with invalid cells filtered out
    """

    if data.surface_areas is None or len(data.surface_areas) == 0:
        logger.warning("Surface areas are empty, skipping filter")
        return data

    if data.surface_normals is None or len(data.surface_normals) == 0:
        logger.warning("Surface normals are empty, skipping filter")
        return data

    # Calculate initial count
    n_total = len(data.surface_areas)

    # Create validity masks
    valid_area_mask = data.surface_areas > tolerance
    normal_norms = np.linalg.norm(data.surface_normals, axis=1)
    valid_normal_mask = normal_norms > tolerance

    # Combine masks (both conditions must be true)
    valid_mask = valid_area_mask & valid_normal_mask

    # Count filtered cells
    n_valid = valid_mask.sum()
    n_filtered = n_total - n_valid
    n_area_filtered = (~valid_area_mask).sum()
    n_normal_filtered = (~valid_normal_mask).sum()

    # Log filtering statistics
    if n_filtered == 0:
        logger.info(f"No invalid surface cells found (all {n_total} cells are valid)")
        return data

    if n_valid == 0:
        logger.error(
            f"All {n_total} surface cells filtered out! "
            f"({n_area_filtered} due to area, {n_normal_filtered} due to normals). "
            "Check tolerance and data quality."
        )
        raise ValueError("Filtering removed all surface cells")

    logger.info(
        f"Filtered {n_filtered} invalid surface cells "
        f"({n_filtered / n_total * 100:.2f}% of {n_total} total cells):"
    )
    logger.info(f"  - {n_area_filtered} cells with area <= {tolerance}")
    logger.info(f"  - {n_normal_filtered} cells with normal L2-norm <= {tolerance}")
    logger.info(f"  - {n_valid} valid cells remaining")

    # Apply filter to all surface arrays
    data.surface_mesh_centers = data.surface_mesh_centers[valid_mask]
    data.surface_normals = data.surface_normals[valid_mask]
    data.surface_areas = data.surface_areas[valid_mask]
    data.surface_fields = data.surface_fields[valid_mask]

    return data


def normalize_surface_normals(
    data: PlasmaModelingExtractedDataInMemory,
) -> PlasmaModelingExtractedDataInMemory:
    """Normalize surface normals to unit vectors."""

    if data.surface_normals is None or data.surface_normals.shape[0] == 0:
        logger.error(f"Surface normals are empty: {data.surface_normals}")
        return data

    # Normalize cell normals
    norms = np.linalg.norm(data.surface_normals, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 1e-10, norms, 1.0)
    data.surface_normals = data.surface_normals / norms

    return data


def non_dimensionalize_surface_fields(
    data: PlasmaModelingExtractedDataInMemory,
) -> PlasmaModelingExtractedDataInMemory:
    """
    Non-dimensionalize surface fields using min-max scaling with precomputed bounds.

    Each variable is scaled to [0, 1] range using:
        x_scaled = (x - min) / (max - min)
    
    Bounds are defined in constants.DryResistConstants and retrieved
    via get_normalization_bounds().
    
    Note: Division is performed in float64 for precision (values span many orders
    of magnitude), then cast back to float32.

    Args:
        data: Plasma modeling data with surface fields

    Returns:
        Data with non-dimensionalized surface fields (values in [0, 1] range)
    """
    from constants import get_normalization_bounds

    if data.surface_fields is None or data.surface_fields.shape[0] == 0:
        logger.error(f"Surface fields are empty: {data.surface_fields}")
        return data

    bounds = get_normalization_bounds()
    
    # Cast to float64 for precision during division
    fields_f64 = data.surface_fields.astype(np.float64)
    
    for i, var_name in enumerate(data.surface_variable_names):
        if var_name in bounds:
            _, vmax = bounds[var_name]
            vmax = np.float64(vmax)
            if vmax != 0:
                fields_f64[:, i] = fields_f64[:, i] / vmax
            else:
                logger.warning(f"Zero vmax for {var_name}, skipping normalization")
        else:
            logger.warning(f"No normalization bounds for '{var_name}', skipping")

    # Cast back to float32
    data.surface_fields = fields_f64.astype(np.float32)

    return data


def update_surface_data_to_float32(
    data: PlasmaModelingExtractedDataInMemory,
) -> PlasmaModelingExtractedDataInMemory:
    """Update surface data to float32."""

    # Update processed surface data
    data.surface_mesh_centers = to_float32(data.surface_mesh_centers)
    data.surface_normals = to_float32(data.surface_normals)
    data.surface_areas = to_float32(data.surface_areas)
    data.surface_fields = to_float32(data.surface_fields)

    return data


def validate_surface_sample_quality(
    data: PlasmaModelingExtractedDataInMemory,
    statistical_tolerance: float = 7.0,
    pressure_max: float = 1e6,  # Pa - adjust for plasma simulations
) -> Optional[PlasmaModelingExtractedDataInMemory]:
    """
    Validate surface sample quality and reject entire sample if it fails checks.

    This validator checks:
    1. Statistical outliers: If all data points are beyond mean ± tolerance*std
    2. Physics bounds: If max pressure exceeds threshold

    Args:
        data: Plasma modeling data with surface information
        statistical_tolerance: Number of standard deviations for outlier detection (default: 7.0)
        pressure_max: Maximum allowed pressure (default: 1e6 Pa)

    Returns:
        Data unchanged if valid, None if sample should be rejected
    """

    if data.surface_fields is None or len(data.surface_fields) == 0:
        logger.warning(
            f"[{data.metadata.filename}] Surface fields are empty, skipping validation"
        )
        return data

    # 1. Check field statistics and perform statistical outlier filtering
    is_invalid, vmax, vmin, n_filtered, n_total = check_field_statistics(
        data.surface_fields, field_type="surface", tolerance=statistical_tolerance
    )

    if is_invalid:
        logger.error(
            f"[{data.metadata.filename}] Sample rejected: "
            f"Statistical outlier detection (mean ± {statistical_tolerance}σ) "
            f"filtered all {n_total} cells"
        )
        return None

    # Log statistics with filename
    logger.info(
        f"[{data.metadata.filename}] Surface field statistics: "
        f"vmax={vmax}, vmin={vmin} "
        f"(filtered {n_filtered}/{n_total} statistical outliers)"
    )

    # 2. Check physics-based bounds
    exceeds_bounds, error_msg = check_surface_physics_bounds(
        vmax, pressure_max=pressure_max
    )

    if exceeds_bounds:
        logger.error(f"[{data.metadata.filename}] Sample rejected: {error_msg}")
        return None

    logger.info(f"[{data.metadata.filename}] Surface sample passed quality checks")
    return data


def decimate_mesh(
    data: PlasmaModelingExtractedDataInMemory,
    algo: str = None,
    reduction: float = 0.0,
    **kwargs,
) -> PlasmaModelingExtractedDataInMemory:
    """Decimate mesh using pyvista.

    Args:
        data: Plasma modeling data with surface polydata
        algo: Decimation algorithm ('decimate_pro' or 'decimate')
        reduction: Target reduction ratio (0.0 = no reduction, 0.9 = 90% reduction)
        **kwargs: Additional arguments passed to decimation function

    Returns:
        Data with decimated surface mesh
    """

    if reduction < 0:
        logger.error(f"Reduction must be >= 0: {reduction}")
        return data

    if not algo or reduction == 0:
        return data

    mesh = data.surface_polydata

    # Need point_data to interpolate target mesh node values.
    mesh = mesh.cell_data_to_point_data()
    # Decimation algos require tri-mesh.
    mesh = mesh.triangulate()
    match algo:
        case "decimate_pro":
            mesh = mesh.decimate_pro(reduction, **kwargs)
        case "decimate":
            if mesh.n_points > 400_000:
                warnings.warn("decimate algo may hang on meshes of size more than 400K")
            mesh = mesh.decimate(
                reduction,
                attribute_error=True,
                scalars=True,
                vectors=True,
                **kwargs,
            )
        case _:
            logger.error(f"Unsupported decimation algo {algo}")
            return data

    # Compute cell data.
    data.surface_polydata = mesh.point_data_to_cell_data()

    # Update metadata
    data.metadata.decimation_algo = algo
    data.metadata.decimation_reduction = reduction

    return data
