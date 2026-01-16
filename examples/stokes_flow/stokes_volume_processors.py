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

"""
Volume data processing functions for Stokes flow pipeline.

Key difference from external_aerodynamics: Stokes flow data is POINT-CENTERED,
not cell-centered. We extract point_data directly without any conversion.
"""

import logging

import numpy as np
from schemas import StokesFlowExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_volume_processing_for_stokes_flow(
    data: StokesFlowExtractedDataInMemory,
    volume_variables: dict[str, str],
) -> StokesFlowExtractedDataInMemory:
    """Default volume processing for Stokes flow.

    Extracts point-centered volume data from VTU:
    - volume_mesh_points: mesh vertex coordinates (N_points, 3)
    - volume_fields: field values at vertices (N_points, N_fields)

    Args:
        data: Stokes flow data with volume_polydata loaded
        volume_variables: Dict mapping field names to types (e.g., {'u': 'scalar', ...})
            Iterating yields field names.

    Returns:
        Data with volume arrays populated
    """
    mesh = data.volume_polydata

    # Extract point coordinates directly (no cell center conversion!)
    data.volume_mesh_points = np.array(mesh.points, dtype=np.float32)

    # Extract point data fields
    fields = []
    for var_name in volume_variables:
        if var_name not in mesh.point_data:
            raise KeyError(
                f"Field '{var_name}' not found in point_data. "
                f"Available fields: {list(mesh.point_data.keys())}"
            )
        field = np.array(mesh.point_data[var_name])
        # Ensure 2D shape (N_points, 1) for scalar fields
        if field.ndim == 1:
            field = field[:, np.newaxis]
        fields.append(field)

    # Concatenate all fields: (N_points, N_fields)
    data.volume_fields = np.concatenate(fields, axis=-1).astype(np.float32)

    # Update metadata
    data.metadata.num_volume_points = len(data.volume_mesh_points)

    var_names = list(volume_variables.keys())
    logger.info(
        f"[{data.metadata.filename}] Volume data: "
        f"{data.metadata.num_volume_points} points, "
        f"{len(var_names)} fields {var_names}"
    )

    return data


def update_volume_to_float32(
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Ensure volume arrays are float32.

    Args:
        data: Stokes flow data with volume arrays

    Returns:
        Data with float32 volume arrays
    """
    if data.volume_mesh_points is not None:
        data.volume_mesh_points = data.volume_mesh_points.astype(np.float32)
    if data.volume_fields is not None:
        data.volume_fields = data.volume_fields.astype(np.float32)

    return data


def filter_invalid_points(
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Filter out points with NaN or inf values.

    Args:
        data: Stokes flow data with volume arrays

    Returns:
        Data with invalid points removed
    """
    if data.volume_mesh_points is None or data.volume_fields is None:
        return data

    n_total = len(data.volume_mesh_points)

    # Create validity masks
    valid_coords = ~np.any(np.isnan(data.volume_mesh_points), axis=1)
    valid_fields = np.all(np.isfinite(data.volume_fields), axis=1)
    valid_mask = valid_coords & valid_fields

    n_valid = valid_mask.sum()
    n_filtered = n_total - n_valid

    if n_filtered == 0:
        logger.info(f"[{data.metadata.filename}] All {n_total} points are valid")
        return data

    if n_valid == 0:
        raise ValueError(
            f"[{data.metadata.filename}] All {n_total} points filtered out!"
        )

    logger.warning(
        f"[{data.metadata.filename}] Filtered {n_filtered}/{n_total} invalid points"
    )

    data.volume_mesh_points = data.volume_mesh_points[valid_mask]
    data.volume_fields = data.volume_fields[valid_mask]
    data.metadata.num_volume_points = len(data.volume_mesh_points)

    return data


def shuffle_volume_data(
    data: StokesFlowExtractedDataInMemory,
    seed: int = 42,
) -> StokesFlowExtractedDataInMemory:
    """Shuffle volume data for randomized sequential access during training.

    Args:
        data: Stokes flow data with volume arrays
        seed: Random seed for reproducibility

    Returns:
        Data with shuffled volume arrays
    """
    if data.volume_mesh_points is None or data.volume_fields is None:
        return data

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data.volume_mesh_points))
    data.volume_mesh_points = data.volume_mesh_points[indices]
    data.volume_fields = data.volume_fields[indices]

    return data
