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
Data schemas for Stokes flow pipeline.

Key difference from external_aerodynamics: Volume data is point-centered,
not cell-centered. Uses volume_mesh_points instead of volume_mesh_centers.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv
import zarr
from constants import ModelType


@dataclass
class StokesFlowMetadata:
    """Metadata for Stokes flow simulation data.

    Version history:
    - 1.0: Initial version with expected metadata fields.
    """

    # Simulation identifiers
    filename: str
    dataset_type: ModelType

    # Physics constants (placeholder - data is already non-dimensional)
    physics_constants: Optional[dict[str, float]] = None

    # Geometry bounds (from STL)
    x_bound: Optional[tuple[float, float]] = None
    y_bound: Optional[tuple[float, float]] = None
    z_bound: Optional[tuple[float, float]] = None

    # Mesh statistics
    num_stl_points: Optional[int] = None
    num_stl_faces: Optional[int] = None
    num_volume_points: Optional[int] = None

    # Zarr format version
    zarr_format: Optional[int] = None


@dataclass
class StokesFlowExtractedDataInMemory:
    """Container for Stokes flow data extracted from simulation files.

    Key difference: volume data is point-centered (volume_mesh_points)
    rather than cell-centered.

    Version history:
    - 1.0: Initial version with STL geometry + point-centered volume data.
    """

    # Metadata
    metadata: StokesFlowMetadata

    # Raw data from files
    stl_polydata: Optional[pv.PolyData] = None
    volume_polydata: Optional[pv.UnstructuredGrid] = None

    # Processed STL geometry data
    stl_coordinates: Optional[np.ndarray] = None  # (N_vertices, 3)
    stl_centers: Optional[np.ndarray] = None  # (N_faces, 3)
    stl_faces: Optional[np.ndarray] = None  # (N_faces * 3,) flattened
    stl_areas: Optional[np.ndarray] = None  # (N_faces,)

    # Processed volume data - POINT-CENTERED (not cell-centered!)
    volume_mesh_points: Optional[np.ndarray] = None  # (N_points, 3)
    volume_fields: Optional[np.ndarray] = None  # (N_points, 3) for [u, v, p]


@dataclass(frozen=True)
class PreparedZarrArrayInfo:
    """Information for preparing an array for Zarr storage.

    Version history:
    - 1.0: Initial version with compression and chunking info
    """

    data: np.ndarray
    chunks: tuple[int, ...]
    compressor: zarr.abc.codec
    shards: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class StokesFlowZarrDataInMemory:
    """Container for Stokes flow data prepared for Zarr storage.

    Version history:
    - 1.0: Initial version with STL geometry + point-centered volume data
    """

    # Metadata
    metadata: StokesFlowMetadata

    # STL geometry data
    stl_coordinates: PreparedZarrArrayInfo
    stl_centers: PreparedZarrArrayInfo
    stl_faces: PreparedZarrArrayInfo
    stl_areas: PreparedZarrArrayInfo

    # Volume data - point-centered
    volume_mesh_points: Optional[PreparedZarrArrayInfo] = None
    volume_fields: Optional[PreparedZarrArrayInfo] = None
