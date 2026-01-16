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
STL geometry processing functions for Stokes flow pipeline.
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


def default_geometry_processing_for_stokes_flow(
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Default geometry processing for Stokes flow.

    Extracts:
    - stl_coordinates: vertex positions (N_vertices, 3)
    - stl_faces: face vertex indices, flattened (N_faces * 3,)
    - stl_areas: face areas (N_faces,)
    - stl_centers: face center positions (N_faces, 3)

    Also updates metadata with bounds and counts.

    Args:
        data: Stokes flow data with stl_polydata loaded

    Returns:
        Data with geometry arrays populated
    """
    stl = data.stl_polydata

    # Extract vertex coordinates
    data.stl_coordinates = np.array(stl.points, dtype=np.float32)

    # Extract face indices (assuming triangular faces)
    # PyVista faces format: [n_pts, idx0, idx1, idx2, n_pts, ...]
    faces_raw = np.array(stl.faces)
    # Reshape to (N_faces, 4) and take columns 1:4 for vertex indices
    data.stl_faces = faces_raw.reshape((-1, 4))[:, 1:].astype(np.int32).flatten()

    # Compute face areas
    areas_mesh = stl.compute_cell_sizes(length=False, area=True, volume=False)
    data.stl_areas = np.array(areas_mesh.cell_data["Area"], dtype=np.float32)

    # Compute face centers
    data.stl_centers = np.array(stl.cell_centers().points, dtype=np.float32)

    # Update metadata
    bounds = stl.bounds
    data.metadata.x_bound = (bounds[0], bounds[1])
    data.metadata.y_bound = (bounds[2], bounds[3])
    data.metadata.z_bound = (bounds[4], bounds[5])
    data.metadata.num_stl_points = len(data.stl_coordinates)
    data.metadata.num_stl_faces = len(data.stl_areas)

    logger.info(
        f"[{data.metadata.filename}] STL geometry: "
        f"{data.metadata.num_stl_points} vertices, {data.metadata.num_stl_faces} faces"
    )

    return data


def update_geometry_to_float32(
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Ensure geometry arrays are float32 (faces remain int32).

    Args:
        data: Stokes flow data with geometry arrays

    Returns:
        Data with float32 geometry arrays
    """
    if data.stl_coordinates is not None:
        data.stl_coordinates = data.stl_coordinates.astype(np.float32)
    if data.stl_centers is not None:
        data.stl_centers = data.stl_centers.astype(np.float32)
    if data.stl_areas is not None:
        data.stl_areas = data.stl_areas.astype(np.float32)
    # stl_faces stays as int32

    return data
