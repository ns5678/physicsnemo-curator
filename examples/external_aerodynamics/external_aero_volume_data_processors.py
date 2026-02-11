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
import time
from typing import Optional

import numpy as np
import vtk
from numba import njit
from tqdm import tqdm

from constants import (
    PhysicsConstantsCarAerodynamics,
    PhysicsConstantsHLPW,
)
from external_aero_utils import get_volume_data, to_float32
from external_aero_validation_utils import (
    check_field_statistics,
    check_volume_physics_bounds,
)
from schemas import ExternalAerodynamicsExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_volume_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    volume_variables: list[str],
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default volume processing for External Aerodynamics."""
    data.volume_mesh_centers, data.volume_fields = get_volume_data(
        data.volume_unstructured_grid, volume_variables
    )
    data.volume_fields = np.concatenate(data.volume_fields, axis=-1)
    return data


def filter_volume_invalid_cells(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Filter out invalid volume cells based on NaN and inf criteria.

    Removes cells where:
    - Coordinates contain NaN values
    - Field values contain NaN or inf values

    Args:
        data: External aerodynamics data with volume information

    Returns:
        Data with invalid cells filtered out
    """

    if data.volume_mesh_centers is None or len(data.volume_mesh_centers) == 0:
        logger.warning("Volume mesh centers are empty, skipping volume filter")
        return data

    if data.volume_fields is None or len(data.volume_fields) == 0:
        logger.warning("Volume fields are empty, skipping volume filter")
        return data

    # Calculate initial count
    n_total = len(data.volume_mesh_centers)

    # Create validity masks
    # Check for NaN in coordinates (any NaN in any dimension makes the cell invalid)
    valid_coords_mask = ~np.any(np.isnan(data.volume_mesh_centers), axis=1)

    # Check for NaN/inf in fields (any non-finite value makes the cell invalid)
    valid_fields_mask = np.all(np.isfinite(data.volume_fields), axis=1)

    # Combine masks (both conditions must be true)
    valid_mask = valid_coords_mask & valid_fields_mask

    # Count filtered cells
    n_valid = valid_mask.sum()
    n_filtered = n_total - n_valid
    n_coords_filtered = (~valid_coords_mask).sum()
    n_fields_filtered = (~valid_fields_mask).sum()

    # Log filtering statistics
    if n_filtered == 0:
        logger.info(f"No invalid volume cells found (all {n_total} cells are valid)")
        return data

    if n_valid == 0:
        logger.error(
            f"All {n_total} volume cells filtered out! "
            f"({n_coords_filtered} due to NaN coords, {n_fields_filtered} due to NaN/inf fields). "
            "Check data quality."
        )

    logger.info(
        f"Filtered {n_filtered} invalid volume cells "
        f"({n_filtered / n_total * 100:.2f}% of {n_total} total cells):"
    )
    logger.info(f"  - {n_coords_filtered} cells with NaN in coordinates")
    logger.info(f"  - {n_fields_filtered} cells with NaN/inf in fields")
    logger.info(f"  - {n_valid} valid cells remaining")

    # Apply filter to all volume arrays
    data.volume_mesh_centers = data.volume_mesh_centers[valid_mask]
    data.volume_fields = data.volume_fields[valid_mask]

    return data


def non_dimensionalize_volume_fields(
    data: ExternalAerodynamicsExtractedDataInMemory,
    air_density: PhysicsConstantsCarAerodynamics.AIR_DENSITY,
    stream_velocity: PhysicsConstantsCarAerodynamics.STREAM_VELOCITY,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize volume fields."""

    if data.volume_fields.shape[0] == 0:
        logger.error(f"Volume fields are empty: {data.volume_fields}")
        return data

    if air_density <= 0:
        logger.error(f"Air density must be > 0: {air_density}")
    if stream_velocity <= 0:
        logger.error(f"Stream velocity must be > 0: {stream_velocity}")

    stl_vertices = data.stl_polydata.points
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
    data.volume_fields[:, :3] = data.volume_fields[:, :3] / stream_velocity
    data.volume_fields[:, 3:4] = data.volume_fields[:, 3:4] / (
        air_density * stream_velocity**2.0
    )
    data.volume_fields[:, 4:] = data.volume_fields[:, 4:] / (
        stream_velocity * length_scale
    )

    return data


def non_dimensionalize_volume_fields_hlpw(
    data: ExternalAerodynamicsExtractedDataInMemory,
    pref: PhysicsConstantsHLPW.PREF,
    tref: PhysicsConstantsHLPW.TREF,
    uref: PhysicsConstantsHLPW.UREF,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize volume fields."""

    # Pressure
    data.volume_fields[:, :1] = data.volume_fields[:, :1] / pref

    # Temperature
    data.volume_fields[:, 1:2] = data.volume_fields[:, 1:2] / tref

    # Velocity
    data.volume_fields[:, 2:] = data.volume_fields[:, 2:] / uref

    return data


def update_volume_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update volume data to float32."""
    data.volume_mesh_centers = to_float32(data.volume_mesh_centers)
    data.volume_fields = to_float32(data.volume_fields)
    return data


def validate_volume_sample_quality(
    data: ExternalAerodynamicsExtractedDataInMemory,
    statistical_tolerance: float = 7.0,
    velocity_max: float = 3.5,
    pressure_max: float = 4.0,
) -> Optional[ExternalAerodynamicsExtractedDataInMemory]:
    """
    Validate volume sample quality and reject entire sample if it fails checks.

    This validator checks:
    1. Statistical outliers: If all data points are beyond mean ± tolerance*std
    2. Physics bounds: If max non-dimensionalized values exceed thresholds

    Note: This should be applied AFTER non-dimensionalization.

    Args:
        data: External aerodynamics data with volume information
        statistical_tolerance: Number of standard deviations for outlier detection (default: 7.0)
        velocity_max: Maximum allowed non-dimensionalized velocity magnitude (default: 3.5)
        pressure_max: Maximum allowed non-dimensionalized pressure (default: 4.0)

    Returns:
        Data unchanged if valid, None if sample should be rejected
    """

    if data.volume_fields is None or len(data.volume_fields) == 0:
        logger.warning(
            f"[{data.metadata.filename}] Volume fields are empty, skipping validation"
        )
        return data

    # 1. Check field statistics and perform statistical outlier filtering
    is_invalid, vmax, vmin, n_filtered, n_total = check_field_statistics(
        data.volume_fields, field_type="volume", tolerance=statistical_tolerance
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
        f"[{data.metadata.filename}] Volume field statistics: "
        f"vmax={vmax}, vmin={vmin} "
        f"(filtered {n_filtered}/{n_total} statistical outliers)"
    )

    # 2. Check physics-based bounds
    exceeds_bounds, error_msg = check_volume_physics_bounds(
        vmax, velocity_max=velocity_max, pressure_max=pressure_max
    )

    if exceeds_bounds:
        logger.error(f"[{data.metadata.filename}] Sample rejected: {error_msg}")
        return None

    logger.info(f"[{data.metadata.filename}] Volume sample passed quality checks")
    return data


def shuffle_volume_data(
    data: ExternalAerodynamicsExtractedDataInMemory,
    seed: int = 42,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Shuffle volume data.

    This is useful because instead of randomly accessing the data upon read,
    we can shuffle the data during preprocessing, and do sequential reads.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data.volume_mesh_centers))
    data.volume_mesh_centers = data.volume_mesh_centers[indices]
    data.volume_fields = data.volume_fields[indices]

    return data


# ---------------------------------------------------------------------------
# FVM face connectivity extraction
# Ported from sn/stokes-flow branch (stokes_volume_processors.py)
# ---------------------------------------------------------------------------


@njit(fastmath=True)
def _compute_face_geometry_numba(point_ids, points, cell_center):
    """Compute face area, normal, and center using numba."""
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)

    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec[0] += edge1[1] * edge2[2] - edge1[2] * edge2[1]
        area_vec[1] += edge1[2] * edge2[0] - edge1[0] * edge2[2]
        area_vec[2] += edge1[0] * edge2[1] - edge1[1] * edge2[0]

    area_mag = np.sqrt(area_vec[0] ** 2 + area_vec[1] ** 2 + area_vec[2] ** 2)
    if area_mag < 1e-30:
        return 0.0, np.zeros(3), np.zeros(3)

    area = 0.5 * area_mag
    normal = area_vec / area_mag

    face_center = np.zeros(3)
    for i in range(n_points):
        face_center += points[point_ids[i]]
    face_center /= n_points

    # Orient normal outward from owner cell
    face_to_cell = cell_center - face_center
    if (
        normal[0] * face_to_cell[0]
        + normal[1] * face_to_cell[1]
        + normal[2] * face_to_cell[2]
        > 0
    ):
        normal = -normal

    return area, normal, face_center


def _build_face_connectivity(ugrid):
    """Build face connectivity from VTK unstructured grid."""
    from vtk.util import numpy_support

    logger.info("Building face connectivity...")
    t0 = time.time()

    n_cells = ugrid.GetNumberOfCells()
    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)

    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ugrid)
    cell_centers_filter.Update()
    cell_centers = numpy_support.vtk_to_numpy(
        cell_centers_filter.GetOutput().GetPoints().GetData()
    ).astype(np.float64)

    face_to_cells = {}
    for cell_idx in tqdm(range(n_cells), desc="  Extracting faces", unit="cells"):
        cell = ugrid.GetCell(cell_idx)
        n_faces = cell.GetNumberOfFaces()
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids = face.GetPointIds()
            n_pts = face_point_ids.GetNumberOfIds()
            face_pts = tuple(sorted([face_point_ids.GetId(i) for i in range(n_pts)]))
            if face_pts not in face_to_cells:
                face_to_cells[face_pts] = []
            face_to_cells[face_pts].append(
                (
                    cell_idx,
                    np.array(
                        [face_point_ids.GetId(i) for i in range(n_pts)], dtype=np.int64
                    ),
                )
            )

    face_owner = []
    face_neighbor = []
    face_area = []
    face_normal = []
    face_center_list = []

    for face_pts_sorted, cell_list in tqdm(
        face_to_cells.items(), desc="  Computing geometry", unit="faces"
    ):
        if len(cell_list) == 2:
            (cell1, pts1), (cell2, pts2) = cell_list
            owner, neighbor = (cell1, cell2) if cell1 < cell2 else (cell2, cell1)
            face_pts = pts1
        elif len(cell_list) == 1:
            owner = cell_list[0][0]
            neighbor = -1
            face_pts = cell_list[0][1]
        else:
            continue

        area, normal, fc = _compute_face_geometry_numba(
            face_pts, points, cell_centers[owner]
        )
        if area < 1e-30:
            continue

        face_owner.append(owner)
        face_neighbor.append(neighbor)
        face_area.append(area)
        face_normal.append(normal)
        face_center_list.append(fc)

    logger.info(f"  {len(face_owner):,} faces built in {time.time() - t0:.2f}s")

    return {
        "n_faces": len(face_owner),
        "face_owner": np.array(face_owner, dtype=np.int32),
        "face_neighbor": np.array(face_neighbor, dtype=np.int32),
        "face_area": np.array(face_area, dtype=np.float32),
        "face_normal": np.array(face_normal, dtype=np.float32),
        "face_centers": np.array(face_center_list, dtype=np.float32),
    }


def compute_fvm_connectivity(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Compute FVM face connectivity from volume mesh.

    Builds face-based connectivity arrays needed for FVM residual kernels.
    Must be called before volume_unstructured_grid is released.

    Args:
        data: data object with volume_unstructured_grid loaded

    Returns:
        Data with FVM connectivity arrays populated
    """
    import pyvista as pv

    ugrid = data.volume_unstructured_grid
    if ugrid is None:
        logger.warning("No volume_unstructured_grid available for FVM connectivity")
        return data

    # Build face connectivity (expensive VTK traversal)
    logger.info(f"[{data.metadata.filename}] Building FVM face connectivity...")
    face_data = _build_face_connectivity(ugrid)

    data.face_owner = face_data["face_owner"]
    data.face_neighbor = face_data["face_neighbor"]
    data.face_area = face_data["face_area"]
    data.face_normal = face_data["face_normal"]
    data.face_centers = face_data["face_centers"]

    # Cell geometry
    mesh = pv.wrap(ugrid)
    data.cell_centers = np.array(mesh.cell_centers().points, dtype=np.float32)
    sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    data.cell_volumes = np.array(sized.cell_data["Volume"], dtype=np.float32)

    logger.info(
        f"[{data.metadata.filename}] FVM: {len(data.cell_centers)} cells, "
        f"{face_data['n_faces']} faces"
    )

    return data
