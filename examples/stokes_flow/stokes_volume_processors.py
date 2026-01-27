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

Volume data is cell-centered: we extract cell centers and convert point_data
to cell_data using averaging interpolation.
"""

import logging
import time

import numpy as np
from numba import njit
from schemas import StokesFlowExtractedDataInMemory
from tqdm import tqdm

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

    Extracts cell-centered volume data from VTU:
    - volume_mesh_points: cell center coordinates (N_cells, 3)
    - volume_fields: field values at cell centers (N_cells, N_fields)

    Point data is interpolated to cell centers using averaging.

    Args:
        data: Stokes flow data with volume_polydata loaded
        volume_variables: Dict mapping field names to types (e.g., {'u': 'scalar', ...})
            Iterating yields field names.

    Returns:
        Data with volume arrays populated
    """
    mesh = data.volume_polydata

    # Extract cell center coordinates
    data.volume_mesh_points = np.array(mesh.cell_centers().points, dtype=np.float32)

    # Convert point data to cell data (averaging interpolation)
    mesh_cell = mesh.point_data_to_cell_data()

    # Extract cell data fields
    fields = []
    for var_name in volume_variables:
        if var_name not in mesh_cell.cell_data:
            raise KeyError(
                f"Field '{var_name}' not found in cell_data after conversion. "
                f"Available fields: {list(mesh_cell.cell_data.keys())}"
            )
        field = np.array(mesh_cell.cell_data[var_name])
        # Ensure 2D shape (N_cells, 1) for scalar fields
        if field.ndim == 1:
            field = field[:, np.newaxis]
        fields.append(field)

    # Concatenate all fields: (N_cells, N_fields)
    data.volume_fields = np.concatenate(fields, axis=-1).astype(np.float32)

    # Update metadata
    data.metadata.num_volume_cells = len(data.volume_mesh_points)

    var_names = list(volume_variables.keys())
    logger.info(
        f"[{data.metadata.filename}] Volume data: "
        f"{data.metadata.num_volume_cells} cells, "
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


def filter_invalid_cells(
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Filter out cells with NaN or inf values.

    Args:
        data: Stokes flow data with volume arrays

    Returns:
        Data with invalid cells removed
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
        logger.info(f"[{data.metadata.filename}] All {n_total} cells are valid")
        return data

    if n_valid == 0:
        raise ValueError(
            f"[{data.metadata.filename}] All {n_total} cells filtered out!"
        )

    logger.warning(
        f"[{data.metadata.filename}] Filtered {n_filtered}/{n_total} invalid cells"
    )

    data.volume_mesh_points = data.volume_mesh_points[valid_mask]
    data.volume_fields = data.volume_fields[valid_mask]
    data.metadata.num_volume_cells = len(data.volume_mesh_points)

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
    import vtk
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
    data: StokesFlowExtractedDataInMemory,
) -> StokesFlowExtractedDataInMemory:
    """Compute FVM face connectivity from volume mesh.

    Builds face-based connectivity arrays needed for FVM residual kernels.
    Must be called before volume_polydata is released.

    Args:
        data: Stokes flow data with volume_polydata loaded

    Returns:
        Data with FVM connectivity arrays populated
    """
    mesh = data.volume_polydata
    if mesh is None:
        logger.warning("No volume_polydata available for FVM connectivity")
        return data

    # Build face connectivity (expensive VTK traversal)
    logger.info(f"[{data.metadata.filename}] Building FVM face connectivity...")
    face_data = _build_face_connectivity(mesh)

    data.face_owner = face_data["face_owner"]
    data.face_neighbor = face_data["face_neighbor"]
    data.face_area = face_data["face_area"]
    data.face_normal = face_data["face_normal"]
    data.face_centers = face_data["face_centers"]

    # Cell geometry
    data.cell_centers = np.array(mesh.cell_centers().points, dtype=np.float32)
    sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    data.cell_volumes = np.array(sized.cell_data["Volume"], dtype=np.float32)

    logger.info(
        f"[{data.metadata.filename}] FVM: {len(data.cell_centers)} cells, "
        f"{face_data['n_faces']} faces"
    )

    return data
