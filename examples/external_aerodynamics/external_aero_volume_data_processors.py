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
from typing import Optional

import numpy as np
import vtk
from constants import (
    PhysicsConstantsCarAerodynamics,
    PhysicsConstantsHLPW,
)
from external_aero_utils import get_volume_data, to_float32
from external_aero_validation_utils import (
    check_field_statistics,
    check_volume_physics_bounds,
)
from numba import njit
from schemas import ExternalAerodynamicsExtractedDataInMemory
from tqdm import tqdm

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


@njit(fastmath=True, cache=True)
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
        if area < 1e-12:
            continue

        face_owner.append(owner)
        face_neighbor.append(neighbor)
        face_area.append(area)
        face_normal.append(normal)
        face_center_list.append(fc)

    logger.info(f"  {len(face_owner):,} faces built")

    return {
        "n_faces": len(face_owner),
        "face_owner": np.array(face_owner, dtype=np.int32),
        "face_neighbor": np.array(face_neighbor, dtype=np.int32),
        "face_area": np.array(face_area, dtype=np.float32),
        "face_normal": np.array(face_normal, dtype=np.float32),
        "face_centers": np.array(face_center_list, dtype=np.float32),
    }


def _build_face_connectivity_optimized(ugrid):
    """Build face connectivity from VTK unstructured grid (optimized version).

    TODO: Optimize the face extraction loop (lines 330-348 equivalent).
    """
    from vtk.util import numpy_support

    logger.info("Building face connectivity (optimized)...")

    n_cells = ugrid.GetNumberOfCells()
    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)

    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ugrid)
    cell_centers_filter.Update()
    cell_centers = numpy_support.vtk_to_numpy(
        cell_centers_filter.GetOutput().GetPoints().GetData()
    ).astype(np.float64)

    face_to_cells = {}
    for cell_idx in range(n_cells):
        cell = ugrid.GetCell(cell_idx)
        n_faces = cell.GetNumberOfFaces()
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids = face.GetPointIds()
            n_pts = face_point_ids.GetNumberOfIds()

            # Fetch point IDs once (was: two separate list comprehensions)
            face_pts_arr = np.array(
                [face_point_ids.GetId(i) for i in range(n_pts)], dtype=np.int64
            )
            face_pts_key = tuple(np.sort(face_pts_arr))

            if face_pts_key not in face_to_cells:
                face_to_cells[face_pts_key] = []
            face_to_cells[face_pts_key].append((cell_idx, face_pts_arr))

    face_owner = []
    face_neighbor = []
    face_area = []
    face_normal = []
    face_center_list = []

    for cell_list in face_to_cells.values():
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

    logger.info(f"  {len(face_owner):,} faces built")

    return {
        "n_faces": len(face_owner),
        "face_owner": np.array(face_owner, dtype=np.int32),
        "face_neighbor": np.array(face_neighbor, dtype=np.int32),
        "face_area": np.array(face_area, dtype=np.float32),
        "face_normal": np.array(face_normal, dtype=np.float32),
        "face_centers": np.array(face_center_list, dtype=np.float32),
    }


def _build_adjacency_vtk(ugrid):
    """Build cell adjacency graph via VTK face sharing.

    Two cells are adjacent if they share a face (exact, no heuristic).
    Uses the same GetCell/GetFace loop as partition_mesh.py but skips
    geometry computation — only builds the face-to-cell mapping needed
    for adjacency.

    Args:
        ugrid: vtkUnstructuredGrid

    Returns:
        adjacency: list of lists, adjacency[i] = neighbor cell IDs for cell i
    """
    n_cells = ugrid.GetNumberOfCells()

    # Map each unique face (sorted point IDs) to the cells that share it
    face_to_cells = {}
    for cell_id in tqdm(range(n_cells), desc="  Building adjacency", unit="cells"):
        cell = ugrid.GetCell(cell_id)
        n_faces = cell.GetNumberOfFaces()
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids = face.GetPointIds()
            n_pts = face_point_ids.GetNumberOfIds()
            face_key = tuple(sorted(face_point_ids.GetId(i) for i in range(n_pts)))
            if face_key not in face_to_cells:
                face_to_cells[face_key] = []
            face_to_cells[face_key].append(cell_id)

    # Derive adjacency: two cells sharing a face are neighbors
    adjacency = [[] for _ in range(n_cells)]
    n_internal = 0
    n_boundary = 0
    for cell_list in face_to_cells.values():
        if len(cell_list) == 2:
            a, b = cell_list
            adjacency[a].append(b)
            adjacency[b].append(a)
            n_internal += 1
        elif len(cell_list) == 1:
            n_boundary += 1

    logger.info(
        f"  VTK adjacency: {n_cells:,} cells, "
        f"{n_internal:,} internal faces, {n_boundary:,} boundary faces"
    )

    return adjacency


def _process_partition(sub_mesh):
    """Build face connectivity and geometry for a single partition sub-mesh.

    Intended to run inside a joblib worker process. Reuses
    _build_face_connectivity_optimized on the extracted sub-mesh.

    Args:
        sub_mesh: PyVista UnstructuredGrid from mesh.extract_cells().
            Expected cell_data keys:
                "is_halo"        — int8 array (0=owned, 1=halo)
                "n_owned"        — int32 array (constant, n_owned cells)
                "volume_fields"  — float32 array (n_cells, n_vars)

    Returns:
        VolumePartitionData or None if sub-mesh is empty.
    """
    from schemas import VolumePartitionData

    n_cells = sub_mesh.n_cells
    if n_cells == 0:
        return None

    n_owned = int(sub_mesh.cell_data["n_owned"][0])
    is_halo = sub_mesh.cell_data["is_halo"]
    volume_fields = sub_mesh.cell_data["volume_fields"]

    # Reuse existing optimized face connectivity builder
    face_data = _build_face_connectivity_optimized(sub_mesh)

    # Cell centers (fast VTK filter on small sub-mesh)
    cell_centers = np.array(sub_mesh.cell_centers().points, dtype=np.float32)

    return VolumePartitionData(
        cell_centers=cell_centers,
        cell_fields=volume_fields,
        is_halo=is_halo,
        n_owned_cells=n_owned,
        face_owner=face_data["face_owner"],
        face_neighbor=face_data["face_neighbor"],
        face_area=face_data["face_area"],
        face_normal=face_data["face_normal"],
        face_centers=face_data["face_centers"],
        cell_volumes=None,
    )


def compute_fvm_and_partition_parallel(
    data: ExternalAerodynamicsExtractedDataInMemory,
    num_partitions: int = 100,
    halo_depth: int = 1,
    n_jobs: int = -1,
    batch_size: int = 32,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Partition mesh via METIS, then build face connectivity per partition in parallel.

    Replaces the serial compute_fvm_connectivity + partition_volume_mesh pipeline.
    Face connectivity is never computed globally — it is built per partition
    inside parallel workers, reducing both wall-clock time and peak memory.

    Architecture:
        Phase 1 (serial): VTK adjacency → METIS → halo expansion
        Phase 2 (batched parallel): extract batch_size sub-meshes from parent,
            dispatch to joblib workers for face connectivity, free batch, repeat
        Phase 3 (serial): cleanup, store results on data object

    Memory strategy: sub-meshes are extracted in batches of batch_size from the
    parent mesh (which stays alive until all batches are done). Each batch's
    sub-meshes are freed after processing. This keeps peak memory at
    parent + batch_size sub-meshes rather than parent + num_partitions sub-meshes.
    See docs/process_parallelization_design.md for details.

    Args:
        data: data object with volume_unstructured_grid and volume_fields
        num_partitions: number of METIS partitions
        halo_depth: ghost cell layers (1 for first-order FVM)
        n_jobs: parallel workers (-1 = all CPUs)
        batch_size: sub-meshes extracted and dispatched per joblib call.
            Controls peak memory. Default 32 (2x a typical n_jobs=16).

    Returns:
        data with volume_partitions populated, volume_unstructured_grid set to None
    """
    import time as _time

    import pymetis
    import pyvista as pv
    from joblib import Parallel, delayed

    ugrid = data.volume_unstructured_grid
    if ugrid is None:
        raise ValueError("volume_unstructured_grid is None. Cannot partition.")

    n_cells = ugrid.GetNumberOfCells()
    logger.info(
        f"[{data.metadata.filename}] compute_fvm_and_partition_parallel: "
        f"{n_cells:,} cells → {num_partitions} partitions, "
        f"halo_depth={halo_depth}, n_jobs={n_jobs}, batch_size={batch_size}"
    )

    # === Phase 1: Serial setup ===

    # Step 1-2: Build adjacency via VTK face sharing (exact)
    _t0 = _time.perf_counter()
    logger.info("  Phase 1: Building adjacency (VTK face loop)...")
    adjacency = _build_adjacency_vtk(ugrid)
    _t1 = _time.perf_counter()
    logger.info(f"  [timer] adjacency: {_t1 - _t0:.2f}s")

    # Step 3: METIS partitioning
    logger.info("  Phase 1: METIS partitioning...")
    adjacency_copy = [list(nbrs) for nbrs in adjacency]  # pymetis mutates input
    n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency_copy)
    del adjacency_copy  # consumed by pymetis
    _t2 = _time.perf_counter()
    logger.info(f"  METIS: {n_cuts:,} edge cuts")
    logger.info(f"  [timer] METIS: {_t2 - _t1:.2f}s")

    # Step 4: Identify interface cells and BFS halo expansion
    logger.info("  Phase 1: Halo expansion...")
    interface_cells = [set() for _ in range(num_partitions)]
    for cell_id in range(n_cells):
        for nbr in adjacency[cell_id]:
            if membership[cell_id] != membership[nbr]:
                interface_cells[membership[cell_id]].add(cell_id)
                break  # cell is interface, no need to check more neighbors

    partition_cells = []  # list of (owned_list, halo_list)
    for part_id in range(num_partitions):
        owned = {i for i, p in enumerate(membership) if p == part_id}
        if len(owned) == 0:
            partition_cells.append(([], []))
            continue

        halo = set()
        frontier = interface_cells[part_id]

        for _ in range(halo_depth):
            next_frontier = set()
            for cell_id in frontier:
                for nbr in adjacency[cell_id]:
                    if nbr not in owned and nbr not in halo:
                        halo.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        partition_cells.append((sorted(owned), sorted(halo)))

    _t3 = _time.perf_counter()
    logger.info(f"  [timer] halo expansion: {_t3 - _t2:.2f}s")

    # Free Phase 1 intermediates — only partition_cells needed from here
    del adjacency, interface_cells, membership

    # Warm up Numba cache in main process before worker dispatch
    dummy_pts = np.array([0, 1, 2], dtype=np.int64)
    dummy_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    dummy_center = np.array([0.33, 0.33, 0.0], dtype=np.float64)
    _compute_face_geometry_numba(dummy_pts, dummy_points, dummy_center)

    # === Phase 2: Batched extraction + parallel face connectivity ===
    # Wrap parent mesh once. Attach volume_fields so extract_cells auto-slices it.
    mesh = pv.wrap(ugrid)
    mesh.cell_data["volume_fields"] = data.volume_fields

    n_batches = (num_partitions + batch_size - 1) // batch_size
    logger.info(
        f"  Phase 2: {num_partitions} partitions in {n_batches} batches "
        f"(batch_size={batch_size}, n_jobs={n_jobs})"
    )

    partitions = []
    _t_phase2_start = _time.perf_counter()

    for batch_idx, batch_start in enumerate(range(0, num_partitions, batch_size)):
        batch_end = min(batch_start + batch_size, num_partitions)
        _tb0 = _time.perf_counter()

        # Step A: Extract this batch's sub-meshes from parent
        batch_sub_meshes = []
        batch_part_ids = []
        for part_id in range(batch_start, batch_end):
            owned, halo = partition_cells[part_id]
            all_cells = owned + halo
            if len(all_cells) == 0:
                continue

            n_owned = len(owned)
            sub_mesh = mesh.extract_cells(all_cells)

            is_halo = np.zeros(len(all_cells), dtype=np.int8)
            is_halo[n_owned:] = 1
            sub_mesh.cell_data["is_halo"] = is_halo
            sub_mesh.cell_data["n_owned"] = np.full(
                len(all_cells), n_owned, dtype=np.int32
            )

            batch_sub_meshes.append(sub_mesh)
            batch_part_ids.append(part_id)

        if len(batch_sub_meshes) == 0:
            continue

        # Step B: Dispatch to joblib workers
        batch_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_process_partition)(sm) for sm in batch_sub_meshes
        )

        # Step C: Collect results, free this batch's sub-meshes
        for part_id, result in zip(batch_part_ids, batch_results):
            if result is not None:
                partitions.append(result)
                logger.info(
                    f"  Partition {part_id}: "
                    f"{result.n_owned_cells} owned + "
                    f"{len(result.is_halo) - result.n_owned_cells} halo cells, "
                    f"{len(result.face_owner)} faces"
                )

        del batch_sub_meshes
        del batch_results

        _tb1 = _time.perf_counter()
        logger.info(
            f"  [timer] batch {batch_idx + 1}/{n_batches} "
            f"(partitions {batch_start}-{batch_end - 1}): {_tb1 - _tb0:.2f}s"
        )

    _t_phase2_end = _time.perf_counter()
    logger.info(f"  [timer] Phase 2 total: {_t_phase2_end - _t_phase2_start:.2f}s")
    logger.info(f"  {len(partitions)} partitions built")

    # === Phase 3: Cleanup ===
    del mesh
    del partition_cells
    data.volume_partitions = partitions
    data.volume_unstructured_grid = None
    data.volume_fields = None

    return data


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
    import time as _time

    import pyvista as pv

    ugrid = data.volume_unstructured_grid
    if ugrid is None:
        logger.warning("No volume_unstructured_grid available for FVM connectivity")
        return data

    # Build face connectivity (expensive VTK traversal)
    _t0 = _time.perf_counter()
    logger.info(f"[{data.metadata.filename}] Building FVM face connectivity...")
    face_data = _build_face_connectivity_optimized(ugrid)
    _t1 = _time.perf_counter()
    logger.info(f"  [timer] face connectivity + geometry: {_t1 - _t0:.2f}s")

    data.face_owner = face_data["face_owner"]
    data.face_neighbor = face_data["face_neighbor"]
    data.face_area = face_data["face_area"]
    data.face_normal = face_data["face_normal"]
    data.face_centers = face_data["face_centers"]

    # Cell geometry
    mesh = pv.wrap(ugrid)
    data.cell_centers = np.array(mesh.cell_centers().points, dtype=np.float32)
    _t2 = _time.perf_counter()
    logger.info(f"  [timer] cell centers: {_t2 - _t1:.2f}s")

    # Cell volumes: only compute if needed (expensive for large meshes)
    compute_cell_volumes = False  # Set True for unsteady/source term problems
    if compute_cell_volumes:
        sized = mesh.compute_cell_sizes(
            length=False, area=False, vertex_count=False, volume=True
        )
        data.cell_volumes = np.array(sized.cell_data["Volume"], dtype=np.float32)
    else:
        data.cell_volumes = None

    logger.info(
        f"[{data.metadata.filename}] FVM: {len(data.cell_centers)} cells, "
        f"{face_data['n_faces']} faces"
    )

    return data


# ---------------------------------------------------------------------------
# METIS mesh partitioning
# ---------------------------------------------------------------------------


def _build_single_partition(
    part_id: int,
    membership: list,
    adjacency: list,
    interface_cells_for_part: set,
    face_candidates: list,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    face_area: np.ndarray,
    face_normal: np.ndarray,
    face_centers: np.ndarray,
    cell_centers: np.ndarray,
    volume_fields: np.ndarray,
    cell_volumes: np.ndarray | None,
    halo_depth: int,
):
    """Build a single partition (Steps 4-8). Used by parallel version."""
    from schemas import VolumePartitionData

    # Step 4: BFS halo expansion from interface cells
    owned = {i for i, p in enumerate(membership) if p == part_id}

    if len(owned) == 0:
        return None  # Empty partition

    halo = set()
    frontier = interface_cells_for_part

    for _ in range(halo_depth):
        next_frontier = set()
        for cell_id in frontier:
            for nbr in adjacency[cell_id]:
                if nbr not in owned and nbr not in halo:
                    halo.add(nbr)
                    next_frontier.add(nbr)
        frontier = next_frontier

    # Step 5: Order cells — owned first, then halo
    owned_list = sorted(owned)
    halo_list = sorted(halo)
    all_cells = owned_list + halo_list
    n_owned = len(owned_list)

    # Step 6: Global-to-local cell index map
    global_to_local = {g: loc for loc, g in enumerate(all_cells)}
    cell_set = set(all_cells)

    # Step 7: Filter and remap faces
    local_face_owner = []
    local_face_neighbor = []
    local_face_area = []
    local_face_normal = []
    local_face_centers = []

    for face_idx in face_candidates:
        o = face_owner[face_idx]
        n = face_neighbor[face_idx]

        if n == -1:
            if o in cell_set:
                local_face_owner.append(global_to_local[o])
                local_face_neighbor.append(-1)
                local_face_area.append(face_area[face_idx])
                local_face_normal.append(face_normal[face_idx])
                local_face_centers.append(face_centers[face_idx])
        else:
            if o in cell_set and n in cell_set:
                local_face_owner.append(global_to_local[o])
                local_face_neighbor.append(global_to_local[n])
                local_face_area.append(face_area[face_idx])
                local_face_normal.append(face_normal[face_idx])
                local_face_centers.append(face_centers[face_idx])

    # Step 8: Build partition data
    all_cells_arr = np.array(all_cells, dtype=np.int64)

    is_halo = np.zeros(len(all_cells), dtype=np.int8)
    is_halo[n_owned:] = 1

    partition = VolumePartitionData(
        cell_centers=cell_centers[all_cells_arr],
        cell_fields=volume_fields[all_cells_arr],
        is_halo=is_halo,
        n_owned_cells=n_owned,
        face_owner=np.array(local_face_owner, dtype=np.int32),
        face_neighbor=np.array(local_face_neighbor, dtype=np.int32),
        face_area=np.array(local_face_area, dtype=np.float32),
        face_normal=(
            np.vstack(local_face_normal).astype(np.float32)
            if local_face_normal
            else np.empty((0, 3), dtype=np.float32)
        ),
        face_centers=(
            np.vstack(local_face_centers).astype(np.float32)
            if local_face_centers
            else np.empty((0, 3), dtype=np.float32)
        ),
        cell_volumes=cell_volumes[all_cells_arr] if cell_volumes is not None else None,
    )

    n_halo = len(halo_list)
    n_faces = len(local_face_owner)
    return (part_id, partition, n_owned, n_halo, n_faces)


def partition_volume_mesh_parallel(
    data: ExternalAerodynamicsExtractedDataInMemory,
    num_partitions: int = 100,
    halo_depth: int = 1,
    n_jobs: int = -1,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Parallel version of partition_volume_mesh using joblib.

    Args:
        data: data object with face connectivity arrays populated
        num_partitions: number of METIS partitions
        halo_depth: number of ghost cell layers (1 for first-order FVM)
        n_jobs: number of parallel jobs (-1 = all CPUs)

    Returns:
        data with volume_partitions populated
    """
    import time as _time
    from collections import defaultdict

    import pymetis
    from joblib import Parallel, delayed

    if data.face_owner is None or data.face_neighbor is None:
        raise ValueError(
            "Face connectivity not found on data object. "
            "Run compute_fvm_connectivity before partition_volume_mesh."
        )

    n_cells = len(data.cell_centers)
    face_owner = data.face_owner
    face_neighbor = data.face_neighbor

    logger.info(
        f"[{data.metadata.filename}] Partitioning {n_cells} cells into "
        f"{num_partitions} partitions with halo_depth={halo_depth} (parallel, n_jobs={n_jobs})"
    )

    _t0 = _time.perf_counter()

    # Step 1: Derive cell adjacency from face_owner / face_neighbor
    adjacency = [[] for _ in range(n_cells)]
    internal_mask = face_neighbor >= 0
    for o, n in zip(face_owner[internal_mask], face_neighbor[internal_mask]):
        adjacency[o].append(n)
        adjacency[n].append(o)

    # Step 2: METIS partitioning
    adjacency_copy = [list(nbrs) for nbrs in adjacency]
    n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency_copy)
    logger.info(f"  METIS: {n_cuts} edge cuts")

    # Step 3: Identify interface cells per partition
    interface_cells = [set() for _ in range(num_partitions)]
    for face_idx in range(len(face_owner)):
        o = face_owner[face_idx]
        n = face_neighbor[face_idx]
        if n < 0:
            continue
        if membership[o] != membership[n]:
            interface_cells[membership[o]].add(o)
            interface_cells[membership[n]].add(n)

    # Step 3b: Build face-to-partition candidate index
    partition_face_candidates = defaultdict(list)
    for face_idx in range(len(face_owner)):
        o = face_owner[face_idx]
        n = face_neighbor[face_idx]
        if n == -1:
            partition_face_candidates[membership[o]].append(face_idx)
        else:
            partition_face_candidates[membership[o]].append(face_idx)
            if membership[n] != membership[o]:
                partition_face_candidates[membership[n]].append(face_idx)

    _t1 = _time.perf_counter()
    logger.info(f"  [timer] METIS + partition setup: {_t1 - _t0:.2f}s")

    # Steps 4-8: Build partitions in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_build_single_partition)(
            part_id=part_id,
            membership=membership,
            adjacency=adjacency,
            interface_cells_for_part=interface_cells[part_id],
            face_candidates=partition_face_candidates[part_id],
            face_owner=face_owner,
            face_neighbor=face_neighbor,
            face_area=data.face_area,
            face_normal=data.face_normal,
            face_centers=data.face_centers,
            cell_centers=data.cell_centers,
            volume_fields=data.volume_fields,
            cell_volumes=data.cell_volumes,
            halo_depth=halo_depth,
        )
        for part_id in range(num_partitions)
    )
    _t2 = _time.perf_counter()
    logger.info(f"  [timer] parallel partition work: {_t2 - _t1:.2f}s")

    # Collect results, filter out empty partitions
    partitions = []
    for result in results:
        if result is not None:
            part_id, partition, n_owned, n_halo, n_faces = result
            partitions.append(partition)
            logger.info(
                f"  Partition {part_id}: "
                f"{n_owned} owned + {n_halo} halo cells, {n_faces} faces"
            )

    logger.info(f"  {len(partitions)} partitions built")

    # Step 9: Store partitions, release intermediate arrays
    data.volume_partitions = partitions
    data.face_owner = None
    data.face_neighbor = None
    data.face_area = None
    data.face_normal = None
    data.face_centers = None
    data.cell_centers = None
    data.cell_volumes = None

    return data


def partition_volume_mesh(
    data: ExternalAerodynamicsExtractedDataInMemory,
    num_partitions: int = 100,
    halo_depth: int = 1,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Partition volume mesh into METIS subdomains with halo cells.

    Requires compute_fvm_connectivity to have run first.
    Replaces shuffle_volume_data in the processor chain.

    Each partition is a self-contained VolumePartitionData with
    partition-local face indices. Owned cells are ordered first,
    then halo cells.

    Args:
        data: data object with face connectivity arrays populated
        num_partitions: number of METIS partitions
        halo_depth: number of ghost cell layers (1 for first-order FVM)

    Returns:
        data with volume_partitions populated, intermediate face/cell
        connectivity arrays set to None
    """
    from collections import defaultdict

    import pymetis
    from schemas import VolumePartitionData

    if data.face_owner is None or data.face_neighbor is None:
        raise ValueError(
            "Face connectivity not found on data object. "
            "Run compute_fvm_connectivity before partition_volume_mesh."
        )

    n_cells = len(data.cell_centers)
    face_owner = data.face_owner
    face_neighbor = data.face_neighbor

    logger.info(
        f"[{data.metadata.filename}] Partitioning {n_cells} cells into "
        f"{num_partitions} partitions with halo_depth={halo_depth}"
    )

    # Step 1: Derive cell adjacency from face_owner / face_neighbor
    adjacency = [[] for _ in range(n_cells)]
    internal_mask = face_neighbor >= 0
    for o, n in zip(face_owner[internal_mask], face_neighbor[internal_mask]):
        adjacency[o].append(n)
        adjacency[n].append(o)

    # Step 2: METIS partitioning
    adjacency_copy = [list(nbrs) for nbrs in adjacency]  # pymetis mutates input
    n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency_copy)
    logger.info(f"  METIS: {n_cuts} edge cuts")

    # Step 3: Identify interface cells per partition
    interface_cells = [set() for _ in range(num_partitions)]
    for face_idx in range(len(face_owner)):
        o = face_owner[face_idx]
        n = face_neighbor[face_idx]
        if n < 0:
            continue
        if membership[o] != membership[n]:
            interface_cells[membership[o]].add(o)
            interface_cells[membership[n]].add(n)

    # Step 3b: Build face-to-partition candidate index
    partition_face_candidates = defaultdict(list)
    for face_idx in range(len(face_owner)):
        o = face_owner[face_idx]
        n = face_neighbor[face_idx]
        if n == -1:
            partition_face_candidates[membership[o]].append(face_idx)
        else:
            partition_face_candidates[membership[o]].append(face_idx)
            if membership[n] != membership[o]:
                partition_face_candidates[membership[n]].append(face_idx)

    # Steps 4-8: Build each partition
    partitions = []

    for part_id in range(num_partitions):
        # Step 4: BFS halo expansion from interface cells
        owned = {i for i, p in enumerate(membership) if p == part_id}

        if len(owned) == 0:
            logger.warning(f"  Partition {part_id} is empty, skipping")
            continue

        halo = set()
        frontier = interface_cells[part_id]

        for layer in range(halo_depth):
            next_frontier = set()
            for cell_id in frontier:
                for nbr in adjacency[cell_id]:
                    if nbr not in owned and nbr not in halo:
                        halo.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        # Step 5: Order cells — owned first, then halo
        owned_list = sorted(owned)
        halo_list = sorted(halo)
        all_cells = owned_list + halo_list
        n_owned = len(owned_list)

        # Step 6: Global-to-local cell index map
        global_to_local = {g: loc for loc, g in enumerate(all_cells)}
        cell_set = set(all_cells)

        # Step 7: Filter and remap faces using precomputed index
        local_face_owner = []
        local_face_neighbor = []
        local_face_area = []
        local_face_normal = []
        local_face_centers = []

        for face_idx in partition_face_candidates[part_id]:
            o = face_owner[face_idx]
            n = face_neighbor[face_idx]

            if n == -1:
                if o in cell_set:
                    local_face_owner.append(global_to_local[o])
                    local_face_neighbor.append(-1)
                    local_face_area.append(data.face_area[face_idx])
                    local_face_normal.append(data.face_normal[face_idx])
                    local_face_centers.append(data.face_centers[face_idx])
            else:
                if o in cell_set and n in cell_set:
                    local_face_owner.append(global_to_local[o])
                    local_face_neighbor.append(global_to_local[n])
                    local_face_area.append(data.face_area[face_idx])
                    local_face_normal.append(data.face_normal[face_idx])
                    local_face_centers.append(data.face_centers[face_idx])

        # Step 8: Build partition data
        all_cells_arr = np.array(all_cells, dtype=np.int64)

        is_halo = np.zeros(len(all_cells), dtype=np.int8)
        is_halo[n_owned:] = 1

        partition = VolumePartitionData(
            cell_centers=data.cell_centers[all_cells_arr],
            cell_fields=data.volume_fields[all_cells_arr],
            cell_volumes=data.cell_volumes[all_cells_arr]
            if data.cell_volumes is not None
            else None,
            is_halo=is_halo,
            n_owned_cells=n_owned,
            face_owner=np.array(local_face_owner, dtype=np.int32),
            face_neighbor=np.array(local_face_neighbor, dtype=np.int32),
            face_area=np.array(local_face_area, dtype=np.float32),
            face_normal=(
                np.vstack(local_face_normal).astype(np.float32)
                if local_face_normal
                else np.empty((0, 3), dtype=np.float32)
            ),
            face_centers=(
                np.vstack(local_face_centers).astype(np.float32)
                if local_face_centers
                else np.empty((0, 3), dtype=np.float32)
            ),
        )
        partitions.append(partition)

        n_halo = len(halo_list)
        n_faces = len(local_face_owner)
        logger.info(
            f"  Partition {part_id}: "
            f"{n_owned} owned + {n_halo} halo cells, {n_faces} faces"
        )

    logger.info(f"  {len(partitions)} partitions built")

    # Step 9: Store partitions, release intermediate arrays
    data.volume_partitions = partitions
    data.face_owner = None
    data.face_neighbor = None
    data.face_area = None
    data.face_normal = None
    data.face_centers = None
    data.cell_centers = None
    data.cell_volumes = None

    return data
