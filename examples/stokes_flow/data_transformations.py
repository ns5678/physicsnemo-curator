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
Data transformations for Stokes flow ETL pipeline.
"""

import logging
import warnings
from typing import Callable, Optional

import numpy as np
import zarr
from schemas import (
    PreparedZarrArrayInfo,
    StokesFlowExtractedDataInMemory,
    StokesFlowZarrDataInMemory,
)
from stokes_geometry_processors import default_geometry_processing_for_stokes_flow
from stokes_volume_processors import default_volume_processing_for_stokes_flow

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class StokesFlowSTLTransformation(DataTransformation):
    """Transforms STL geometry data for Stokes flow."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        geometry_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.geometry_processors = geometry_processors
        self.logger = logging.getLogger(__name__)

    def transform(
        self, data: StokesFlowExtractedDataInMemory
    ) -> StokesFlowExtractedDataInMemory:
        """Transform STL data.

        Always applies default geometry processing first, then any
        additional processors from config.

        Args:
            data: Stokes flow data with stl_polydata loaded

        Returns:
            Data with geometry arrays populated
        """
        # Regardless of whether there are any additional geometry processors,
        # we always apply the default geometry processing.
        # This will ensure that the bare minimum criteria for geometry data is met.
        # That is - The geometry data (vertices, faces, areas and centers) are present.
        data = default_geometry_processing_for_stokes_flow(data)

        # Apply any additional processors
        if self.geometry_processors is not None:
            for processor in self.geometry_processors:
                data = processor(data)

        # Release raw polydata to save memory
        data.stl_polydata = None

        return data


class StokesFlowVolumeTransformation(DataTransformation):
    """Transforms volume data for Stokes flow (point-centered)."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        volume_variables: Optional[dict[str, str]] = None,
        volume_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.volume_variables = volume_variables
        self.volume_processors = volume_processors
        self.logger = logging.getLogger(__name__)

        if volume_variables is None:
            self.logger.error("Volume variables are empty!")
            raise ValueError("Volume variables are empty!")

        self.logger.info(
            f"Initializing StokesFlowVolumeTransformation with "
            f"volume_variables: {volume_variables} and volume_processors: {volume_processors}"
        )
        self.logger.info("This will only be processed if the model_type is volume.")

    def transform(
        self, data: StokesFlowExtractedDataInMemory
    ) -> StokesFlowExtractedDataInMemory:
        """Transform volume data.

        Extracts point-centered fields directly from VTU point_data.

        Args:
            data: Stokes flow data with volume_polydata loaded

        Returns:
            Data with volume arrays populated
        """
        if data.volume_polydata is not None:
            # Regardless of whether there are any additional volume processors,
            # we always apply the default volume processing.
            # This will ensure that the bare minimum criteria for volume data is met.
            # That is - The volume data (mesh points and fields) are present.
            data = default_volume_processing_for_stokes_flow(
                data, self.volume_variables
            )

            # Apply any additional processors
            if self.volume_processors is not None:
                for processor in self.volume_processors:
                    data = processor(data)

            # Release raw polydata to save memory
            data.volume_polydata = None

        return data


class StokesFlowZarrTransformation(DataTransformation):
    """Transforms Stokes flow data for Zarr storage format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        compression_method: str = "zstd",
        compression_level: int = 5,
        chunk_size_mb: float = 1.0,
        chunks_per_shard: int = 1000,
    ):
        super().__init__(cfg)
        self.compressor = zarr.codecs.BloscCodec(
            cname=compression_method,
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )
        self.chunk_size_mb = chunk_size_mb
        self.chunks_per_shard = chunks_per_shard

        if chunk_size_mb < 1.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too small.",
                UserWarning,
            )

    def _prepare_array(self, array: np.ndarray) -> PreparedZarrArrayInfo:
        """Prepare array for Zarr storage with compression, chunking, sharding."""
        if array is None:
            return None

        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)
        item_size = array.itemsize
        shape = array.shape

        if len(shape) == 1:
            chunk_size = min(shape[0], target_chunk_size // item_size)
            chunks = (chunk_size,)
            ideal_shard_size = chunks[0] * self.chunks_per_shard
            if shape[0] <= ideal_shard_size:
                num_chunks = (shape[0] + chunks[0] - 1) // chunks[0]
                shard_size = num_chunks * chunks[0]
            else:
                shard_size = ideal_shard_size
            shards = (shard_size,)
        else:
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            chunks = (chunk_rows, shape[1])
            ideal_shard_rows = chunks[0] * self.chunks_per_shard
            if shape[0] <= ideal_shard_rows:
                num_chunks = (shape[0] + chunks[0] - 1) // chunks[0]
                shard_rows = num_chunks * chunks[0]
            else:
                shard_rows = ideal_shard_rows
            shards = (shard_rows, shape[1])

        return PreparedZarrArrayInfo(
            data=np.float32(array),
            chunks=chunks,
            compressor=self.compressor,
            shards=shards,
        )

    def _prepare_int_array(self, array: np.ndarray) -> PreparedZarrArrayInfo:
        """Prepare integer array (faces) for Zarr storage."""
        if array is None:
            return None

        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)
        item_size = array.itemsize
        chunk_size = min(len(array), target_chunk_size // item_size)
        chunks = (chunk_size,)

        ideal_shard_size = chunks[0] * self.chunks_per_shard
        if len(array) <= ideal_shard_size:
            num_chunks = (len(array) + chunks[0] - 1) // chunks[0]
            shard_size = num_chunks * chunks[0]
        else:
            shard_size = ideal_shard_size
        shards = (shard_size,)

        return PreparedZarrArrayInfo(
            data=np.int32(array),
            chunks=chunks,
            compressor=self.compressor,
            shards=shards,
        )

    def transform(
        self, data: StokesFlowExtractedDataInMemory
    ) -> StokesFlowZarrDataInMemory:
        """Transform data for Zarr storage format.

        Args:
            data: Stokes flow extracted data

        Returns:
            Data prepared for Zarr storage
        """
        return StokesFlowZarrDataInMemory(
            metadata=data.metadata,
            # STL geometry
            stl_coordinates=self._prepare_array(data.stl_coordinates),
            stl_centers=self._prepare_array(data.stl_centers),
            stl_faces=self._prepare_int_array(data.stl_faces),
            stl_areas=self._prepare_array(data.stl_areas),
            # Volume data (point-centered)
            volume_mesh_points=self._prepare_array(data.volume_mesh_points),
            volume_fields=self._prepare_array(data.volume_fields),
            # FVM connectivity
            cell_centers=self._prepare_array(data.cell_centers),
            cell_volumes=self._prepare_array(data.cell_volumes),
            face_owner=self._prepare_int_array(data.face_owner),
            face_neighbor=self._prepare_int_array(data.face_neighbor),
            face_area=self._prepare_array(data.face_area),
            face_normal=self._prepare_array(data.face_normal),
            face_centers=self._prepare_array(data.face_centers),
        )
