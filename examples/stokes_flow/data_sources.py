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
Data source for reading and writing Stokes flow simulation data.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pyvista as pv
import zarr
from constants import DatasetKind, ModelType, get_physics_constants
from paths import get_path_getter
from schemas import (
    StokesFlowExtractedDataInMemory,
    StokesFlowMetadata,
    StokesFlowZarrDataInMemory,
)
from zarr.storage import LocalStore

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class StokesFlowDataSource(DataSource):
    """Data source for reading and writing Stokes flow simulation data."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        kind: DatasetKind | str = DatasetKind.STOKES,
        model_type: Optional[ModelType | str] = ModelType.VOLUME,
        serialization_method: str = "zarr",
        overwrite_existing: bool = True,
    ):
        super().__init__(cfg)

        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.kind = DatasetKind(kind.lower()) if isinstance(kind, str) else kind
        self.model_type = (
            ModelType(model_type.lower()) if isinstance(model_type, str) else model_type
        )
        self.serialization_method = serialization_method
        self.overwrite_existing = overwrite_existing

        # Validate directories
        if self.input_dir and not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.path_getter = get_path_getter(kind)

    def get_file_list(self) -> list[str]:
        """Get list of simulation directories to process.

        Returns:
            List of directory names (e.g., ['res_0', 'res_1', ...])
        """
        return sorted(d.name for d in self.input_dir.iterdir() if d.is_dir())

    def read_file(self, dirname: str) -> StokesFlowExtractedDataInMemory:
        """Read Stokes flow simulation data from a directory.

        Args:
            dirname: Name of the simulation directory (e.g., 'res_0')

        Returns:
            StokesFlowExtractedDataInMemory containing raw simulation data

        Raises:
            FileNotFoundError: If STL or VTU file is not found.
        """
        sim_dir = self.input_dir / dirname

        # Load STL geometry
        stl_path = self.path_getter.geometry_path(sim_dir)
        if not stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        stl_polydata = pv.read(str(stl_path))

        # Load volume data
        volume_polydata = None
        if self.model_type == ModelType.VOLUME:
            volume_path = self.path_getter.volume_path(sim_dir)
            if not volume_path.exists():
                raise FileNotFoundError(f"Volume data file not found: {volume_path}")

            volume_polydata = pv.read(str(volume_path))

        metadata = StokesFlowMetadata(
            filename=dirname,
            dataset_type=self.model_type,
            physics_constants=get_physics_constants(self.kind),
        )

        return StokesFlowExtractedDataInMemory(
            stl_polydata=stl_polydata,
            volume_polydata=volume_polydata,
            metadata=metadata,
        )

    def _get_output_path(self, filename: str) -> Path:
        """Get the final output path for a given filename.

        Args:
            filename: Name of the simulation case

        Returns:
            Path to the output file/directory
        """
        if self.serialization_method == "zarr":
            return self.output_dir / f"{filename}.zarr"
        else:
            raise ValueError(
                f"Unsupported serialization method: {self.serialization_method}"
            )

    def _write_impl_temp_file(
        self,
        data: StokesFlowZarrDataInMemory,
        output_path: Path,
    ) -> None:
        """Write transformed data to the specified output path.

        Args:
            data: Transformed data to write
            output_path: Path where data should be written
        """
        if self.serialization_method == "zarr":
            if not isinstance(data, StokesFlowZarrDataInMemory):
                raise TypeError(
                    "Expected StokesFlowZarrDataInMemory for zarr serialization"
                )
            self._write_zarr(data, output_path)
        else:
            raise ValueError(
                f"Unsupported serialization method: {self.serialization_method}"
            )

    def _write_zarr(self, data: StokesFlowZarrDataInMemory, output_path: Path) -> None:
        """Write data in Zarr format.

        Args:
            data: Data to write in Zarr format
            output_path: Path where the .zarr directory should be written
        """
        zarr_store = LocalStore(output_path)
        root = zarr.open_group(store=zarr_store, mode="w")

        # Write metadata as attributes
        data.metadata.zarr_format = zarr.__version__
        root.attrs.update(asdict(data.metadata))

        # Write STL geometry arrays
        for field in ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]:
            array_info = getattr(data, field)
            root.create_array(
                name=field,
                data=array_info.data,
                chunks=array_info.chunks,
                shards=array_info.shards,
                compressors=array_info.compressor if array_info.compressor else None,
            )

        # Write volume arrays if present
        for field in ["volume_mesh_points", "volume_fields"]:
            array_info = getattr(data, field)
            if array_info is not None:
                root.create_array(
                    name=field,
                    data=array_info.data,
                    chunks=array_info.chunks,
                    shards=array_info.shards,
                    compressors=(
                        array_info.compressor if array_info.compressor else None
                    ),
                )
            else:
                self.logger.warning(f"{field} is absent in the dataset")

        # Write FVM connectivity arrays if present
        fvm_fields = [
            "cell_centers",
            "cell_volumes",
            "face_owner",
            "face_neighbor",
            "face_area",
            "face_normal",
            "face_centers",
        ]
        for field in fvm_fields:
            array_info = getattr(data, field, None)
            if array_info is not None:
                root.create_array(
                    name=field,
                    data=array_info.data,
                    chunks=array_info.chunks,
                    shards=array_info.shards,
                    compressors=(
                        array_info.compressor if array_info.compressor else None
                    ),
                )

    def should_skip(self, filename: str) -> bool:
        """Check whether the file should be skipped.

        Args:
            filename: Name of the file to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(filename)
        if output_path.exists():
            self.logger.info(f"Skipping {filename} - File already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary files from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        pattern = "*.zarr_temp"
        for temp_file in self.output_dir.glob(pattern):
            self.logger.warning(f"Removing orphaned temp file: {temp_file}")
            if temp_file.is_dir():
                shutil.rmtree(temp_file)
            else:
                temp_file.unlink()
