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
Path definitions for the Stokes flow dataset.

Directory structure:
    {split}/res_{index}/body.stl
    {split}/res_{index}/volume.vtu
"""

from pathlib import Path

from constants import DatasetKind


class StokesFlowPaths:
    """Utility class for handling Stokes flow dataset file paths.

    Stokes flow file naming pattern:
    - Directory: res_0, res_1, res_10, etc.
    - Geometry: body.stl
    - Volume: volume.vtu
    """

    GEOMETRY_FILE = "body.stl"
    VOLUME_FILE = "volume.vtu"

    @staticmethod
    def geometry_path(sim_dir: Path) -> Path:
        """Get path to the STL geometry file.

        Args:
            sim_dir: Base directory for the simulation (e.g., train/res_0)

        Returns:
            Path to the STL file containing body geometry
        """
        return sim_dir / StokesFlowPaths.GEOMETRY_FILE

    @staticmethod
    def volume_path(sim_dir: Path) -> Path:
        """Get path to the volume data file.

        Args:
            sim_dir: Base directory for the simulation (e.g., train/res_0)

        Returns:
            Path to the VTU file containing volume data (u, v, p fields)
        """
        return sim_dir / StokesFlowPaths.VOLUME_FILE


def get_path_getter(kind: DatasetKind):
    """Returns path getter for a given dataset type.

    Args:
        kind: The dataset kind

    Returns:
        Path getter class with geometry_path and volume_path methods

    Raises:
        ValueError: If dataset kind is unknown.
    """
    match kind:
        case DatasetKind.STOKES:
            return StokesFlowPaths
        case _:
            raise ValueError(f"Unknown dataset kind: {kind}")
