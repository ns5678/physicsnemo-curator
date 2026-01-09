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
This module contains the path definitions for the Plasma Modeling datasets.
"""

from pathlib import Path

from constants import DatasetKind


class DryResistPaths:
    """Utility class for handling DryResist dataset file paths.

    DryResist file naming pattern:
    - Directory: s_001, s_002, etc.
    - Geometry: body.stl
    - Surface: surface.vtp
    - Parameters: parameters.yaml
    """

    GEOMETRY_FILE = "body.stl"
    SURFACE_FILE = "surface.vtp"
    PARAMETERS_FILE = "parameters.yaml"

    @staticmethod
    def geometry_path(case_dir: Path) -> Path:
        """Get path to the STL geometry file.

        Args:
            case_dir: Base directory for the simulation case

        Returns:
            Path: Path to the STL file containing geometry
        """
        return case_dir / DryResistPaths.GEOMETRY_FILE

    @staticmethod
    def surface_path(case_dir: Path) -> Path:
        """Get path to the surface data file.

        Args:
            case_dir: Base directory for the simulation case

        Returns:
            Path: Path to the VTP file containing surface data
        """
        return case_dir / DryResistPaths.SURFACE_FILE

    @staticmethod
    def parameters_path(case_dir: Path) -> Path:
        """Get path to the parameters file.

        Args:
            case_dir: Base directory for the simulation case

        Returns:
            Path: Path to the YAML file containing global parameters
        """
        return case_dir / DryResistPaths.PARAMETERS_FILE

def get_path_getter(kind: DatasetKind):
    """Returns path getter for a given dataset type."""

    match kind:
        case DatasetKind.DRYRESIST:
            return DryResistPaths