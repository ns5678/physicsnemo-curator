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
Constants and enums for Stokes flow datasets.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PhysicsConstantsStokes:
    """Physical constants for Stokes flow simulations.

    Note: Stokes flow data is already non-dimensionalized, so these
    are placeholder reference values (unity).
    """

    REF_VELOCITY: float = 1.0  # Reference velocity (unity - already non-dim)
    REF_PRESSURE: float = 1.0  # Reference pressure (unity - already non-dim)


class ModelType(str, Enum):
    """Types of models that can be processed."""

    VOLUME = "volume"  # STL geometry + volume data (default for Stokes)


class DatasetKind(str, Enum):
    """Types of Stokes flow datasets."""

    STOKES = "stokes"


@dataclass(frozen=True)
class DefaultVariables:
    """Default variables to extract from the simulation."""

    # Point-centered volume fields: u (x-velocity), v (y-velocity), p (pressure)
    VOLUME: tuple[str, ...] = ("u", "v", "p")


def get_physics_constants(kind: DatasetKind) -> dict[str, float]:
    """Get physics constants dict based on dataset kind.

    Args:
        kind: The dataset kind

    Returns:
        Dictionary of physics constant names to values.

    Raises:
        ValueError: If dataset kind is unknown.
    """
    if kind == DatasetKind.STOKES:
        c = PhysicsConstantsStokes()
        return {"ref_velocity": c.REF_VELOCITY, "ref_pressure": c.REF_PRESSURE}
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
