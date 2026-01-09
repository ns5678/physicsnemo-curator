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
This module contains constants and enums for plasma modeling datasets.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class DryResistConstants:
    """Physics constants and normalization bounds for DryResist plasma simulations.
    
    Contains:
    - Reference values for global parameters (flow rates, pressure)
    - Min/max bounds for surface variables (computed from training dataset)
    """
    
    # Reference values for global parameters
    SAGA_FLOW_RATE: float = 1.82e-4  # kg/s reference SAGA flow rate
    WATER_FLOW_RATE: float = 1.97e-4  # kg/s reference water flow rate
    PRESSURE_REF: float = 132.3305  # Pa reference chamber pressure
    
    # Surface variable normalization bounds (min/max from training data)
    DENSITY_MIN: float = 8.180378e-05
    DENSITY_MAX: float = 1.543953e-01
    X_VELOCITY_MIN: float = -9.867145e+01
    X_VELOCITY_MAX: float = 1.647426e+02
    Y_VELOCITY_MIN: float = -2.197884e+02
    Y_VELOCITY_MAX: float = 2.098510e+00
    TEMPERATURE_MIN: float = 2.707102e+02
    TEMPERATURE_MAX: float = 3.741057e+02
    PRESSURE_MIN: float = 6.295084e+00
    PRESSURE_MAX: float = 1.178301e+04
    RHO_E_MIN: float = 1.000521e+01
    RHO_E_MAX: float = 1.770302e+04
    RHO_U_MIN: float = -5.698159e-01
    RHO_U_MAX: float = 5.614316e-01
    RHO_V_MIN: float = -3.384178e+00
    RHO_V_MAX: float = 7.402311e-04
    
    # Species densities
    RHO_SAGA_MIN: float = 1.967241e-26
    RHO_SAGA_MAX: float = 3.087630e-03
    RHO_H2O_MIN: float = 3.437572e-12
    RHO_H2O_MAX: float = 7.948344e-03
    RHO_AR_MIN: float = 7.707079e-05
    RHO_AR_MAX: float = 1.513077e-01
    RHO_RSN_N_CH323H2O_MIN: float = 4.472000e-35
    RHO_RSN_N_CH323H2O_MAX: float = 9.977530e-14
    RHO_NH_CH32_MIN: float = 2.250324e-19
    RHO_NH_CH32_MAX: float = 2.638095e-06
    RHO_RSN_N_CH322OH_MIN: float = 2.655966e-31
    RHO_RSN_N_CH322OH_MAX: float = 1.852237e-08
    RHO_RSN_N_CH322OHH2O_MIN: float = 4.083973e-35
    RHO_RSN_N_CH322OHH2O_MAX: float = 9.103174e-14
    RHO_RSN_N_CH32_OH2_MIN: float = 2.386634e-36
    RHO_RSN_N_CH32_OH2_MAX: float = 1.664407e-13
    RHO_RSN_N_CH32_OH2H2O_MIN: float = 3.696015e-35
    RHO_RSN_N_CH32_OH2H2O_MAX: float = 8.238415e-14
    RHO_RSN_OH3_MIN: float = 4.387191e-27
    RHO_RSN_OH3_MAX: float = 1.185502e-06
    RHO_RSN_OH2O_OH2SNR_MIN: float = 9.203996e-34
    RHO_RSN_OH2O_OH2SNR_MAX: float = 1.730956e-07
    RHO_RSNO2_OH2_MIN: float = -1.000000e-06
    RHO_RSNO2_OH2_MAX: float = 1.536241e-06
    RHO_RSNO3_OH3_MIN: float = -1.000000e-06
    RHO_RSNO3_OH3_MAX: float = 5.806100e-07
    RHO_RSNO4O_OH2_MIN: float = -2.439547e-07
    RHO_RSNO4O_OH2_MAX: float = 5.067207e-08
    RHO_RSNO5O2_OH_MIN: float = -1.000000e-06
    RHO_RSNO5O2_OH_MAX: float = 1.550101e-06
    RHO_RSNO6O2_OH2_MIN: float = -1.000000e-06
    RHO_RSNO6O2_OH2_MAX: float = 1.830909e-06
    RHO_RSNO12O4_OH4_MIN: float = -6.080963e-07
    RHO_RSNO12O4_OH4_MAX: float = 9.121080e-09


class ModelType(str, Enum):
    """Types of models that can be processed."""

    SURFACE = "surface"
    VOLUME = "volume"
    COMBINED = "combined"


class DatasetKind(str, Enum):
    """Types of datasets that can be processed."""
   
    DRYRESIST = "dryresist"


@dataclass(frozen=True)
class DefaultVariables:
    """Default variables to extract from the simulation."""

    SURFACE: tuple[str, ...] = ("pMean", "wallShearStress")
    VOLUME: tuple[str, ...] = ("UMean", "pMean")


def get_normalization_bounds() -> dict[str, tuple[float, float]]:
    """Get normalization bounds as dict mapping VTP field names to (min, max) tuples.
    
    Returns:
        Dict mapping surface variable names to (min, max) bounds.
    """
    b = DryResistConstants()
    return {
        "density(kg/m3)": (b.DENSITY_MIN, b.DENSITY_MAX),
        "x_velocity(m/s)": (b.X_VELOCITY_MIN, b.X_VELOCITY_MAX),
        "y_velocity(m/s)": (b.Y_VELOCITY_MIN, b.Y_VELOCITY_MAX),
        "temperature(K)": (b.TEMPERATURE_MIN, b.TEMPERATURE_MAX),
        "pressure(Pa)": (b.PRESSURE_MIN, b.PRESSURE_MAX),
        "rho_e(J/m3)": (b.RHO_E_MIN, b.RHO_E_MAX),
        "rho_u(kg/m2-s)": (b.RHO_U_MIN, b.RHO_U_MAX),
        "rho_v(kg/m2-s)": (b.RHO_V_MIN, b.RHO_V_MAX),
        "rho_SAGA(kg/m3)": (b.RHO_SAGA_MIN, b.RHO_SAGA_MAX),
        "rho_H2O(kg/m3)": (b.RHO_H2O_MIN, b.RHO_H2O_MAX),
        "rho_AR(kg/m3)": (b.RHO_AR_MIN, b.RHO_AR_MAX),
        "rho_RSn(N(CH3)2)3H2O(kg/m3)": (b.RHO_RSN_N_CH323H2O_MIN, b.RHO_RSN_N_CH323H2O_MAX),
        "rho_NH(CH3)2(kg/m3)": (b.RHO_NH_CH32_MIN, b.RHO_NH_CH32_MAX),
        "rho_RSn(N(CH3)2)2OH(kg/m3)": (b.RHO_RSN_N_CH322OH_MIN, b.RHO_RSN_N_CH322OH_MAX),
        "rho_RSn(N(CH3)2)2OHH2O(kg/m3)": (b.RHO_RSN_N_CH322OHH2O_MIN, b.RHO_RSN_N_CH322OHH2O_MAX),
        "rho_RSn(N(CH3)2)(OH)2(kg/m3)": (b.RHO_RSN_N_CH32_OH2_MIN, b.RHO_RSN_N_CH32_OH2_MAX),
        "rho_RSn(N(CH3)2)(OH)2H2O(kg/m3)": (b.RHO_RSN_N_CH32_OH2H2O_MIN, b.RHO_RSN_N_CH32_OH2H2O_MAX),
        "rho_RSn(OH)3(kg/m3)": (b.RHO_RSN_OH3_MIN, b.RHO_RSN_OH3_MAX),
        "rho_RSn(OH)2O(OH)2SnR(kg/m3)": (b.RHO_RSN_OH2O_OH2SNR_MIN, b.RHO_RSN_OH2O_OH2SNR_MAX),
        "rho_(RSnO)2(OH)2(kg/m3)": (b.RHO_RSNO2_OH2_MIN, b.RHO_RSNO2_OH2_MAX),
        "rho_(RSnO)3(OH)3(kg/m3)": (b.RHO_RSNO3_OH3_MIN, b.RHO_RSNO3_OH3_MAX),
        "rho_(RSnO)4O(OH)2(kg/m3)": (b.RHO_RSNO4O_OH2_MIN, b.RHO_RSNO4O_OH2_MAX),
        "rho_(RSnO)5O2(OH)(kg/m3)": (b.RHO_RSNO5O2_OH_MIN, b.RHO_RSNO5O2_OH_MAX),
        "rho_(RSnO)6O2(OH)2(kg/m3)": (b.RHO_RSNO6O2_OH2_MIN, b.RHO_RSNO6O2_OH2_MAX),
        "rho_(RSnO)12O4(OH)4(kg/m3)": (b.RHO_RSNO12O4_OH4_MIN, b.RHO_RSNO12O4_OH4_MAX),
    }


def get_physics_constants(kind: DatasetKind) -> dict[str, float]:
    """Get physics constants dict based on dataset kind. Add a branch
    to the if-elif pipeline below to populate metadata with values
    used for non-dimensionalization.

    Args:
        kind: The dataset kind (from config etl.common.kind)

    Returns:
        Dictionary of physics constant names to values.

    Raises:
        ValueError: If dataset kind is unknown.
    """
    if kind == DatasetKind.DRYRESIST:
        c = DryResistConstants()
        return {
            "saga_flow_rate": c.SAGA_FLOW_RATE,
            "water_flow_rate": c.WATER_FLOW_RATE,
            "pressure": c.PRESSURE_REF,
        }
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
