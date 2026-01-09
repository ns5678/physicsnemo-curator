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

import numpy as np
from schemas import PlasmaModelingExtractedDataInMemory

logging.basicConfig(
    format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def default_global_params_processing_for_plasma(
    data: PlasmaModelingExtractedDataInMemory,
    global_parameters: dict,
) -> PlasmaModelingExtractedDataInMemory:
    """Default global parameters processing for Plasma Modeling.

    Extracts and flattens global parameter references from config into a 1D numpy array.
    Handles both vector and scalar parameter types.

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with structure:
            {
                "param_name": {
                    "type": "vector" or "scalar",
                    "reference": value or list
                }
            }

    Returns:
        Updated `data` with global_params_reference set
    """

    # Build dictionaries for types and reference values
    global_params_types = {
        name: params["type"] for name, params in global_parameters.items()
    }

    global_params_reference_dict = {
        name: params["reference"] for name, params in global_parameters.items()
    }

    # Arrange global parameters reference in a list based on the type of the parameter
    global_params_reference_list = []
    for name, param_type in global_params_types.items():
        if param_type == "vector":
            global_params_reference_list.extend(global_params_reference_dict[name])
        elif param_type == "scalar":
            global_params_reference_list.append(global_params_reference_dict[name])
        else:
            raise ValueError(
                f"Global parameter '{name}' has unsupported type '{param_type}'. "
                f"Must be 'vector' or 'scalar'."
            )

    # Convert to numpy array and store in data container
    data.global_params_reference = np.array(
        global_params_reference_list, dtype=np.float32
    )

    return data


def process_global_params(
    data: PlasmaModelingExtractedDataInMemory,
    global_parameters: dict,
) -> PlasmaModelingExtractedDataInMemory:
    """Extract global parameters from DryResist simulation data.

    For DryResist, global parameters are read from parameters.yaml file
    which is loaded into metadata.simulation_params by data_sources.py.

    The parameters extracted are:
    - saga_flow_rate: SAGA precursor flow rate (kg/s)
    - water_flow_rate: Water vapor flow rate (kg/s)
    - pressure: Chamber pressure (Pa)

    Args:
        data: Container with simulation data and metadata
        global_parameters: Dict from config with parameter definitions

    Returns:
        Updated `data` with global_params_values extracted from simulation
    """
    # Get simulation parameters loaded from parameters.yaml
    sim_params = data.metadata.simulation_params
    if sim_params is None:
        raise ValueError(
            "simulation_params is None. Ensure parameters.yaml was loaded by data_sources."
        )

    # Map config parameter names to keys in parameters.yaml
    # parameters.yaml has: saga_flow_rate, water_flow_rate, pressure
    extracted_values = {}

    for name in global_parameters.keys():
        if name in sim_params:
            extracted_values[name] = sim_params[name]
            logger.info(f"Extracted {name}={sim_params[name]} from parameters.yaml")
        else:
            raise ValueError(
                f"Global parameter '{name}' not found in parameters.yaml. "
                f"Available keys: {list(sim_params.keys())}"
            )

    # Build the flattened array using the same logic as reference processing
    global_params_values_list = []
    for name, params in global_parameters.items():
        param_type = params["type"]
        value = extracted_values[name]

        if param_type == "vector":
            if isinstance(value, list):
                global_params_values_list.extend(value)
            else:
                global_params_values_list.append(value)
        elif param_type == "scalar":
            global_params_values_list.append(value)
        else:
            raise ValueError(
                f"Global parameter '{name}' has unsupported type '{param_type}'. "
                f"Must be 'vector' or 'scalar'."
            )

    data.global_params_values = np.array(global_params_values_list, dtype=np.float32)

    return data
