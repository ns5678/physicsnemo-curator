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
Utilities for processing Plasma Modeling data.
"""

from typing import Optional, TypeAlias

import numpy as np

OptionalNDArray: TypeAlias = Optional[np.ndarray]


def to_float32(array: OptionalNDArray) -> OptionalNDArray:
    """Convert array to float32 if not None.

    Args:
        array: Input array or None

    Returns:
        Array converted to float32 or None if input was None
    """
    return np.float32(array) if array is not None else None
