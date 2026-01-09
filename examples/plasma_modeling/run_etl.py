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

"""Plasma Modeling ETL pipeline runner.

This script handles component instantiation and runs the ETL orchestrator
for processing plasma modeling datasets.

Usage:
    python run_etl.py \\
        etl.source.input_dir=/data/plasma/ \\
        etl.sink.output_dir=/data/plasma.processed
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def setup_logging() -> None:
    """Configure logging for multiprocess ETL pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | P%(process)d | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Reduce noise from external libraries
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pyvista").setLevel(logging.WARNING)


setup_logging()

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from physicsnemo_curator.etl.etl_orchestrator import ETLOrchestrator
from physicsnemo_curator.etl.processing_config import ProcessingConfig
from physicsnemo_curator.utils import utils as curator_utils

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="config", config_name="plasma_DryResist")
def main(cfg: DictConfig) -> None:
    """Run the Plasma Modeling ETL pipeline."""
    # Set multiprocessing start method
    curator_utils.setup_multiprocessing()
    
    logger.info("Starting Plasma Modeling ETL pipeline")
    logger.info(f"Input: {cfg.etl.source.input_dir}")
    logger.info(f"Output: {cfg.etl.sink.output_dir}")
    logger.info(f"Workers: {cfg.etl.processing.num_processes}")

    # Create processing config with common settings
    processing_config = ProcessingConfig(**cfg.etl.processing)

    # Create and run validator (if configured)
    validator = None
    if "validator" in cfg.etl:
        validator = instantiate(
            cfg.etl.validator,
            processing_config,
            **{k: v for k, v in cfg.etl.source.items() if not k.startswith("_")},
        )

    # Instantiate source
    source = instantiate(cfg.etl.source, processing_config)

    # Instantiate sink
    sink = instantiate(cfg.etl.sink, processing_config)

    # Instantiate transformations
    # Need to pass processing_config to each transformation, see:
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    cfgs = {k: {"_args_": [processing_config]} for k in cfg.etl.transformations.keys()}
    transformations = instantiate(cfg.etl.transformations, **cfgs)

    # Create and run orchestrator with instantiated components
    orchestrator = ETLOrchestrator(
        source=source,
        sink=sink,
        transformations=transformations,
        processing_config=processing_config,
        validator=validator,
    )
    orchestrator.run()
    
    logger.info("ETL pipeline completed")


if __name__ == "__main__":
    main()
