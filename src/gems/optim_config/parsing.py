# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from pydantic import Field, ValidationError
from yaml import safe_load

from gems.utils import ModifiedBaseModel

if TYPE_CHECKING:
    from gems.study.network import Network

OPTIM_CONFIG_FILENAME = "optim-config.yml"


class ElementLocation(str, Enum):
    MASTER = "master"
    SUBPROBLEMS = "subproblems"
    MASTER_AND_SUBPROBLEMS = "master-and-subproblems"


class ElementLocationConfig(ModifiedBaseModel):
    id: str
    location: ElementLocation


class ModelDecompositionConfig(ModifiedBaseModel):
    variables: List[ElementLocationConfig] = Field(default_factory=list)
    constraints: List[ElementLocationConfig] = Field(default_factory=list)
    objective_contributions: List[ElementLocationConfig] = Field(default_factory=list)


class ModelOptimConfig(ModifiedBaseModel):
    id: str
    model_decomposition: Optional[ModelDecompositionConfig] = None


class ResolutionMode(str, Enum):
    SEQUENTIAL_SUBPROBLEMS = "sequential-subproblems"
    BENDERS_DECOMPOSITION = "benders-decomposition"


class OptimConfig(ModifiedBaseModel):
    resolution_mode: ResolutionMode = ResolutionMode.SEQUENTIAL_SUBPROBLEMS
    models: List[ModelOptimConfig] = Field(default_factory=list)


def load_optim_config(components_path: Path) -> Optional[OptimConfig]:
    """Load optim-config.yml from the same directory as components_path.

    Returns None if the file does not exist.
    Raises ValueError on parsing or validation failure.
    """
    config_path = components_path.parent / OPTIM_CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        with config_path.open() as f:
            return OptimConfig.model_validate(safe_load(f))
    except ValidationError as e:
        raise ValueError(f"Invalid {OPTIM_CONFIG_FILENAME}: {e}")


def validate_optim_config(config: OptimConfig, network: "Network") -> None:
    """Cross-validate optim-config entries against the resolved network.

    Checks that every referenced model ID, variable ID, constraint ID, and
    objective-contribution ID actually exists.  Raises ValueError listing all
    unknown identifiers.
    """
    models_in_network = {c.model.id: c.model for c in network.all_components}
    errors: List[str] = []

    for mc in config.models:
        model = models_in_network.get(mc.id)
        if model is None:
            errors.append(f"Model '{mc.id}' not found in network")
            continue
        if mc.model_decomposition is None:
            continue
        d = mc.model_decomposition
        for v in d.variables:
            if v.id not in model.variables:
                errors.append(f"Variable '{v.id}' not found in model '{mc.id}'")
        for c in d.constraints:
            if c.id not in model.constraints and c.id not in model.binding_constraints:
                errors.append(f"Constraint '{c.id}' not found in model '{mc.id}'")
        obj_keys = set(model.objective_contributions or {})
        for o in d.objective_contributions:
            if o.id not in obj_keys:
                errors.append(
                    f"Objective-contribution '{o.id}' not found in model '{mc.id}'"
                )

    if errors:
        raise ValueError(
            f"Errors in {OPTIM_CONFIG_FILENAME}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
