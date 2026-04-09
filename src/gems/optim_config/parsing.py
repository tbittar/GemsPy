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
from typing import TYPE_CHECKING, List, Optional, Set

from pydantic import Field, ValidationError
from yaml import safe_load

from gems.expression.expression import (
    AdditionNode,
    BinaryOperatorNode,
    ExpressionNode,
    MaxNode,
    MinNode,
    UnaryOperatorNode,
    VariableNode,
)
from gems.utils import ModifiedBaseModel

if TYPE_CHECKING:
    from gems.model.model import Model
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


_MASTER_LOCS: Set[ElementLocation] = {
    ElementLocation.MASTER,
    ElementLocation.MASTER_AND_SUBPROBLEMS,
}


def _collect_variable_names(expr: ExpressionNode) -> Set[str]:
    """Recursively collect all variable names referenced in an expression."""
    if isinstance(expr, VariableNode):
        return {expr.name}
    if isinstance(expr, (AdditionNode, MaxNode, MinNode)):
        result: Set[str] = set()
        for op in expr.operands:
            result |= _collect_variable_names(op)
        return result
    if isinstance(expr, UnaryOperatorNode):
        return _collect_variable_names(expr.operand)
    if isinstance(expr, BinaryOperatorNode):
        return _collect_variable_names(expr.left) | _collect_variable_names(expr.right)
    return set()


def _check_id_existence(
    d: ModelDecompositionConfig, model: "Model", mc_id: str, errors: List[str]
) -> None:
    for v in d.variables:
        if v.id not in model.variables:
            errors.append(f"Variable '{v.id}' not found in model '{mc_id}'")
    for c in d.constraints:
        if c.id not in model.constraints and c.id not in model.binding_constraints:
            errors.append(f"Constraint '{c.id}' not found in model '{mc_id}'")
    obj_keys = set(model.objective_contributions or {})
    for o in d.objective_contributions:
        if o.id not in obj_keys:
            errors.append(
                f"Objective-contribution '{o.id}' not found in model '{mc_id}'"
            )


def _check_master_variables_not_time_dependent(
    d: ModelDecompositionConfig, model: "Model", mc_id: str, errors: List[str]
) -> None:
    """Variables assigned to master or master-and-subproblems must not depend on time."""
    for v in d.variables:
        if v.location in _MASTER_LOCS and v.id in model.variables:
            if model.variables[v.id].structure.time:
                errors.append(
                    f"Variable '{v.id}' in model '{mc_id}' is time-dependent "
                    f"but is assigned to '{v.location.value}'; "
                    "master variables must not depend on time"
                )


def _check_master_constraints_use_master_variables(
    d: ModelDecompositionConfig, model: "Model", mc_id: str, errors: List[str]
) -> None:
    """Constraints in master must only reference variables in master or master-and-subproblems."""
    master_var_ids = {
        v.id
        for v in d.variables
        if v.location in _MASTER_LOCS and v.id in model.variables
    }
    for c in d.constraints:
        if c.location != ElementLocation.MASTER:
            continue
        constraint = model.constraints.get(c.id) or model.binding_constraints.get(c.id)
        if constraint is None:
            continue
        for name in sorted(
            _collect_variable_names(constraint.expression) - master_var_ids
        ):
            errors.append(
                f"Constraint '{c.id}' in model '{mc_id}' references variable '{name}' "
                "which is not assigned to master or master-and-subproblems"
            )


def _check_master_objectives_use_master_variables(
    d: ModelDecompositionConfig, model: "Model", mc_id: str, errors: List[str]
) -> None:
    """Objective contributions in master must only reference variables in master or master-and-subproblems."""
    master_var_ids = {
        v.id
        for v in d.variables
        if v.location in _MASTER_LOCS and v.id in model.variables
    }
    obj_contribs = model.objective_contributions or {}
    for o in d.objective_contributions:
        if o.location != ElementLocation.MASTER:
            continue
        expr = obj_contribs.get(o.id)
        if expr is None:
            continue
        for name in sorted(_collect_variable_names(expr) - master_var_ids):
            errors.append(
                f"Objective contribution '{o.id}' in model '{mc_id}' references variable '{name}' "
                "which is not assigned to master or master-and-subproblems"
            )


def validate_optim_config(config: OptimConfig, network: "Network") -> None:
    """Cross-validate optim-config entries against the resolved network.

    Checks that every referenced ID exists, that master variables do not
    depend on time, and that master constraints and objectives only reference
    variables assigned to master or master-and-subproblems.
    Raises ValueError listing all violations.
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
        _check_id_existence(d, model, mc.id, errors)
        _check_master_variables_not_time_dependent(d, model, mc.id, errors)
        _check_master_constraints_use_master_variables(d, model, mc.id, errors)
        _check_master_objectives_use_master_variables(d, model, mc.id, errors)

    if errors:
        raise ValueError(
            f"Errors in {OPTIM_CONFIG_FILENAME}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
