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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from gems.study.data import DataBase
from gems.study.system import System

if TYPE_CHECKING:
    from gems.optim_config.parsing import OptimConfig


@dataclass
class Study:
    """
    Container that pairs a System (component topology and connections) with a
    DataBase (parameter values for those components).

    These two objects are always used together to build an optimisation
    problem.  ``Study`` gathers them into a single, coherent unit and
    provides the cross-validation logic that was previously spread between
    ``DataBase.requirements_consistency`` and the callers of
    ``build_problem``.

    ``optim_config`` is optional; it is populated by :func:`~gems.study.folder.load_study`
    when an ``optim-config.yml`` is present alongside the study inputs.
    """

    system: System
    database: DataBase
    optim_config: Optional[OptimConfig] = field(default=None)

    def check_consistency(self) -> None:
        """Validate that the database supplies data for every parameter of every
        component defined in the system.

        Raises
        ------
        ValueError
            If a required data entry is missing or its time/scenario structure
            does not match what the model parameter expects.
        """
        for component in self.system.components:
            for param in component.model.parameters.values():
                data_structure = self.database.get_data(component.id, param.name)

                if not data_structure.check_requirement(
                    component.model.parameters[param.name].structure.time,
                    component.model.parameters[param.name].structure.scenario,
                ):
                    raise ValueError(
                        f"Data inconsistency for component: {component.id}, "
                        f"parameter: {param.name}. Requirement not met."
                    )
