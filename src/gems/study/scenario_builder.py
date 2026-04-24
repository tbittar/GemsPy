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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ScenarioBuilder:
    """Maps Monte-Carlo scenario IDs to time-series column indices.

    Each component carries a ``scenario_group`` key.  The builder holds a
    table loaded from a three-column ``.dat`` file
    (``scenario_group``, ``scenario_id``, ``column_id``) and resolves a
    (scenario_group, MC scenario ID) pair to the corresponding column index.

    Currently a pass-through: ``scenario_id == column_id`` for every group.
    The .dat file format and the resolution logic are not yet implemented.
    """

    def resolve(self, scenario_group: Optional[str], scenario_id: int) -> int:
        """Return the time-series column index for a given MC scenario ID.

        Pass-through for now.  Future implementation will look up
        ``column_id`` in the (scenario_group, scenario_id) → column_id table
        parsed from the scenariobuilder.dat file.
        """
        return scenario_id

    @classmethod
    def load(cls, path: Path) -> "ScenarioBuilder":
        """Load from a .dat file (not yet implemented — returns pass-through).

        Future: parse rows of (scenario_group, scenario_id, column_id) and
        build the internal mapping table.
        """
        # TODO: parse (scenario_group, scenario_id, column_id) rows
        return cls()
