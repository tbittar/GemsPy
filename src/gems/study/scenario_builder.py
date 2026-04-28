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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class ScenarioBuilder:
    """Maps MC scenario indices to data-series column indices (0-based).

    Loaded from a ``scenariobuilder.dat`` file whose lines have the form::

        scenario_group, mc_scenario = time_serie_number

    where ``time_serie_number`` is 1-based (column 1 = first column of the
    data-series file).  Internally the mapping is stored as one numpy array
    per group so that ``resolve_vectorized`` is a pure array index — no
    Python loop over scenarios.

    When no file is present the builder is empty and ``resolve_vectorized``
    returns the MC scenario indices unchanged (identity mapping).
    """

    _group_arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def resolve_vectorized(
        self, scenario_group: Optional[str], mc_scenarios: np.ndarray
    ) -> np.ndarray:
        """Return 0-based col_idx array for a vector of MC scenario indices.

        Falls back to identity (col_idx == mc_scenario) when the group is
        absent from the mapping.  No Python loop — pure numpy array indexing.
        """
        if scenario_group is None or scenario_group not in self._group_arrays:
            return mc_scenarios
        return self._group_arrays[scenario_group][mc_scenarios]

    @classmethod
    def load(cls, path: Path) -> "ScenarioBuilder":
        """Parse a ``scenariobuilder.dat`` file.

        Each non-blank, non-comment line must follow::

            group_name, mc_scenario = time_serie_number

        ``time_serie_number`` is 1-based; it is stored internally as a
        0-based column index.
        """
        raw: Dict[str, Dict[int, int]] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                left, _, right = line.partition("=")
                parts = [p.strip() for p in left.split(",")]
                group, mc_scenario = parts[0], int(parts[1])
                raw.setdefault(group, {})[mc_scenario] = int(right.strip()) - 1

        group_arrays: Dict[str, np.ndarray] = {}
        for group, mapping in raw.items():
            max_mc = max(mapping)
            arr = np.arange(max_mc + 1, dtype=int)  # identity by default
            for mc, col in mapping.items():
                arr[mc] = col
            group_arrays[group] = arr

        return cls(group_arrays)
