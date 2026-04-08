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

from abc import ABC, abstractmethod
from typing import Generator, Optional

from gems.expression import ExpressionNode
from gems.model import Constraint, Model, ProblemContext, Variable


class ModelSelectionStrategy(ABC):
    """
    Abstract class to specify the strategy of the created problem.
    Its derived class selects variables and constraints for the optimization problem:
        - MergedProblemStrategy: Keep all variables and constraints
    """

    def get_variables(self, model: Model) -> Generator[Variable, None, None]:
        for variable in model.variables.values():
            if self._keep_from_context(variable.context):
                yield variable

    def get_constraints(self, model: Model) -> Generator[Constraint, None, None]:
        for constraint in model.get_all_constraints():
            if self._keep_from_context(constraint.context):
                yield constraint

    @abstractmethod
    def _keep_from_context(self, context: ProblemContext) -> bool:
        ...

    @abstractmethod
    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        ...


class MergedProblemStrategy(ModelSelectionStrategy):
    def _keep_from_context(self, context: ProblemContext) -> bool:
        return True

    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        yield model.objective_operational_contribution
        yield model.objective_investment_contribution


class RiskManagementStrategy(ABC):
    """
    Abstract functor class for risk management.
    Its derived class implements the default (uniform) risk measure:
        - UniformRisk: All expressions have the same weight
    """

    def __call__(self, expr: ExpressionNode) -> ExpressionNode:
        return self._modify_expression(expr)

    @abstractmethod
    def _modify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        ...


class UniformRisk(RiskManagementStrategy):
    def _modify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        return expr
