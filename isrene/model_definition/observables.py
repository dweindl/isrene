"""Classes for observables"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

from .model_definition import (
    Compartment,
    Model,
    ModelComponent,
    Observable,
    SpeciesType,
)

logger = logging.getLogger(__file__)

__all__ = ["MdvObservable", "AmountObservable", "RawMdvObservable"]


class MdvObservable(Observable):
    """
    Observable MDV corrected for natural isotope abundance.

    Observable mass distribution vector (MDV),
    a.k.a. mass isotopomer distribution (MID), corrected for natural isotope
    abundance.
    """

    def __init__(
        self,
        species: SpeciesType,
        compartments: List[Compartment],
        model: Model = None,
    ):
        # TODO remove model argument, since we can deduce it from other
        #  arguments
        if model:
            from warnings import warn

            warn(
                f"Passing the `model` argument to {self.__class__.__name__} "
                "is no longer required and deprecated.",
                DeprecationWarning,
            )
        # deduce model from arguments
        model_from_components = {
            compartment.model for compartment in compartments
        }
        model_from_components.add(species.model)
        if len(model_from_components) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{set(model_from_components)}"
            )
        model = model_from_components.pop()

        # TODO: include compartment here
        #  (-> also requires compartments in measurement metadata)
        # self.id = f"observable_mdv_{species.id}_{'_'.join(c.name for c in compartments)}"
        id_ = f"observable_mdv_{species.id}"

        ModelComponent.__init__(self, id_=id_, model=model)

        self.species = species
        self.compartments = compartments
        self.model.add_observable(self)

    @property
    def max_labels(self):
        from .model_definition_hl import LABELING_STATES

        return len(
            [
                sti_id
                for sti_id, sti_st in self.species.species_type_instances
                if list(sti_st.sites.values()) == [LABELING_STATES]
            ]
        )

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.species),
            repr(self.compartments),
        )


class RawMdvObservable(Observable):
    """
    Observable MDV not corrected for natural isotope abundance.

    Observable mass distribution vector (MDV),
    a.k.a. mass isotopomer distribution (MID), not corrected for natural isotope
    abundance.
    The observation function accounts for natural isotope abundance.
    """

    def __init__(
        self,
        species: SpeciesType,
        compartments: Union[Compartment, List[Compartment]],
        sum_formula: str,
        label: str = "C",
        model: Model = None,
        num_measurements: int = None,
    ):
        # TODO remove model argument, since we can deduce it from other
        #  arguments
        if model:
            from warnings import warn

            warn(
                f"Passing the `model` argument to {self.__class__.__name__} "
                "is no longer required and deprecated.",
                DeprecationWarning,
            )
        # deduce model from arguments
        model_from_components = {
            compartment.model for compartment in compartments
        }
        model_from_components.add(species.model)
        if len(model_from_components) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{set(model_from_components)}"
            )
        model = model_from_components.pop()

        # TODO: include compartment here
        #  (-> also requires compartments in measurement metadata)
        # self.id = f"observable_mdv_{species.id}_{'_'.join(c.name for c in compartments)}"
        id_ = f"observable_mdv_raw_{species.id}"

        ModelComponent.__init__(self, id_=id_, model=model)

        self.species = species
        self.compartments = (
            [compartments]
            if isinstance(compartments, Compartment)
            else compartments
        )
        self.sum_formula = sum_formula
        self.label = label
        # usually not more than 3 extra isotope peaks for the fully labeled
        #  compound are detected
        self.num_measurements = num_measurements or self.max_labels + 4

        self.model.add_observable(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.species),
            repr(self.compartments),
        )

    @property
    def max_labels(self):
        from .model_definition_hl import LABELING_STATES

        return len(
            [
                sti_id
                for sti_id, sti_st in self.species.species_type_instances
                if list(sti_st.sites.values()) == [LABELING_STATES]
            ]
        )

    def get_correction_matrix(self) -> np.array:
        from ..mid.correction import get_correction_matrix

        return get_correction_matrix(
            sum_formula=self.sum_formula,
            num_rows=self.num_measurements,
            max_labels=self.max_labels,
            label=self.label,
            row_normalize=False,
        )


class AmountObservable(Observable):
    """Observable for total amount of a species, independent of its site
    states, across the specified compartments"""

    def __init__(
        self,
        species: SpeciesType,
        compartments: List[Compartment],
        model: Model = None,
        relative: bool = False,
    ):
        # TODO remove model argument, since we can deduce it from other
        #  arguments
        if model:
            from warnings import warn

            warn(
                f"Passing the `model` argument to {self.__class__.__name__} "
                "is no longer required and deprecated.",
                DeprecationWarning,
            )
        # deduce model from arguments
        model_from_components = {
            compartment.model for compartment in compartments
        }
        model_from_components.add(species.model)
        if len(model_from_components) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{set(model_from_components)}"
            )
        model = model_from_components.pop()

        id_ = (
            f"observable_{'relative_' if relative else ''}amount_{species.id}_"
            f"{'_'.join(c.id for c in compartments)}"
        )
        ModelComponent.__init__(self, id_=id_, model=model)

        self.species = species
        self.compartments = compartments
        self.relative = relative

        self.model.add_observable(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.species),
            repr(self.compartments),
        )
