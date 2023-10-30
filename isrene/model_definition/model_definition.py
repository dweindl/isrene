"""Classes for model entities"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from itertools import chain
from numbers import Number
from pprint import pprint
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pint.errors
import sympy as sp
from sbmlmath import SpeciesSymbol, rate_of

from .. import Q_, ureg

logger = logging.getLogger(__file__)

__all__ = [
    "Expression",
    "Compartment",
    "ReactionGenerator",
    "InitialConcentration",
    "Model",
    "SpeciesPattern",
    "SpeciesType",
    "FeatureMapping",
    "DontCare",
    "ReactionPattern",
    "RateLawGenerator",
]

# For type hints, any subclass of ModelComponent
ModelComponentType = TypeVar("ModelComponentType", bound="ModelComponent")


class DuplicateIdError(ValueError):
    ...


class ModelComponent:
    """Base class for a component of a ``Model``

    Attributes:
        id: Some ID that would be a valid identifier in Python
        model: The model to which this component belongs
        name: Some free text name of this component
        sbo_term: Systems Biology Ontology term
    """

    def __init__(
        self,
        id_: str,
        model: Model,
        name: Optional[str] = None,
        sbo_term: Optional[str] = None,
    ):
        if not isinstance(model, Model):
            raise ValueError("Must provide a valid model.")

        self.id = id_
        self.name = name
        self.model = model
        self.sbo_term = sbo_term
        self.annotations = {}


class DontCare:
    """Represents the "don't care" state of a species feature"""

    def __str__(self):
        return self.__class__.__name__


class Expression(sp.Symbol, ModelComponent):
    """A mathematical expression in the model.

    There is no distinction between "parameters" and "expressions". This
    makes it easier to modify expressions during model constructions.
    Exporters should convert expressions to parameters if the right-hand-side
    and the target language permit.

    Attributes:
        expr: Right-hand-side of the expression
        estimated: Whether the right-hand-side is to be estimated. Only
            meaningful if the right-hand-side is a numerical constant.
        non_negative: Whether the right-hand-side is assumed to be
            non-negative. Only relevant if `estimated` and if the
            right-hand-side is a numerical constant.
    """

    # Required due sp.Symbol base
    def __new__(
        cls,
        id_: str,
        expr: Union[float, sp.Expr, ureg.Quantity],
        model: Model = None,
        estimated: bool = False,
        non_negative: bool = True,
        sbo_term: Optional[str] = None,
        name_: str = None,
    ):
        return super().__new__(cls, id_, real=True)

    def __init__(
        self,
        id_: str,
        expr: Union[float, sp.Expr],
        model: Model,
        estimated: bool = True,
        non_negative: bool = True,
        sbo_term: Optional[str] = None,
        name_: str = None,
    ):
        if not isinstance(model, Model):
            raise ValueError("Must provide a valid model.")

        # Note that currently `name` must be equal to ID for expressions,
        #  as `name` is used to identify the sympy Symbol. We use name_
        #  instead. Not sure how to handle that in a better way.
        ModelComponent.__init__(
            self, id_=id_, model=model, name=id_, sbo_term=sbo_term
        )

        # TODO: or can be derived from expression?
        # self.unit =
        assert expr is not None
        self.expr = expr
        self.annotations["parameter"] = {}
        # TODO default to False if expr is a symbolic expression?
        self.estimated = estimated
        self.non_negative = non_negative
        self.name_ = name_

        self.model.add_expression(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.expr),
        )

    @property
    def estimated(self):
        return self._estimated

    @estimated.setter
    def estimated(self, value: bool):
        self._estimated = value
        self.annotations["parameter"]["estimated"] = (
            "true" if self._estimated else "false"
        )

    @property
    def non_negative(self):
        return self._non_negative

    @non_negative.setter
    def non_negative(self, value: bool):
        self._non_negative = value
        self.annotations["parameter"]["non_negative"] = (
            "true" if self._non_negative else "false"
        )

    @property
    def unit(self) -> ureg.Unit:
        """Get the unit of this expression."""
        if isinstance(self.expr, ureg.Quantity):
            return self.expr.u

        if isinstance(self.expr, Number):
            return ureg.dimensionless

        if isinstance(self.expr, sp.Expr):
            # derive unit from symbols

            # TODO: how to handle functions?
            # for now, replace all exp(..)
            # TODO warn if not dimensionless inside exp(..)
            from sympy.functions.elementary.exponential import ExpBase, ExpMeta

            expr = self.expr.subs(
                [
                    (x, 1)
                    for x in sp.preorder_traversal(self.expr)
                    if isinstance(x, (ExpMeta, ExpBase))
                ]
            )
            if isinstance(expr, sp.core.power.Pow):
                # TODO: check if arguments are dimensionless
                return ureg.dimensionless

            # need to sympify the unit here, to avoid sympification
            #  inside subs with strict=True, which fails for units
            subs = [
                (x, sp.sympify(str(self.model.unit_of_component(x))))
                for x in expr.free_symbols
            ]

            res = expr.subs(subs)

            if rate_ofs := expr.find(rate_of):
                # TODO: we currently assume that rateOf always refers to concentration-based species
                res = expr.subs(
                    [
                        (
                            rate_expr,
                            sp.sympify(
                                str(
                                    self.model.concentration_unit
                                    / self.model.time_unit
                                )
                            ),
                        )
                        for rate_expr in rate_ofs
                    ]
                )

            try:
                quantity = ureg.parse_expression(str(res))
            except pint.errors.UndefinedUnitError as e:
                # We cannot handle min(exp(..)) at the moment
                if res.__class__.__name__ == "Min":
                    # arg_idx = 1 if res.args[0].is_Number else 0
                    # quantity = ureg.parse_expression(str(res.args[arg_idx]))
                    logger.warning(
                        f"Unable to determine unit for {self}: "
                        f"{res}. Assuming dimensionless."
                    )
                    return ureg.dimensionless
                else:
                    raise ValueError(
                        f"Unable to determine unit for {self}: {res}"
                    ) from e
            if not isinstance(quantity, ureg.Quantity):
                return ureg.dimensionless

            # Only interested in the unit
            return quantity.units

        raise NotImplementedError(f"Unable to determine unit for {self}")


class Compartment(ModelComponent):
    """A compartment.

    Attributes:
        size: The compartment size. Volume or area, depending on `dimensions`.
        parent: The surrounding compartment, if any.
        electric_potential: Electric potential of the compartment
        ph: pH value in this compartment (= -log10(c_H^+); c_H^+ in mol/L)
        ionic_strength: Ionic strength in this compartment
        c_mg: Total Mg2+ concentration in this compartment
    """

    def __init__(
        self,
        id_: str,
        size: Union[Q_, Expression],
        dimensions: Optional[int] = None,
        parent: Union["Compartment", None] = None,
        name: str = None,
        model: Model = None,
        electric_potential: Q_ = None,
        ph: Q_ = None,
        ionic_strength: Q_ = None,
        temperature: Q_ = None,
        c_mg: Q_ = None,
    ):
        if not isinstance(model, Model):
            raise ValueError("Must provide a valid model.")

        super().__init__(
            id_=id_, model=model, name=name, sbo_term="SBO:0000290"
        )
        assert dimensions in {2, 3, None}
        if dimensions is None:
            if not isinstance(size, Q_):
                raise ValueError("`dimensions` required")
            if size.check("[volume]"):
                dimensions = 3
            elif size.check("[area]"):
                dimensions = 2
            else:
                raise ValueError(
                    "Invalid unit for `size`, neither area nor " "volume."
                )
        # assert Q_(1, size.units) \
        #        == Q_(1, model.volume_unit if dimensions == 3 else model.area_unit)

        self.size = size
        self.dimensions = dimensions
        self.parent = parent

        if electric_potential is not None:
            electric_potential.check("[electric_potential]")
        self.electric_potential = electric_potential

        if ph is not None:
            if isinstance(ph, Q_):
                ph.check("[dimensionless]")
            else:
                ph = Q_(ph)
        self.ph = ph

        if ionic_strength is not None:
            ionic_strength.check("[concentration]")
        self.ionic_strength = ionic_strength

        if temperature is not None:
            temperature.check("[temperature]")
        elif parent:
            temperature = parent.temperature
        self.temperature = temperature

        if c_mg is not None:
            c_mg.check("[concentration]")
        self.c_mg = c_mg

        self.model.add_compartment(self)

    def __repr__(self):  # sourcery skip: replace-interpolation-with-fstring
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.size),
        )


class ReactionGenerator(ModelComponent, ABC):
    """A ReactionGenerator is an entity that can be converted into a number
    of basic reactions."""

    # TODO needs refactoring.
    #  should (not) derive from ModelComponent?
    #  could be reduced to having to_elementary?!

    def __init__(
        self,
        id_: str,
        reversible: bool = False,
        # TODO handle elsewhere
        hmr_id: Optional[str] = None,
        # TODO
        rate_law_generator: Optional["RateLawGenerator"] = None,
        model: Model = None,
    ):
        super().__init__(model=model, id_=id_)

        self.reversible = reversible
        self.hmr_id = hmr_id
        self.rate_law_generator = rate_law_generator
        self._generated = False

    @abstractmethod
    def to_elementary(self) -> Iterable[ReactionPattern]:
        """Get basic reaction patterns from this potentially
        high-level reaction definition"""
        pass


class Initial:
    """
    Base class for InitialConcentration/InitialAmount
    """

    pass


class InitialConcentration(Initial):
    """An initial concentration of a concrete species (SpeciesPattern)"""

    def __init__(
        self,
        species: "SpeciesPattern",
        value: Union[Q_, Expression],
        constant: bool = False,
        estimated: bool = False,
    ):
        """
        Constructor

        :param species: Concrete species pattern for which the initial value
            is set.
        :param value: Value or expression of the initial concentration.
        :param constant: Whether the concentration is constant, i.e. this is
            a boundary condition.
        :param estimated: Whether the initial concentration is used as is
            or whether it is estimated.
        """
        if not species.is_concrete:
            raise ValueError(
                "Initial concentrations can only be defined for concrete "
                f"species. {species} is not concrete."
            )

        self.species = species
        self.value = value
        self.constant = constant
        self.estimated = estimated
        self.species.template.model.add_initial(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.species),
            repr(self.value),
        )


class Model:
    """A container for the various model entities

    All entities are required to have unique IDs (not always checked yet).

    Note that unit conversion for volumes / area / length is not done
    automatically.
    """

    def __init__(
        self,
        id_: str = "model",
        *,
        time_unit: ureg.Unit = ureg.second,
        amount_unit: ureg.Unit = ureg.mole,
        volume_unit: ureg.Unit = ureg.meter**3,
        area_unit: ureg.Unit = ureg.meter**2,
        length_unit: ureg.Unit = ureg.meter,
    ):
        """
        Initialize model

        Arguments
        ---------
        id_: Some model ID
        time_unit: Model time unit
        amount_unit: Model amount unit
        volume_unit: Model volume unit
        area_unit: Model area unit
        length_unit: Model length unit
        """
        assert Q_(1.0, time_unit).check("[time]")
        assert Q_(1.0, amount_unit).check("[substance]")
        assert Q_(1.0, volume_unit).check("[volume]")
        assert Q_(1.0, area_unit).check("[area]")
        assert Q_(1.0, length_unit).check("[length]")

        self.time_unit = time_unit
        self.amount_unit = amount_unit
        self.volume_unit = volume_unit
        self.concentration_unit: ureg.Unit = (
            self.amount_unit / self.volume_unit
        )
        self.area_unit = area_unit
        self.length_unit = length_unit
        self._molar_energy_unit = ureg.kJ / ureg.mol

        self.id = id_
        self.compartments: List[Compartment] = []
        self.species_types: List[SpeciesType] = []
        self.expressions: List[Expression] = []
        self.reactions: List[ReactionPattern] = []
        self.observables: List[Observable] = []
        self.initials: List[Initial] = []
        self.parameter_factory = ParameterFactory(self)
        # reactants that will be ignored in rate laws (e.g. H2O)
        self.ignored_reactants = set()

    def __getitem__(self, item: str):
        """Access model components by ID.

        Returns
        -------
        The respective model component, or `None` if there is no component
        with the given ID.
        """
        component_lists = [
            self.compartments,
            self.species_types,
            self.expressions,
            self.reactions,
            self.observables,
        ]
        for component in chain(*component_lists):
            if component.id == item:
                return component

    def add(
        self, component: Union[ReactionGenerator, ModelComponentType, Initial]
    ):
        """Add a component to the model."""
        component.model = self

        if isinstance(component, Compartment):
            self.add_compartment(component)
        elif isinstance(component, SpeciesType):
            self.add_species(component)
        elif isinstance(component, Expression):
            self.add_expression(component)
        elif isinstance(component, (ReactionGenerator, ReactionPattern)):
            self.add_reaction(component)
        elif isinstance(component, Observable):
            self.add_observable(component)
        elif isinstance(component, Initial):
            self.add_initial(component)
        else:
            raise TypeError(f"What to do with {component}?")

    def add_compartment(self, compartment: Compartment):
        """Add a compartment to the model."""
        if self[compartment.id]:
            raise DuplicateIdError(str(compartment))

        self.compartments.append(compartment)

    def add_species(self, species: SpeciesType) -> SpeciesType:
        """Add a species to the model"""
        with contextlib.suppress(StopIteration):
            if existing_st := next(
                filter(lambda s: s.id == species.id, self.species_types)
            ):
                raise ValueError(
                    f"SpeciesType `{existing_st.id}` already exists."
                )

        species.model = self
        self.species_types.append(species)
        return species

    def add_expression(self, expression: Expression):
        """Add an expression to the model"""
        if self[expression.id]:
            raise DuplicateIdError(str(expression))
        self.expressions.append(expression)

    def add_reaction(
        self, reaction: Union[ReactionGenerator, ReactionPattern]
    ) -> None:
        """Add a reaction to the model"""
        reaction.model = self
        if isinstance(reaction, ReactionPattern):
            if (existing := self[reaction.id]) and id(existing) != id(
                reaction
            ):
                raise DuplicateIdError(
                    f"New: {reaction}\nexisting: {existing}"
                )
            reaction.model = self
            self.reactions.append(reaction)

        elif isinstance(reaction, ReactionGenerator):
            list(reaction.to_elementary())
        else:
            raise ValueError(
                f"Unknown reaction type {type(reaction)} for " "{reaction}."
            )

    def add_observable(self, observable: Observable) -> None:
        """Add an observable to the model."""
        if self[observable.id]:
            raise DuplicateIdError(f"{observable.id} - {observable}")

        self.observables.append(observable)

    def add_initial(self, initial: Initial) -> Initial:
        """Add an initial amount/concentration to the model."""
        initial.model = self
        assert isinstance(initial, InitialConcentration)
        # TODO check equivalent, not identical units
        if isinstance(initial.value, Q_):
            # Convert to model units
            initial.value.ito(self.concentration_unit)
        elif isinstance(initial.value, Expression):
            if isinstance(initial.value.expr, Q_):
                initial.value.expr.ito(self.concentration_unit)
            assert initial.value.unit == self.concentration_unit
        else:
            raise ValueError(
                f"Unexpected initial value type in {initial}: "
                f"{type(initial.value)}"
            )
        self.initials.append(initial)
        return initial

    def get_expression(self, id_: str) -> Expression:
        """Get expression with the given ID."""
        for e in self.expressions:
            if e.id == id_:
                return e
        raise ValueError(f"Unknown expression: {id_}")

    def print(self) -> None:
        """Print some summary of the model"""
        print("compartments")
        pprint(self.compartments)
        print()
        print("species")
        pprint(self.species_types)
        print()
        print("expressions")
        pprint(self.expressions)
        print()
        print("reactions")
        pprint(self.reactions)
        print()
        print("observables")
        pprint(self.observables)
        print()
        print("initials")
        pprint(self.initials)

    def get_species_symbol(
        self,
        species_instance: SpeciesPattern,
    ) -> SpeciesSymbol:
        """Get symbol for the given species pattern to be used in maths
        expressions."""
        species_id = self._generate_species_id(species_instance)
        return SpeciesSymbol(species_id)

    def get_compartment_symbol(
        self,
        compartment: Compartment,
    ) -> sp.Symbol:
        """
        Get symbol for the given compartment to be used in maths expressions.
        """
        return sp.Symbol(compartment.id)

    def _generate_species_id(
        self,
        species_instance: SpeciesPattern,
        encode_state: bool = True,
        encode_compartment: bool = True,
    ) -> str:
        """Generate an ID for the given concrete species.

        NOTE:
        This has to be, but is currently not ensured, to be model-wide unique.
        """
        # TODO ID may not be unique!
        # base ID
        id_ = species_instance.template.id

        # compartment suffix
        if encode_compartment and species_instance.compartment:
            compartment = species_instance.compartment.id
            id_ = f"{id_}_{compartment}"

        # state suffix
        if encode_state and species_instance.site_states:
            site_config_str = "_".join(
                f"{site}_{config}"
                for site, config in species_instance.site_states.items()
            )
            id_ = f"{id_}_{site_config_str}"
        # TODO (check if has species_type)

        return id_

    def _generate_species_name(
        self,
        species_instance: SpeciesPattern,
        encode_state: bool = True,
        encode_compartment: bool = True,
    ) -> str:
        """Create a name for the given species."""
        name = species_instance.template.name or species_instance.template.id

        # compartment suffix
        if encode_compartment and species_instance.compartment:
            compartment = species_instance.compartment.id
            name = f"{name} in {compartment}"

        # state suffix
        if encode_state and species_instance.site_states:
            site_config_str = ", ".join(
                f"{site}={config}"
                for site, config in species_instance.site_states.items()
            )
            name = f"{name} ({site_config_str})"

        return name

    def unit_of_component(
        self, component_id: [str, SpeciesSymbol, sp.Symbol]
    ) -> ureg.Unit:
        """Get the unit of a model component with the given ID"""
        if isinstance(component_id, SpeciesSymbol):
            # FIXME we assume we are only dealing with concentrations, and that
            #  all model species use the same unit
            return self.concentration_unit

        component_id = str(component_id)

        # Is it a compartment?
        with contextlib.suppress(StopIteration):
            if comp := next(
                filter(lambda x: x.name == component_id, self.compartments)
            ):
                if comp.dimensions == 3:
                    return self.volume_unit
                if comp.dimensions == 2:
                    return self.area_unit
                raise AssertionError(
                    f"Invalid dimensions {comp.dimensions} "
                    f"for compartment {comp.id}"
                )

        # Is it an expression?
        with contextlib.suppress(StopIteration):
            if expr := next(
                filter(lambda x: x.name == component_id, self.expressions)
            ):
                return expr.unit

        # Is it a reaction?
        with contextlib.suppress(StopIteration):
            if expr := next(
                filter(lambda x: x.id == component_id, self.reactions)
            ):
                return expr.unit

        raise ValueError(f"What is {component_id}?")


class SpeciesType(ModelComponent):
    """A SpeciesType describes the structure of a type of Species, i.e. things
    like binding or labelling sites.
    """

    def __init__(
        self,
        id_: str,
        sites: Optional[Dict] = None,
        model: Model = None,
        name: str = None,
        species_type_instances: List[Tuple[str, "SpeciesType"]] = None,
    ):
        if not isinstance(model, Model):
            raise ValueError("Must provide a valid model.")

        super().__init__(model=model, id_=id_, name=name)

        if sites is None:
            sites = {}
        self.sites: Dict[str, str] = sites

        self.species_type_instances: List[str, "SpeciesType"] = (
            species_type_instances or []
        )

        model.add(self)

    def __call__(
        self,
        site_states: Optional[Dict] = None,
        compartment: Optional[Compartment] = None,
    ):
        """Create a pattern."""
        return SpeciesPattern(self, site_states, compartment)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.sites),
        )


class SpeciesPattern:
    """A species pattern, concrete or non-concrete.

    If no SpeciesType is set, then this is automatically concrete.

    Attributes
    ----------
    template: The species type this pattern refers to
    site_states: The state of the sites of the underlying species type
    compartment: The compartment
    """

    def __init__(
        self,
        species_type: SpeciesType = None,
        site_states: Optional[Dict] = None,
        compartment: Optional[Compartment] = None,
    ):
        # TODO: doesn't really make sense to have unset species type, right?!
        if species_type is None and site_states:
            raise ValueError(
                "Must not provide site states if no species " "type is given"
            )

        if site_states is None:
            site_states = {}

        self.template = species_type

        self._check_site_values(site_states)

        self.site_states = site_states
        self.compartment = compartment

    def __repr__(self):
        return "%s(%s, %s, %s)" % (
            self.__class__.__name__,
            repr(self.template),
            repr(self.site_states),
            repr(self.compartment),
        )

    def __eq__(self, other):
        return self.__class__ == other.__class__ and all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("template", "site_states", "compartment")
        )

    @property
    def is_concrete(self):
        """Check if this pattern is concrete.

        i.e. matches only a single species.
        """
        # TODO anything else to check?
        if self.compartment is None:
            return False

        # check if all sites are defined
        if self.template is None:
            return True

        undefined_sites = set(self.template.sites.keys()) | set(
            component_id
            for component_id, _ in self.template.species_type_instances
        ) - set(self.site_states.keys())
        return not undefined_sites

    def _check_site_values(self, actual_states):
        """Ensure the given site configuration is valid"""
        # TODO handle hierarchical species types properly
        possible_states = self.template.sites.copy()
        for component_id, species_type in self.template.species_type_instances:
            if len(species_type.sites) > 1:
                raise NotImplementedError()
            if len(species_type.sites) > 0:
                possible_states[component_id] = next(
                    iter(
                        self.template.species_type_instances[0][
                            1
                        ].sites.values()
                    )
                )

        for site, site_value in actual_states.items():
            try:
                cur_possible_states = possible_states[site]
            except KeyError as e:
                raise ValueError(
                    f"Non-existing site specified: {site} "
                    f" (allowed: {list(possible_states.keys())})"
                ) from e

            if site_value not in cur_possible_states:
                raise ValueError(
                    f"Non-existing site state '{site_value}' "
                    f"specified for site '{site}' "
                    f"(allowed: {list(cur_possible_states)})"
                )


class Observable(ModelComponent):
    """Specifies an observable"""

    pass


class FeatureMapping:
    """Mapping of features of reaction substrates to reaction products.

    Attributes:
        substrates: List of tuples of unique reaction IDs and species pattern.
            The unique ID is used to distinguish between reactants, in case
            there are multiple reactants of the same name.
        products: List of tuples of unique reaction IDs and species pattern.
        mapping: Dictionary with feature mapping. Dictionary mapping
            product IDs (specified in products) to dictionaries of product
            feature IDs to tuples of reactant IDs and reactant features.
    """

    def __init__(
        self,
        substrates: List[Tuple[str, SpeciesPattern]],
        products: List[Tuple[str, SpeciesPattern]],
        mapping: Dict[str, Dict[str, Tuple[str, str]]],
    ):
        self.substrates = substrates
        self.products = products
        self.mapping = mapping

        # Check that mapping uses existing reactant references
        for product_reference in mapping:
            try:
                next(
                    filter(
                        lambda x: x[0] != product_reference, self.substrates
                    )
                )
            except StopIteration:
                raise ValueError(
                    "Use of undefined reactant reference "
                    f"{product_reference}"
                )
        # TODO further checks

    @staticmethod
    def trivial_mapping(
        substrates: List[SpeciesPattern],
        products: List[SpeciesPattern],
        prefix: str,
    ):
        """Create mapping for a reaction in which there is a 1:1 correspondence
        of substrate and product features and features are just transferred"""
        mapping = {}
        assert len(substrates) == len(products)
        substrates_new = []
        products_new = []
        for idx, (substrate, product) in enumerate(zip(substrates, products)):
            # this may not necessarily be unique
            substrate_id = f"{prefix}_substrate_{idx}"
            product_id = f"{prefix}_product_{idx}"
            substrates_new.append((substrate_id, substrate))
            products_new.append((product_id, product))
            assert product.template == substrate.template
            mapping[product_id] = {
                site: (substrate_id, site) for site in product.template.sites
            }

        return FeatureMapping(substrates_new, products_new, mapping)

    # TODO def fully_defined


class ReactionPattern(ModelComponent):
    """A (possibly concrete) reaction pattern

    Attributes:

    """

    def __init__(
        self,
        id_: str,
        substrates: List[SpeciesPattern],
        products: List[SpeciesPattern],
        reversible: bool,
        mapping: Optional[FeatureMapping] = None,
        group_id: str = None,
        rate_coefficient: sp.Expr = 1,
        rate_law: sp.Expr = None,
        rate_law_generator: RateLawGenerator = None,
        model: Model = None,
        enzyme: SpeciesPattern = None,
        name: str = None,
        add: bool = True,
    ):
        super().__init__(model=model, id_=id_, name=name)

        assert mapping is None or isinstance(mapping, FeatureMapping)
        self.group_id = group_id if group_id is not None else self.id
        assert self.group_id is not None
        self.substrates = substrates
        self.products = products

        self.feature_mapping = mapping
        self.reversible = reversible
        self.enzyme = enzyme
        # some optional extra coefficient to be combined with self.rate_law
        self.rate_coefficient = rate_coefficient

        self.rate_law = None

        if rate_law and rate_law_generator:
            raise ValueError(
                "Provide only one of rate_law or rate_law_generator"
            )
        self.rate_law = rate_law
        self.rate_law_generator = rate_law_generator

        if add:
            self.model.add_reaction(self)

    def __repr__(self):
        substrates = " + ".join(str(r) for r in self.substrates)
        products = " + ".join(str(r) for r in self.products)
        arrow = "<->" if self.reversible else "->"
        return f"<{self.id}: {substrates} {arrow} {products}>"

    def get_substrate_symbols(
        self, symbol_generator: Callable[[SpeciesPattern], sp.Symbol]
    ) -> List[sp.Symbol]:
        return list(map(symbol_generator, self.substrates))

    def get_substrate_compartment_symbols(
        self,
    ) -> List[sp.Symbol]:
        return [sp.Symbol(x.compartment.id) for x in self.substrates]

    def product_symbols(
        self, symbol_generator: Callable[[SpeciesPattern], sp.Symbol]
    ) -> List[sp.Symbol]:
        return list(map(symbol_generator, self.products))

    def assert_balanced(self):
        """Check mass and charge balance"""
        from .model_definition_hl import Metabolite

        for reactant in chain(self.substrates, self.products):
            species_type = reactant.template
            if not isinstance(species_type, Metabolite):
                raise ValueError(
                    "assert_balanced is only supported for reactions "
                    "of which all reactants are of type Metabolite. "
                    f"{species_type.id} is of type {type(species_type)}."
                )
            if species_type.charge is None:
                raise ValueError(
                    f"`charge` is not specified for {species_type.id}"
                )
            if species_type.sum_formula is None:
                raise ValueError(
                    f"`sum_formula` is not specified for {species_type.id}"
                )

        from ..sum_formula import parse_sum_formula, sum_formula_diff

        sum_products = parse_sum_formula(
            "".join(x.template.sum_formula for x in self.products)
        )
        sum_substrates = parse_sum_formula(
            "".join(x.template.sum_formula for x in self.substrates)
        )
        if diff := sum_formula_diff(sum_substrates, sum_products):
            raise ValueError(f"Mass balance failure in {self}: {diff}")

        sum_charges = sum(x.template.charge for x in self.products) - sum(
            x.template.charge for x in self.substrates
        )
        if sum_charges != 0:
            raise ValueError(
                f"Charge balance failure in {self}: {sum_charges}"
            )

    @property
    def involved_compartments(self) -> Set[Compartment]:
        """Get the compartments of the reactants"""
        return {
            reactant.compartment
            for reactant in chain(self.substrates, self.products)
        }

    @property
    def unit(self) -> ureg.Unit:
        """Get the unit of the reaction rate"""
        return self.model.amount_unit / self.model.time_unit

    def get_reactant_symbols(
        self,
        filtered: bool = False,
        representation_type: str = None,
    ) -> List[sp.Symbol]:
        """
        Get symbols for reactants to be used in maths expressions.

        In the order of the reactants in ``rule.substrates``.
        Expected to be globally unique.
        """

        def sym(reactant: SpeciesPattern, idx: int, representation_type):
            return SpeciesSymbol(
                self.model._generate_species_id(reactant),
                # TODO centralize species-reference-id-generation
                # TODO s/substrate/reactant/ for all species-references
                # setting speciesReference is meaningless if
                #  representationType=sum?! (and speciesReference is only
                #  allowed inside kineticLaws)
                species_reference=None
                if representation_type or reactant.is_concrete
                else f"{self.id}_substrate_{idx}",
                representation_type=representation_type,
            )

        return [
            sym(reactant, idx, representation_type=representation_type)
            for idx, reactant in enumerate(self.substrates)
            if not filtered
            or reactant.template not in self.model.ignored_reactants
        ]

    def get_product_symbols(
        self,
        filtered: bool = False,
        representation_type: str = None,
    ) -> List[sp.Symbol]:
        """
        Get symbols for products to be used in maths expressions.

        In the order of the reactants in ``rule.products``.
        Expected to be globally unique.
        """

        def sym(reactant: SpeciesPattern, idx: int, representation_type):
            return SpeciesSymbol(
                self.model._generate_species_id(reactant),
                # TODO centralize species-reference-id-generation
                # TODO s/substrate/reactant/ for all species-references
                # setting speciesReference is meaningless if
                #  representationType=sum?! (and speciesReference is only
                #  allowed inside kineticLaws)
                species_reference=None
                if representation_type or reactant.is_concrete
                else f"{self.id}_product_{idx}",
                representation_type=representation_type,
            )

        return [
            sym(reactant, idx, representation_type=representation_type)
            for idx, reactant in enumerate(self.products)
            if not filtered
            or reactant.template not in self.model.ignored_reactants
        ]


class RateLawGenerator(ABC):
    """Generates rate laws for ReactionPatterns"""

    @abstractmethod
    def expression(self, reaction: ReactionPattern) -> sp.Expr:
        """As sympy expression"""
        pass


class ParameterFactory:
    def __init__(self, model: Model):
        self._model = model

    def michaelis_constant(
        self, enzyme_id: str, reactant_id: str
    ) -> Expression:
        expr_id = f"k_r_m_{enzyme_id}_{reactant_id}"
        return self._model[expr_id] or Expression(
            id_=expr_id,
            expr=Q_(1.0, self._model.concentration_unit),
            sbo_term="SBO:0000027",
            estimated=True,
            non_negative=True,
            model=self._model,
        )

    def forward_rate_constant(
        self, reaction_id: str, order: int
    ) -> Expression:
        expr_id = f"kf_{reaction_id}"
        return self._model[expr_id] or Expression(
            expr_id,
            Q_(
                1.0,
                self._model.concentration_unit ** (1 - order)
                / self._model.time_unit,
            ),
            estimated=True,
            non_negative=True,
            sbo_term="SBO:0000320",
            model=self._model,
        )

    def backward_rate_constant(
        self, reaction_id: str, order: int
    ) -> Expression:
        return Expression(
            f"kb_{reaction_id}",
            Q_(
                1.0,
                self._model.concentration_unit ** (1 - order)
                / self._model.time_unit,
            ),
            estimated=True,
            non_negative=True,
            sbo_term="SBO:0000321",
            model=self._model,
        )

    def equilibrium_constant(self, reaction_id: str, delta: int) -> Expression:
        expr_id = f"k_eq_{reaction_id}"
        return self._model[expr_id] or Expression(
            expr_id,
            Q_(1.0, self._model.concentration_unit**delta),
            sbo_term="SBO:0000281",
            model=self._model,
        )

    def chemical_potential(self, reactant: SpeciesPattern) -> Expression:
        """Return mu0 parameter for the given pattern"""
        # TODO: compartment specific?!
        expr_id = f"mu0_{reactant.template.id}"
        return self._model[expr_id] or Expression(
            expr_id,
            Q_(1.0, self._model._molar_energy_unit),
            estimated=True,
            non_negative=False,
            sbo_term="SBO:0000463",
            model=reactant.template.model,
        )


def filter_reactants(reactants: List[SpeciesPattern]) -> List[SpeciesPattern]:
    """Return list without ignored reactants"""
    return list(
        filter(
            lambda x: x.template not in x.template.model.ignored_reactants,
            reactants,
        )
    )
