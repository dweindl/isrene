"""Classes for higher-level model entities"""
import itertools
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import sympy as sp

from .. import Q_
from .model_definition import *

logger = logging.getLogger(__file__)

__all__ = [
    "AtomMappingReaction",
    "AtomMappingReactionSymmetric",
    "SynthesisReaction",
    "DecayReaction",
    "DiffusionReaction",
    "FacilitatedDiffusionReaction",
    "Metabolite",
    "Enzyme",
    # "MdvObservable", "AmountObservable"
]

# State value for a labeled site
LABELED_STATE = "l"
# State value for an unlabeled site
UNLABELED_STATE = "u"
# State values for isotopic labeling sites
LABELING_STATES = [UNLABELED_STATE, LABELED_STATE]


def atom_site_id(index: int) -> str:
    """Return the ID string for an atom site with the given 1-based index"""
    return f"C{index}"


class Metabolite(SpeciesType):
    """Describes the structure of a Metabolite, such as number and states of
    (isotopically labelable) atoms.
    """

    def __init__(
        self,
        id_: str,
        num_labelable_carbons: Optional[int] = None,
        kegg_id: Optional[str] = None,
        labelable: Optional[bool] = True,
        sum_formula: str = None,
        charge: int = None,
        name: str = None,
        model: Model = None,
    ):
        assert num_labelable_carbons is not None or labelable is False

        site_states = {}
        species_type_instances = []
        if labelable:
            # supporting only carbon labelling for now
            # find or add C_atom species type
            for st in model.species_types:
                if st.id == "C_atom":
                    c_atom_species_type = st
                    break
            else:
                c_atom_species_type = SpeciesType(
                    id_="C_atom",
                    sites={"C_label": LABELING_STATES},
                    model=model,
                )
            # add carbon sites
            species_type_instances = [
                (atom_site_id(i), c_atom_species_type)
                for i in range(1, num_labelable_carbons + 1)
            ]

        # TODO binding sites yet unsupported
        # # bound to anything?
        # site_states['b'] = None

        super().__init__(
            id_=id_,
            sites=site_states,
            name=name,
            model=model,
            species_type_instances=species_type_instances,
        )
        self.num_carbons = num_labelable_carbons
        self.kegg_id = kegg_id
        self.labelable = labelable
        self.sbo_term = "SBO:0000247"
        self.sum_formula = sum_formula
        self.charge = charge

    def unlabeled(self, compartment: Compartment) -> SpeciesPattern:
        """Create a pattern matching only the unlabeled state of this type."""
        if not self.labelable:
            return SpeciesPattern(self, compartment=compartment)

        def unlabeled(states):
            return UNLABELED_STATE if states == LABELING_STATES else states

        site_states = {
            site: unlabeled(possible_states)
            for site, possible_states in self.sites.items()
        }
        for sti_id, sti_st in self.species_type_instances:
            if list(sti_st.sites.values()) == [LABELING_STATES]:
                site_states[sti_id] = UNLABELED_STATE

        return SpeciesPattern(self, site_states, compartment)

    def labeled(self, compartment: Compartment) -> SpeciesPattern:
        """Create a pattern matching only the fully labeled state of this type."""

        def labeled(states):
            return LABELED_STATE if states == LABELING_STATES else states

        site_states = {
            site: labeled(possible_states)
            for site, possible_states in self.sites.items()
        }
        for sti_id, sti_st in self.species_type_instances:
            if list(sti_st.sites.values()) == [LABELING_STATES]:
                site_states[sti_id] = LABELED_STATE
        return SpeciesPattern(self, site_states, compartment)

    @property
    def num_hydrogens(self):
        from ..sum_formula import parse_sum_formula

        if self.sum_formula is None:
            raise ValueError(f"No sum formula specified for {self.id}.")
        return parse_sum_formula(self.sum_formula).get("H", 0)


class Enzyme(SpeciesType):
    def __init__(
        self,
        id_: str,
        sites: Dict = None,
        name: str = None,
        model: Model = None,
    ):
        super().__init__(id_=id_, sites=sites, name=name, model=model)

        # self.sbo_term = 'SBO:0000014'
        self.sbo_term = "SBO:0000245"


class AtomMappingReaction(ReactionGenerator):
    """A reaction pattern based on atom mappings between substrates and products."""

    def __init__(
        self,
        id_: str,
        substrates: List[
            Union["SpeciesPattern", Tuple["SpeciesPattern", str]]
        ],
        products: List[Union["SpeciesPattern", Tuple["SpeciesPattern", str]]],
        parameters: List[Union[Expression]] = None,
        reversible: bool = False,
        compartment: Compartment = None,
        hmr_id: str = None,
        rate_law_generator: Optional[RateLawGenerator] = None,
        rate_law: Optional[Expression] = None,
        group_id: Optional[str] = None,
        enzyme: SpeciesPattern = None,
        name: str = None,
    ):
        if not isinstance(substrates, list):
            substrates = [substrates]
        if not isinstance(products, list):
            products = [products]

        # all should be tuples with (potentially empty) mapping
        def fix_reactants(reactants):
            for i in range(len(reactants)):
                if not isinstance(reactants[i], Tuple):
                    # add empty mapping
                    reactants[i] = (reactants[i], "")

        fix_reactants(substrates)
        fix_reactants(products)

        def check_reactants(reactants: List[Tuple["SpeciesPattern", str]]):
            for reactant, mapping in reactants:
                if mapping and reactant.site_states:
                    # TODO: how to handle partially defined state in
                    #  combination with atom mapping?
                    #  disallow for now
                    raise NotImplementedError(
                        "Atom mapping with partially set site states "
                        "not supported"
                    )

                # If metabolite is labelable, there must be a full mapping
                # defined
                if isinstance(
                    metabolite_type := reactant.template, Metabolite
                ):
                    if (
                        metabolite_type.labelable
                        and len(mapping) != metabolite_type.num_carbons
                    ):
                        raise ValueError(
                            f"Invalid atom mapping: {reactant}"
                            f"has {metabolite_type.num_carbons} labelable "
                            f"positions, but mapping was specified for "
                            f"{len(mapping)} positions."
                        )

        check_reactants(substrates)
        check_reactants(products)

        # deduce model from reactants
        model_from_reactants = {
            r.template.model
            for r, _ in itertools.chain(substrates, products)
            if r.template
        }
        if len(model_from_reactants) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{set(model_from_reactants)}"
            )
        model = model_from_reactants.pop()

        super().__init__(
            id_=id_, reversible=reversible, hmr_id=hmr_id, model=model
        )

        # set compartment on individual reactants
        if compartment:
            for r, _ in itertools.chain(substrates, products):
                r.compartment = compartment

        self.substrates = substrates
        self.products = products
        self.parameters = parameters
        self.compartment = compartment
        self.rate_law = rate_law
        self.rate_law_generator = rate_law_generator
        self.group_id = group_id
        self.enzyme = enzyme
        self.name = name
        self.model.add_reaction(self)

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        # convert mapping
        mapping = {}
        for i_product, (product, product_atoms) in enumerate(self.products):
            product_id = f"{self.id}_product_{i_product}"
            cur_prod_map = {}

            for i_product_atom, product_atom in enumerate(product_atoms):
                for i_substrate, (substrate, substrate_atoms) in enumerate(
                    self.substrates
                ):
                    assert isinstance(substrate.template, Metabolite)
                    if not substrate.template.labelable:
                        continue
                    if (pos := substrate_atoms.find(product_atom)) >= 0:
                        # TODO: centralize speciesReference ID generation
                        substrate_id = f"{self.id}_substrate_{i_substrate}"
                        cur_prod_map[atom_site_id(i_product_atom + 1)] = (
                            substrate_id,
                            atom_site_id(pos + 1),
                        )
                        break
            mapping[product_id] = cur_prod_map
        mapping = FeatureMapping(
            [
                (f"{self.id}_substrate_{idx}", reactant)
                for idx, reactant in enumerate(self.substrates)
            ],
            [
                (f"{self.id}_product_{idx}", product)
                for idx, product in enumerate(self.products)
            ],
            mapping,
        )

        def _get_reactants(reactant_list):
            return [reactant[0] for reactant in reactant_list]

        substrates = _get_reactants(self.substrates)
        products = _get_reactants(self.products)
        yield ReactionPattern(
            id_=self.id,
            substrates=substrates,
            products=products,
            group_id=self.group_id,
            reversible=self.reversible,
            mapping=mapping,
            rate_law=self.rate_law,
            rate_law_generator=self.rate_law_generator,
            model=self.model,
            enzyme=self.enzyme,
            name=self.name,
        )


class AtomMappingReactionSymmetric(ReactionGenerator):
    """A reaction pattern based on multiple possible atom mappings due to
    molecular symmetry.
    """

    def __init__(
        self,
        id_: str,
        reactants: List[
            Tuple[
                List[Union["SpeciesPattern", Tuple["SpeciesPattern", str]]],
                List[Union["SpeciesPattern", Tuple["SpeciesPattern", str]]],
            ]
        ],
        reversible: bool = None,
        compartment: Compartment = None,
        hmr_id: str = None,
        rate_law_generator: Optional[RateLawGenerator] = None,
        enzyme: SpeciesPattern = None,
    ):
        # all should be tuples with (potentially empty) mapping
        def fix_reactants(reactants):
            for i in range(len(reactants)):
                if not isinstance(reactants[i], Tuple):
                    # add empty mapping
                    reactants[i] = (reactants[i], "")

        for reactant_set in reactants:
            fix_reactants(reactant_set[0])
            fix_reactants(reactant_set[1])

        # deduce model from reactants
        model_from_reactants = {
            r.template.model
            for reactant_set in reactants
            for r, _ in itertools.chain(reactant_set[0], reactant_set[1])
            if r.template
        }
        if len(model_from_reactants) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{set(model_from_reactants)}"
            )
        model = model_from_reactants.pop()

        super().__init__(
            id_=id_, reversible=reversible, hmr_id=hmr_id, model=model
        )
        self.reactants = reactants
        self.compartment = compartment
        self.rate_law_generator = rate_law_generator
        self.enzyme = enzyme
        self.model.add_reaction(self)

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        # TODO account for coefficient in rate law
        coefficient = 1.0 / len(self.reactants)

        for i_sub, (substrates, products) in enumerate(self.reactants):
            # parameters_sub = [
            #     self.expression(
            #         f"_{parameter_total.name}_sym_{i_sub}",
            #         coefficient * parameter_total
            #     ) for parameter_total in parameters_total
            # ]
            # TODO: could save a couple of reactions by not going via
            #  intermediary AtomMappingReaction.
            #  e.g. ⚫⚫⚫⚫ -> ⚫⚫⚫⚫ and ⚪⚪⚪⚪ -> ⚪⚪⚪⚪ will be
            #  generated `len(self.reactant)`-times. also other reactions,
            #  depending on the specific atom mapping
            atr = AtomMappingReaction(
                id_=f"{self.id}_sym_{i_sub}",
                substrates=substrates,
                products=products,
                reversible=self.reversible,
                compartment=self.compartment,
                hmr_id=self.hmr_id,
                rate_law_generator=self.rate_law_generator,
                group_id=self.id,
                enzyme=self.enzyme,
            )
            for r in atr.to_elementary():
                # Correct for reaction multiplicity
                # TODO "libSBML SBML unit consistency (Warning): In situations where a mathematical expression contains literal numbers or parameters whose units have not been declared, it is not possible to verify accurately the consistency of the units in the expression. "
                #  modify AST to set "dimensionless"? do for all scalars?
                #  or introduce parameters to avoid magic numbers altogether? reaction_multiplicity_{num_reactions}
                r.rate_coefficient = coefficient
                yield r


class SynthesisReaction(ReactionGenerator):
    """A reaction creating some species out of nothing"""

    def __init__(
        self,
        id_: str,
        species: "SpeciesPattern",
        expr: Expression,
        model: Model,
    ):
        super().__init__(id_=id_, reversible=False, model=model)
        self.species = species
        self.expr = expr
        self.model.add_reaction(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.species),
        )

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        yield ReactionPattern(
            id_=self.id,
            substrates=[],
            products=[self.species],
            rate_law=sp.Symbol(self.expr.id),
            reversible=False,
            # dependencies=[self.expr]
        )


class DecayReaction(ReactionGenerator):
    """A reaction destroying some species"""

    # TODO 1st order vs 0-order

    def __init__(
        self,
        id_: str,
        species: "SpeciesPattern",
        expr: Expression,
        model: Model,
    ):
        super().__init__(id_, model=model)
        self.species = species
        self.expr = expr
        self.model.add_reaction(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.species),
        )

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        yield ReactionPattern(
            id_=self.id,
            substrates=[self.species],
            products=[],
            reversible=False,
            rate_law=sp.Symbol(self.expr.id)
            * sp.Symbol(Model._generate_species_id(None, self.species))
            # dependencies=[self.expr]
        )


class DiffusionReaction(ReactionGenerator):
    """A passive diffusion reaction across a membrane between two adjacent 3D
    compartments"""

    def __init__(
        self,
        id_: str,
        species_pattern: "SpeciesPattern",
        compartment1: Compartment,
        compartment2: Compartment,
        hmr_id: Optional[str] = None,
    ):
        # deduce model from arguments
        model_from_components = {compartment1.model, compartment2.model}
        if species_pattern.template:
            model_from_components.add(species_pattern.template.model)
        if len(model_from_components) != 1:
            raise AssertionError(
                "Must not mix model components of different models: "
                f"{model_from_components}"
            )
        model = model_from_components.pop()

        super().__init__(id_=id_, hmr_id=hmr_id, model=model)
        self.species_pattern: "SpeciesPattern" = species_pattern
        self.compartment1 = compartment1
        self.compartment2 = compartment2
        self.model.add_reaction(self)

    def __repr__(self):
        return "%s(%s, %s, %s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.species_pattern),
            repr(self.compartment1),
            repr(self.compartment2),
        )

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        # TODO ensure state is concentration, not amount
        src = SpeciesPattern(
            species_type=self.species_pattern.template,
            site_states=self.species_pattern.site_states,
            compartment=self.compartment1,
        )
        dest = SpeciesPattern(
            species_type=self.species_pattern.template,
            site_states=self.species_pattern.site_states,
            compartment=self.compartment2,
        )
        mapping = FeatureMapping.trivial_mapping(
            substrates=[src], products=[dest], prefix=self.id
        )

        rate_law = (
            None
            if self.rate_law_generator
            else self._get_rate_law(src=src, dest=dest)
        )

        yield ReactionPattern(
            id_=self.id,
            name=self.name,
            substrates=[src],
            products=[dest],
            reversible=True,
            mapping=mapping,
            rate_law=rate_law,
            rate_law_generator=self.rate_law_generator,
            model=self.model,
        )

    def _get_rate_law(
        self, src: SpeciesPattern, dest: SpeciesPattern
    ) -> sp.Expr:
        """Generate rate expression"""
        src_sym = self.model.get_species_symbol(src)
        dest_sym = self.model.get_species_symbol(dest)
        # kf absorbs diffusivity, area, ...
        kf = Expression(
            f"k_{self.id}",
            Q_(1.0, 1 / self.model.time_unit),
            estimated=True,
            non_negative=True,
            model=self.model,
        )
        # equilibrium might be != 1.0 in case of charged species
        # Note: unlike elsewhere we don't need to correct K_eq for
        #  potentially different units, as they anyway cancel out
        keq = self.model.parameter_factory.equilibrium_constant(
            self.id, delta=0
        )
        rate_law = kf * (src_sym - dest_sym / keq)
        # amount-change for kinetic law
        rate_law *= self.model.get_compartment_symbol(self.compartment2)
        return rate_law


class FacilitatedDiffusionReaction(ReactionGenerator):
    """A facilitated (enzyme-limited) diffusion reaction across a membrane
    between two adjacent 3D compartments"""

    def __init__(
        self,
        id_: str,
        transporter: "SpeciesPattern",
        cargo: "SpeciesPattern",
        compartment1: Compartment,
        compartment2: Compartment,
        model: Model,
    ):
        super().__init__(id_=id_, model=model)

        self.transporter = transporter
        self.cargo = cargo
        self.compartment1 = compartment1
        self.compartment2 = compartment2
        self.model.add_reaction(self)

    def __repr__(self):
        return "%s(%s, %s, %s, %s, %s)" % (
            self.__class__.__name__,
            repr(self.id),
            repr(self.cargo),
            repr(self.transporter),
            repr(self.compartment1),
            repr(self.compartment2),
        )

    def to_elementary(self) -> Iterable[ReactionPattern]:
        if self._generated:
            return
        self._generated = True

        raise NotImplementedError()
