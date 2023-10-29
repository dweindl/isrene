"""Convenience functions for libsbml multi extension"""
import itertools
import logging
from itertools import chain
from pprint import pprint
from typing import Dict, List, Tuple

import libsbml

from ..sbml.sbml_wrappers import _check

logger = logging.getLogger(__file__)


def create_species_type(
    multi_model: libsbml.MultiModelPlugin,
    id_: str,
    name: str = None,
) -> libsbml.MultiSpeciesType:
    """Create an SBML multi SpeciesType"""
    st = multi_model.createMultiSpeciesType()
    _check(st.setId(id_))
    if name:
        _check(st.setName(name))
    return st


def create_species_type_instance(
    species_type: libsbml.MultiSpeciesType,
    sub_species_type: str,
    id_: str,
) -> libsbml.SpeciesTypeInstance:
    """Create an instance of the given SpeciesType"""
    sti = species_type.createSpeciesTypeInstance()
    _check(sti.setId(id_))
    _check(sti.setSpeciesType(sub_species_type))
    return sti


def create_species_feature_type(
    species_type: libsbml.MultiSpeciesType,
    id_: str,
    occur: int = 1,
    values: List[str] = None,
) -> libsbml.SpeciesFeatureType:
    """Create a feature type for the given SpeciesType"""
    sft = species_type.createSpeciesFeatureType()
    _check(sft.setId(id_))
    _check(sft.setOccur(occur))

    if values:
        for value in values:
            create_possible_species_feature_value(sft, value)
    return sft


def create_possible_species_feature_value(
    sft: libsbml.SpeciesFeatureType, id_: str
):
    """Create possible feature values for the given SpeciesFeatureType"""
    psfv = sft.createPossibleSpeciesFeatureValue()
    _check(psfv.setId(id_))
    return psfv


def create_species_feature(
    species: libsbml.Species,
    species_feature_type: str,
    value: str,
    occur: int = 1,
    component: str = None,
) -> None:
    """Create a SpeciesFeature for the given Species"""

    if occur > 1:
        raise NotImplementedError("..")

    sf = species.createSpeciesFeature()
    _check(sf.setSpeciesFeatureType(species_feature_type))
    _check(sf.setOccur(occur))
    if component:
        _check(sf.setComponent(component))

    sfv = sf.createSpeciesFeatureValue()
    _check(sfv.setValue(value))


class MultiModel:
    """
    Parts of an SBML-multi model that are relevant to network generation

    Classes changed by Multi:
    * Model
    * Compartment [skip]
    * Species [required]
    * Reaction
    * SimpleSpeciesReference
    * SpeciesReference.

    New classes:
    * CompartmentReference
    * SpeciesType: (potentially hierarchical) list of species types.
      PySB Monomers.
    * SpeciesFeatureType: declares sites. configuration is specified via
      ``SpeciesFeature`` for a ``Species``
    * PossibleSpeciesFeatureValue: Declares the different values a
      ``SpeciesFeatureType`` can have. Specified via ``SpeciesFeatureValue``
      of a ``Species``.
    * SpeciesTypeInstance
    * InSpeciesTypeBond
    * SpeciesTypeComponentIndex
    * SubListOfSpeciesFeatures
    * OutwardBindingSite
    * SpeciesFeature: a "site" in a ``Species`` as defined in the
      ``SpeciesType`` via ``SpeciesFeatureType``
    * SpeciesFeatureValue: the state of a site in ``SpeciesFeature``. One of
      ``
    * SpeciesTypeComponentMapInProduct:
      in ``listOfSpeciesTypeComponentMapsInProduct`` in ``listOfProducts``

    """

    def __init__(self, sbml_model: libsbml.Model):
        self.sbml_model = sbml_model
        self.multi_model: libsbml.MultiModelPlugin = sbml_model.getPlugin(
            "multi"
        )

        self.rules: List[Rule] = []
        # SBML species ID -> ``Species``
        self.species_by_id: Dict[str, "Species"] = {}
        # SBML speciesType ID -> ``Species``
        self.species: Dict[str, List["Species"]] = {}
        self.species_types: Dict[str, SpeciesType] = {}

        self._check_unsupported()

        self._collect_species_types()
        self._collect_initial_species()
        self._collect_rules()

    def _check_unsupported(self):
        # TODO complain about all unsupported multi features
        for compartment in self.sbml_model.getListOfCompartments():
            multi_compartment = compartment.getPlugin("multi")
            if (
                multi_compartment.isSetIsType()
                and multi_compartment.getIsType()
            ):
                raise NotImplementedError("Cannot handle compartment types.")

    def _collect_initial_species(self):
        # TODO: only with non-zero initial concentrations?!
        for sbml_species in self.sbml_model.getListOfSpecies():
            s = Species.from_sbml(sbml_species, self.species_types)
            if s.species_type:
                # TODO not compatible with hierarchy
                try:
                    self.species[s.species_type.sbml_id].append(s)
                except KeyError:
                    self.species[s.species_type.sbml_id] = [s]

                self.species_by_id[s.sbml_id] = s

        print("Initials:")
        pprint(
            [
                s
                for s in itertools.chain.from_iterable(self.species.values())
                if s.is_concrete()
            ]
        )

    def _collect_rules(self):
        for reaction in self.sbml_model.getListOfReactions():
            if (
                reaction.getTypeCode()
                == libsbml.SBML_MULTI_INTRA_SPECIES_REACTION
            ):
                raise NotImplementedError("SBML_MULTI_INTRA_SPECIES_REACTION")
            # TODO only if uses multi?
            self.rules.append(Rule.from_sbml(self.sbml_model, reaction, self))

        print("Reactions:")
        pprint(self.rules)

    def _collect_species_types(self):
        self.species_types = {}
        # FIXME: must happen in topological order, because they may depend on each other,
        #  for now we just hope that everything is already in topological order in the sbml model
        self.species_types = {
            st.getId(): SpeciesType.from_sbml(
                st, species_types=self.species_types
            )
            for st in self.sbml_model.getPlugin(
                "multi"
            ).getListOfMultiSpeciesTypes()
        }
        print("Species types:")
        pprint(self.species_types)

    def num_concrete_species(self) -> int:
        res = 0
        for ss in self.species.values():
            for s in ss:
                if s.is_concrete():
                    res += 1
        return res

    def get_concrete_species(self) -> List["Species"]:
        """Get list of all concrete Species"""
        return [s for s in chain(*self.species.values()) if s.is_concrete()]

    def get_non_concrete_species(self) -> List["Species"]:
        """Get list of all non-concrete, i.e., pattern Species"""
        return [
            s for s in chain(*self.species.values()) if not s.is_concrete()
        ]


class SpeciesType:
    """SBML-multi SpeciesType - describes the structure of any derived Species"""

    def __init__(
        self,
        sbml_id: str,
        feature_types: List["SpeciesFeatureType"],
        species_type_instances: List["SpeciesTypeInstance"] = None,
    ):
        self.sbml_id: str = sbml_id

        list_of_feature_ids = [feature.id for feature in feature_types]
        if len(list_of_feature_ids) != len(set(list_of_feature_ids)):
            raise ValueError(
                f"Feature IDs of {sbml_id} are not unique: "
                + str(list_of_feature_ids)
            )
        self.own_features: List[SpeciesFeatureType] = feature_types
        self.species_type_instances: List[SpeciesTypeInstance] = (
            species_type_instances or []
        )

    @property
    def features(self):
        # FIXME we need to handle components properly
        #  either we need a feature tree, or need to flatten it properly by creating unique feature IDs
        #  for now we just assume there is only one feature type per component

        for sti in self.species_type_instances:
            if len(sti.species_type.features) > 1:
                raise NotImplementedError(
                    "Lacking proper support for hierarchical species types."
                )

        def _hack(f: SpeciesFeatureType, sti_id: str):
            import copy

            res = copy.deepcopy(f)
            res.id = sti_id
            return res

        derived_features = list(
            chain.from_iterable(
                [_hack(f, sti.id) for f in sti.species_type.features]
                for sti in self.species_type_instances
            )
        )
        return self.own_features + derived_features

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({repr(self.sbml_id)}, {repr(self.features)})"
        )

    @staticmethod
    def from_sbml(
        species_type: libsbml.MultiSpeciesType,
        species_types: Dict[str, "SpeciesType"],
    ):
        if species_type.isSetCompartment():
            raise NotImplementedError(
                "SpeciesType compartments are currently not handled."
            )

        if species_type.getNumSpeciesTypeComponentIndexes():
            raise NotImplementedError(
                "SpeciesType SpeciesTypeComponentIndexes are currently not "
                "handled."
            )

        if species_type.getNumInSpeciesTypeBonds():
            raise NotImplementedError(
                "SpeciesType InSpeciesTypeBonds are currently not " "handled."
            )

        if species_type.getTypeCode() != libsbml.SBML_MULTI_SPECIES_TYPE:
            # might be SBML_MULTI_BINDING_SITE_SPECIES_TYPE
            raise NotImplementedError(
                "Only SBML_MULTI_SPECIES_TYPE SpeciesType is supported."
            )

        features = [
            SpeciesFeatureType.from_feature_type(sft)
            for sft in species_type.getListOfSpeciesFeatureTypes()
        ]
        species_type_instances = [
            SpeciesTypeInstance.from_sbml(sti, species_types)
            for sti in species_type.getListOfSpeciesTypeInstances()
        ]
        st = SpeciesType(
            species_type.getId(), features, species_type_instances
        )
        # TODO move out of here
        species_types[st.sbml_id] = st
        return st

    def __str__(self):
        return self.sbml_id

    def __eq__(self, other):
        return self.sbml_id == other.sbml_id


class SpeciesFeatureType:
    """SBML multi SpeciesFeatureType

    Defines the ID, number of occurrences and possible values a feature of a
    SpeciesType can take.
    """

    def __init__(self, id: str, occur: int, values: List[str]):
        self.id: str = id
        self.occur: int = occur
        self.possible_values: List[str] = values

    @staticmethod
    def from_feature_type(feature_type: libsbml.SpeciesFeatureType):
        values = [
            feat_val.getId()
            for feat_val in feature_type.getListOfPossibleSpeciesFeatureValues()
        ]

        return SpeciesFeatureType(
            feature_type.getId(), feature_type.getOccur(), values
        )

    def __repr__(self):
        return f"{self.id}({self.possible_values})[{self.occur}]"


class SpeciesTypeInstance:
    """SBML-multi SpeciesTypeInstance"""

    def __init__(self, id_: str, species_type: SpeciesType):
        self.id: str = id_
        self.species_type: SpeciesType = species_type

    @staticmethod
    def from_sbml(
        sti: libsbml.SpeciesTypeInstance, species_types: Dict[str, SpeciesType]
    ):
        if sti.isSetCompartmentReference():
            raise NotImplementedError(
                "Compartment references are currently not supported."
            )
        return SpeciesTypeInstance(
            sti.getId(), species_types[sti.getSpeciesType()]
        )


class SpeciesReference:
    """SpeciesReference in a reaction"""

    # TODO: currently also used for or ModifierSpeciesReference, which does
    #  not have feature_map and stoichiometry
    def __init__(
        self, id_: str, species: "Species", reaction: "Rule", feature_map: Dict
    ):
        # species reference ID
        self.id: str = id_
        self.species: Species = species
        self.reaction: Rule = reaction
        self.feature_map: Dict[str, str] = feature_map

    @staticmethod
    def from_sbml(
        multi_model: MultiModel,
        species_reference: libsbml.SpeciesReference,
        reaction: "Rule",
    ):
        if (
            isinstance(species_reference, libsbml.SpeciesReference)
            and species_reference.getStoichiometry() != 1
        ):
            raise NotImplementedError()

        if species_reference.getPlugin("multi").isSetCompartmentReference():
            raise NotImplementedError()

        multi_species_reference = species_reference.getPlugin("multi")
        species = multi_model.species_by_id[species_reference.getSpecies()]
        mapping = (
            _read_feature_map(multi_species_reference, reaction.reactants)
            if isinstance(species_reference, libsbml.SpeciesReference)
            else {}
        )
        return SpeciesReference(
            id_=species_reference.getId(),
            species=species,
            reaction=reaction,
            feature_map=mapping,
        )

    def __repr__(self):
        return f"ref:{self.species.sbml_id}"


class Rule:
    """
    A reaction rule.

    A libsbml.Reaction that has reactants or products that have
    ``speciesType`` set"""

    def __init__(
        self,
        id_: str,
        reactants: List["SpeciesReference"],
        products: List["SpeciesReference"],
        modifiers: List["SpeciesReference"],
        # TODO sbml reversible has no effect
        reversible: bool,
    ):
        self.id = id_
        # TODO species references?
        self.reactants: List[SpeciesReference] = reactants
        self.products: List[SpeciesReference] = products
        self.modifiers: List[SpeciesReference] = modifiers
        self.reversible = reversible

    @staticmethod
    def from_sbml(
        sbml_model: libsbml.Model,
        sbml_reaction: libsbml.Reaction,
        multi_model: MultiModel,
    ):
        """``Rule`` from an SBML Reaction"""
        rule = Rule(
            sbml_reaction.getId(),
            reactants=[],
            products=[],
            modifiers=[],
            reversible=sbml_reaction.getReversible(),
        )

        for reactant in sbml_reaction.getListOfReactants():
            # TODO only if multi?
            rule.reactants.append(
                SpeciesReference.from_sbml(multi_model, reactant, rule)
            )

        for product in sbml_reaction.getListOfProducts():
            # TODO only if multi?
            rule.products.append(
                SpeciesReference.from_sbml(multi_model, product, rule)
            )

        for modifier in sbml_reaction.getListOfModifiers():
            # TODO only if multi?
            rule.modifiers.append(
                SpeciesReference.from_sbml(multi_model, modifier, rule)
            )

        # TODO clean up
        # For reversible reactions, also set feature mapping for the reactants
        if rule.reversible:
            for reactant in rule.reactants:
                reactant.feature_map = invert_feature_mapping(
                    reactant, rule.products
                )

        return rule

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    def is_concrete(self):
        return all(
            reactant.species.is_concrete()
            for reactant in itertools.chain(self.reactants, self.products)
        )


def invert_feature_mapping(
    reactant: SpeciesReference, products: List[SpeciesReference]
) -> Dict[str, Tuple]:
    mapping = {}
    for product in products:
        for product_feature, (
            mapped_reactant,
            reactant_feature,
        ) in product.feature_map.items():
            if reactant == mapped_reactant:
                mapping[reactant_feature] = (product, product_feature)

    required_features = {
        ft.id for ft in reactant.species.species_type.features
    }
    mapped_features = set(mapping.keys())
    if diff := required_features - mapped_features:
        raise AssertionError(f"{reactant=}\n{products=}\n{diff=}")

    return mapping


class Species:
    """An SBML-multi extended Species, i.e. a concrete pool or a pattern

    NOTE: it is currently assumed (not checked) that there exists only one
    Species of each SpeciesType with a given state and compartment.
    """

    def __init__(
        self,
        sbml_id: str,
        compartment: str,
        species_type: SpeciesType,
        feature_values: Dict[str, List[str]],
        sbml_name: str = None,
        sbo_term: str = None,
    ):
        self.sbml_id: str = sbml_id
        self.sbml_name: str = sbml_name if sbml_name is not None else sbml_id
        self.compartment: str = compartment
        self.species_type: SpeciesType = species_type
        self.feature_values: Dict[str, List[str]] = feature_values
        self.sbo_term: str = sbo_term

    @staticmethod
    def from_sbml(
        sbml_species: libsbml.Species, species_types: Dict[str, SpeciesType]
    ):
        multi_species = sbml_species.getPlugin("multi")

        if multi_species.getNumOutwardBindingSites():
            raise NotImplementedError(
                "OutwardBindingSites are currently not supported."
            )

        if multi_species.getNumSubListOfSpeciesFeatures():
            raise NotImplementedError(
                "SubListOfSpeciesFeatures are currently not supported."
            )

        feature_values = feature_values_to_dict(multi_species)

        return Species(
            sbml_species.getId(),
            sbml_species.getCompartment(),
            species_types[multi_species.getSpeciesType()],
            feature_values,
            sbml_name=sbml_species.getName(),
            sbo_term=sbml_species.getSBOTermID(),
        )

    def is_concrete(self):
        return species_is_concrete(self)

    def __repr__(self):
        features = ",".join(f"{k}:{v}" for k, v in self.feature_values.items())
        if features:
            return f"{self.sbml_id}({features})@{self.compartment}"
        return f"{self.sbml_id}@{self.compartment}"

    def get_matches(self, candidates: List["Species"]) -> List["Species"]:
        """Get concrete species that match this pattern"""
        if self.is_concrete():
            return [self]

        matches = []
        for candidate in candidates:
            if not candidate.is_concrete():
                continue
            if candidate.matches_pattern(self):
                matches.append(candidate)
        return matches

    def matches_pattern(self, pattern: "Species") -> bool:
        """Check whether this species matches the given pattern Species"""
        if pattern.sbml_id == self.sbml_id:
            return True

        # TODO hierarchical (in that case SpeciesTypeId matching may be
        #   insufficient (see specs sec. 3.14)
        # TODO assumes compartment is set
        if (
            pattern.species_type.sbml_id == self.species_type.sbml_id
            and pattern.compartment == self.compartment
        ):
            if not pattern.feature_values:
                return True

            # for all defined sites, the site configuration needs to match
            return all(
                self.feature_values[site] == config
                for site, config in pattern.feature_values.items()
            )

        return False

    def all_features_defined(self) -> bool:
        """Check whether all features of this species are defined

        i.e. none is in "don't care state"
        """
        # TODO: needs species type set
        # TODO hierarchical
        for feature_type in self.species_type.features:
            if feature_type.id not in self.feature_values.keys():
                return False

            if feature_type.occur != 1:
                raise NotImplementedError(
                    "Species features with occurrence >1 not supported."
                )
        return True


def feature_values_to_dict(
    multi_species: libsbml.MultiSpeciesPlugin,
) -> Dict[str, List[str]]:
    """Parse feature values of the given SBML MultiSpecies

    SpeciesFeature ID => [value1, value2, ...]

    TODO: "don't care" features are just omitted in the returned dict
    """
    feature_values = {}
    for species_feature in multi_species.getListOfSpeciesFeatures():
        if species_feature.getOccur() != 1:
            raise NotImplementedError(
                "Only occur=1 is currently supported for SpeciesFeatures"
            )
        values = [
            sfv.getValue()
            for sfv in species_feature.getListOfSpeciesFeatureValues()
        ]
        # FIXME: handle "component" attribute properly
        #  ~~we just prefix the speciesfeaturetype with the component - this assumes that the component is always set or always unset for the respective feature type; this will fail otherwise~~
        #  we just use the component ID, assuming this component has only one single feature (which is the case for any isrene model)
        component_id = species_feature.getComponent()
        # sft_id = species_feature.getSpeciesFeatureType()
        feature_values[component_id] = values

    return feature_values


def species_is_concrete(species: Species) -> bool:
    """Check whether the given species is "concrete" (or "fully defined")

    See SBML-multi specifications section 3.19.
    """

    # Species is concrete if (1)-(5) are fulfilled (see SBML-multi specs)

    # 1) "All outwardBindingSites must be free (bindingStatus=“unbound”),
    #    since “bound” sites imply that there is a non-specified binding
    #    partner."

    # TODO outwardBindingSites
    #  all_outward_unbound = all(s == UNBOUND
    #  for s in species.outward_binding_sites)
    # if not all_outward_unbound:
    #     return False

    # 2) "Each speciesFeature occurrence can only have one speciesFeatureValue,
    #    and every occurrence of every speciesFeatureType of every component
    #    of the referenced speciesType must be referenced by exactly one
    #    speciesFeature occurrence."

    # 3) "Only “and” values are allowed for the relation attributes of the
    #    SubListOfSpeciesFeatures objects."

    # TODO SubListOfSpeciesFeatures

    # 4) "Only one single SpeciesFeatureValue object is allowed for any
    #    speciesFeature."

    if not species.all_features_defined():
        return False

    # 5) "The referenced compartment cannot be a compartment type, which means
    #    the value of the isType attribute of the referenced compartment can
    #    only be “false”."

    # TODO must be non-type compartment

    return True


def _read_feature_map(
    multi_species_reference, reactants: List[SpeciesReference]
) -> Dict[str, Tuple]:
    """Read feature mapping for the given product

    Arguments:
        multi_species_reference: Reference to reaction product
        reactants: List of reactants for the given reactions
    Returns:
        Feature mapping for given product (dict keys are product component
        IDs, values are tuple of (reactant, component).
    """
    mapping = {}
    for (
        component_mapping
    ) in multi_species_reference.getListOfSpeciesTypeComponentMapInProducts():
        reactant = None
        for reactant in reactants:
            if reactant.id == component_mapping.getReactant():
                break
        assert reactant is not None
        # TODO non-unique?
        mapping[component_mapping.getProductComponent()] = (
            reactant,
            component_mapping.getReactantComponent(),
        )
    return mapping


def _has_nonzero_concentration(species: libsbml.Species) -> bool:
    """
    Determine whether the given species has non-zero initial
    concentration/amount.
    """
    # Non-zero initialAmount/initialConcentration?
    if species.getHasOnlySubstanceUnits():
        if species.getInitialAmount():
            return True
    elif species.getInitialConcentration():
        return True

    # any InitialAssignments?
    model = species.getModel()
    if not (ia := model.getInitialAssignmentBySymbol(species.getId())):
        return False

    # check if the assigment is nonzero
    from sbmlmath import sbml_math_to_sympy

    expr = sbml_math_to_sympy(ia)
    # is the initial assignment expression a single parameter whose value is
    #  non-zero?
    if p := model.getParameter(str(expr)):
        return bool(p.getValue())
    return False
