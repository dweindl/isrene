"""Functions for generating reactions networks from SBML-multi models
and exporting to SBML-core models"""

# allow executing this file directly
if __name__ == "__main__":
    __package__ = "isrene.sbml_multi"

import itertools
import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Sequence, Tuple, Union

import libsbml
import sympy as sp
from more_itertools import partition
from sbmlmath import SpeciesSymbol, sbml_math_to_sympy, sympy_to_sbml_math

from ..sbml.sbml_wrappers import (
    _check,
    _check_lib_sbml_errors,
    create_reaction,
    create_species,
    remove_species_reference_ids,
)
from .sbml_multi_wrappers import (
    MultiModel,
    Rule,
    Species,
    SpeciesReference,
    _has_nonzero_concentration,
)

logger = logging.getLogger(__file__)


class NetworkGenerator:
    """
    Network generation based on SBML-multi

    Takes an SBML-multi model, expands the rules and generates an SBML-core
    model.
    """

    # TODO: different modes:
    #   1) just collect species
    #   2) create expanded sbml with all reactions
    #   3) create expanded sbml with RateRules instead of loads of reactions
    def __init__(self, model: Union[libsbml.Model, str, Path]):
        # load model from file?
        if isinstance(model, (str, Path)):
            sbml_reader: libsbml.SBMLReader = libsbml.SBMLReader()
            self.sbml_document = sbml_reader.readSBMLFromFile(str(model))
            model = self.sbml_document.getModel()

        self.sbml_model = model
        self.multi_model = MultiModel(self.sbml_model)
        self._generated = False
        self.species = {}

    def expand_model(self) -> libsbml.Model:
        """Determines all possible species and adds them to ``multi_model``

        Returns a list of concrete species in the model (both, existing ones
        and those generated from rules).
        """
        # collect seed species
        # TODO: add option to consider additional species with 0-concentration
        #  as seed species
        self.species = {
            species_type_id: [
                pattern
                for pattern in patterns
                if pattern.is_concrete
                and _has_nonzero_concentration(
                    self.sbml_model.getSpecies(pattern.sbml_id)
                )
            ]
            for species_type_id, patterns in self.multi_model.species.items()
        }
        from pprint import pformat

        logger.debug(f"Seed species:\n{pformat(self.species)}")

        rule_matchers = [RuleMatcher(rule) for rule in self.multi_model.rules]

        def num_species():
            return sum(map(len, self.species.values()))

        # TODO track number of reactions
        iteration = -1
        while True:
            iteration += 1
            logger.info(f"Iteration {iteration} " f"({num_species()} species)")

            # apply all rules and check if species set got extended
            num_added = 0
            for rule_matcher in rule_matchers:
                try:
                    num_added += rule_matcher.apply(self.species)
                except ValueError as e:
                    raise ValueError(
                        f"Error processing rule {rule_matcher.rule}."
                    ) from e

            # nothing added -> quit
            if num_added == 0:
                break

        logger.info(
            f"Finished after {iteration} iterations: "
            f"{num_species()} species"
        )

        self._generated = True
        return [s for s in chain(*self.species.values()) if s.is_concrete()]

    def _create_pattern_to_species_map(
        self,
    ) -> Dict["Species", List["Species"]]:
        """Create map from non-concrete species to all matching concrete
        species"""
        non_concrete_species, concrete_species = partition(
            lambda x: x.is_concrete(),
            chain(*self.multi_model.species.values(), *self.species.values()),
        )
        # remove duplicates
        #  (seed species are in self.multi_model.species and in self.species)
        seen = set()
        concrete_species = [
            x
            for x in concrete_species
            if not (x.sbml_id in seen or seen.add(x.sbml_id))
        ]
        seen = set()
        non_concrete_species = [
            x
            for x in non_concrete_species
            if not (x.sbml_id in seen or seen.add(x.sbml_id))
        ]

        return {
            pattern: pattern.get_matches(concrete_species)
            for pattern in non_concrete_species
        }

    def create_sbml_core_model(self, filename: str):
        """Create an SBML core compatible model from the expanded network"""
        logger.debug("Generating SBML core model...")

        if not self._generated:
            self.expand_model()

        # create new model (and document?) without multi
        new_sbml_doc = self.sbml_model.getSBMLDocument().clone()
        new_sbml_model = new_sbml_doc.getModel()

        # Remove all multi:SpeciesTypes
        # species_types = [
        #     species_type.getId() for species_type
        #     in new_sbml_model.getPlugin('multi').getListOfMultiSpeciesTypes()
        # ]
        # map(new_sbml_model.removeSpeciesType, species_types)
        new_sbml_model.getPlugin("multi").getListOfMultiSpeciesTypes().clear()

        pattern_mappings = self._create_pattern_to_species_map()
        self._create_concrete_species(new_sbml_model)

        logger.debug("Writing reactions ...")
        self._create_reactions(new_sbml_model, pattern_mappings)

        logger.debug("Expanding math elements ...")
        # TODO: can skip reactions here, they have been handled before
        _expand_all_math(new_sbml_model, pattern_mappings)

        logger.debug("Removing non-concrete patterns ...")
        # remove non-concrete species
        for non_concrete_species in pattern_mappings.keys():
            non_concrete_species_id = non_concrete_species.sbml_id
            new_sbml_model.removeSpecies(non_concrete_species_id)

        # remove multi info from all species
        for species in new_sbml_model.getListOfSpecies():
            species.getPlugin("multi").unsetSpeciesType()
            species.getPlugin("multi").getListOfSpeciesFeatures().clear()

        # remove multi info from all compartments
        for compartment in new_sbml_model.getListOfCompartments():
            compartment.getPlugin("multi").unsetIsType()

        # disable multi
        _check(
            new_sbml_doc.disablePackage(
                new_sbml_doc.getPlugin("multi").getURI(), "multi"
            )
        )

        remove_species_reference_ids(new_sbml_model)

        # verify there is no multi left
        # if new_sbml_model.all_elements_from_plugins:
        # logger.warning(new_sbml_model.all_elements_from_plugins)
        for x in new_sbml_model.getListOfAllElementsFromPlugins():
            logger.warning(f"Plugin element: {x}")

        if (
            libsbml.SBMLWriter().writeSBMLToFile(new_sbml_doc, filename)
            is not True
        ):
            raise IOError(f"Error writing SBML model to file `{filename}`.")

        new_sbml_doc.validateSBML()
        _check_lib_sbml_errors(new_sbml_doc, show_warnings=False)

        logger.debug(f"Num species: {new_sbml_model.getNumSpecies()}")
        logger.debug(f"Num parameters: {new_sbml_model.getNumParameters()}")
        logger.debug(f"Num reactions: {new_sbml_model.getNumReactions()}")

    def _create_concrete_species(self, new_sbml_model: libsbml.Model):
        """Create concrete Species for the expanded model"""
        concrete_species = [
            s for s in chain(*self.species.values()) if s.is_concrete()
        ]

        for s in concrete_species:
            if new_sbml_model.getSpecies(s.sbml_id) is not None:
                # exists already
                continue

            logger.debug(f"Creating species {s.sbml_id}")

            create_species(
                new_sbml_model,
                sbml_id=s.sbml_id,
                compartment=s.compartment,
                name=s.sbml_name,
                sbo_term=s.sbo_term,
                # TODO other attributes from species(type)?
                # TODO annotations from species type?
                # TODO data from any other packages?
            )

    def _create_reactions(
        self,
        new_sbml_model: libsbml.Model,
        pattern_mapping: Dict["Species", List["Species"]],
    ):
        """Create concrete reactions after network expansion

        Replaces all reaction rules by concrete instances.
        """
        pattern_mapping_str = {
            pattern.sbml_id: [x.sbml_id for x in concrete_matches]
            for pattern, concrete_matches in pattern_mapping.items()
        }

        for rule in self.multi_model.rules:
            if rule.is_concrete():
                # nothing to do
                logger.debug(f"Skipping concrete reaction {rule.id}")
                continue

            logger.debug(f"Expanding reaction {rule.id}")

            reaction_template = self.sbml_model.getReaction(rule.id)
            # add concrete reaction for all combinations of reactants
            matcher = RuleMatcher(rule)
            matching_reactant_sets = _get_matching_reactant_sets(
                self.species, matcher.reactants
            )

            # counts concrete reaction instances of the current rule
            i_rxn = -1
            # current sbml reaction instance
            sbml_rxn = None
            # math expression of the current rule
            kl_sympy_math = sbml_math_to_sympy(
                reaction_template.getKineticLaw()
            )

            modifier_matches = list(
                chain.from_iterable(
                    modifier.species.get_matches(
                        self.species[modifier.species.species_type.sbml_id]
                    )
                    for modifier in rule.modifiers
                )
            )

            for i_rxn, reactant_set in enumerate(matching_reactant_sets):
                # create concrete reaction
                sbml_rxn = create_reaction(
                    model=new_sbml_model,
                    reaction_id=f"{rule.id}_{i_rxn}",
                    reversible=rule.reversible,
                    fast=reaction_template.getFast(),
                )
                # Replace reactant/product/modifier patterns by concrete
                # matches
                # Also collect substitutions for kinetic laws:
                # - species reference IDs in pattern => concrete reactants
                # - species pattern IDs (without representation type sum)
                #   in pattern => concrete reactants
                species_reference_substitutions = {}
                for idx, (reactant, reactant_pattern) in enumerate(
                    zip(reactant_set, rule.reactants)
                ):
                    species_ref = sbml_rxn.createReactant()
                    species_ref.setId(f"{sbml_rxn.getId()}_substrate_{idx}")
                    species_ref.setSpecies(reactant.sbml_id)
                    # TODO according to template
                    species_ref.setConstant(True)
                    # species_ref.setStoichiometry(coeff)
                    species_reference_substitutions[
                        reactant_pattern.species.sbml_id
                    ] = reactant.sbml_id
                    species_reference_substitutions[
                        f"{rule.id}_substrate_{idx}"
                    ] = reactant.sbml_id

                for idx, (product, product_pattern) in enumerate(
                    zip(matcher.products, rule.products)
                ):
                    # features of reaction product
                    feature_values = matcher._get_product_feature_values(
                        product, reactant_set, matcher.reactants
                    )
                    try:
                        product_species = create_species_from_template(
                            product.species, feature_values=feature_values
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Error creating species {product.species} for "
                            f"rule {rule}."
                        ) from e
                    product_species = _get_species(
                        self.species, product_species
                    )
                    if not product_species:
                        raise AssertionError(f"Species missing {product}")
                    species_ref = sbml_rxn.createProduct()
                    species_ref.setId(f"{sbml_rxn.getId()}_product_{idx}")
                    species_ref.setSpecies(product_species.sbml_id)
                    # TODO according to template
                    species_ref.setConstant(True)
                    # species_ref.setStoichiometry(coeff)
                    species_reference_substitutions[
                        product_pattern.species.sbml_id
                    ] = product_species.sbml_id
                    species_reference_substitutions[
                        f"{rule.id}_product_{idx}"
                    ] = product_species.sbml_id

                # handle modifier:
                for modifier in modifier_matches:
                    sbml_modifier = sbml_rxn.createModifier()
                    sbml_modifier.setSpecies(modifier.sbml_id)

                # update species references in kinetic laws
                # TODO handle <ci multi:representationType="numericValue">
                #   for possibleSpeciesFeatureValue
                for sym in kl_sympy_math.free_symbols:
                    if isinstance(sym, SpeciesSymbol):
                        if sym.representation_type == "sum":
                            # replace by sum of concentrations/amounts of
                            # species matching the given pattern
                            try:
                                matching_elements = [
                                    sp.Symbol(s)
                                    for s in pattern_mapping_str[sym.name]
                                ]
                            except KeyError:
                                # if the given key doesn't exist, it means we
                                #  already have a concrete species
                                matching_elements = [sp.Symbol(sym.name)]
                            species_reference_substitutions[sym] = (
                                sp.Add(*matching_elements)
                                if matching_elements
                                else sp.Float(0)
                            )
                        elif sym.species_reference:
                            # replace species reference by concrete species
                            #  matching that reactant in the current instance
                            species_reference_substitutions[
                                sym
                            ] = species_reference_substitutions[
                                sym.species_reference
                            ]

                kl = sbml_rxn.createKineticLaw()
                if kl_sympy_math:
                    sbml_math = sympy_to_sbml_math(
                        kl_sympy_math.subs(species_reference_substitutions)
                    )
                    _check(kl.setMath(sbml_math))

            if i_rxn < 0:
                logger.warning(f"No matching reactants for rule {rule}")
            elif i_rxn == 0:
                # only one instance was added, simplify id
                sbml_rxn.setId(rule.id)
            # Remove reaction pattern
            new_sbml_model.removeReaction(rule.id)


class RuleMatcher:
    """Match reaction rules against current list of species and create products"""

    def __init__(self, rule: Rule):
        self.rule: Rule = rule

        # TODO: could ignore any species that do not have a SpeciesType
        #  for increased efficiency, but currently elsewhere all reactants are
        #  expected
        self.reactants: List["SpeciesReference"] = [
            r for r in self.rule.reactants if r.species.species_type
        ]
        self.products: List["SpeciesReference"] = [
            r for r in self.rule.products if r.species.species_type
        ]

    def apply(self, species_list: Dict[str, List["Species"]]) -> int:
        """Apply rule to species_list and add any new products there

        Returns:
            The number of added concrete species
        """
        # Apply for both forward and backward reaction
        num_added = self._apply(species_list, self.reactants, self.products)
        if not self.rule.reversible:
            return num_added
        return num_added + self._apply(
            species_list, self.products, self.reactants
        )

    def _apply(
        self,
        species_list: Dict[str, List["Species"]],
        reactants: List["SpeciesReference"],
        products: List["SpeciesReference"],
    ) -> int:
        """Apply rule to species_list and add any new products there

        Returns:
            The number of added concrete species
        """
        num_added = 0
        # for each potential set of educts
        for reactant_set in _get_matching_reactant_sets(
            species_list, reactants
        ):
            # create new species?
            for product in products:
                # features of reaction product
                feature_values = self._get_product_feature_values(
                    product, reactant_set, reactants
                )
                product_species = create_species_from_template(
                    product.species, feature_values=feature_values
                )

                # does species with these features exist?
                if _get_species(species_list, product_species) is None:
                    # No - create
                    logger.debug(
                        f"{self.rule.id} Adding new species: "
                        f"{product_species} ({product_species.species_type})"
                    )
                    species_list[product_species.species_type.sbml_id].append(
                        product_species
                    )
                    num_added += 1
                else:
                    pass
                    # logger.debug(f"\t\tAlready exists: {product_species} {product_species.feature_values}")

        return num_added

    def _get_product_feature_values(
        self,
        product: SpeciesReference,
        reactant_set: Sequence[Species],
        reactants: Sequence[SpeciesReference],
    ) -> Dict[str, List[str]]:
        """Get product feature values based on the feature values of the
        provided reactants"""
        feature_values = {}
        # TODO ordering should match feature definition
        for product_feature, (
            reactant_ref,
            reactant_feature,
        ) in product.feature_map.items():
            # find reaction that determines the current feature
            feature_source_reactant_idx = next(
                i for i, r in enumerate(reactants) if r.id == reactant_ref.id
            )
            feature_source_reactant = reactant_set[feature_source_reactant_idx]
            try:
                feature_values[
                    product_feature
                ] = feature_source_reactant.feature_values[reactant_feature]
            except KeyError as e:
                # not fully defined
                logger.warning(
                    (
                        self.rule.id,
                        feature_values,
                        product_feature,
                        feature_source_reactant.feature_values,
                        reactant_feature,
                    ),
                )
                raise e
        return feature_values


def _get_matching_reactant_sets(
    species_list: Dict[str, List["Species"]],
    reactants: Iterable["SpeciesReference"],
) -> Generator[Tuple["Species"], None, None]:
    """Generate sets of matching reactions for the current rule

    Arguments:
        species_list: Dict of available species
        reactants: List of required species
    """
    reactant_matches = []

    for reactant in reactants:
        cur_matches = reactant.species.get_matches(
            species_list[reactant.species.species_type.sbml_id]
        )

        if not cur_matches:
            # reaction can't take place
            return
        # logger.debug(f"\t Match {reactant}: {cur_matches}")
        reactant_matches.append(cur_matches)

    # for each potential set of educts
    yield from itertools.product(*reactant_matches)


def _expand_all_math(
    new_sbml_model: libsbml.Model,
    pattern_mapping: Dict["Species", List["Species"]],
):
    """In all math elements, replace multi-elements

    i.e. replace <ci multi:representationType="sum"> by sums of matching
    species
    """
    # create ID-based mapping from non-concrete species to matching concrete
    # species
    pattern_mapping_str = {
        pattern.sbml_id: [x.sbml_id for x in concrete_matches]
        for pattern, concrete_matches in pattern_mapping.items()
    }

    # process all math elements of the model
    for element in new_sbml_model.getListOfAllElements():
        if get_math := getattr(element, "getMath", None):
            replacements_made = _expand_math(
                get_math(), new_sbml_model, pattern_mapping_str
            )
            if not replacements_made:
                continue

            # simplify to get rid of expressions that reduce to zero after
            #  expansion
            sympy_math = sbml_math_to_sympy(element)
            # For some expressions, sympy gets stuck during simplification
            # Skip simplification for larger expressions (the threshold is
            # totally arbitrary, but did the job)
            if len(sympy_math.free_symbols) < 10:
                simplified_sympy_math = sp.simplify(sympy_math)
                simplified_math = sympy_to_sbml_math(simplified_sympy_math)
                _check(element.setMath(simplified_math))


def _expand_math(
    ast_node: libsbml.ASTNode,
    sbml_model: libsbml.Model,
    pattern_mapping: Dict[str, List[str]],
) -> bool:
    """Replace <ci multi:representationType="sum"> by sums of matching
    concrete species

    Returns:
         True if any replacements were made, False otherwise.
    """
    replacements_made = False
    # If name-type node, check if it needs to be replaced
    if (
        ast_node.getType() == libsbml.AST_NAME
        and sbml_model.getSpecies(name := ast_node.getName())
        and (
            multi_ast_node := ast_node.getPlugin("multi")
        ).isSetRepresentationType()
    ):
        if (repr_type := multi_ast_node.getRepresentationType()) == "sum":
            # construct new ASTNode for sum of mapping species
            try:
                matching_elements = [
                    sp.Symbol(s) for s in pattern_mapping[name]
                ]
            except KeyError:
                # if the given key doesn't exist, it means we already
                # have a concrete species
                matching_elements = [sp.Symbol(name)]

            # print("Matches:", name, matching_elements)
            expansion = (
                sp.Add(*matching_elements)
                if matching_elements
                else sp.Float(0)
            )
            expanded_node = sympy_to_sbml_math(expansion)
            ast_node.replaceArgument(name, expanded_node)
            replacements_made = True
            # print("replaced:", ast_node, libsbml.formulaToL3String(ast_node))

            # Unset. (Otherwise this is not happening if replacement
            # above is a single node)
            multi_ast_node.unsetRepresentationType()
        else:
            raise NotImplementedError(
                f"Unknown representationType: {repr_type}"
            )

    # descend the tree
    for i_child in range(ast_node.getNumChildren()):
        child = ast_node.getChild(i_child)
        replacements_made |= _expand_math(child, sbml_model, pattern_mapping)
    return replacements_made


def create_species_from_template(
    template: Species,
    feature_values: Dict[str, List[str]],
) -> Species:
    """Create a copy of this species with updated state and ID"""

    # TODO generate some unique ID based on compartment and feature values
    def feature_vals_to_id_str(feature, values):
        return f"{feature}_{'_'.join(values)}"

    try:
        ft = "_".join(
            map(
                lambda feature: feature_vals_to_id_str(
                    feature.id, feature_values[feature.id]
                ),
                template.species_type.features,
            )
        )
    except KeyError as e:
        raise ValueError(
            "Undefined features. "
            f"Missing feature value for feature {e.args[0]}"
            f" of {template} (defined features: {feature_values})."
        )
    id_str = f"{template.species_type.sbml_id}_{template.compartment}_{ft}"

    def feature_vals_to_name_str(feature, values):
        return f"{feature}={','.join(values)}"

    ft = ", ".join(
        feature_vals_to_name_str(feature.id, feature_values[feature.id])
        for feature in template.species_type.features
    )
    # TODO species type name?
    name_str = (
        f"{template.species_type.sbml_id} in {template.compartment} " f"({ft})"
    )

    species = Species(
        # TODO
        sbml_id=id_str,
        sbml_name=name_str,
        compartment=template.compartment,
        species_type=template.species_type,
        feature_values=feature_values,
        sbo_term=template.sbo_term,
    )
    assert species.is_concrete()
    return species


def _get_species(
    species_list: Dict[str, List["Species"]], other: "Species"
) -> Union["Species", None]:
    """Get species matching the argument or None"""
    return next(
        (
            s
            for s in species_list[other.species_type.sbml_id]
            if (
                s.species_type == other.species_type
                and s.feature_values == other.feature_values
                and s.compartment == other.compartment
            )
        ),
        None,
    )


def _console_logging(log_level=logging.DEBUG):
    """Set up `logging` to print to console"""
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main():
    """Usage: python __file__ path-to-model-to-expand.xml"""
    import sys

    _console_logging()
    file = sys.argv[1]

    # load model
    sbml_reader: libsbml.SBMLReader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBMLFromFile(file)
    sbml_model = sbml_doc.getModel()

    # generate and export network
    nwg = NetworkGenerator(model=sbml_model)
    nwg.expand_model()
    core_model_name = "expanded.xml"
    nwg.create_sbml_core_model(core_model_name)

    # test import via amici
    from amici.sbml_import import SbmlImporter

    SbmlImporter(core_model_name).sbml2amici(
        model_name="deleteme", verbose=True
    )


if __name__ == "__main__":
    main()
