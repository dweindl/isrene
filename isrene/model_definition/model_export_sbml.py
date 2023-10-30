"""Functionality for model export to SBML multi"""

import contextlib
import logging
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict, Tuple, Union

import libsbml
import sympy as sp
from pint.errors import DimensionalityError
from sbmlmath import SpeciesSymbol, sympy_to_sbml_math
from sympy.utilities.iterables import multiset_permutations

from .. import ureg
from ..sbml.annotations import set_annotations
from ..sbml.sbml_pint import unit_from_pint
from ..sbml.sbml_wrappers import (
    _check,
    _check_lib_sbml_errors,
    create_parameter,
    create_species,
)
from ..sbml_multi import enable_multi
from ..sbml_multi.sbml_multi_wrappers import (
    create_possible_species_feature_value,
    create_species_feature,
    create_species_feature_type,
    create_species_type,
    create_species_type_instance,
)
from ..sum_formula import get_nominal_mass
from . import (
    AmountObservable,
    Compartment,
    Expression,
    InitialConcentration,
    MdvObservable,
    Model,
    RawMdvObservable,
    ReactionPattern,
    SpeciesPattern,
    SpeciesType,
)
from .model_definition_hl import (
    LABELED_STATE,
    LABELING_STATES,
    UNLABELED_STATE,
)

Q_ = ureg.Quantity
logger = logging.getLogger(__file__)

# XML namespace URI and prefix for SBML annotations elements
annotation_xmlns = "some_ns_uri"
annotation_prefix = "myannot"


class ModelExporterSbmlMulti:
    """Exports a model to SBML-multi"""

    def __init__(
        self, model: Model, sbml_version: int = 3, sbml_level: int = 2
    ):
        self.model = model
        # Note: False is not really supported; would only be possible
        #  if no atom mapping reactions were included
        self._multi = True
        self.sbml_document = None
        self.sbml_model = None
        self.sbml_version = sbml_version
        self.sbml_level = sbml_level

    def export(self, filename: Union[str, Path]) -> None:
        """Export to SBML file"""
        filename = Path(filename)
        if self.sbml_model is None:
            self.generate_sbml()
        logger.debug(f"Writing model to {filename}")
        filename.parent.mkdir(parents=True, exist_ok=True)
        if not libsbml.writeSBMLToFile(self.sbml_document, str(filename)):
            with contextlib.suppress(RuntimeError):
                _check_lib_sbml_errors(self.sbml_document)
            raise RuntimeError(f"Failed to write model to {filename}")

        self.validate()

    def generate_sbml(self) -> libsbml.Model:
        """Generate SBML model"""
        self.sbml_document = libsbml.SBMLDocument(
            self.sbml_version, self.sbml_level
        )
        if self._multi:
            enable_multi(self.sbml_document)
        self.sbml_model = self.sbml_document.createModel()
        self.sbml_model.setId(self.model.id)

        self._create_unit_definitions()
        self._create_compartments()

        if self._multi:
            self._create_species_types()

        # NOTE: reactions need to be created before processing expressions, as
        #  the latter may refer to species that are only created here from the
        #  reactant patterns
        self._create_reactions()

        self._process_expressions()

        self._process_initials()

        self._process_observables()

        # TODO annotations

        logger.debug(f"Num species: {self.sbml_model.getNumSpecies()}")
        logger.debug(f"Num parameters: {self.sbml_model.getNumParameters()}")
        logger.debug(f"Num reactions: {self.sbml_model.getNumReactions()}")
        num_species_types = self.sbml_model.getPlugin(
            "multi"
        ).getNumMultiSpeciesTypes()
        logger.debug(f"Num species types: {num_species_types}")

        for rxn in self.sbml_model.getListOfReactions():
            math = libsbml.formulaToL3String(rxn.getKineticLaw().getMath())
            logger.debug(f"\t{rxn.getId()}: {math}")
        return self.sbml_model

    def _create_unit_definitions(self) -> None:
        """Create the basic unit definitions and set default units"""

        # time unit
        assert self.model.time_unit == ureg.seconds
        self.sbml_model.setTimeUnits("second")

        # amount unit
        sbml_units = unit_from_pint(
            self.model.amount_unit, self.sbml_model, ureg=ureg
        )
        self.sbml_model.setExtentUnits(sbml_units)
        self.sbml_model.setSubstanceUnits(sbml_units)

        sbml_units = unit_from_pint(
            self.model.length_unit, self.sbml_model, ureg=ureg
        )
        self.sbml_model.setLengthUnits(sbml_units)

        sbml_units = unit_from_pint(
            self.model.area_unit, self.sbml_model, ureg=ureg
        )
        self.sbml_model.setAreaUnits(sbml_units)

        sbml_units = unit_from_pint(
            self.model.volume_unit, self.sbml_model, ureg=ureg
        )
        self.sbml_model.setVolumeUnits(sbml_units)

    def _create_compartments(self) -> None:
        """Create SBML compartments"""
        for c in self.model.compartments:
            sbml_comp = self.sbml_model.createCompartment()
            _check(sbml_comp.setId(c.id))
            sbml_comp.setConstant(True)
            if c.dimensions == 3:
                unit = self.model.volume_unit
            elif c.dimensions == 2:
                unit = self.model.area_unit
            else:
                raise ValueError("dimensions")

            sbml_comp.setSpatialDimensions(c.dimensions)
            sbml_comp.setUnits(
                unit_from_pint(unit, self.sbml_model, ureg=ureg)
            )
            # TODO: initial assignment or set directly
            if isinstance(c.size, Q_):
                # set directly
                if c.dimensions == 2:
                    sbml_comp.setSize(c.size.m_as(self.model.area_unit))
                elif c.dimensions == 3:
                    sbml_comp.setSize(c.size.m_as(self.model.volume_unit))
            else:
                size_par = self._expression_to_sbml(c.size)
                size_par.setUnits(sbml_comp.getUnits())
                rule = self.sbml_model.createInitialAssignment()
                rule.setSymbol(sbml_comp.getId())
                # math_ast = _check(libsbml.parseL3Formula(size_par.getId()))
                math_ast = libsbml.parseL3Formula(size_par.getId())
                _check(rule.setMath(math_ast))

            if c.sbo_term:
                sbml_comp.setSBOTerm(c.sbo_term)

            if self._multi:
                # currently all compartments are regular SBML-core compartments
                sbml_comp.getPlugin("multi").setIsType(False)

    def _process_expressions(self) -> None:
        for expression in self.model.expressions:
            self._expression_to_sbml(expression)

    def _process_initials(self) -> None:
        """Create concrete species, initial value parameters and
        InitialAssignments for all provided initial concentrations/amounts
        """
        for initial in self.model.initials:
            if isinstance(initial, InitialConcentration):
                self._process_initial_concentration(initial)
            else:
                raise NotImplementedError(type(initial))

    def _process_initial_concentration(self, initial: InitialConcentration):
        """Create concrete species, initial value parameters and
        InitialAssignments for the given InitialConcentration

        TODO: If the initial value is a numeric value, this is set on the SBML
         species directly, otherwise an InitialAssignment is created.
        """
        sbml_species = self._create_species(initial.species)
        _check(sbml_species.setBoundaryCondition(initial.constant))
        _check(sbml_species.setConstant(initial.constant))
        rule = self.sbml_model.createInitialAssignment()
        rule.setSymbol(sbml_species.getId())

        if isinstance(initial.value, Q_):
            # create parameter
            par_id = (
                f"{sbml_species.getId()}_"
                f"{initial.species.compartment.id}_initial_concentration"
            )
            species_name = sbml_species.getName() or sbml_species.getId()
            par_name = (
                f"Initial concentration of {species_name} "
                f"in {initial.species.compartment.id}"
            )
            # TODO handle area concentrations properly
            try:
                units = unit_from_pint(
                    self.model.concentration_unit, self.sbml_model, ureg=ureg
                )
                value = initial.value.m_as(self.model.concentration_unit)
            except DimensionalityError:
                units = "dimensionless"
                value = initial.value.m
            p = create_parameter(
                self.sbml_model,
                par_id,
                constant=False,
                units=units,
                value=value,
                name=par_name,
            )
            set_annotations(
                p,
                {
                    "parameter": {
                        "estimated": str(initial.estimated),
                        "non_negative": "true",
                    },
                },
            )
            p.setSBOTerm("SBO:0000196")
            math_ast = libsbml.parseL3Formula(par_id)
        elif isinstance(initial.value, Expression):
            # assign expression directly
            math_ast = libsbml.parseL3Formula(initial.value.id)
        else:
            raise NotImplementedError(initial)

        rule.setMath(math_ast)

    def _create_species(
        self, species_instance: SpeciesPattern
    ) -> libsbml.Species:
        """Create an SBML Species for the given species instance

        or return existing one
        """
        sbml_id = self.model._generate_species_id(species_instance)

        # exists?
        if s := self.sbml_model.getSpecies(sbml_id):
            return s

        # create
        name = self.model._generate_species_name(species_instance)
        # TODO: need to update MultiModel?
        assert species_instance.compartment is not None
        sbml_species = create_species(
            self.sbml_model,
            sbml_id=sbml_id,
            name=name,
            compartment=species_instance.compartment.id,
            sbo_term=species_instance.template.sbo_term,
        )

        if self._multi:
            multi_species = sbml_species.getPlugin("multi")
            multi_species.setSpeciesType(species_instance.template.id)

            if species_instance.site_states:
                for site, state in species_instance.site_states.items():
                    # FIXME proper handling if hierarchical speciesTypes
                    component_id_to_feature_type_id = {}
                    for (
                        sti_id,
                        sti_st,
                    ) in species_instance.template.species_type_instances:
                        assert len(sti_st.sites) <= 1
                        if len(sti_st.sites) == 1:
                            component_id_to_feature_type_id[sti_id] = next(
                                iter(sti_st.sites)
                            )

                    if site not in component_id_to_feature_type_id.keys():
                        # it's a feature defined on this species_type directly, not on a SpeciesTypeInstance
                        create_species_feature(multi_species, site, state)
                    else:
                        # SpeciesTypeInstance feature
                        component = site
                        feature_type = component_id_to_feature_type_id[
                            component
                        ]
                        create_species_feature(
                            multi_species,
                            feature_type,
                            state,
                            component=component,
                        )

                # TODO binding sites not yet supported -
                #   requires supporting hierarchical speciestypes

        if species_instance.template.sbo_term:
            sbml_species.setSBOTerm(species_instance.template.sbo_term)
        return sbml_species

    def _create_species_types(self):
        """Create all SBML-multi SpeciesTypes"""
        multi_model = self.sbml_model.getPlugin("multi")
        for species_type in self.model.species_types:
            st = create_species_type(
                multi_model,
                species_type.id,
                name=species_type.name or species_type.id,
            )

            for site, possible_states in species_type.sites.items():
                sft = create_species_feature_type(st, site)
                for possible_state in possible_states:
                    create_possible_species_feature_value(sft, possible_state)

            # species type instances
            for sti_id, sti_st in species_type.species_type_instances:
                create_species_type_instance(st, sti_st.id, sti_id)

            # TODO binding sites

    def _expression_to_sbml(self, expr: Expression):
        """Convert all model expressions to SBML parameters or AssignmentRules"""
        if element := self.sbml_model.getElementBySId(expr.id):
            # TODO warn?
            # TODO assert it's parameter
            return element
        if isinstance(expr.expr, Number):
            parameter = create_parameter(
                self.sbml_model,
                expr.id,
                name=expr.name_,
                constant=False,
                # TODO in correct unit?
                value=float(expr.expr),
                # TODO
                units="dimensionless",
            )
            set_annotations(parameter, expr.annotations)

            if expr.sbo_term:
                parameter.setSBOTerm(expr.sbo_term)

            return parameter

        sbml_units = unit_from_pint(expr.unit, self.sbml_model, ureg=ureg)
        if isinstance(expr.expr, Q_) and isinstance(expr.expr.m, Number):
            # TODO
            parameter = create_parameter(
                self.sbml_model,
                expr.id,
                name=expr.name_,
                constant=False,
                value=expr.expr.m,
                # TODO
                units=sbml_units,
            )
            set_annotations(parameter, expr.annotations)
            if expr.sbo_term:
                parameter.setSBOTerm(expr.sbo_term)
            return parameter

        if isinstance(expr.expr, sp.Expr) or isinstance(expr.expr, Q_):
            # TODO check if can be floatified, then we don't need an assignment
            #  rule
            parameter = create_parameter(
                self.sbml_model,
                expr.id,
                name=expr.name_,
                constant=False,
                value=1.0,
                # TODO
                units=sbml_units,
            )
            ast_node = sympy_to_sbml_math(
                expr.expr if isinstance(expr.expr, sp.Expr) else expr.expr.m
            )
            ar = self.sbml_model.createAssignmentRule()
            ar.setVariable(expr.id)
            ar.setMath(ast_node)
            if expr.sbo_term:
                parameter.setSBOTerm(expr.sbo_term)
            return parameter

        raise NotImplementedError(f"{expr} {expr.expr} ({type(expr.expr)}")

    def _create_reactions(self):
        """Create SBML reactions for all model reactions"""
        if self._multi:
            for reaction in self.model.reactions:
                self._handle_reaction_pattern(reaction)

    def _handle_reaction_pattern(
        self,
        rxn: ReactionPattern,
    ):
        rxn_id = rxn.id

        sbml_rxn = self.sbml_model.createReaction()
        _check(sbml_rxn.setId(rxn_id))
        sbml_rxn.setReversible(rxn.reversible)
        reactant_list = [(rxn.substrates, True), (rxn.products, False)]
        for reactants, is_substrate in reactant_list:
            for i_reactant, reactant in enumerate(reactants):
                sbml_species = self._create_species(reactant)
                species_id = sbml_species.getId()
                if is_substrate:
                    species_ref = sbml_rxn.createReactant()
                    # TODO set ID in reaction
                    _check(
                        species_ref.setId(f"{rxn_id}_substrate_{i_reactant}")
                    )
                else:
                    species_ref = sbml_rxn.createProduct()
                    # TODO set ID in reaction
                    _check(species_ref.setId(f"{rxn_id}_product_{i_reactant}"))
                _check(species_ref.setSpecies(species_id))
                species_ref.setConstant(True)
                species_ref.setStoichiometry(1)

                if rxn.feature_mapping:
                    if cur_product_map := rxn.feature_mapping.mapping.get(
                        species_ref.getId()
                    ):
                        # Write atom mappings to products
                        multi_sr = species_ref.getPlugin("multi")
                        _feature_mapping_to_sbml(multi_sr, cur_product_map)
        # modifiers
        if rxn.enzyme is not None:
            modifier = sbml_rxn.createModifier()
            modifier.setSpecies(self._create_species(rxn.enzyme).getId())

        self._set_kinetic_law(rxn, sbml_rxn)

    def _set_kinetic_law(
        self, rxn: ReactionPattern, sbml_rxn: libsbml.Reaction
    ):
        if rxn.rate_law is None and rxn.rate_law_generator is None:
            logger.warning(f"No rate law set for reaction {rxn.id}")
            return

        # Rate law
        if rxn.rate_law:
            rate_law = rxn.rate_coefficient * rxn.rate_law
        else:
            rate_law = (
                rxn.rate_coefficient * rxn.rate_law_generator.expression(rxn)
            )

        # TODO all these parameters should have been created for the rate law
        #  already
        for sym in rate_law.free_symbols:
            # model expressions will be processed later;
            #  postpone kinetic law processing until after that?
            if self.sbml_model.getElementBySId(sym.name) is None and all(
                e.id != sym.name for e in self.model.expressions
            ):
                warnings.warn(
                    f"Creating parameter {sym} which should already exist."
                )
                create_parameter(
                    self.sbml_model,
                    str(sym),
                    constant=True,
                    value=1.0,
                    units="dimensionless",
                )

        kinetic_law = sbml_rxn.createKineticLaw()
        _check(kinetic_law.setMath(sympy_to_sbml_math(rate_law)))

    def validate(self) -> None:
        self.sbml_document.validateSBML()
        _check_lib_sbml_errors(self.sbml_document, True)

    def _process_observables(self) -> None:
        """Create AssignmentRules for Observables"""
        for observable in self.model.observables:
            if isinstance(observable, MdvObservable):
                self._process_mdv_observable(observable)
            elif isinstance(observable, RawMdvObservable):
                self._process_raw_mdv_observable(observable)
            elif isinstance(observable, AmountObservable):
                self._process_amount_observable(observable)
            else:
                raise NotImplementedError(f"{observable} ({type(observable)})")

    def _process_mdv_observable(self, observable: MdvObservable) -> None:
        """Create AssignmentRules for MdvObservable"""
        # collect expressions for total amounts per compartment
        total_amount_exprs = [
            SpeciesSymbol(sid, representation_type="sum")
            if sbml_species.getHasOnlySubstanceUnits()
            else SpeciesSymbol(sid, representation_type="sum")
            * sp.Symbol(compartment.id)
            for compartment in observable.compartments
            if (
                sbml_species := self.sbml_model.getSpecies(
                    (sid := f"{observable.species.id}_{compartment.id}")
                )
            )
            # sp.Symbol(
            #     self._add_total_amount_expression(observable.species, c)
            # ) for c in observable.compartments
        ]
        len_mdv = observable.max_labels

        for mass in range(len_mdv + 1):
            pattern = LABELED_STATE * mass + UNLABELED_STATE * (len_mdv - mass)

            # create expression for M+i for each compartment
            exprs_cur_mass = []

            for compartment in observable.compartments:
                for state in multiset_permutations(pattern):
                    site_states = {
                        # FIXME: improve hierarchical species type handling; assumes there is only one site
                        sti_id: state_
                        for (sti_id, sti_st), state_ in zip(
                            observable.species.species_type_instances, state
                        )
                        if list(sti_st.sites.values()) == [LABELING_STATES]
                    }
                    assert len(site_states) == len(state)
                    cur_species = self._create_species(
                        SpeciesPattern(
                            observable.species,
                            compartment=compartment,
                            site_states=site_states,
                        )
                    )
                    exprs_cur_mass.append(
                        sp.Symbol(cur_species.getId())
                        * sp.Symbol(compartment.id)
                    )

            # The relative amount is measured
            rel_amount_expr = sp.Add(*exprs_cur_mass) / sp.Add(
                *total_amount_exprs
            )

            target = create_parameter(
                self.sbml_model,
                parameter_id=f"{observable.id}_M{mass}",
                name=(
                    f"{observable.species.name or observable.species.id} in "
                    f"{','.join(c.name or c.id for c in observable.compartments)} "
                    f"M+{mass}"
                ),
                constant=False,
                units="dimensionless",
                value=0.0,
            )

            ast_node = sympy_to_sbml_math(rel_amount_expr)
            ar = self.sbml_model.createAssignmentRule()
            ar.setVariable(target.getId())
            ar.setMath(ast_node)

    def _process_raw_mdv_observable(
        self, observable: RawMdvObservable
    ) -> None:
        """Create AssignmentRules for RawMdvObservable

        Observable functions contain the correction terms and a scaling
        parameter.
        """

        if observable.label != "C":
            raise ValueError("Only C-labeling is supported so far.")

        # nominal mass of M+0
        mass = get_nominal_mass(observable.sum_formula)
        # maximum number of artificial labels
        max_labels = observable.max_labels

        correction_matrix = observable.get_correction_matrix()
        correction_matrix = sp.Matrix(correction_matrix)
        # absolute mass distribution vector in amount substance,
        #  not accounting for natural isotope abundance
        mdv_model = sp.zeros(rows=max_labels + 1, cols=1)

        for mass_increment in range(max_labels + 1):
            # e.g. 'llu'
            pattern = LABELED_STATE * mass_increment + UNLABELED_STATE * (
                max_labels - mass_increment
            )
            # collect amount expressions for the observed species in all
            #  observed compartments if it matches the given mass increment
            for compartment in observable.compartments:
                for state in multiset_permutations(pattern):
                    # translate permutation to site states
                    assert len(
                        observable.species.species_type_instances
                    ) == len(state)
                    site_states = {
                        # FIXME: improve hierarchical species type handling; assumes there is only one site
                        sti_id: state_
                        for (sti_id, sti_st), state_ in zip(
                            observable.species.species_type_instances, state
                        )
                        if list(sti_st.sites.values()) == [LABELING_STATES]
                    }
                    assert len(site_states) == len(state)

                    # We need to create the species to use it in the observable
                    #  formula
                    cur_species = self._create_species(
                        SpeciesPattern(
                            observable.species,
                            compartment=compartment,
                            site_states=site_states,
                        )
                    )
                    mdv_model[mass_increment] += sp.Symbol(
                        cur_species.getId()
                    ) * sp.Symbol(compartment.id)

        # mass distribution vector accounting for natural isotope abundance
        convoluted_mid = correction_matrix * mdv_model

        # create individual M+i, i=0,... observables
        for mass_increment in range(convoluted_mid.shape[0]):
            # TODO fragment label
            observable_id = f"{observable.id}_{mass}_M{mass_increment}"
            observable_name = (
                f"Signal of m/z {mass}+{mass_increment} of "
                f"{observable.species.id} in "
                f"{' and '.join(c.name or c.id for c in observable.compartments)}"
            )

            # assuming measurements are raw MS intensities (non-absolute),
            #  we need a scaling parameter (response factor)
            scaling_par_id = f"observableParameter1_{observable_id}"
            scaling_par_name = f"Scaling parameter for {observable_id}"
            sbml_units = unit_from_pint(
                # signal per amount
                self.model.amount_unit**-1,
                self.sbml_model,
                ureg=ureg,
            )
            create_parameter(
                self.sbml_model,
                parameter_id=scaling_par_id,
                name=scaling_par_name,
                constant=False,
                units=sbml_units,
                value=1.0,
            )

            # parameter as assignment rule target
            target = create_parameter(
                self.sbml_model,
                parameter_id=observable_id,
                name=observable_name,
                constant=False,
                units="dimensionless",
                value=0.0,
            )
            # actual assignment rule
            # scaled observable expression
            expr = convoluted_mid[mass_increment] * sp.Symbol(scaling_par_id)
            ast_node = sympy_to_sbml_math(expr)
            ar = self.sbml_model.createAssignmentRule()
            ar.setVariable(target.getId())
            ar.setMath(ast_node)

    def _process_amount_observable(self, observable: AmountObservable) -> None:
        """Create AssignmentRule for AmountObservable"""

        # gather expressions for amounts in each compartment
        # TODO: pattern species has to exist in every relevant compartment
        #  already!
        # compartment_amounts = [sp.Symbol(
        #     self._add_total_amount_expression(observable.species, c)
        # ) for c in observable.compartments]
        compartment_amounts = [
            SpeciesSymbol(sid, representation_type="sum")
            if sbml_species.getHasOnlySubstanceUnits()
            else SpeciesSymbol(sid, representation_type="sum")
            * sp.Symbol(compartment.id)
            for compartment in observable.compartments
            if (
                sbml_species := self.sbml_model.getSpecies(
                    (sid := f"{observable.species.id}_{compartment.id}")
                )
            )
        ]
        expr = sp.Add(*compartment_amounts)

        if observable.relative:
            # add scaling parameter (response factor) for relative measurements
            scaling_par_id = f"observableParameter1_{observable.id}"
            scaling_par_name = f"Scaling parameter for {observable.id}"
            sbml_units = unit_from_pint(
                # signal per amount
                self.model.amount_unit**-1,
                self.sbml_model,
                ureg=ureg,
            )
            create_parameter(
                self.sbml_model,
                parameter_id=scaling_par_id,
                name=scaling_par_name,
                constant=False,
                units=sbml_units,
                value=1.0,
            )
            expr = sp.Symbol(scaling_par_id) * expr

            sbml_units = unit_from_pint(
                ureg.dimensionless, self.sbml_model, ureg
            )
        else:
            # units of the observable (amount)
            sbml_units = unit_from_pint(
                self.model.amount_unit, self.sbml_model, ureg
            )

        # assignment rule target for observable
        target = create_parameter(
            self.sbml_model,
            parameter_id=observable.id,
            constant=False,
            units=sbml_units,
            value=0.0,
        )

        # create assignment rule for observable
        ast_node = sympy_to_sbml_math(expr)
        ar = self.sbml_model.createAssignmentRule()
        ar.setVariable(target.getId())
        ar.setMath(ast_node)

    def _add_total_amount_expression(
        self, species: SpeciesType, compartment: Compartment
    ) -> str:
        """Create a parameter and corresponding AssignmentRule for a
        compartment-specific total amount
        of species matching the given pattern (if not already exists)"""
        expr_id = f"_{species.id}_{compartment.id}_amount"

        # exists?
        if self.sbml_model.getElementBySId(expr_id):
            return expr_id

        # create
        sbml_units = unit_from_pint(
            self.model.amount_unit, self.sbml_model, ureg
        )
        target = create_parameter(
            self.sbml_model,
            parameter_id=expr_id,
            constant=False,
            units=sbml_units,
            value=0.0,
        )
        set_annotations(
            target,
            {
                "parameter": {"estimated": "false", "non_negative": "true"},
            },
        )

        sid = f"{species.id}_{compartment.id}"
        if not (sbml_species := self.sbml_model.getSpecies(sid)):
            # TODO: if the species does not exist, we should skip creating that
            #  parameter entirely
            return expr_id

        # Only create assignment rule if the given species exists in the
        #  given compartment
        species_symbol = SpeciesSymbol(sid, representation_type="sum")
        compartment_symbol = sp.Symbol(compartment.id)
        expr = species_symbol
        if not sbml_species.getHasOnlySubstanceUnits():
            expr *= compartment_symbol
        ast_node = sympy_to_sbml_math(expr)

        ar = self.sbml_model.createAssignmentRule()
        ar.setVariable(target.getId())
        ar.setMath(ast_node)

        return expr_id


def _feature_mapping_to_sbml(
    multi_sr: libsbml.SpeciesReference,
    mapping: Dict[str, Tuple[str, str]],
) -> None:
    """Write feature mappings for the given product of a reaction to the SBML
    model"""
    for product_feature, (reactant_id, reactant_feature) in mapping.items():
        stcmip = multi_sr.createSpeciesTypeComponentMapInProduct()
        stcmip.setReactant(reactant_id)
        stcmip.setReactantComponent(reactant_feature)
        stcmip.setProductComponent(product_feature)
