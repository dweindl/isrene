"""Convenience functions for libsbml core"""

import logging
from itertools import chain
from typing import Iterable, Optional, Tuple, Union

import libsbml
import sympy as sp

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


retval_to_str = {
    getattr(libsbml, attr): attr
    for attr in (
        "LIBSBML_DUPLICATE_OBJECT_ID",
        "LIBSBML_INDEX_EXCEEDS_SIZE",
        "LIBSBML_INVALID_ATTRIBUTE_VALUE",
        "LIBSBML_INVALID_OBJECT",
        "LIBSBML_INVALID_XML_OPERATION",
        "LIBSBML_LEVEL_MISMATCH",
        "LIBSBML_NAMESPACES_MISMATCH",
        "LIBSBML_OPERATION_FAILED",
        "LIBSBML_UNEXPECTED_ATTRIBUTE",
        "LIBSBML_PKG_UNKNOWN",
        "LIBSBML_PKG_VERSION_MISMATCH",
        "LIBSBML_PKG_CONFLICTED_VERSION",
    )
}


def create_species(
    model: libsbml.Model,
    sbml_id: str,
    compartment: str,
    name: Optional[str] = None,
    constant=False,
    initial_concentration=0.0,
    substance_units=None,
    boundary_condition=False,
    has_only_substance_units=False,
    sbo_term: str = None,
) -> libsbml.Species:
    """Create SBML species"""
    s = model.createSpecies()
    _check(s.setId(sbml_id))
    if name:
        _check(s.setName(name))
    _check(s.setCompartment(compartment))
    _check(s.setConstant(constant))
    _check(s.setInitialConcentration(initial_concentration))
    if substance_units:
        _check(s.setSubstanceUnits(substance_units))
    _check(s.setBoundaryCondition(boundary_condition))
    _check(s.setHasOnlySubstanceUnits(has_only_substance_units))

    if sbo_term:
        _check(s.setSBOTerm(sbo_term))
    return s


def _check_lib_sbml_errors(
    sbml_doc: libsbml.SBMLDocument, show_warnings: bool = False
) -> None:
    """
    Checks the error log in the current self.sbml_doc.

    :param sbml_doc:
        SBML document

    :param show_warnings:
        display SBML warnings
    """
    num_warning = sbml_doc.getNumErrors(libsbml.LIBSBML_SEV_WARNING)
    num_error = sbml_doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR)
    num_fatal = sbml_doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL)

    if num_warning + num_error + num_fatal:
        severity_to_log_level = {
            libsbml.LIBSBML_SEV_INFO: logging.INFO,
            libsbml.LIBSBML_SEV_WARNING: logging.WARNING,
        }

        for i_error in range(sbml_doc.getNumErrors()):
            error = sbml_doc.getError(i_error)
            # we ignore any info messages for now
            if (
                severity := error.getSeverity()
            ) >= libsbml.LIBSBML_SEV_ERROR or (
                show_warnings and severity >= libsbml.LIBSBML_SEV_WARNING
            ):
                logger.log(
                    severity_to_log_level.get(severity, logging.ERROR),
                    f"libSBML {error.getCategoryAsString()} "
                    f"({error.getSeverityAsString()}):"
                    f" {error.getMessage()}",
                )

    if num_error + num_fatal:
        raise RuntimeError("SBML Error (see error messages above)")


def create_parameter(
    model: libsbml.Model,
    parameter_id: str,
    constant: bool,
    value: float,
    units: str,
    name: str = None,
) -> libsbml.Parameter:
    """Add parameter to SBML model"""
    k = model.createParameter()
    _check(k.setId(parameter_id))
    _check(k.setName(name or parameter_id))
    _check(k.setConstant(constant))
    _check(k.setValue(value))
    _check(k.setUnits(units))
    return k


def add_unit(
    unit_definition: libsbml.UnitDefinition,
    kind: int,
    exponent=1,
    scale=0,
    multiplier=1,
) -> libsbml.Unit:
    """Add unit to SBML unit definition"""
    unit = unit_definition.createUnit()
    _check(unit.setKind(kind))
    _check(unit.setExponent(exponent))
    _check(unit.setScale(scale))
    _check(unit.setMultiplier(multiplier))
    return unit


def _check(res):
    if res != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"libsbml error: {retval_to_str.get(res, res)}")


def create_reaction(
    model: libsbml.Model,
    reaction_id: str,
    reactants: Optional[Iterable[Tuple[int, str]]] = None,
    products: Optional[Iterable[Tuple[int, str]]] = None,
    formula: Optional[str] = None,
    reversible: bool = False,
    fast: bool = False,
) -> libsbml.Reaction:
    """Add reaction to SBML model"""
    r = model.createReaction()
    _check(r.setId(reaction_id))
    _check(r.setReversible(reversible))
    # FIXME: checking setFast, as this gives an LIBSBML_UNEXPECTED_ATTRIBUTE
    #  error because the model package is not available for level3 version2,
    #  so we need to use version 1
    # _check(r.setFast(fast))
    r.setFast(fast)

    if reactants is None:
        reactants = {}

    if products is None:
        products = {}

    for coeff, name in reactants:
        species_ref = r.createReactant()
        _check(species_ref.setSpecies(name))
        _check(species_ref.setConstant(True))  # TODO ?
        _check(species_ref.setStoichiometry(coeff))

    for coeff, name in products:
        species_ref = r.createProduct()
        _check(species_ref.setSpecies(name))
        _check(species_ref.setConstant(True))  # TODO ?
        _check(species_ref.setStoichiometry(coeff))

    if formula:
        math_ast = libsbml.parseL3Formula(formula)
        kinetic_law = r.createKineticLaw()
        _check(kinetic_law.setMath(math_ast))
    return r


def create_assigment_rule(model: libsbml.Model, name: str, formula: str):
    """Create an assignment rule"""
    rule = model.createAssignmentRule()
    _check(rule.setId(name))
    _check(rule.setName(name))
    _check(rule.setVariable(name))
    _check(rule.setFormula(formula))
    return rule


def sbml_math_to_sympy_via_str(
    sbml_obj: Union[libsbml.SBase, libsbml.ASTNode]
) -> sp.Expr:
    """Convert SBML MathML to sympy using intermediary plain formula string
    representation."""
    parser_settings = libsbml.L3ParserSettings()
    parser_settings.setParseUnits(libsbml.L3P_NO_UNITS)

    if not isinstance(sbml_obj, libsbml.ASTNode):
        sbml_obj = sbml_obj.getMath()
    return sp.sympify(
        libsbml.formulaToL3StringWithSettings(sbml_obj, parser_settings)
    )


def remove_species_reference_ids(model: libsbml.Model) -> None:
    """Remove the ID attribute of all species references in the model.

    Speeds up AMICI model import. Does not check whether the species reference ID is actually used anywhere.
    """
    for reaction in model.getListOfReactions():
        for reference in chain(
            reaction.getListOfReactants(),
            reaction.getListOfProducts(),
            reaction.getListOfModifiers(),
        ):
            _check(reference.unsetIdAttribute())
