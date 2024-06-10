"""Functionality for interconverting pint and sbml units

SBML units may not always be compatible with pint units, because pint.Unit
cannot have a multiplier. Therefore, we use pint.Quantity to represent SBML
unit definitions.
"""
import operator
from functools import reduce
from math import log10

import libsbml
import pint

from .sbml_wrappers import _check, add_unit

predefined_sbml_units = [
    "ampere",
    "avogadro",
    "gram",
    "katal",
    "metre",
    "second",
    "watt",
    "becquerel",
    "gray",
    "kelvin",
    "mole",
    "siemens",
    "weber",
    "candela",
    "henry",
    "kilogram",
    "newton",
    "sievert",
    "coulomb",
    "hertz",
    "litre",
    "ohm",
    "steradian",
    "dimensionless",
    "item",
    "lumen",
    "pascal",
    "tesla",
    "farad",
    "joule",
    "lux",
    "radian",
    "volt",
]


def _unit_definition_from_pint(
    unit: pint.Unit, sbml_model: libsbml.Model, ureg: pint.UnitRegistry
) -> libsbml.UnitDefinition:
    """Create an UnitDefinition in the given SBML model for the given pint unit"""
    if unit.dimensionless is True:
        # since pint 0.24.0, str(Unit("dimensionless")) == ""
        unit_id = "dimensionless"
    else:
        unit_id = str(unit)
        unit_id = unit_id.replace(" / ", "_per_")
        unit_id = unit_id.replace(" * ", "_times_")
        unit_id = unit_id.replace(" ** ", "_power_of_")
        unit_id = unit_id.replace(".", "_")

        # turn into a valid SBML ID
        unit_id = unit_id.removeprefix("1_")

    if not libsbml.SyntaxChecker_isValidSBMLSId(unit_id):
        raise AssertionError(
            f"Generated unit ID for '{unit}' is invalid: " f"{unit_id}"
        )

    Q_ = ureg.Quantity

    # exists?
    if unit_definition := sbml_model.getUnitDefinition(unit_id):
        # ID generation may not be unique. ensure units match up.
        assert (
            Q_(1, unit).to_base_units()
            == Q_(
                1, sbml_unit_definition_to_pint(unit_definition, ureg)
            ).to_base_units()
        )
        return unit_definition

    # create
    unit_definition = sbml_model.createUnitDefinition()
    _check(unit_definition.setId(unit_id))

    # handle compound units
    for unit_str, exponent in unit._units.items():
        sub_unit = getattr(ureg, unit_str)

        if Q_(1, sub_unit).check("[substance]"):
            scale = _get_scale(sub_unit, ureg.mole, ureg)
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_MOLE,
                scale=scale,
                exponent=exponent,
            )
        elif Q_(1, sub_unit).check("[volume]"):
            scale = _get_scale(sub_unit, ureg.litre, ureg)
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_LITRE,
                scale=scale,
                exponent=exponent,
            )
        elif Q_(1, sub_unit).check("[concentration]"):
            # TODO use actual unit, not  base units
            coefficient = (
                Q_(1.0, sub_unit).to_base_units()
                / Q_(1.0, ureg.mole / ureg.litre).to_base_units()
            ).m
            scale = log10(coefficient)
            int_scale = int(scale)
            assert int_scale == scale
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_MOLE,
                scale=int_scale,
                exponent=exponent,
            )
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_LITRE,
                exponent=-1 * exponent,
            )

        elif Q_(1, sub_unit).check("[length]"):
            scale = _get_scale(sub_unit, ureg.metre, ureg)
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_METRE,
                exponent=exponent,
                scale=scale,
            )
        elif Q_(1, sub_unit).check("[time]"):
            scale = _get_scale(sub_unit, ureg.seconds, ureg)
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_SECOND,
                exponent=exponent,
                scale=scale,
            )
        elif sub_unit == ureg.dimensionless:
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_DIMENSIONLESS,
                exponent=0,
                scale=0,
            )
        elif Q_(1, sub_unit).check("[energy]"):
            scale = _get_scale(sub_unit, ureg.joule, ureg)
            add_unit(
                unit_definition,
                kind=libsbml.UNIT_KIND_JOULE,
                exponent=exponent,
                scale=scale,
            )
        else:
            raise NotImplementedError(
                f"Missing unit conversion for {unit} ({dict(unit._units)})"
            )

    return unit_definition


def _get_scale(
    unit: pint.Unit, reference_unit: pint.Unit, ureg: pint.UnitRegistry
) -> int:
    """Get the scale to define ``unit`` in terms of ``reference unit``"""
    Q_ = ureg.Quantity

    ratio = Q_(1.0, unit) / Q_(1.0, reference_unit)
    ratio = ratio.to_base_units()
    assert ratio.u == ureg.dimensionless
    coefficient = ratio.m
    scale = log10(coefficient)
    int_scale = int(scale)
    assert int_scale == scale
    return int_scale


def unit_from_pint(
    unit: pint.Unit, sbml_model: libsbml.Model, ureg: pint.UnitRegistry
) -> str:
    """Returns the value for a 'unit' attribute for the given ``pint.Unit``
    and creates a new UnitDefinition if necessary"""

    # predefined and not allowed as ID for unitDefinitions
    if unit.dimensionless is True:
        # since pint 0.24.0, str(Unit("dimensionless")) == ""
        return "dimensionless"

    unit_id = str(unit)
    if unit_id in predefined_sbml_units:
        return unit_id
    return _unit_definition_from_pint(unit, sbml_model, ureg).getId()


def sbml_units_to_pint(
    sbml_units: str, sbml_model: libsbml.Model, ureg: pint.UnitRegistry
) -> pint.Unit:
    """Convert the value of a `units` attribute to a pint unit"""
    if sbml_units in predefined_sbml_units:
        return getattr(ureg, sbml_units)

    unit_definition = sbml_model.getUnitDefinition(sbml_units)
    return sbml_unit_definition_to_pint(unit_definition, ureg)


def sbml_unit_kind_id_to_str(sbml_kind: int) -> str:
    return next(
        x
        for x in dir(libsbml)
        if x.startswith("UNIT_KIND_") and getattr(libsbml, x) == sbml_kind
    )


def sbml_unit_to_pint(
    sbml_unit: libsbml.Unit, ureg: pint.UnitRegistry
) -> pint.Unit:
    """Convert an SBML Unit to a pint Unit"""
    # a unit in SBML is defined as (multiplier * 10^scale * kind)^exponent
    sbml_kind = sbml_unit.getKind()
    scale = sbml_unit.getScale()
    multiplier = sbml_unit.getMultiplier()
    exponent = sbml_unit.getExponent()

    if sbml_kind in (libsbml.UNIT_KIND_LITER, libsbml.UNIT_KIND_LITRE):
        pint_kind = "liter"
    elif sbml_kind == libsbml.UNIT_KIND_MOLE:
        pint_kind = "mole"
    elif sbml_kind == libsbml.UNIT_KIND_SECOND:
        pint_kind = "second"
    elif sbml_kind in (libsbml.UNIT_KIND_METRE, libsbml.UNIT_KIND_METER):
        pint_kind = "metre"
    elif sbml_kind == libsbml.UNIT_KIND_JOULE:
        pint_kind = "joule"
    elif sbml_kind == libsbml.UNIT_KIND_DIMENSIONLESS:
        assert scale == 0
        assert multiplier == 1
        assert exponent == 0
        return ureg.dimensionless
    else:
        kind_id = sbml_unit_kind_id_to_str(sbml_kind)
        raise NotImplementedError(f"Unit kind {kind_id}")

    scale_map = {
        3: "kilo",
        0: "",
        -3: "milli",
        -6: "micro",
        -9: "nano",
        -12: "pico",
        -15: "femto",
    }

    if multiplier != 1:
        raise NotImplementedError("Multiplier not supported")

    try:
        pint_str = f"{scale_map[scale]}{pint_kind}"
        return getattr(ureg, pint_str) ** exponent
    except Exception as e:
        print(e)
        raise NotImplementedError(
            f"Unable to convert unit {sbml_unit_kind_id_to_str(sbml_kind)} "
            f"multiplier: {multiplier}, scale: {scale}, exponent: {exponent}"
        )


def sbml_unit_definition_to_pint(
    unit_definition: libsbml.UnitDefinition, ureg: pint.UnitRegistry
) -> pint.Unit:
    return reduce(
        operator.mul,
        (
            sbml_unit_to_pint(unit, ureg=ureg)
            for unit in unit_definition.getListOfUnits()
        ),
    )
