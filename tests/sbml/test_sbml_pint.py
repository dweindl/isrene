import pytest

from isrene import ureg
from isrene.sbml.sbml_pint import *
from isrene.sbml.sbml_pint import unit_from_pint


@pytest.mark.parametrize(
    "expected_unit",
    [
        ureg.dimensionless,
        ureg.liter,
        ureg.milliliter,
        ureg.mole,
        ureg.micromole,
        1 / ureg.femtomole,
        ureg.nanomolar,
        ureg.mole / ureg.second,
        ureg.millimeter**2,
        1 / ureg.millimeter**2,
    ],
)
def test_cycle_units(expected_unit):
    """Cycle units between pint and SBML"""
    sbml_document = libsbml.SBMLDocument(3, 1)
    sbml_model = sbml_document.createModel()

    actual_unit = sbml_units_to_pint(
        unit_from_pint(expected_unit, sbml_model, ureg), sbml_model, ureg
    )

    assert (
        ureg.Quantity(1, actual_unit).to_base_units()
        == ureg.Quantity(1, expected_unit).to_base_units()
    )
