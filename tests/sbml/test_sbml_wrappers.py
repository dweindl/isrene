import libsbml
import pytest

from isrene.sbml import get_free_symbols
from isrene.sbml.sbml_wrappers import _check


def test_free_symbols():
    sbml_math = libsbml.parseL3Formula("2 * p1")
    assert get_free_symbols(sbml_math) == {"p1"}

    sbml_math = libsbml.parseL3Formula("sqrt(1 / p2) + p1 * pow(3, p3)")
    assert get_free_symbols(sbml_math) == {"p1", "p2", "p3"}


def test_error_checking_retval_to_str():
    sbml_document = libsbml.SBMLDocument(3, 2)
    sbml_model = sbml_document.createModel()
    with pytest.raises(RuntimeError, match="LIBSBML_INVALID_ATTRIBUTE_VALUE"):
        _check(sbml_model.setId("invälid"))
