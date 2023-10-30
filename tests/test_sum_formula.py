import pytest

from isrene.sum_formula import *


def test_parse_sum_formula():
    assert parse_sum_formula("") == {}
    assert parse_sum_formula("C") == {"C": 1}
    assert parse_sum_formula("CH") == {"C": 1, "H": 1}
    assert parse_sum_formula("Co") == {"Co": 1}
    assert parse_sum_formula("NaNaNa2") == {"Na": 4}
    assert parse_sum_formula("H2O") == {"H": 2, "O": 1}
    assert parse_sum_formula("CH2CH2") == {"C": 2, "H": 4}

    # TODO should spaces be allowed? or only lowercase letters?
    # with pytest.raises(ValueError):
    #     # No spaces allowed
    #     parse_sum_formula("C O")


def test_sum_formula_diff():
    assert sum_formula_diff({}, {}) == {}
    assert sum_formula_diff({"C": 1}, {"C": 1}) == {}
    assert sum_formula_diff({"H": 1}, {"C": 1}) == {"C": -1, "H": 1}


def test_nominal_mass():
    assert get_nominal_mass("") == 0
    assert get_nominal_mass("C") == 12
    assert get_nominal_mass("CO2") == 44
    assert get_nominal_mass("COO") == 44
    assert get_nominal_mass("C6H12O6") == 180
