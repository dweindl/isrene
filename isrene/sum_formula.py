"""Functions for working with (chemical) sum formulas"""
import contextlib
import re
from numbers import Number
from typing import Dict


def parse_sum_formula(x: str) -> Dict[str, Number]:
    """Convert sum formula to dict (element -> count)"""
    s = re.split(r"([A-Z][a-z]*)", x)
    res = {}
    last_token = 0
    for token in s:
        with contextlib.suppress(ValueError):
            token = float(token)
            token = int(token) if int(token) - token == 0 else token
        if isinstance(token, str):
            if last_token and isinstance(last_token, str):
                # set previous to 1
                res[last_token] = res.get(last_token, 0) + 1
        elif token:
            res[last_token] = res.get(last_token, 0) + token
        last_token = token

    return res


def sum_formula_diff(s1: Dict, s2: Dict) -> Dict[str, Number]:
    """Get difference of two sum formulas

    :returns: Dictionary of elements of differing occurrences mapping to the
    difference.
    """
    res = {
        k: s1.get(k, 0) - s2.get(k, 0) for k in set(s1.keys()) | set(s2.keys())
    }
    res = dict(filter(lambda elem: elem[1] != 0, res.items()))
    return res


def get_nominal_mass(sum_formula: str):
    """Compute nominal mass for the given sum formula"""
    masses = {
        "C": 12,
        "N": 14,
        "H": 1,
        "O": 16,
        "Si": 28,
    }
    composition = parse_sum_formula(sum_formula)
    return sum(
        count * masses[element] for element, count in composition.items()
    )
