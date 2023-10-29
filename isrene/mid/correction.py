"""Functionality for MID correction"""
from typing import Dict

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from ..sum_formula import parse_sum_formula

# mean of ranges from
# https://www.degruyter.com/document/doi/10.1515/pac-2015-0503/html?lang=en
natural_abundances = {
    # isotope mass distributions [M0, M1, M2, ...]
    "C": np.array([(0.9884 + 0.9904) / 2, (0.0096 + 0.0116) / 2, 0]),
    "H": np.array([(0.99972 + 0.99999) / 2, (0.00001 + 0.00028) / 2, 0]),
    "N": np.array([(0.99578 + 0.9966) / 2, (0.00337 + 0.00422) / 2, 0]),
    "O": np.array(
        [
            (0.99738 + 0.99776) / 2,
            (0.000367 + 0.000400) / 2,
            (0.00187 + 0.00222) / 2,
        ]
    ),
    "Si": np.array(
        [
            (0.92191 + 0.92318) / 2,
            (0.04645 + 0.04699) / 2,
            (0.03037 + 0.03110) / 2,
        ]
    ),
}


def get_correction_matrix(
    sum_formula: str,
    num_rows: int,
    max_labels: int,
    label: str,
    row_normalize=False,
) -> np.array:
    """Get correction matrix.

    Parameters
    ----------
    sum_formula:
        Sum formula for the species for which to create the correction matrix
    num_rows:
        Number of rows to return (usually the size of the measured isotope
        cluster).
    max_labels:
        Maximum number of artificial labels that could occur in the given
        species. (Determines the number of columns of the correction matrix,
        which is ``max_labels + 1``).
    label:
        The artificially enriched element, e.g. "C" for 13C labeling
    row_normalize:
        Normalize to row-sum == 1. Default is col-sum == 1.
    """
    composition = parse_sum_formula(sum_formula)
    if (num_labelable := composition.get(label, 0)) < max_labels:
        raise ValueError(
            f"Maximum of {max_labels} {label} labels specified "
            f"for {sum_formula}, but only {num_labelable} "
            f"{label} present."
        )

    corr = np.zeros(shape=(num_rows, max_labels + 1))

    # each column corresponds to the MID of the original composition minus one
    #  labeled atom per column and shifted down by the number of labels
    #  (assuming the label adds 1 mass unit)
    for num_labels in range(max_labels + 1):
        cur_composition = composition.copy()
        cur_composition[label] -= num_labels
        mid = get_natural_isotope_distribution(cur_composition)
        length = min(num_rows - num_labels, len(mid))
        corr[num_labels : num_labels + length, num_labels] = mid[:length]

    if row_normalize:
        corr = corr / np.sum(corr, axis=1).reshape(-1, 1)

    return corr


def get_natural_isotope_distribution(composition: Dict):
    """Determine natural isotope distribution for the given
    elemental composition.

    See e.g. https://doi.org/10.3390/metabo11050310
    """
    res = Polynomial(1)
    for element, count in composition.items():
        res *= Polynomial(natural_abundances[element]) ** count
    return res.coef
