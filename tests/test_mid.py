"""Test MID correction"""
from numpy.testing import assert_allclose

from isrene.mid.correction import *


def test_get_natural_isotope_distribution():
    assert_allclose(
        get_natural_isotope_distribution({"C": 1}), natural_abundances["C"][:2]
    )

    assert_allclose(
        get_natural_isotope_distribution({"C": 1, "O": 1}),
        np.array(
            [
                # M+0
                natural_abundances["C"][0] * natural_abundances["O"][0],
                # M+1
                natural_abundances["C"][0] * natural_abundances["O"][1]
                + natural_abundances["C"][1] * natural_abundances["O"][0],
                # M+2
                natural_abundances["C"][1] * natural_abundances["O"][1]
                + natural_abundances["C"][0] * natural_abundances["O"][2],
                # M+3
                natural_abundances["C"][1] * natural_abundances["O"][2],
            ]
        ),
    )


def test_get_correction_matrix():
    expected = np.array(
        [
            get_natural_isotope_distribution(parse_sum_formula("C4O"))[:4],
            [
                0,
                *get_natural_isotope_distribution(parse_sum_formula("C3O"))[
                    :3
                ],
            ],
            [
                0,
                0,
                *get_natural_isotope_distribution(parse_sum_formula("C2O"))[
                    :2
                ],
            ],
        ]
    ).T

    # probabilities add up to approx 1?
    assert np.all(
        np.abs(1 - np.sum(get_correction_matrix("C4O", 8, 2, "C"), axis=0))
        < 0.01
    )
    # Probability of impossible isotopologues is 0
    assert np.all(get_correction_matrix("C4O", 8, 2, "C")[-1] == 0)

    # matches manually created
    assert_allclose(expected, get_correction_matrix("C4O", 4, 2, "C"))


def test_get_correction_matrix_row_normalize():
    assert np.all(
        np.abs(
            1
            - np.sum(
                get_correction_matrix("C4O", 4, 2, "C", row_normalize=True),
                axis=1,
            )
        )
        < 0.01
    )
