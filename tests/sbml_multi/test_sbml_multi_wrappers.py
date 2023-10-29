import pytest

from isrene.sbml_multi.sbml_multi_wrappers import *


def test_species_type():
    with pytest.raises(ValueError):
        # that could in principle be allowed, as SBML multi can have 'occur'>1,
        # but currently sites are handled as dict keys, therefore this is not
        # currently allowed
        SpeciesType(
            "st",
            feature_types=[
                SpeciesFeatureType("sft1", 1, ["a", "b"]),
                SpeciesFeatureType("sft1", 1, ["a", "b"]),
            ],
        )


def test_pattern_matching():
    species_type = SpeciesType(
        "st",
        feature_types=[
            SpeciesFeatureType("sft1", 1, ["a", "b"]),
            SpeciesFeatureType("sft2", 1, ["a", "b"]),
        ],
    )
    pattern = Species(
        "pattern",
        compartment="comp1",
        species_type=species_type,
        feature_values={"sft1": ["a"]},
    )
    assert not pattern.is_concrete()
    assert pattern.matches_pattern(pattern)
    # matches more general pattern
    assert pattern.matches_pattern(
        Species(
            "pattern2",
            compartment="comp1",
            species_type=species_type,
            feature_values={},
        )
    )

    concrete1 = Species(
        "concrete1",
        compartment="comp1",
        species_type=species_type,
        # TODO check if we really need a list for the
        #  configuration
        feature_values={"sft1": ["a"], "sft2": ["a"]},
    )

    assert concrete1.is_concrete()

    # different compartment
    assert not concrete1.matches_pattern(
        Species(
            "concrete2",
            compartment="comp2",
            species_type=species_type,
            feature_values={"sft1": ["a"], "sft2": ["a"]},
        )
    )

    # different configuration
    assert not concrete1.matches_pattern(
        Species(
            "concrete2",
            compartment="comp2",
            species_type=species_type,
            feature_values={"sft1": ["b"]},
        )
    )

    # different type, same sites
    species_type2 = SpeciesType(
        "st2",
        feature_types=[
            SpeciesFeatureType("sft1", 1, ["a", "b"]),
            SpeciesFeatureType("sft2", 1, ["a", "b"]),
        ],
    )
    assert not concrete1.matches_pattern(
        Species(
            "concrete3",
            compartment="comp2",
            species_type=species_type2,
            feature_values={"sft1": ["a"], "sft2": ["a"]},
        )
    )
