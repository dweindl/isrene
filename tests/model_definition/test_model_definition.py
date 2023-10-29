import pytest

from isrene import Q_, ureg
from isrene.model_definition.model_definition import *
from isrene.model_definition.model_definition_hl import *


def test_atom_mapping_reaction_requires_mapping():
    model = Model()
    a = Metabolite("A", 1, model=model)
    b = Metabolite("B", 1, model=model)
    AtomMappingReaction("r", [(a(), "a")], [(b(), "a")])
    with pytest.raises(ValueError):
        # missing mapping
        AtomMappingReaction("r", [a()], [b()])


def test_cycle_model():
    """Just check that creating the internal model passes without error."""
    model = Model("cycle")

    compartment = Compartment("C", ureg.Quantity(1, ureg.mL), 3, model=model)

    S = Metabolite("S", num_labelable_carbons=3, model=model)

    AtomMappingReaction(
        "abc_bca",
        [(S(compartment=compartment), "abc")],
        [(S(compartment=compartment), "bca")],
        reversible=True,
        # rate_law_generator=MassActionRateLawGenerator(),
    )

    InitialConcentration(
        S({"C1": "l", "C2": "u", "C3": "u"}, compartment=compartment),
        Q_(1.0, ureg.mM),
    )
