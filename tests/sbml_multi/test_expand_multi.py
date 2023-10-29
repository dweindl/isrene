"""Test SBML-multi model expansion"""

from tempfile import TemporaryDirectory

import libsbml
import pytest

from isrene.sbml.sbml_wrappers import _check_lib_sbml_errors, create_species
from isrene.sbml_multi import enable_multi
from isrene.sbml_multi.sbml_multi_expand import NetworkGenerator
from isrene.sbml_multi.sbml_multi_wrappers import (
    create_possible_species_feature_value,
    create_species_feature,
    create_species_feature_type,
    create_species_type,
    create_species_type_instance,
)

# FIXME: problem with uniqueness of multi:possibleSpeciesFeatureValue multi:id -> cannot be "u", "l" for all
#  either introduce C-site, and give each metabolite multiple ones (-> occur=N, must be implemented, can be named C1, C2, ...?)
#  or make all possibleSpeciesFeatureValue ids unique (speciesType_C#_u, speciesType_C#_l)
# TODO: but why does that not fail in the notebook? libsbml version?


@pytest.fixture
def new_multi_model():
    """Create an SBML-multi model, with some basic settings"""
    sbml_document = libsbml.SBMLDocument(3, 1)
    enable_multi(sbml_document)
    sbml_model = sbml_document.createModel()
    sbml_model.setTimeUnits("second")

    sbml_comp = sbml_model.createCompartment()
    sbml_comp.setId("compartment1")
    sbml_comp.setConstant(True)

    return sbml_document, sbml_model


@pytest.fixture
def example_model_1(new_multi_model):
    """Create example model manually"""
    sbml_document, sbml_model = new_multi_model

    # Species types
    multi_model = sbml_model.getPlugin("multi")
    # C-atom with labeled and unlabeled state
    st = create_species_type(multi_model, "C")
    create_species_feature_type(
        st,
        "C_isotope",
        values=["_12C", "_13C"],
    )

    # Some 2-carbon metabolite
    st = create_species_type(multi_model, "ST1")
    for i in range(2):
        create_species_type_instance(
            st,
            "C",
            f"C{i + 1}",
        )
    # Some other 2-carbon metabolite
    st = create_species_type(multi_model, "ST2")
    for i in range(2):
        create_species_type_instance(
            st,
            "C",
            f"C{i + 1}",
        )

    # Non-concrete species
    s = create_species(sbml_model, sbml_id="S1", compartment="compartment1")
    multi_species = s.getPlugin("multi")
    multi_species.setSpeciesType("ST1")

    s = create_species(sbml_model, sbml_id="S2", compartment="compartment1")
    multi_species = s.getPlugin("multi")
    multi_species.setSpeciesType("ST2")

    # Reaction
    # S1(a,b) -> S2(b,a)
    sbml_rxn = sbml_model.createReaction()
    sbml_rxn.setId("R1")
    sbml_rxn.setReversible(False)

    species_ref = sbml_rxn.createReactant()
    species_ref.setId("R1_substrate_S1")
    species_ref.setSpecies("S1")
    species_ref.setConstant(True)
    species_ref.setStoichiometry(1)

    species_ref = sbml_rxn.createProduct()
    species_ref.setId("R1_product_S1")
    species_ref.setSpecies("S2")
    species_ref.setConstant(True)
    species_ref.setStoichiometry(1)

    # mappings
    multi_sr = species_ref.getPlugin("multi")
    stcmip = multi_sr.createSpeciesTypeComponentMapInProduct()
    stcmip.setReactant(f"R1_substrate_S1")
    stcmip.setReactantComponent("C1")
    stcmip.setProductComponent("C2")
    stcmip = multi_sr.createSpeciesTypeComponentMapInProduct()
    stcmip.setReactant("R1_substrate_S1")
    stcmip.setReactantComponent("C2")
    stcmip.setProductComponent("C1")

    # kinetic law
    kl = sbml_rxn.createKineticLaw()
    kl.setMath(libsbml.parseL3Formula("1.0"))

    # Initials
    s = create_species(
        sbml_model,
        sbml_id="S1_C1_l_C2_u",
        compartment="compartment1",
        initial_concentration=1.0,
    )
    multi_species = s.getPlugin("multi")
    multi_species.setSpeciesType("ST1")
    create_species_feature(multi_species, "C_isotope", "_13C", component="C1")
    create_species_feature(multi_species, "C_isotope", "_12C", component="C2")

    print(libsbml.writeSBMLToString(sbml_document))

    sbml_document.validateSBML()
    _check_lib_sbml_errors(sbml_document, show_warnings=False)

    return sbml_document, sbml_model


def test_basic_network_generation(example_model_1):
    # TODO cleanup, test multi reactions, multiple reactants
    sbml_document, sbml_model = example_model_1
    nwg = NetworkGenerator(model=sbml_model)
    species_list = nwg.expand_model()
    assert nwg.multi_model.species_by_id["S1"].is_concrete() is False
    assert nwg.multi_model.species_by_id["S2"].is_concrete() is False
    assert nwg.multi_model.species_by_id["S1_C1_l_C2_u"].is_concrete() is True
    assert len(species_list) == 2
    assert set(map(str, species_list)) == {
        "S1_C1_l_C2_u(C1:['_13C'],C2:['_12C'])@compartment1",
        "ST2_compartment1_C1__12C_C2__13C(C2:['_13C'],C1:['_12C'])@compartment1",
    }

    core_model_name = "expanded.xml"
    nwg.create_sbml_core_model(core_model_name)

    # # test import via amici
    # model_name = "testmodel"
    # from amici.sbml_import import SbmlImporter
    #
    # with TemporaryDirectory(prefix=model_name) as d:
    #     SbmlImporter(core_model_name).sbml2amici(
    #         model_name=model_name, output_dir=d, verbose=True
    #     )
