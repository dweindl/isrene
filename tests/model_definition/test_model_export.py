import libsbml
import pytest

from isrene.model_definition.model_export_sbml import *
from isrene.sbml.annotations import get_annotations, set_annotations


@pytest.fixture
def new_model():
    """Create an SBML model"""
    sbml_document = libsbml.SBMLDocument(3, 1)
    sbml_model = sbml_document.createModel()
    return sbml_document, sbml_model


def test_annotations_roundtrip(new_model):
    sbml_document, sbml_model = new_model
    expected_annotations = {
        "foo": {"bar": "2", "baz": "true"},
        "bar": {"bar": "3", "baz": "true"},
    }
    set_annotations(sbml_model, expected_annotations)
    actual_annotations = get_annotations(sbml_model.getAnnotation())

    assert expected_annotations == actual_annotations
