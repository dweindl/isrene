from typing import Dict

import libsbml

from ..sbml.sbml_wrappers import _check

# XML namespace URI and prefix for SBML annotations elements
annotation_xmlns = "some_ns_uri"
annotation_prefix = "myannot"


def set_annotations(
    element: libsbml.SBase, annotations: Dict[str, Dict[str, str]]
) -> None:
    """Set SBML annotation for the given element"""
    if not annotations:
        return

    def get_attributes(attrs: Dict[str, str]) -> str:
        return " ".join(
            f'{annotation_prefix}:{attrib_id}="{attrib_value}"'
            for attrib_id, attrib_value in attrs.items()
        )

    for annot_type, attrs in annotations.items():
        annot_str = (
            f"<{annotation_prefix}:{annot_type} "
            f'xmlns:{annotation_prefix}="{annotation_xmlns}" '
            f"{get_attributes(attrs)}/>"
        )
        _check(element.appendAnnotation(annot_str))


def get_annotations(annot_node: libsbml.XMLNode) -> Dict:
    """Retrieve annotations from an SBML annotation block

    Return dict with item names mapping to dicts of attribute IDs/values
    """
    if annot_node is None:
        return {}

    res = {}
    for child_idx in range(annot_node.getNumChildren()):
        child = annot_node.getChild(child_idx)
        if child.getPrefix() != annotation_prefix:
            continue

        xml_attrs = child.getAttributes()
        attrs = {
            xml_attrs.getName(attr_idx): xml_attrs.getValue(attr_idx)
            for attr_idx in range(xml_attrs.getNumAttributes())
        }
        res[child.getName()] = attrs
    return res
