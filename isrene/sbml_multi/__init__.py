"""Functionality for working with SBML-multi models"""
from ..sbml.sbml_wrappers import _check


def enable_multi(
    sbml_document: "libsbml.SBMLDocument",
    multi_version: int = 1,
    prefix: str = "multi",
):
    """Enable the SBML-multi extension in the given SBML document"""
    pkg_uri = (
        f"http://www.sbml.org/sbml/level{sbml_document.getLevel()}"
        # FIXME version{sbml_document.getVersion()} yields LIBSBML_PKG_UNKNOWN
        f"/version1/"
        f"multi/version{multi_version}"
    )
    _check(sbml_document.enablePackage(pkg_uri, prefix, True))

    # Has to be set for multi package
    sbml_document.getPlugin(prefix).setRequired(True)
