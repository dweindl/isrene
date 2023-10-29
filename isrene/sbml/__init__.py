"""Various helper functions for dealing with SBML (core) models"""
from itertools import filterfalse
from operator import methodcaller
from typing import Set

import libsbml

__all__ = [
    "get_free_symbols",
    "get_unused_parameters",
    "get_used_symbols",
]


def get_free_symbols(ast_node: libsbml.ASTNode) -> Set[str]:
    """Get all symbols (MathML `<ci>` elements) occurring in the given math
    element"""
    result = set()
    if ast_node.getType() == libsbml.AST_NAME:
        result.add(ast_node.getName())

    for i_child in range(ast_node.getNumChildren()):
        result |= get_free_symbols(ast_node.getChild(i_child))

    return result


def get_used_symbols(model: libsbml.Model) -> Set[str]:
    """Get IDs of symbols used in any math element"""
    used_symbols = set()
    for element in model.getListOfAllElements():
        if get_math := getattr(element, "getMath", None):
            used_symbols |= get_free_symbols(get_math())

    return used_symbols


def get_unused_parameters(
    model: libsbml.Model, ignore_rule_targets: bool = True
) -> Set[str]:
    """Get IDs of parameters not used in any math element"""
    used_symbols = get_used_symbols(model)
    parameters = set(map(methodcaller("getId"), model.getListOfParameters()))
    unused_parameters = parameters - used_symbols

    return (
        set(filterfalse(model.getRuleByVariable, unused_parameters))
        if ignore_rule_targets
        else unused_parameters
    )
