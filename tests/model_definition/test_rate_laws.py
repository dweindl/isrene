from itertools import chain

import sympy as sp
from sbmlmath import SpeciesSymbol
from sympy import init_printing

from isrene import Q_
from isrene.model_definition import *
from isrene.model_definition.model_definition_rate_laws import keq_corr

init_printing(num_columns=200)


def test_keq_from_chem_potential():
    model = Model(id_="test1")
    Compartment("C1", Q_(1, model.volume_unit), model=model)
    SpeciesType("A", model=model)
    SpeciesType("B", model=model)

    # A <-> B
    reaction = ReactionPattern(
        "R1", [model["A"]()], [model["B"]()], reversible=True, model=model
    )
    keq = keq_from_chem_potential(reaction)
    assert isinstance(keq, Expression)
    assert keq.expr.free_symbols == {model["mu0_A"], model["mu0_B"]}
    # same chemical potential -> K_eq = 1
    assert keq.expr.subs(model["mu0_B"], model["mu0_A"]) == 1

    # 2 A <-> B
    reaction = ReactionPattern(
        "R2",
        [model["A"](), model["A"]()],
        [model["B"]()],
        reversible=True,
        model=model,
    )
    keq = keq_from_chem_potential(reaction)
    assert keq.expr.free_symbols == {model["mu0_A"], model["mu0_B"]}
    # same chemical potential -> K_eq = 1
    assert keq.expr.subs(model["mu0_B"], 2 * model["mu0_A"]) == 1


def test_keq_from_chem_potential_2():
    model = Model(id_="test1")
    Compartment("C1", Q_(1, model.volume_unit), model=model)
    SpeciesType("Ac", model=model)
    SpeciesType("OAA", model=model)
    SpeciesType("Cit", model=model)

    # Acetate(aq) + Oxaloacetate(aq) ⇌ Citrate(aq)
    # Estimated ΔrG'° 	-3.8 ± 0.9 [kJ/mol]     K'eq = 5
    # https://equilibrator.weizmann.ac.il/search?query=acetate+%2B+oxaloacetate+%E2%87%8C+citrate
    reaction = ReactionPattern(
        "R1",
        [model["Ac"](), model["OAA"]()],
        [model["Cit"]()],
        reversible=True,
        model=model,
    )
    keq = keq_from_chem_potential(reaction)
    assert keq.expr.free_symbols == {
        model["mu0_Ac"],
        model["mu0_OAA"],
        model["mu0_Cit"],
    }
    # same chemical potential -> K_eq = 1
    from math import isclose

    assert isclose(
        keq.expr.subs(
            {
                # ΔfG'° -- kJ/mol
                model["mu0_Ac"]: -238.4,
                model["mu0_OAA"]: -709.7,
                model["mu0_Cit"]: -952.0,
            }
        ),
        5,
        rel_tol=5 / 100,
    )


def test_keq_corr():
    model = Model(id_="test1")
    Compartment("C1", Q_(1, model.volume_unit), model=model)
    SpeciesType("A", model=model)
    SpeciesType("B", model=model)

    # A <-> B
    reaction = ReactionPattern(
        "R1", [model["A"]()], [model["B"]()], reversible=True, model=model
    )
    assert 1.0 == keq_corr(reaction)

    # 2A <-> B
    reaction = ReactionPattern(
        "R2",
        [model["A"](), model["A"]()],
        [model["B"]()],
        reversible=True,
        model=model,
    )
    # keq = 2M / (3M * 4M)
    # <-> keq * keq_corr = 2000 / (3000 * 4000)
    assert 2 / (3 * 4) * keq_corr(reaction) == 2000 / (3000 * 4000)
    # M / (M*M) => mM / (mM*mM) == 1/1000
    assert 1 / 1000 == keq_corr(reaction)


def test_ThermodynamicsRateLawGenerator():
    model = Model(id_="test1")
    Compartment("C1", Q_(1, model.volume_unit), model=model)
    SpeciesType("A", model=model)
    SpeciesType("B", model=model)

    # 2A <-> B
    reaction = ReactionPattern(
        "R1",
        [model["A"](), model["A"]()],
        [model["B"]()],
        reversible=True,
        model=model,
    )
    # 0 net flux at equilibrium
    # if keq == (mul(prod) / mul(subs))
    # kf * mul(subs) - kf / keq * mul(prod) := 0

    flux = ThermodynamicsRateLawGenerator().expression(reaction)
    print(flux)
    substrate_syms = reaction.get_reactant_symbols(filtered=True)
    product_syms = reaction.get_product_symbols(filtered=True)

    assert 0 == flux.subs(
        model["k_eq_R1"],
        sp.Mul(*product_syms) / sp.Mul(*substrate_syms) / keq_corr(reaction),
    )


def test_modular_rate_laws():
    model = Model(id_="test1")
    Compartment("C1", Q_(1, model.volume_unit), model=model)
    SpeciesType("A", model=model)
    SpeciesType("B", model=model)
    SpeciesType("C", model=model)
    E = Enzyme("E", model=model)

    # A <-> B + C
    reaction = ReactionPattern(
        "R1",
        [model["A"]()],
        [model["B"](), model["C"]()],
        reversible=True,
        model=model,
        enzyme=E(compartment=model["C1"]),
    )
    rate_expr = ModularRateLawGenerator(
        version="hal", sub_rate_law="CM"
    ).expression(reaction)

    rate_expr = rate_expr.subs(model["mrl_Dr_E"], model["mrl_Dr_E"].expr)

    # remove "vmax" part
    rate_expr /= model.get_species_symbol(E(compartment=model["C1"]))
    rate_expr /= model.get_compartment_symbol(model["C1"])
    rate_expr /= model["k_r_v_R1"]

    # no flux at equilibrium
    substrate_syms = reaction.get_reactant_symbols(filtered=True)
    product_syms = reaction.get_product_symbols(filtered=True)
    assert sp.solve(rate_expr) == [
        {
            model["k_eq_R1"]: sp.Mul(*product_syms)
            / sp.Mul(*substrate_syms)
            / keq_corr(reaction)
        }
    ]

    # K_eq_app < K_eq -> v > 0
    # K_eq_app > K_eq -> v < 0

    subs = {
        model.get_expression("k_eq_R1"): 1.1,
        model.get_expression("k_r_m_E_A"): 1,
        model.get_expression("k_r_m_E_B"): 1,
        model.get_expression("k_r_m_E_C"): 1,
        SpeciesSymbol("A", representation_type="sum"): 1000,
        SpeciesSymbol("B", representation_type="sum"): 1000,
        SpeciesSymbol("C", representation_type="sum"): 1000,
        **{x: 1000 for x in chain(substrate_syms, product_syms)},
    }
    assert rate_expr.subs(subs) > 0
    subs[model.get_expression("k_eq_R1")] = 0.9
    assert rate_expr.subs(subs) < 0

    # higher substrate concentration -> higher flux
    subs = {
        model.get_expression("k_eq_R1"): 1,
        model.get_expression("k_r_m_E_A"): 1,
        model.get_expression("k_r_m_E_B"): 1,
        model.get_expression("k_r_m_E_C"): 1,
        SpeciesSymbol("A", representation_type="sum"): 1000,
        SpeciesSymbol("B", representation_type="sum"): 1000,
        SpeciesSymbol("C", representation_type="sum"): 1000,
        **{
            x: 1000
            for x in chain(substrate_syms, product_syms)
            if x != substrate_syms[0]
        },
    }
    assert (rate_expr.subs(subs).diff(substrate_syms[0]) > 0) is sp.true

    # higher product concentration -> smaller flux
    subs = {
        model.get_expression("k_eq_R1"): 1,
        model.get_expression("k_r_m_E_A"): 1,
        model.get_expression("k_r_m_E_B"): 1,
        model.get_expression("k_r_m_E_C"): 1,
        SpeciesSymbol("A", representation_type="sum"): 1000,
        SpeciesSymbol("B", representation_type="sum"): 1000,
        SpeciesSymbol("C", representation_type="sum"): 1000,
        **{
            x: 1000
            for x in chain(substrate_syms, product_syms)
            if x != product_syms[0]
        },
    }
    assert (rate_expr.subs(subs).diff(product_syms[0]) < 0) is sp.true
