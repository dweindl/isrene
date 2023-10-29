from __future__ import annotations

from itertools import chain
from typing import List, Literal

import sympy as sp

from .. import Q_, standard_temperature, ureg
from .model_definition import (
    Expression,
    Model,
    RateLawGenerator,
    ReactionPattern,
    SpeciesPattern,
    filter_reactants,
)

__all__ = [
    "ThermodynamicsRateLawGenerator",
    "ThermodynamicsRateLawGeneratorPanGaw2021",
    "ModularRateLawGenerator",
    "MassActionRateLawGenerator",
    "keq_from_chem_potential",
]


class MassActionRateLawGenerator(RateLawGenerator):
    r"""Generates simple mass action rate laws for a Reaction

    * Adds reaction rate coefficient k_$reactionId
    * Adds rate expression k_$reactionId * \prod_i substrate_i
    * Does not handle scaling of reaction rate by compartment sizes
    """

    def expression(self, reaction: ReactionPattern) -> sp.Expr:
        """As sympy expression"""
        if reaction.enzyme:
            from warnings import warn

            warn(
                f"{self.__class__.__name__} will generate one-step reactions "
                f"and ignore catalyzing enzymes ({reaction.id})."
            )

        substrates = filter_reactants(reaction.substrates)
        assert substrates, f"No substrates: {reaction}"
        substrate_syms = reaction.get_reactant_symbols(filtered=True)

        substrate_compartment_syms = {
            reaction.model.get_compartment_symbol(species.compartment)
            for species in substrates
        }

        kf = reaction.model.parameter_factory.forward_rate_constant(
            reaction_id=reaction.group_id, order=len(substrates)
        )

        if not reaction.reversible:
            return (
                kf * sp.Mul(*substrate_syms)
            ) * substrate_compartment_syms.pop()

        products = filter_reactants(reaction.products)
        assert products, f"No products: {reaction}"
        product_syms = reaction.get_product_symbols(filtered=True)

        # TODO alternatively from chemical potentials
        #   -- replaces ThermodynamicsRateLawGenerator?
        keq = reaction.model.parameter_factory.equilibrium_constant(
            reaction_id=reaction.group_id,
            delta=len(products) - len(substrates),
        )

        # k_eq correction for concentration unit
        # Equilibrium constants are conventionally computed from the
        # mass action ratio with concentrations in mol/L. If the model
        # uses a different concentration unit, we need to rescale those
        # concentrations, or more conveniently k_eq, otherwise the net flux
        # will not be 0 at the actual equilibrium concentrations.
        # (matters only if #reactants != #products)
        conv_factor = (
            Q_(1, ureg.molar) / Q_(1, reaction.model.concentration_unit)
        ).m_as(ureg.dimensionless)
        keq_corr = sp.Float(conv_factor) ** (len(products) - len(substrates))

        # need amount-changes for SBML export
        # (multiply by the volume of any involved compartment)
        rate = (
            kf * sp.Mul(*substrate_syms)
            - kf / (keq * keq_corr) * sp.Mul(*product_syms)
        ) * substrate_compartment_syms.pop()

        return rate


class ThermodynamicsRateLawGenerator(RateLawGenerator):
    r"""Generate rate expressions based on thermodynamic principles

    * Adds parameters `mu0_${molecule}` for chemical potentials at
      standard state (1M, 298K, 1atm)
      (compartment-independent)
      (ignores impact of temperature)
    * Adds Parameter `kf_${reaction_name}` and rate law:
      `v = kf_${reaction_name} * (
        (\prod c_${reactant}) - (\prod c_${products}) / K)`
    ` where K = exp(\sum_products mu_0 - \sum_educts mu_0)
    """

    def expression(self, reaction: ReactionPattern) -> sp.Expr:
        """As sympy expression"""
        assert reaction.reversible

        substrate_syms = reaction.get_reactant_symbols(filtered=True)
        product_syms = reaction.get_product_symbols(filtered=True)

        kf = reaction.model.parameter_factory.forward_rate_constant(
            reaction_id=reaction.group_id, order=len(substrate_syms)
        )
        fwd_rate = kf * sp.Mul(*substrate_syms)

        keq = keq_corr(reaction) * keq_from_chem_potential(reaction)

        bwd_rate = kf / keq * sp.Mul(*product_syms)
        return fwd_rate - bwd_rate


class ThermodynamicsRateLawGeneratorPanGaw2021(RateLawGenerator):
    r"""Generate rate expressions based on thermodynamic principles

    * Adds Parameters `K_${molecule}` as in chemical potential
      \mu = RT \ln(K * x)
      (compartment-independent)
    * Adds Parameter `kappa_${reaction_name}` and Expression
      `{reaction_id}_k{f,b} =
       kappa_${reaction_name} * \prod K_${reactant}`
      and uses those as rate expression

    Compare https://doi.org/10.1101/2021.07.26.453900
    """

    def expression(self, reaction: ReactionPattern) -> sp.Expr:
        """As sympy expression"""
        # TODO: units from model
        kappa = Expression(
            f"kappa_{reaction.group_id}",
            Q_(1.0, reaction.model.amount_unit / reaction.model.time_unit),
            estimated=True,
            non_negative=True,
            model=reaction.model,
        )

        substrate_syms = reaction.get_reactant_symbols(filtered=True)
        product_syms = reaction.get_product_symbols(filtered=True)

        K_exprs_substrates = [
            self._get_k(reactant, reaction.model)
            for reactant in reaction.substrates
        ]
        K_exprs_products = [
            self._get_k(reactant, reaction.model)
            for reactant in reaction.products
        ]

        rate_fwd = kappa * sp.Mul(
            *(
                k * substrate_sym
                for k, substrate_sym in zip(K_exprs_substrates, substrate_syms)
            )
        )
        rate_bwd = kappa * sp.Mul(
            *(
                k * product_sym
                for k, product_sym in zip(K_exprs_products, product_syms)
            )
        )
        return rate_fwd - rate_bwd

    @staticmethod
    def _get_k(reactant: SpeciesPattern, model: Model):
        """Return K parameter for the given pattern"""
        # TODO: assumes no complex
        # TODO: unit from model
        return Expression(
            f"K_{reactant.template.id}",
            Q_(1.0, 1 / model.concentration_unit),
            model=model,
        )


class ModularRateLawGenerator(RateLawGenerator):
    r"""
    See LiebermeisterUhl2010:

    ‘common’ (CM), ‘direct binding’ (DM), ‘simultaneous binding’ (SM),
    ‘power-law’ (PM) and ‘force-dependent’ (FM)

    common:
        generalized reversible Michaelis-Menten; random-order mechanism;
        +/- convenience kinetics
    direct binding:
        ...
    simultaneous binding:
        ...
    power-law:
        corresponds to mass action (no enzyme-saturation)
    force-dependent:
        ...

    Different versions:
    cat: Explicit
    hal: Haldane-compliant
    weg: Wegscheider-compliant

    Introduces the following parameters:

    k_r_m_{rxn_id}:
        :math:`K_M` values for each reaction x reactant
    k_plus_r_{rxn_id}, k_minus_r_{rxn_id} ('cat'-only):
        forward and backward turnover rates, per reaction
    k_r_v_{rxn_id} ('hal','weg'-only):
        reaction velocity constant, per reaction
    k_r_eq_{rxn_id} ('hal'-only):
        reaction equilibrium constant, per reaction
    mu0_{reactant_id}:
        standard chemical potential, per reactant (same for all reactions)

    NOTE: cooperativity factor not yet supported (i.e. hard-coded to 1.0)
    """

    def __init__(
        self,
        # reaction: Reaction,
        sub_rate_law: Literal["CM", "DM", "SM", "FM", "PM"] = "CM",
        version: Literal["cat", "hal", "weg"] = "weg",
    ):
        if sub_rate_law in {"CM", "DM", "SM", "FM", "PM"}:
            self.sub_rate_law = sub_rate_law
        else:
            raise ValueError(
                f"Unknown modular rate law subtype: {sub_rate_law}"
            )

        if version in {"cat", "hal", "weg"}:
            self.version = version
        else:
            raise ValueError(
                f"Unknown modular rate law version: {sub_rate_law}"
            )

    def expression(self, reaction: ReactionPattern) -> sp.Expr:
        """As sympy expression"""
        # See Supplementary table A.1
        # TODO: assumes states are concentrations (verify or handle amounts)
        # NOTE: stoichiometric coefficients are omitted, as reactants are
        #  replicated in the reaction (implicit coefficient = 1)
        if not reaction.enzyme:
            raise ValueError(
                "Modular rates laws require an enzyme, but none was set for "
                f"reaction {reaction.id}."
            )
        if not reaction.reversible:
            raise NotImplementedError(
                "Modular rate laws for irreversible reactions are not "
                "supported."
            )

        substrates = filter_reactants(reaction.substrates)
        products = filter_reactants(reaction.products)
        assert substrates
        assert products

        # cooperativity factor
        sym_h_r = sp.Integer(1)
        # note: h_r is so far omitted in m_{ri}^\pm

        substrate_k_r_m_syms = self.get_krm_symbols(reaction, substrates)
        product_k_r_m_syms = self.get_krm_symbols(reaction, products)

        # concentration symbols
        # NOTE: assumes all species are in concentrations (not amounts)
        substrate_c_syms = reaction.get_reactant_symbols(filtered=True)
        product_c_syms = reaction.get_product_symbols(filtered=True)

        assert len(substrate_c_syms) == len(substrate_k_r_m_syms)
        assert len(product_c_syms) == len(product_k_r_m_syms)

        # Numerator T_r
        if self.version == "cat":
            # turnover rates
            sym_k_plus_r = Expression(
                f"k_plus_r_{reaction.group_id}",
                1.0,
                sbo_term="SBO:0000320",
                model=reaction.model,
            )
            sym_k_minus_r = Expression(
                f"k_minus_r_{reaction.group_id}",
                1.0,
                sbo_term="SBO:0000321",
                model=reaction.model,
            )

            sym_T_r = sym_k_plus_r * sp.Mul(
                *[
                    (c / k_r_m) ** sym_h_r
                    for c, k_r_m in zip(substrate_c_syms, substrate_k_r_m_syms)
                ]
            ) + sym_k_minus_r * sp.Mul(
                *[
                    (c / k_r_m) ** sym_h_r
                    for c, k_r_m in zip(product_c_syms, product_k_r_m_syms)
                ]
            )
        elif self.version in ("hal", "weg"):
            # Velocity constant
            sym_k_r_v_id = f"k_r_v_{reaction.group_id}"
            sym_k_r_v = reaction.model[sym_k_r_v_id] or Expression(
                sym_k_r_v_id,
                Q_(1.0, 1 / reaction.model.time_unit),
                name_=f"{reaction.group_id} modular rate law catalytic rate "
                "constant geometric mean",
                sbo_term="SBO:0000482",
                model=reaction.model,
            )
            if self.version == "hal":
                # equilibrium constant as constant
                keq = reaction.model.parameter_factory.equilibrium_constant(
                    reaction_id=reaction.group_id,
                    delta=len(products) - len(substrates),
                )
            elif self.version == "weg":
                keq = keq_from_chem_potential(reaction)

            keq *= keq_corr(reaction)

            # # above eq 25
            # kf = sym_k_r_v * (keq * keq_corr * sp.Mul(*substrate_k_r_m_syms)
            #                   / sp.Mul(*product_k_r_m_syms)) ** (sym_h_r / 2)
            #
            # kb = sym_k_r_v * (keq * keq_corr * sp.Mul(*substrate_k_r_m_syms)
            #                   / sp.Mul(*product_k_r_m_syms)) ** (-sym_h_r / 2)
            #
            # # k_r_v = sqrt(kf * kb) by definition
            # if sym_k_r_v ** 2 != kf * kb:
            #     raise AssertionError(
            #         "Wrong expressions: "
            #         f"{sym_k_r_v ** 2=} != {kf * kb=}"
            #     )
            # # Haldane compliant (eq 15)
            # assert keq_corr * keq ** sym_h_r == (
            #         kf / kb
            #         * sp.Mul(*product_k_r_m_syms) ** sym_h_r
            #         / sp.Mul(*substrate_k_r_m_syms)** sym_h_r
            # )

            #  unit: 1/time * concentration^\pm(n_subs-n_prods)
            # see eq 2 or 'cat' in Suppl Table A.1
            # sym_T_r = (
            #         kf * (sp.Mul(*substrate_c_syms)
            #               / sp.Mul(*substrate_k_r_m_syms)) ** sym_h_r
            #         - kb * (sp.Mul(*product_c_syms)
            #                 / sp.Mul(*product_k_r_m_syms)) ** sym_h_r
            # )
            # see 'hal' in Suppl Table A.1
            sym_T_r = (
                sym_k_r_v
                * (
                    # TODO k_eq correction to be applied to keq? -> sqrt
                    #      or to concentrations -> no sqrt
                    keq ** (sym_h_r / 2) * sp.Mul(*substrate_c_syms) ** sym_h_r
                    - keq ** (-sym_h_r / 2)
                    * sp.Mul(*product_c_syms) ** sym_h_r
                )
                / sp.Mul(*chain(substrate_k_r_m_syms, product_k_r_m_syms))
                ** (sym_h_r / 2)
            )
        else:
            raise AssertionError(f"Invalid sub_rate_law '{self.sub_rate_law}'")

        # Denominator D_r
        # Accounts for different binding states of the enzyme, reducing its
        # effective concentration.
        # Here we need to account for any competing substrates.
        # Therefore, this term is not reaction-specific, but enzyme-specific.
        # NOTE: Currently assuming all enzyme states/complexes have the same
        #  enzymatic activity
        D_r_expr_id = f"mrl_Dr_{reaction.enzyme.template.id}"
        try:
            # Generated previously?
            sym_D_r = reaction.model.get_expression(D_r_expr_id)
        except ValueError:
            # Create new
            sym_D_r = self._generate_D_r(
                model=reaction.model,
                enzyme=reaction.enzyme,
                sym_h_r=sym_h_r,
                D_r_expr_id=D_r_expr_id,
            )

        # Enzyme amount(!) u_r
        # TODO if we ever support complex patterns, this needs to be expanded
        #   (need to set representationType=sum)
        sym_u_r = reaction.model.get_species_symbol(
            reaction.enzyme
        ) * reaction.model.get_compartment_symbol(reaction.enzyme.compartment)

        # allosteric regulation f_r
        #  (complete/non-competitive activation/inhibition)
        sym_f_r = sp.Integer(1)

        # Specific activation/inhibition
        sym_D_r_reg = sp.Float(0)

        # net flux
        expr_v_r = sym_u_r * sym_f_r * sym_T_r / (sym_D_r + sym_D_r_reg)

        return expr_v_r

    def _generate_D_r(
        self,
        model: Model,
        enzyme: SpeciesPattern,
        D_r_expr_id: str,
        sym_h_r,
    ) -> sp.Expr:
        """Generate denominator D_r for the given enzyme

        Accounts for different binding states of the enzyme, reducing its
        effective concentration.
        """
        # We need to consider the concentration of all isotopologues
        #  that can act as reactants. And also other reactions catalyzed by the
        #  same enzyme.

        # We sum up all D_r expressions for the individual reactions catalyzed
        # by the given enzyme, and apply some corrections for redundant states.
        # See also LiebermeisterUhl2010 S1 Eq B.12

        sym_D_r = 0
        # Reactants that occurred so far
        seen_reactants = []
        # Number of reactions found to be catalyzed by the given enzyme
        num_matched_reactions = 0
        # Does any substrate in any reaction have
        # stoichiometric coefficients != 1
        non_unity_stoichiometry = False
        # Does any species occur in any reaction
        # on both substrate and product side?
        reactant_product_overlap = False

        for reaction in model.reactions:
            if reaction.enzyme != enzyme:
                continue
            # we check whether we already had the same reactants.
            # this is an issue for reactions that can yield different sets
            # of products from the same set of substrates (e.g. A->B and A->C).
            # we must not count them twice.
            # this seems is a bit difficult to handle for
            # substrates with stoichiometric coefficients>1, as we would
            # need information on which binding sites which substrate
            # binds to.
            substrates = filter_reactants(reaction.substrates)
            products = filter_reactants(reaction.products)
            reactant_product_overlap |= any(
                map(substrates.__contains__, products)
            )
            non_unity_stoichiometry |= _has_duplicates(
                substrates
            ) or _has_duplicates(products)
            sym_D_r += self._sym_Dr_single_reaction(reaction, sym_h_r)

            # correct for already considered reactants
            if any(map(seen_reactants.__contains__, substrates)) or any(
                map(seen_reactants.__contains__, substrates)
            ):
                # sym_D_r based on a dummy reaction with the already
                # considered substrates/products will give us the redundant
                # expressions we need to remove.
                # this also takes care of reactions with same substrates,
                # but different products.
                # this will fail if we have multiple instances of the same
                # reactant on the substrate or the product side
                sym_D_r -= (
                    self._sym_Dr_single_reaction(
                        ReactionPattern(
                            "",
                            substrates=[
                                x for x in substrates if x in seen_reactants
                            ],
                            products=[
                                x for x in products if x in seen_reactants
                            ],
                            reversible=True,
                            model=reaction.model,
                            enzyme=reaction.enzyme,
                            # make sure this does not get added to the model
                            add=False,
                        ),
                        sym_h_r,
                    )
                    - 1
                )
            seen_reactants.extend(
                chain(reaction.substrates, reaction.products)
            )
            num_matched_reactions += 1

        # account for free enzyme that was potentially counted several times
        # See also LiebermeisterUhl2010 S1 Eq B.12
        sym_D_r -= num_matched_reactions - 1

        if num_matched_reactions > 1 and (
            reactant_product_overlap or non_unity_stoichiometry
        ):
            raise NotImplementedError(
                "Cannot handle multiple reactions catalyzed by the "
                f"same enzyme {enzyme.template.id} because substrates "
                "and products overlap or reactants have "
                "stoichiometric coefficients != 1."
            )

        # Current workaround for taking into account enzyme saturation
        #  from competing isotopologue substrates is adding this term as a
        #  model expression (instead of directly in the kinetic law), for
        #  which representationType=SUM in SBML multi is set automatically
        #  during export. We currently can't represent representationType
        #  in sympy expressions, so it can't go into the kinetic law.
        if sym_D_r != 1:
            sym_D_r = Expression(
                D_r_expr_id,
                sym_D_r,
                estimated=False,
                name_=f"Modular rate law D_r term for {enzyme.template.id}",
                model=reaction.model,
            )
        return sym_D_r

    def _sym_Dr_single_reaction(
        self,
        reaction: ReactionPattern,
        sym_h_r,
    ) -> sp.Expr:
        """Generate sympy expression for D_r for the given reaction.

        (Not accounting for other reactions catalyzed by the same enzyme)
        """
        if self.sub_rate_law == "PM":
            return sp.Integer(1)

        substrates = filter_reactants(reaction.substrates)
        products = filter_reactants(reaction.products)
        substrate_k_r_m_syms = self.get_krm_symbols(reaction, substrates)
        product_k_r_m_syms = self.get_krm_symbols(reaction, products)
        substrate_c_syms = reaction.get_reactant_symbols(
            filtered=True,
            # we need to account for all labelling states
            representation_type="sum",
        )
        product_c_syms = reaction.get_product_symbols(
            filtered=True,
            # we need to account for all labelling states
            representation_type="sum",
        )

        if self.sub_rate_law == "CM":
            return (
                sp.Mul(
                    *(
                        (1 + c / k_r_m) ** sym_h_r
                        for c, k_r_m in zip(
                            substrate_c_syms, substrate_k_r_m_syms
                        )
                    )
                )
                + sp.Mul(
                    *(
                        (1 + c / k_r_m) ** sym_h_r
                        for c, k_r_m in zip(product_c_syms, product_k_r_m_syms)
                    )
                )
                - 1
            )

        if self.sub_rate_law == "SM":
            return sp.Mul(
                *(
                    (1 + c / k_r_m) ** sym_h_r
                    for c, k_r_m in zip(
                        chain(substrate_c_syms, product_c_syms),
                        chain(substrate_k_r_m_syms, product_k_r_m_syms),
                    )
                )
            )

        if self.sub_rate_law == "DM":
            return (
                1
                + sp.Mul(
                    *[
                        (c / k_r_m) ** sym_h_r
                        for c, k_r_m in zip(
                            substrate_c_syms, substrate_k_r_m_syms
                        )
                    ]
                )
                + sp.Mul(
                    *[
                        (c / k_r_m) ** sym_h_r
                        for c, k_r_m in zip(product_c_syms, product_k_r_m_syms)
                    ]
                )
            )

        if self.sub_rate_law == "FM":
            return sp.Mul(
                *[
                    (c / k_r_m) ** (sym_h_r / 2)
                    for c, k_r_m in zip(
                        chain(substrate_c_syms, product_c_syms),
                        chain(substrate_k_r_m_syms, product_k_r_m_syms),
                    )
                ]
            )

        raise AssertionError(f"Invalid sub_rate_law '{self.sub_rate_law}'")

    @staticmethod
    def get_krm_symbols(
        reaction: ReactionPattern, species_patterns: List[SpeciesPattern]
    ):
        """Generate model expression for current reactant/reaction"""
        # reactant constants
        #  (half-saturation concentration / dissociation constants)

        return [
            reaction.model.parameter_factory.michaelis_constant(
                enzyme_id=str(reaction.enzyme.template.id),
                # NOTE: this should NOT be compartment-specific
                reactant_id=reaction.model._generate_species_id(
                    species_pattern,
                    encode_state=False,
                    encode_compartment=False,
                ),
            )
            for species_pattern in species_patterns
        ]


def _has_duplicates(seq):
    """Check for duplicates in sequences of unhashable objects"""
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return len(seq) != len(unique_list)


def keq_from_chem_potential(reaction: ReactionPattern) -> sp.Expr:
    r"""
    Expression for equilibrium constant.

        deltaG0 = \sum mu0_products - \sum mu0_substrates
        K_eq = exp(- deltaG0 / RT)

    Generate / get expression for equilibrium constant for the given reaction
    computed from chemical potentials of the reactants.
    """
    model = reaction.model
    mu_expected_units = model._molar_energy_unit
    parameter_factory = model.parameter_factory

    # NOTE: don't ignore compounds from ignored_substrates for computing K_eq!
    mu0_substrates = list(
        map(parameter_factory.chemical_potential, reaction.substrates)
    )
    mu0_products = list(
        map(parameter_factory.chemical_potential, reaction.products)
    )

    # Check expected units for concentrations and chemical
    #  potentials match (assuming 1 mM as reference state)
    # see Supplement C.1
    # handle float issues:
    if Q_(1, model.concentration_unit) - Q_(
        1, ureg.millimolar
    ).to_base_units() > Q_("1e-14 mM"):
        raise ValueError(
            "Only concentrations in mM and chemical "
            f"potentials in {mu_expected_units} are currently "
            "supported, but concentration unit was "
            f"{model.concentration_unit}."
        )
    for mu0 in chain(mu0_products, mu0_substrates):
        if mu0.expr.units != mu_expected_units:
            raise ValueError(
                "Only concentrations in mM and chemical "
                f"potentials in {mu_expected_units} are currently "
                f"supported, but chemical potential {mu0.name} "
                f"has units {mu0.expr.units}."
            )

    RT = (standard_temperature * ureg.R).m_as(mu_expected_units)
    delta_g_0 = sp.Add(*mu0_products) - sp.Add(*mu0_substrates)

    # unit of k_eq: dimensionless
    #  (vs k_eq = \prod c_prod * \prod c_subs, where unit is
    #          (concentration unit)^(n_prod - n_subs)
    keq = sp.exp(-delta_g_0 / RT)

    # TODO check if already exists
    keq = Expression(
        f"k_eq_{reaction.group_id}",
        keq,
        estimated=False,
        non_negative=True,
        sbo_term="SBO:0000281",
        model=model,
    )
    return keq


def keq_corr(reaction: ReactionPattern) -> float:
    """k_eq correction for concentration unit

    Equilibrium constants are conventionally computed from the
    mass action ratio with concentrations in mol/L. If the model
    uses a different concentration unit, we need to rescale those
    concentrations, or more conveniently k_eq, otherwise the net flux
    will not be 0 at the actual equilibrium concentrations.
    (matters only if #reactants != #products)
    """
    substrates = filter_reactants(reaction.substrates)
    products = filter_reactants(reaction.products)

    if len(substrates) == len(products):
        return 1

    conv_factor = (
        Q_(1, ureg.molar) / Q_(1, reaction.model.concentration_unit)
    ).m_as(ureg.dimensionless)
    keq_corr = sp.Float(conv_factor) ** (len(products) - len(substrates))
    # TODO for SBML unit consistency:
    # keq_corr *= Q_(1, reaction.model.concentration_unit) ** (
    #     len(products) - len(substrates)
    # )

    return keq_corr
