#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
#%%
# Import Python libraries
import pandas as pd
import numpy as np
import math
import time
import json

# Import Pyomo libraries
from pyomo.environ import ConcreteModel, value, Var, Reals, Param, TransformationFactory, \
    Constraint, Expression, Objective, check_optimal_termination, exp, units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES Libraries
import idaes
from idaes.core import FlowsheetBlock
from idaes.models_extra.column_models.MEAsolvent_column import MEAColumn
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock)

from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              unused_variables_set,
                                              large_residuals_set)
from pyomo.util.infeasible import log_infeasible_constraints
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale

from idaes.core import FlowDirection

import logging

logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)
import sys

sys.path.insert(0, 'flowsheets')
from mea_properties import (
    MEALiquidParameterBlock,
    FlueGasParameterBlock,
    scale_mea_liquid_params,
    scale_mea_vapor_params,
    switch_liquid_to_parmest_params,
    initialize_inherent_reactions
)

# -----------------------------------------------------------------------------
solver = get_solver()
# solver.options["bound_push"] = 1e-22

# NCCC pilot test data
# Reference: Development of a Rigorous Modeling Framework for Solvent based CO2 capture. Part 2
# Steady state validation and uncertainty quantification with pilot plant data
# Ind. Eng. Chem. Res. 2018, 57, 10464-10481

NCCC_test = {'K13': {'diameter': 0.64, 'length': 18,
                     'vapor_inlet': {'flow_mol': 21.349296001274,
                                     'temperature': 315.24,
                                     'pressure': 107560,
                                     'mole_frac_comp': {'CO2': 0.0935693229788952,
                                                        'H2O': 0.0737536427126972,
                                                        'N2': 0.745948398741712,
                                                        'O2': 0.0867286355666952}},
                     'liquid_inlet': {'flow_mol': 81.0566893462779,
                                      'temperature': 315.08,
                                      'pressure': 107560,
                                      'mole_frac_comp': {'CO2': 0.0192670770845385,
                                                         'H2O': 0.869252058393728,
                                                         'MEA': 0.111480864521734}}},
             'K17': {'diameter': 0.64, 'length': 12,
                     'vapor_inlet': {'flow_mol': 21.3521601734185,
                                     'temperature': 314.18,
                                     'pressure': 108120,
                                     'mole_frac_comp': {'CO2': 0.0919717766725138,
                                                        'H2O': 0.0701377703889619,
                                                        'N2': 0.748200074176874,
                                                        'O2': 0.0896903787616504}},
                     'liquid_inlet': {'flow_mol': 80.6281332727093,
                                      'temperature': 315.18,
                                      'pressure': 108120,
                                      'mole_frac_comp': {'CO2': 0.0210508190313722,
                                                         'H2O': 0.865667140600553,
                                                         'MEA': 0.113282040368075}}},
             'K18': {'diameter': 0.64, 'length': 6,
                     'vapor_inlet': {'flow_mol': 21.7367507327159,
                                     'temperature': 319.22,
                                     'pressure': 109180,
                                     'mole_frac_comp': {'CO2': 0.101947863634366,
                                                        'H2O': 0.0912918913073196,
                                                        'N2': 0.734006946649283,
                                                        'O2': 0.072753298409031}},
                     'liquid_inlet': {'flow_mol': 81.3551492717131,
                                      'temperature': 315.39,
                                      'pressure': 109180,
                                      'mole_frac_comp': {'CO2': 0.0164615447958917,
                                                         'H2O': 0.872350449774912,
                                                         'MEA': 0.111188005429196}}},
             'K19': {'diameter': 0.64, 'length': 6,
                     'vapor_inlet': {'flow_mol': 13.7503082525113,
                                     'temperature': 319.33,
                                     'pressure': 108020,
                                     'mole_frac_comp': {'CO2': 0.11002211467436,
                                                        'H2O': 0.0935427361795275,
                                                        'N2': 0.733191112428523,
                                                        'O2': 0.0632440367175896}},
                     'liquid_inlet': {'flow_mol': 143.377802339556,
                                      'temperature': 314.05,
                                      'pressure': 108020,
                                      'mole_frac_comp': {'CO2': 0.0201775176672136,
                                                         'H2O': 0.87979225318062,
                                                         'MEA': 0.100030229152166}}},
             'K20': {'diameter': 0.64, 'length': 6,
                     'vapor_inlet': {'flow_mol': 12.6437596188192,
                                     'temperature': 319.24,
                                     'pressure': 108120,
                                     'mole_frac_comp': {'CO2': 0.109893443469725,
                                                        'H2O': 0.0933084406858741,
                                                        'N2': 0.734195950035033,
                                                        'O2': 0.062602165809368}},
                     'liquid_inlet': {'flow_mol': 39.1256218477082,
                                      'temperature': 318.45,
                                      'pressure': 108120,
                                      'mole_frac_comp': {'CO2': 0.00796534573075643,
                                                         'H2O': 0.891743375496661,
                                                         'MEA': 0.100291278772583}}},
             'K21': {'diameter': 0.64, 'length': 12,
                     'vapor_inlet': {'flow_mol': 13.0874229278781,
                                     'temperature': 319.26,
                                     'pressure': 107760,
                                     'mole_frac_comp': {'CO2': 0.101861100001204,
                                                        'H2O': 0.093471594809597,
                                                        'N2': 0.733235652437409,
                                                        'O2': 0.07143165275179}},
                     'liquid_inlet': {'flow_mol': 39.3621792556852,
                                      'temperature': 318.33,
                                      'pressure': 107760,
                                      'mole_frac_comp': {'CO2': 0.00887656870404782,
                                                         'H2O': 0.894527873673133,
                                                         'MEA': 0.096595557622819}}},
             'SRP': {'diameter': 0.467, 'length': 6,
                     'vapor_inlet': {'flow_mol': 3.52,
                                     'temperature': 320,
                                     'pressure': 109180,
                                     'mole_frac_comp': {'CO2': 0.1,
                                                        'H2O': 0.013,
                                                        'N2': 0.8175115207373272,
                                                        'O2': 0.06948847926267282}},
                     'liquid_inlet': {'flow_mol': 30.006359526818237,
                                      'temperature': 314.0,
                                      'pressure': 109180,
                                      'mole_frac_comp': {'CO2': 0.03353820798950301,
                                                         'H2O': 0.8462531612237478,
                                                         'MEA': 0.12020863078674912}}}
             }


#
# parmest_parameters = {
#     'VLE': {'bic_k_eq_coeff_1': 366.061867998774,
#             'bic_k_eq_coeff_2': -13326.25411,
#             'bic_k_eq_coeff_3': -55.68643292,
#             'car_k_eq_coeff_1': 164.039636,
#             'car_k_eq_coeff_2': -707.0056712,
#             'car_k_eq_coeff_3': -26.40136817,
#             'lwm_coeff_1': -2.076073001,
#             'lwm_coeff_2': 0.037322205,
#             'lwm_coeff_3': -0.00032721,
#             'lwm_coeff_4': -0.111102655,
#             },
#     'surface_tension': {'surf_tens_CO2_coeff_1': -0.00589934906112609,
#                         'surf_tens_CO2_coeff_2': 0.00175020536428591,
#                         'surf_tens_CO2_coeff_3': 0.129650182728177,
#                         'surf_tens_CO2_coeff_4': 0.0000126444768126308,
#                         'surf_tens_CO2_coeff_5': -5.73954817199691E-06,
#                         'surf_tens_CO2_coeff_6': -0.00018969005534195,
#                         'surf_tens_F_coeff_a': 1070.65668317975,
#                         'surf_tens_F_coeff_b': -2578.78134208703,
#                         'surf_tens_F_coeff_c': 3399.24113311222,
#                         'surf_tens_F_coeff_d': -2352.47410135319,
#                         'surf_tens_F_coeff_e': 2960.24753687833,
#                         'surf_tens_F_coeff_f': 3.06684894924048,
#                         'surf_tens_F_coeff_g': -1.79435372759593,
#                         'surf_tens_F_coeff_h': -7.2124219075848,
#                         'surf_tens_F_coeff_i': 2.97502322396621,
#                         'surf_tens_F_coeff_j': -10.5738529301824,
#                         },
#     'molar_volume': {'vol_mol_liq_comp_coeff_a': -10.5792012186177,
#                      'vol_mol_liq_comp_coeff_b': -2.02049415703576,
#                      'vol_mol_liq_comp_coeff_c': 3.1506793296904,
#                      'vol_mol_liq_comp_coeff_d': 192.012600751473,
#                      'vol_mol_liq_comp_coeff_e': -695.384861676286,
#                      },
#     'viscosity': {'visc_d_coeff_a': -0.0854041877181552,
#                   'visc_d_coeff_b': 2.72913373574306,
#                   'visc_d_coeff_c': 35.1158892542595,
#                   'visc_d_coeff_d': 1805.52759876533,
#                   'visc_d_coeff_e': 0.00716025669867574,
#                   'visc_d_coeff_f': 0.0106488402285381,
#                   'visc_d_coeff_g': -0.0854041877181552,
#                   },
# }


def build_column_model(x_nfe_list):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Set up property package
    m.fs.vapor_properties = FlueGasParameterBlock()
    params = MEALiquidParameterBlock(ions=True)
    m.fs.liquid_properties = MEALiquidParameterBlock(ions=True)

    # Create an instance of the column in the flowsheet
    m.fs.unit = MEAColumn(
        finite_elements=len(x_nfe_list) - 1,
        length_domain_set=x_nfe_list,
        vapor_phase={
            "property_package": m.fs.vapor_properties},
        liquid_phase={
            "property_package": m.fs.liquid_properties})

    return m


def get_uniform_grid(nfe):
    # Finite element list in the spatial domain
    x_nfe_list = [i / nfe for i in range(nfe + 1)]

    return x_nfe_list


def get_custom_grid_dx(x_locations=[0.3], dx=[0.01, 0.1]):
    # This function assumes that the spatial domain is between 0 and 1. 
    # x_locations is a list of the points in the spatial domain between 
    # which the custom mesh should be created (anchors). The end points 0 and 1
    # should not be included in x_locations.
    # dx is a list of the desired step sizes between the anchors.

    x_nfe_list = []

    for k in range(len(x_locations)):
        if k == 0:
            x_nfe_list_temp = np.linspace(0, x_locations[k],
                                          round(x_locations[k] / dx[k]) + 1).tolist()
        else:
            x_nfe_list_temp = np.linspace(x_locations[k - 1], x_locations[k],
                                          round((x_locations[k] -
                                                 x_locations[k - 1]) / dx[k]) + 1).tolist()

        x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))

    x_nfe_list_temp = np.linspace(x_locations[k], 1,
                                  round((1 - x_locations[k]) / dx[k + 1]) + 1).tolist()
    x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))

    # nfe = len(x_nfe_list) -1

    return x_nfe_list.tolist()


def get_custom_grid_nfe(x_locations=[0.3], x_nfe=[30, 10]):
    # This function assumes that the spatial domain is between 0 and 1. 
    # x_locations is a list of the points in the spatial domain between 
    # which the custom mesh should be created (anchors). The end points 0 and 1
    # should not be included in x_locations.
    # x_nfe is a list of the desired number of finite elements between the 
    # anchors.

    x_nfe_list = []

    for k in range(len(x_locations)):
        if k == 0:
            x_nfe_list_temp = np.linspace(0, x_locations[k],
                                          x_nfe[k]).tolist()
        else:
            x_nfe_list_temp = np.linspace(x_locations[k - 1], x_locations[k],
                                          x_nfe[k]).tolist()

        x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))

    x_nfe_list_temp = np.linspace(x_locations[k], 1,
                                  x_nfe[k + 1]).tolist()
    x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))

    # nfe = len(x_nfe_list) -1

    return x_nfe_list.tolist()


def set_inputs(m, input_dic=NCCC_test, case='K13'):
    # Fix column design variables
    # Absorber diameter 
    m.fs.unit.diameter_column.fix(input_dic[case]['diameter'])

    # Absorber length according to number of packed beds 
    # 1 bed is approx 6 m
    m.fs.unit.length_column.fix(input_dic[case]['length'])  # meter

    # Fix operating conditions
    # Flue gas
    m.fs.unit.vapor_inlet.flow_mol.fix(input_dic[case]['vapor_inlet']['flow_mol'])
    m.fs.unit.vapor_inlet.temperature.fix(input_dic[case]['vapor_inlet']['temperature'])
    m.fs.unit.vapor_inlet.pressure.fix(input_dic[case]['vapor_inlet']['pressure'])

    m.fs.unit.vapor_inlet.mole_frac_comp[0, "CO2"].fix(input_dic[case]['vapor_inlet']['mole_frac_comp']['CO2'])
    m.fs.unit.vapor_inlet.mole_frac_comp[0, "H2O"].fix(input_dic[case]['vapor_inlet']['mole_frac_comp']['H2O'])
    m.fs.unit.vapor_inlet.mole_frac_comp[0, "N2"].fix(input_dic[case]['vapor_inlet']['mole_frac_comp']['N2'])
    m.fs.unit.vapor_inlet.mole_frac_comp[0, "O2"].fix(input_dic[case]['vapor_inlet']['mole_frac_comp']['O2'])

    # Solvent liquid
    m.fs.unit.liquid_inlet.flow_mol.fix(input_dic[case]['liquid_inlet']['flow_mol'])
    m.fs.unit.liquid_inlet.temperature.fix(input_dic[case]['liquid_inlet']['temperature'])
    m.fs.unit.liquid_inlet.pressure.fix(input_dic[case]['liquid_inlet']['pressure'])
    m.fs.unit.liquid_inlet.mole_frac_comp[0, "CO2"].fix(input_dic[case]['liquid_inlet']['mole_frac_comp']['CO2'])
    m.fs.unit.liquid_inlet.mole_frac_comp[0, "H2O"].fix(input_dic[case]['liquid_inlet']['mole_frac_comp']['H2O'])
    m.fs.unit.liquid_inlet.mole_frac_comp[0, "MEA"].fix(input_dic[case]['liquid_inlet']['mole_frac_comp']['MEA'])


def strip_statevar_bounds_for_initialization(m):
    # Strip variable bounds
    for t in m.fs.time:
        for x in m.fs.unit.liquid_phase.length_domain:
            for j in m.fs.unit.config.liquid_phase.property_package.true_phase_component_set:
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setub(None)
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].domain = Reals

                m.fs.unit.liquid_phase.properties[t, x].mole_frac_phase_comp_true[j].setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].mole_frac_phase_comp_true[j].setub(None)

            for j in m.fs.unit.config.liquid_phase.property_package.apparent_species_set:
                m.fs.unit.liquid_phase.properties[t, x].mole_frac_comp[j].setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].mole_frac_comp[j].setub(None)
                m.fs.unit.liquid_phase.properties[t, x].phase_frac['Liq'].setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].phase_frac['Liq'].setub(None)

            for j in m.fs.unit.config.liquid_phase.property_package.apparent_phase_component_set:
                m.fs.unit.liquid_phase.properties[t, x].mole_frac_phase_comp.setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].mole_frac_phase_comp.setub(None)

            m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase['Liq'].setlb(None)
            m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase['Liq'].domain = Reals

    for t in m.fs.time:
        for x in m.fs.unit.vapor_phase.length_domain:
            for j in m.fs.unit.config.vapor_phase.property_package.component_list:
                m.fs.unit.vapor_phase.properties[t, x].mole_frac_comp[j].setlb(None)
                m.fs.unit.vapor_phase.properties[t, x].mole_frac_comp[j].setub(None)

    for t in m.fs.time:
        for x in m.fs.unit.vapor_phase.length_domain:
            for j in m.fs.unit.config.vapor_phase.property_package._phase_component_set:
                m.fs.unit.vapor_phase.properties[t, x].mole_frac_phase_comp[j].setlb(None)
                m.fs.unit.vapor_phase.properties[t, x].mole_frac_phase_comp[j].setub(None)

    # for t in m.fs.time:    
    #     for x in m.fs.unit.vapor_phase.length_domain:

    #         # Pressure
    #         m.fs.unit.liquid_phase.properties[t, x].temperature.setlb(None)
    #         m.fs.unit.liquid_phase.properties[t, x].temperature.setub(None)
    #         m.fs.unit.liquid_phase.properties[t, x].temperature.domain = Reals
    #         m.fs.unit.vapor_phase.properties[t, x].temperature.setlb(None)
    #         m.fs.unit.vapor_phase.properties[t, x].temperature.setub(None)
    #         m.fs.unit.vapor_phase.properties[t, x].temperature.domain = Reals

    return m


def strip_some_statevar_bounds(m):
    for t in m.fs.time:
        for x in m.fs.unit.vapor_phase.length_domain:

            # Pressure
            m.fs.unit.liquid_phase.properties[t, x].pressure.setlb(None)
            m.fs.unit.liquid_phase.properties[t, x].pressure.setub(None)
            m.fs.unit.liquid_phase.properties[t, x].pressure.domain = Reals
            m.fs.unit.vapor_phase.properties[t, x].pressure.setlb(None)
            m.fs.unit.vapor_phase.properties[t, x].pressure.setub(None)
            m.fs.unit.vapor_phase.properties[t, x].pressure.domain = Reals

            # Molar flowrate
            m.fs.unit.liquid_phase.properties[t, x].flow_mol.setlb(None)
            m.fs.unit.liquid_phase.properties[t, x].flow_mol.setub(None)
            m.fs.unit.liquid_phase.properties[t, x].flow_mol.domain = Reals
            m.fs.unit.vapor_phase.properties[t, x].flow_mol.setlb(None)
            m.fs.unit.vapor_phase.properties[t, x].flow_mol.setub(None)
            m.fs.unit.vapor_phase.properties[t, x].flow_mol.domain = Reals

            # flow_mol_phase
            m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase['Liq'].setlb(None)
            m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase['Liq'].setub(None)
            m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase['Liq'].domain = Reals
            m.fs.unit.vapor_phase.properties[t, x].flow_mol_phase['Vap'].setlb(None)
            m.fs.unit.vapor_phase.properties[t, x].flow_mol_phase['Vap'].setub(None)
            m.fs.unit.vapor_phase.properties[t, x].flow_mol_phase['Vap'].domain = Reals

            # velocity_liq
            m.fs.unit.velocity_liq.setlb(None)
            m.fs.unit.velocity_liq.domain = Reals

            # sqrt_conc_interface_MEA
            # m.fs.unit.sqrt_conc_interface_MEA.setlb(None)
            # m.fs.unit.sqrt_conc_interface_MEA.setub(None)

            # conc_interface_MEA
            # m.fs.unit.conc_interface_MEA.setlb(None)
            # m.fs.unit.conc_interface_MEA.setub(None)

            # Enhancement factor
            # m.fs.unit.enhancement_factor.setlb(None)
            # m.fs.unit.enhancement_factor.setub(None)

            for j in m.fs.unit.config.liquid_phase.property_package.true_phase_component_set:
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setlb(None)
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setub(None)
                m.fs.unit.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].domain = Reals

                m.fs.unit.liquid_phase.properties[t, x].log_conc_mol_phase_comp_true[j].setlb(None)

    return m


def scale_absorber(m, fs):
    xfrm = TransformationFactory("contrib.strip_var_bounds")
    gsf = iscale.get_scaling_factor
    ssf = iscale.set_scaling_factor

    def cst(con, s):
        iscale.constraint_scaling_transform(con, s, overwrite=False)

    mole_frac_vap_scaling_factors = {
        "N2": 1,
        "H2O": 10,
        "CO2": 10,
        "O2": 10,
    }

    mole_frac_liq_scaling_factors = {
        "H2O": 1,
        "MEA": 10,
        "CO2": 20,
    }
    mole_frac_liq_true_scaling_factors = {
        "CO2": 1e4,  # Could go to 1e4 or 3e4
        "H2O": 1,
        "HCO3_-": 1000,
        "MEA": 30,
        "MEACOO_-": 30,
        "MEA_+": 30,
    }

    sf_flow_mol = 1e-2

    for pp in [fs.liquid_properties, fs.vapor_properties]:
        pp.set_default_scaling("enth_mol_phase", 3e-4)
        pp.set_default_scaling("pressure", 1e-5)
        pp.set_default_scaling("temperature", 1)
        pp.set_default_scaling("flow_mol", sf_flow_mol)
        pp.set_default_scaling("flow_mol_phase", sf_flow_mol)

    fs.vapor_properties.set_default_scaling("flow_mol_phase_comp", 2 * sf_flow_mol)
    fs.vapor_properties.set_default_scaling("flow_mass_phase", 1 / 350)
    fs.vapor_properties.set_default_scaling("visc_d_phase", 1 / 1.8186152032921942e-05)
    for comp, sfy in mole_frac_vap_scaling_factors.items():
        fs.vapor_properties.set_default_scaling("mole_frac_comp", sfy, index=comp)
        fs.vapor_properties.set_default_scaling("mole_frac_phase_comp", sfy, index=("Vap", comp))
        fs.vapor_properties.set_default_scaling("flow_mol_phase_comp", sfy * sf_flow_mol, index=("Vap", comp))

    fs.liquid_properties.set_default_scaling("flow_mass_phase", 1 / 600)
    fs.liquid_properties.set_default_scaling("visc_d_phase", 500)
    fs.liquid_properties.set_default_scaling("log_k_eq", 1)

    for pp in [fs.liquid_properties]:
        pp.set_default_scaling("dens_mol_phase", 1 / 43000, index="Liq")
        for comp, sf_x in mole_frac_liq_scaling_factors.items():
            pp.set_default_scaling("mole_frac_comp", sf_x, index=comp)
            pp.set_default_scaling("mole_frac_phase_comp", sf_x, index=("Liq", comp))
            pp.set_default_scaling("flow_mol_phase_comp", sf_x * sf_flow_mol, index=("Liq", comp))

    for comp, sf_x in mole_frac_liq_true_scaling_factors.items():
        fs.liquid_properties.set_default_scaling("mole_frac_phase_comp_true", sf_x, index=("Liq", comp))
        fs.liquid_properties.set_default_scaling("flow_mol_phase_comp_true", sf_x * sf_flow_mol, index=("Liq", comp))

    fs.liquid_properties.set_default_scaling("apparent_inherent_reaction_extent", 1 / 33, index="bicarbonate")
    fs.liquid_properties.set_default_scaling("apparent_inherent_reaction_extent", 1 / 750, index="carbamate")

    # Absorber column
    column = fs.unit
    for t in fs.time:
        for x in column.liquid_phase.length_domain:
            ssf(column.velocity_liq[t, x], 200)
            ssf(column.interphase_mass_transfer[t, x, "CO2"], 1 / 20)
            ssf(column.interphase_mass_transfer[t, x, "H2O"], 1 / 100)
            # ssf(column.pressure_equil[t, x, "CO2"], 1 / 200 * 20)
            # ssf(column.pressure_equil[t, x, "H2O"], 1e-4 * 20)
            # ssf(column.conc_CO2_bulk[t, x], 3)

        for x in column.vapor_phase.length_domain:
            ssf(column.heat_transfer_coeff[t, x], 1 / 3e6)

    # for (t, x), con in column.Hatta_number_eqn.items():
    #     cst(column.Hatta_number_eqn[t, x], 1e6)
    # for (t, x), con in column.conc_CO2_equil_bulk_eqn.items():
    #     cst(column.conc_CO2_equil_bulk_eqn[t, x], 10)
    # for (t, x), con in column.sqrt_conc_interface_MEA_eqn.items():
    #     cst(column.sqrt_conc_interface_MEA_eqn[t, x], 10)
    # for (t, x), con in column.enhancement_factor_eqn1.items():
    #     cst(column.enhancement_factor_eqn1[t, x], 1e-1)
    # for (t, x), con in column.enhancement_factor_eqn2.items():
    #     cst(column.enhancement_factor_eqn2[t, x], 1e-1)

    iscale.calculate_scaling_factors(m)


def define_column_design_parameters(m):
    m.fs.unit.co2_capture = Var(m.fs.time,
                                initialize=98,
                                doc='''CO2 Capture Rate [%]''')

    @m.fs.unit.Constraint(m.fs.time,
                          doc='''Correlation for CO2 Capture Rate''')
    def co2_capture_eqn(b, t):
        return b.co2_capture[t] * (b.vapor_inlet.mole_frac_comp[0, "CO2"]
                                   * b.vapor_inlet.flow_mol[0]) == (
                (b.vapor_inlet.mole_frac_comp[0, "CO2"] * b.vapor_inlet.flow_mol[0] -
                 b.vapor_outlet.mole_frac_comp[0, "CO2"] * b.vapor_outlet.flow_mol[0]) * 100)

    m.fs.HDratio = Expression(
        expr=m.fs.unit.length_column / m.fs.unit.diameter_column,
        doc="Column height to column diameter ratio (-)"
    )
    m.fs.LGratio = Expression(
        expr=m.fs.unit.liquid_inlet.flow_mol[0] / m.fs.unit.vapor_inlet.flow_mol[0],
        doc="Inlet liquid (solvent) flowrate to inlet gas flowrate ratio (-)"
    )
    m.fs.volume_column = Expression(
        expr=math.pi * m.fs.unit.length_column * (m.fs.unit.diameter_column / 2) ** 2,
        doc="Volume of column (cubic m)"
    )
    m.fs.volume_column_withheads = Expression(
        expr=(
                math.pi * (
                m.fs.unit.diameter_column ** 2 * m.fs.unit.length_column
        ) / 4
                + math.pi / 3 * m.fs.unit.diameter_column ** 3
        ),
        doc="Volume of column with heads (cubic m)",
    )

    return m


def check_conservation(m):
    vap_in = m.fs.unit.vapor_phase.properties[0, 0]
    vap_out = m.fs.unit.vapor_phase.properties[0, 1]
    liq_in = m.fs.unit.liquid_phase.properties[0, 1]
    liq_out = m.fs.unit.liquid_phase.properties[0, 0]

    # Material conservation
    for j in ["CO2", "H2O"]:
        mass = abs(value(
            vap_in.get_material_flow_terms("Vap", j) +
            liq_in.get_material_flow_terms("Liq", j) -
            vap_out.get_material_flow_terms("Vap", j) -
            liq_out.get_material_flow_terms("Liq", j)))
        assert 1e-6 >= abs(mass)

    for j in ["N2", "O2"]:
        assert 1e-6 >= abs(value(
            vap_in.get_material_flow_terms("Vap", j) -
            vap_out.get_material_flow_terms("Vap", j)))

    for j in ["MEA"]:
        assert 1e-6 >= abs(value(
            liq_in.get_material_flow_terms("Liq", j) -
            liq_out.get_material_flow_terms("Liq", j)))

    # Energy conservation
    assert 1e-6 >= abs(value(
        vap_in.get_enthalpy_flow_terms("Vap") +
        liq_in.get_enthalpy_flow_terms("Liq") -
        vap_out.get_enthalpy_flow_terms("Vap") -
        liq_out.get_enthalpy_flow_terms("Liq"))
                       / 3000  # Scale the residual to the magnitude of the enthalpy flow terms
                       )


def model_results():
    # Variables to be tested
    print('Liquid phase temperature')
    for x in m.fs.unit.vapor_phase.length_domain:
        print(value(m.fs.unit.liquid_phase.properties[0, x].temperature))

    print('Vapor phase temperature')
    for x in m.fs.unit.vapor_phase.length_domain:
        print(value(m.fs.unit.vapor_phase.properties[0, x].temperature))

    print('Vapor phase partial pressure of CO2')
    for x in m.fs.unit.vapor_phase.length_domain:
        print(value(
            m.fs.unit.vapor_phase.properties[0, x].pressure * m.fs.unit.vapor_phase.properties[0, x].mole_frac_comp[
                'CO2']))

    print('Henrys constant of CO2 in liquid phase')
    for x in m.fs.unit.vapor_phase.length_domain:
        print(value(m.fs.unit.liquid_phase.properties[0, x].henry['Liq', 'CO2']))

    print('Percent CO2 capture')
    print(value(m.fs.unit.co2_capture[0]))

    print('Flood fraction')
    for x in m.fs.unit.vapor_phase.length_domain:
        print(value(m.fs.unit.flood_fraction[0, x]))


def model_robustness_stats():
    for v, sv in iscale.badly_scaled_var_generator(m):
        print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")

    jac, nlp = iscale.get_jacobian(m, scaled=True)

    print("Extreme Jacobian entries:")
    for i in iscale.extreme_jacobian_entries(jac=jac, nlp=nlp, large=1e5):
        print(f"    {i[0]:.2e}, [{i[1]}, {i[2]}]")

    condition_number = iscale.jacobian_cond(m, jac=jac)
    print("Condition Number:")
    print("{:e}".format(condition_number))


def check_scaling(m):
    import idaes.core.util.scaling as iscale
    jac, nlp = iscale.get_jacobian(m, scaled=True)
    # print("Extreme Jacobian entries:")
    sourceFile = open('extreme_jacobian.txt', 'w')
    for i in iscale.extreme_jacobian_entries(
            jac=jac, nlp=nlp, small=1e-6, large=1e3):
        print(f"    {i[0]:.2e}, [{i[1]}, {i[2]}]", file=sourceFile)
    sourceFile.close()
    # print("Unscaled constraints:")
    sourceFile2 = open('unscaled_constraints.txt', 'w')
    for c in iscale.unscaled_constraints_generator(m):
        print(f"    {c}", file=sourceFile2)
    sourceFile2.close()
    sourceFile3 = open('constraints_with_scale_factor.txt', 'w')
    # print("Scaled constraints by factor:")
    for c, s in iscale.constraints_with_scale_factor_generator(m):
        print(f"    {c}, {s}", file=sourceFile3)
    sourceFile3.close()
    # print("Badly scaled variables:")
    sourceFile4 = open('badly_scaled_var.txt', 'w')
    for v, sv in iscale.badly_scaled_var_generator(
            m, large=1e3, small=1e-4, zero=1e-12):
        print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}",
              file=sourceFile4)
    sourceFile.close()
    print(f"Jacobian Condition Number: {iscale.jacobian_cond(jac=jac):.2e}")

    return m


def validation_plot(m, case_K='K13'):
    import matplotlib.pyplot as plt

    validation_data = pd.read_csv(r'data\NCCC_Temperature_Profiles_forValidation.csv')

    # Computed temperature profile(s)
    Tliq = []
    Tvap = []
    for x in m.fs.unit.vapor_phase.length_domain:
        Tliq.append(value(m.fs.unit.liquid_phase.properties[0, x].temperature - 273.15))
        Tvap.append(value(m.fs.unit.vapor_phase.properties[0, x].temperature - 273.15))

    mass_transfer_vap_CO2 = []
    mass_transfer_vap_H2O = []
    for x in m.fs.unit.vapor_phase.length_domain:
        mass_transfer_vap_CO2.append(value(m.fs.unit.vapor_phase.mass_transfer_term[0, x, "Vap", "CO2"]))
        mass_transfer_vap_H2O.append(value(m.fs.unit.vapor_phase.mass_transfer_term[0, x, "Vap", "H2O"]))

    mass_transfer_liq_CO2 = []
    mass_transfer_liq_H2O = []
    for x in m.fs.unit.liquid_phase.length_domain:
        mass_transfer_liq_CO2.append(value(m.fs.unit.liquid_phase.mass_transfer_term[0, x, "Liq", "CO2"]))
        mass_transfer_liq_H2O.append(value(m.fs.unit.liquid_phase.mass_transfer_term[0, x, "Liq", "H2O"]))

    # Experimental temperature profile(s)
    if case_K == "K13":
        x = validation_data['z_3beds']
        y = validation_data[case_K]
    elif case_K == "K17":
        validation_data_no_nan = validation_data.dropna(subset=['z_2beds'])
        x = validation_data_no_nan['z_2beds']
        y = validation_data_no_nan[case_K]
    elif case_K == "K18":
        validation_data_no_nan = validation_data.dropna(subset=['z_1bed'])
        x = validation_data_no_nan['z_1bed']
        y = validation_data_no_nan[case_K]
    elif case_K == "K19":
        validation_data_no_nan = validation_data.dropna(subset=['z_1bed'])
        x = validation_data_no_nan['z_1bed']
        y = validation_data_no_nan[case_K]
    elif case_K == "K20":
        validation_data_no_nan = validation_data.dropna(subset=['z_1bed'])
        x = validation_data_no_nan['z_1bed']
        y = validation_data_no_nan[case_K]
    elif case_K == "K21":
        validation_data_no_nan = validation_data.dropna(subset=['z_2beds'])
        x = validation_data_no_nan['z_2beds']
        y = validation_data_no_nan[case_K]
    else:
        message_to_print_part1 = "The specified case is not among the cases without "
        message_to_print_part2 = "intercooling, please chose between cases "
        message_to_print_part3 = "'K13', 'K17', 'K18', 'K19', 'K20', and 'K21'"
        message_to_print = ' '.join(
            [message_to_print_part1, message_to_print_part2, message_to_print_part3])
        print(message_to_print)

    plt.figure()
    plt.plot(x, y,
             label='measured, $T_{liq}$',
             marker='o',
             markeredgecolor='black',
             markerfacecolor='None',
             markersize=8,
             linestyle='None')
    plt.plot(m.fs.unit.vapor_phase.length_domain, Tliq,
             label='computed, $T_{liq}$',
             linestyle='-',
             color='tab:blue')
    plt.plot(m.fs.unit.vapor_phase.length_domain, Tvap,
             label='computed, $T_{vap}$',
             linestyle='--',
             color='tab:orange')
    plt.legend(loc='best', ncol=1)
    plt.grid()
    plt.ylim([40, 80])
    # plt.xlim([0, 1])
    title = "Case " + case_K
    plt.title(title)
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("Temperature [°C]")

    # Pp_CO2 = []
    # Pp_CO2_equil = []
    # Pp_CO2_equil_He = []
    # for x in m.fs.unit.vapor_phase.length_domain:
    #     Pp_CO2.append(value(m.fs.unit.vapor_phase.properties[0, x].pressure * 
    #                          m.fs.unit.vapor_phase.properties[0, x].mole_frac_comp['CO2'] * 
    #                          1e-3))
    #     Pp_CO2_equil.append(value(m.fs.unit.pressure_equil[0, x, 'CO2'] * 1e-3))
    #     Pp_CO2_equil_He.append(value(m.fs.unit.PpCO2_equil_He[0, x] * 1e-3))

    # plt.figure()
    # plt.plot(m.fs.unit.vapor_phase.length_domain, Pp_CO2,
    #           label='$Pp_{CO2}$', 
    #           linestyle='-',
    #           color='tab:blue')
    # plt.plot(m.fs.unit.vapor_phase.length_domain, Pp_CO2_equil, 
    #           label='$Pp_{CO2}^{*}$', 
    #           linestyle='--',
    #           color='tab:orange')
    # plt.plot(m.fs.unit.vapor_phase.length_domain, Pp_CO2_equil_He, 
    #           label="$Pp_{CO2}^{*} (Henry's Law)'$", 
    #           linestyle=':',
    #           color='tab:red')
    # plt.legend(loc='best',ncol=1)
    # plt.grid()
    # plt.xlabel("Normalized Bed Height [-]")
    # plt.ylabel("CO2 Partial Pressure (kPa)") 

    plt.show()


def print_column_design_parameters(m):
    print("\n ******* Printing some results *******")
    print("\nColumn diameter: ", value(m.fs.unit.diameter_column), "m")
    print("Column height: ", value(m.fs.unit.length_column), "m")
    print("Column volume: ", value(m.fs.volume_column), "m3")
    print("Column volume with heads: ", value(m.fs.volume_column_withheads), "m3")
    print("Column height to diameter ratio: ", value(m.fs.HDratio))
    print("\nSolvent inlet molar flowrate: ", value(m.fs.unit.liquid_inlet.flow_mol[0]), "mol/s")
    print("L/G ratio: ", value(m.fs.LGratio))
    print("\nCO2 capture: ", value(m.fs.unit.co2_capture[0]), "%")


def get_mole_fraction(CO2_loading, amine_concentration):
    MW_MEA = 61.084
    MW_H2O = 18.02

    x_MEA_unloaded = amine_concentration / (MW_MEA / MW_H2O + amine_concentration * (1 - MW_MEA / MW_H2O))
    x_H2O_unloaded = 1 - x_MEA_unloaded

    n_MEA = 100 * x_MEA_unloaded
    n_H2O = 100 * x_H2O_unloaded

    n_CO2 = n_MEA * CO2_loading
    n_tot = n_MEA + n_H2O + n_CO2
    x_CO2, x_MEA, x_H2O = n_CO2 / n_tot, n_MEA / n_tot, n_H2O / n_tot

    return x_CO2, x_MEA, x_H2O


def get_nested_dic(df):
    n_runs = len(df)
    w_MEA = .3
    Tv = 313.15
    # y_CO2 = 0.04226
    # y_H2O = 0.05480
    # y_N2 = 0.76942 + 0.00920
    # y_O2 = 0.12430
    y_CO2 = 0.101947863634366
    y_H2O = 0.0912918913073196
    y_N2 = 0.734006946649283
    y_O2 = 0.072753298409031
    # H = 16
    # D = 13.41
    # Fv_T = 37116.4/4
    H = 6
    D = .64
    Fv_T = 10

    nested_dic = {}
    run_num = 1
    for i, row in df.iterrows():
        Tl, loading, L_G = row['Tl'], row['loading'], row['L/G']
        Fl_T = Fv_T*L_G
        x_CO2, x_MEA, x_H2O = get_mole_fraction(loading, w_MEA)

        dic_2 = {'diameter': D, 'length': H,
                 'vapor_inlet': {'flow_mol': Fv_T,
                                 'temperature': Tl,
                                 'pressure': 109180,
                                 'mole_frac_comp': {'CO2': y_CO2,
                                                    'H2O': y_H2O,
                                                    'N2': y_N2,
                                                    'O2': y_O2}},
                 'liquid_inlet': {'flow_mol': Fl_T,
                                  'temperature': Tv,
                                  'pressure': 109180,
                                  'mole_frac_comp': {'CO2': x_CO2,
                                                     'H2O': x_H2O,
                                                     'MEA': x_MEA}}}
        nested_dic[str(run_num)] = dic_2
        run_num += 1

    return nested_dic


if __name__ == "__main__":

    ts = time.time()

    # Create dataframes to save results
    df_out_myresults = pd.DataFrame()
    df_out_liqtemp = pd.DataFrame()
    df_out_vaptemp = pd.DataFrame()
    df_out_PCO2 = pd.DataFrame()
    df_out_PCO2_equil = pd.DataFrame()
    df_out_vapvel = pd.DataFrame()
    df_out_loading = pd.DataFrame()

    case_number_list = ['K13', 'K17', 'K18', 'K19', 'K20']  #, 'K21']
    case_number_list = ["K18"]


    df = pd.read_csv('enhancement_factor_runs.csv')

    E_cases = get_nested_dic(df)

    nfe = 40
    x_nfe_list = get_uniform_grid(nfe)
    # x_nfe_list = get_custom_grid_dx([0.3, 0.8], [0.005, 0.1, 0.005])
    # x_nfe_list = get_custom_grid_nfe([0.3, 0.8], [50, 10, 40])
    # x_nfe_list = get_custom_grid_nfe([0.01, 0.1], [40, 40, 20])

    m = build_column_model(x_nfe_list)
    set_inputs(m, case='1', input_dic=E_cases)
    strip_statevar_bounds_for_initialization(m)
    switch_liquid_to_parmest_params(m.fs.liquid_properties, ions=True)
    scale_absorber(m, m.fs)
    initialize_inherent_reactions(m.fs.unit.liquid_phase)
    define_column_design_parameters(m)

    # # # Strip all bounds - works only for simulations (square problems)
    # xfrm = TransformationFactory('contrib.strip_var_bounds')
    # xfrm.apply_to(m, reversible=True)
    # # strip_some_statevar_bounds(m)

    bounded_vars = 0
    for v in m.component_data_objects(Var, descend_into=True):
        if v.lb != None or v.ub != None:
            # print(v.name, v.lb, v.ub)
            bounded_vars = bounded_vars + 1
    print('Number of bounded variables: ', value(bounded_vars))

    df_runs = pd.read_csv('enhancement_factor_runs.csv')
    E_cases = get_nested_dic(df_runs)

    dfs = []
    sheetnames = []

    for k in range(len(df_runs)):
        ts_loopstart = time.time()

        # case_number = case_number_list[k]

        df_new = df_runs.iloc[k]
        Tl, loading, L_G = df_new['Tl'], df_new['loading'], df_new['L/G']
        print(Tl, loading, L_G)

        print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))
        set_inputs(m, case=str(k+1), input_dic=E_cases)
        print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))

        try:
            m.fs.unit.initialize(
                outlvl=idaeslog.INFO_HIGH,
                optarg={
                    'nlp_scaling_method': 'user-scaling',
                    'linear_solver': 'ma57',
                    'OF_ma57_automatic_scaling': 'yes',
                    'max_iter': 300,
                    'tol': 1e-8,
                }
            )

            print("\nSolve model, some variable bounds stripped ...")
            print("\n")

            # Solve model
            optarg = {
                'nlp_scaling_method': 'user-scaling',
                'linear_solver': 'ma57',
                'OF_ma57_automatic_scaling': 'yes',
                'max_iter': 300,
                'tol': 1e-8,
            }
            solver.options = optarg

            res = solver.solve(m, tee=False)
            # Check whether mass & energy balances close
            # check_conservation(m)
            # check_scaling(m)

            # constrviol = large_residuals_set(m.fs.unit)
            # print(constrviol)

            assert check_optimal_termination(res)

            # print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))

            # model_results()
            # validation_plot(m, case_number)
            # model_robustness_stats()

            print("\n-------- Simulation Results --------")
            print_column_design_parameters(m)

            import importlib
            import save_run_profiles

            importlib.reload(save_run_profiles)

            df = save_run_profiles.save_run_profiles(m)

            dfs.append(df)
            sheetnames.append(f'Tl={Tl},alpha={loading},L_G ={L_G}')

        except:
            continue

    import xlwings as xw
    filename = 'Simulation_Results/Profiles_Reduced_Enhancement_Factor.xlsx'
    filename = 'Simulation_Results/Profiles_Implicit_Enhancement_Factor.xlsx'
    wb = xw.Book(filename, read_only=False)

    for sheetname, df in zip(sheetnames, dfs):
        try:
            wb.sheets[sheetname].clear()
        except:
            wb.sheets.add(sheetname)
        wb.sheets[sheetname].range("A1").value = df

    for sheet in wb.sheets:
        if sheet.name not in sheetnames:
            sheet.delete()
    wb.save(path=filename)
        #%%
    #     print("\n-------- Simulation Results --------")
    #     print_column_design_parameters(m)
    #
    #     # Save results
    #     my_results = {'CO2_Capture': value(m.fs.unit.co2_capture[0]),
    #                   'IPOPT_Terminal_Condition': res.solver.termination_condition,
    #                   'Simulation_time': value(time.time() - ts_loopstart)}
    #
    #     df_out_myresults_row = pd.DataFrame(my_results, index=[k + 1])
    #     df_out_myresults = pd.concat([df_out_myresults, df_out_myresults_row], axis=0)
    #
    #     # Create data frame to store liquid phase temperature, CO2 partial pressure
    #     # gas phase velocity, and flooding fraction
    #     column_idx_T = "T_" + case_number_list[k]
    #     column_idx_T_vap = "T_vap_" + case_number_list[k]
    #     column_idx_PCO2 = "P_CO2_" + case_number_list[k]
    #     column_idx_mass_xfer_dP = "mass_xfer_dP_" + case_number_list[k]
    #     column_idx_vapvel = "Vap_vel_" + case_number_list[k]
    #     column_idx_loading = "Solvent_loading_" + case_number_list[k]
    #
    #     df_out_liqtemp_column = pd.DataFrame()
    #     df_out_vaptemp_column = pd.DataFrame()
    #     df_out_PCO2_column = pd.DataFrame()
    #     df_out_PCO2_column_equil = pd.DataFrame()
    #     df_out_vapvel_column = pd.DataFrame()
    #     df_out_loading_column = pd.DataFrame()
    #
    #     for z in m.fs.unit.vapor_phase.length_domain:
    #         # Liquid phase temperature
    #         output_liqtemp = {column_idx_T: value(m.fs.unit.liquid_phase.properties[0, z].temperature)}
    #         df_out_liqtemp_row = pd.DataFrame(output_liqtemp, index=[z])
    #         df_out_liqtemp_column = pd.concat([df_out_liqtemp_column, df_out_liqtemp_row], axis=0)
    #         # Vapor phase temperature
    #         output_vaptemp = {column_idx_T: value(m.fs.unit.vapor_phase.properties[0, z].temperature)}
    #         df_out_vaptemp_row = pd.DataFrame(output_vaptemp, index=[z])
    #         df_out_vaptemp_column = pd.concat([df_out_vaptemp_column, df_out_vaptemp_row], axis=0)
    #         # CO2 partial pressure
    #         output_PCO2 = {column_idx_PCO2: value(m.fs.unit.vapor_phase.properties[0, z].pressure *
    #                                               m.fs.unit.vapor_phase.properties[0, z].mole_frac_comp['CO2'])}
    #         df_out_PCO2_row = pd.DataFrame(output_PCO2, index=[z])
    #         df_out_PCO2_column = pd.concat([df_out_PCO2_column, df_out_PCO2_row], axis=0)
    #         # CO2 mass transfer driving force
    #         output_mass_xfer_dP = {column_idx_mass_xfer_dP: value(m.fs.unit.mass_transfer_driving_force[0, z, 'CO2'])}
    #         df_out_mass_xfer_dP_row = pd.DataFrame(output_mass_xfer_dP, index=[z])
    #         df_out_mass_xfer_dP_column = pd.concat([df_out_PCO2_column_equil, df_out_mass_xfer_dP_row], axis=0)
    #         # Vapor phase velocity
    #         output_vapvel = {column_idx_vapvel: value(m.fs.unit.velocity_vap[0, z])}
    #         df_out_vapvel_row = pd.DataFrame(output_vapvel, index=[z])
    #         df_out_vapvel_column = pd.concat([df_out_vapvel_column, df_out_vapvel_row], axis=0)
    #         # Flooding fraction
    #         output_loading = {column_idx_loading: value(
    #             m.fs.unit.liquid_phase.properties[0, z].mole_frac_phase_comp['Liq', 'CO2'] /
    #             m.fs.unit.liquid_phase.properties[0, z].mole_frac_phase_comp['Liq', 'MEA'])}
    #         df_out_loading_row = pd.DataFrame(output_loading, index=[z])
    #         df_out_loading_column = pd.concat([df_out_loading_column, df_out_loading_row], axis=0)
    #         print(value(m.fs.unit.velocity_vap[0, z]))
    #
    #     df_out_liqtemp = pd.concat([df_out_liqtemp, df_out_liqtemp_column], axis=1)
    #     df_out_vaptemp = pd.concat([df_out_vaptemp, df_out_vaptemp_column], axis=1)
    #     df_out_PCO2 = pd.concat([df_out_PCO2, df_out_PCO2_column], axis=1)
    #     df_out_PCO2_equil = pd.concat([df_out_PCO2_equil, df_out_PCO2_column_equil], axis=1)
    #     df_out_vapvel = pd.concat([df_out_vapvel, df_out_vapvel_column], axis=1)
    #     df_out_loading = pd.concat([df_out_loading, df_out_loading_column], axis=1)
    #
    # # # Check if results folder exists (create if it does not exist), save results as .csv
    # # # Directory name
    # import os
    #
    # directory = 'Results_each_case_Putta_parmestparams'
    # CO2_capture = os.path.join(directory, 'results_CO2_Capture.csv')
    # liquid_temperature = os.path.join(directory, 'results_liquid_temperature.csv')
    # vapor_temperature = os.path.join(directory, 'results_vapor_temperature.csv')
    # CO2_partialpressure = os.path.join(directory, 'results_CO2_partialpressure.csv')
    # CO2_partialpressure_equil = os.path.join(directory, 'results_CO2_partialpressure_equil.csv')
    # vapor_phase_velocity = os.path.join(directory, 'results_vapor_phase_velocity.csv')
    # solvent_loading = os.path.join(directory, 'results_solvent_loading.csv')
    #
    # try:
    #     os.mkdir(directory)
    #     df_out_myresults.to_csv(CO2_capture)
    #     df_out_liqtemp.to_csv(liquid_temperature)
    #     df_out_vaptemp.to_csv(vapor_temperature)
    #     df_out_PCO2.to_csv(CO2_partialpressure)
    #     df_out_PCO2_equil.to_csv(CO2_partialpressure_equil)
    #     df_out_vapvel.to_csv(vapor_phase_velocity)
    #     df_out_loading.to_csv(solvent_loading)
    # except:
    #     df_out_myresults.to_csv(CO2_capture)
    #     df_out_liqtemp.to_csv(liquid_temperature)
    #     df_out_vaptemp.to_csv(vapor_temperature)
    #     df_out_PCO2.to_csv(CO2_partialpressure)
    #     df_out_PCO2_equil.to_csv(CO2_partialpressure_equil)
    #     df_out_vapvel.to_csv(vapor_phase_velocity)
    #     df_out_loading.to_csv(solvent_loading)
    #
    # print("\n")
    # print("----------------------------------------------------------")
    # print('Total simulation time: ', value(time.time() - ts), " s")
    # print("----------------------------------------------------------")

    #%%
