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

# Import Python libraries
import pandas as pd
import numpy as np
import copy
import math
import time
import logging
import sys

# Import Pyomo libraries
import pyomo.opt
from pyomo.environ import ConcreteModel, value, Var, Reals, Param, TransformationFactory,\
    Constraint, Expression, Objective, check_optimal_termination, exp, units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES Libraries
from idaes.core import FlowsheetBlock
from idaes.models_extra.column_models.MEAsolvent_column import MEAColumn

from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    unused_variables_set,
    large_residuals_set
)
from pyomo.util.infeasible import log_infeasible_constraints
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core import FlowDirection

sys.path.insert(0, 'flowsheets')
from mea_properties import (
    MEALiquidParameterBlock,
    FlueGasParameterBlock,
    scale_mea_liquid_params,
    scale_mea_vapor_params,
    switch_liquid_to_parmest_params,
    initialize_inherent_reactions
)
from mea_absorber_reformulated import MEAAbsorberFlowsheet
from idaes.core.surrogate.alamopy import AlamoSurrogate

logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
solver = get_solver()
# solver.options["bound_push"] = 1e-22

# 650MW NGCC plant-scale (this absorber represents 1 of the 3 used in parallel)


def build_column_model(x_nfe_list, use_surrogate):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Set up molar flow bounds
    state_bounds = {
        "flow_mol": (0, 1, 1e6, pyunits.mol / pyunits.s)
    }
    
    # Set up property packages
    m.fs.vapor_properties = FlueGasParameterBlock(state_bounds=state_bounds)
    m.fs.liquid_properties = MEALiquidParameterBlock(ions=True, state_bounds=state_bounds)

    if use_surrogate:
        surrogate = AlamoSurrogate.load_from_file("alamo_surrogate_absorber.json")
    else:
        surrogate = None
    # Create an instance of the column in the flowsheet
    m.fs.absorber = MEAColumn(
        finite_elements=len(x_nfe_list)-1, 
        length_domain_set=x_nfe_list,
        vapor_phase={
            "property_package": m.fs.vapor_properties},
        liquid_phase={
            "property_package": m.fs.liquid_properties},
        surrogate_enhancement_factor_model=surrogate,
            
    )

    return m


def get_uniform_grid(nfe):
    # Finite element list in the spatial domain
    x_nfe_list = [i / nfe for i in range(nfe + 1)]
    
    return x_nfe_list

    
def get_custom_grid_dx(x_locations=[0.3], dx = [0.01, 0.1]):
    # This function assumes that the spatial domain is between 0 and 1. 
    # x_locations is a list of the points in the spatial domain between 
    # which the custom mesh should be created (anchors). The end points 0 and 1
    # should not be included in x_locations.
    # dx is a list of the desired step sizes between the anchors.
    
    x_nfe_list = []
    
    for k in range(len(x_locations)):
        if k == 0:
            x_nfe_list_temp = np.linspace(0, x_locations[k], 
                                round(x_locations[k]/dx[k])+1).tolist()
        else:
            x_nfe_list_temp = np.linspace(x_locations[k-1], x_locations[k],
                                round((x_locations[k] - 
                                       x_locations[k-1])/dx[k])+1).tolist()

        x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))
        
    x_nfe_list_temp = np.linspace(x_locations[k], 1,
                            round((1-x_locations[k])/dx[k+1])+1).tolist()
    x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))
        
    # nfe = len(x_nfe_list) -1
        
    return x_nfe_list.tolist()

 
def get_custom_grid_nfe(x_locations=[0.3], x_nfe = [30, 10]):
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
            x_nfe_list_temp = np.linspace(x_locations[k-1], x_locations[k],
                                x_nfe[k]).tolist()

        x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))
        
    x_nfe_list_temp = np.linspace(x_locations[k], 1,
                            x_nfe[k+1]).tolist()
    x_nfe_list = np.unique(np.concatenate((x_nfe_list, x_nfe_list_temp)))
        
    # nfe = len(x_nfe_list) -1
        
    return x_nfe_list.tolist()


def set_inputs(CO2_loading, H2O_loading, lean_flowrate):
# def set_inputs(case = 'K13'):    
    # Fix column design variables
    # Absorber diameter 
    m.fs.absorber.diameter_column.fix(12)
    
    # Absorber length according to number of packed beds 
    # 1 bed is approx 6 m
    m.fs.absorber.length_column.fix(20) # meter

    # Fix operating conditions
    # Flue gas
    m.fs.absorber.vapor_inlet.flow_mol.fix(12000)
    m.fs.absorber.vapor_inlet.temperature.fix(313.15)
    m.fs.absorber.vapor_inlet.pressure.fix(105000)
    m.fs.absorber.vapor_inlet.mole_frac_comp[0, "CO2"].fix(0.042)
    m.fs.absorber.vapor_inlet.mole_frac_comp[0, "H2O"].fix(0.058)
    m.fs.absorber.vapor_inlet.mole_frac_comp[0, "N2"].fix(0.77)
    m.fs.absorber.vapor_inlet.mole_frac_comp[0, "O2"].fix(0.13)
    
    # Solvent liquid
    # Computing mole fractions of MEA, CO2, and H2O from CO2 loading,
    # H2O loading, and total lean solvent flowrates 
    
    xMEA = 1/(CO2_loading + H2O_loading + 1)
    xCO2 = CO2_loading/(CO2_loading + H2O_loading + 1)
    xH2O = H2O_loading/(CO2_loading + H2O_loading + 1)
    
    print("\n*****************************************")
    print("CO2 loading: ", value(CO2_loading))
    print("CO2 mole fraction: ", value(xCO2))
    print("H2O mole fraction: ", value(xH2O))
    print("MEA mole fraction: ", value(xMEA))
    print("\n")
    
    time.sleep(10) 
    
    m.fs.absorber.liquid_inlet.flow_mol.fix(lean_flowrate)
    m.fs.absorber.liquid_inlet.temperature.fix(332.495) #313.15)
    m.fs.absorber.liquid_inlet.pressure.fix(258700) #105000)
    m.fs.absorber.liquid_inlet.mole_frac_comp[0, "MEA"].fix(xMEA)
    m.fs.absorber.liquid_inlet.mole_frac_comp[0, "CO2"].fix(xCO2)
    m.fs.absorber.liquid_inlet.mole_frac_comp[0, "H2O"].fix(xH2O)


def scale_absorber(column):
    gsf = iscale.get_scaling_factor
    ssf = iscale.set_scaling_factor

    def cst(con, s):
        iscale.constraint_scaling_transform(con, s, overwrite=False)

    for t in column.flowsheet().time:
        for x in column.liquid_phase.length_domain:
            ssf(column.velocity_liq[t, x], 20)
            ssf(column.interphase_mass_transfer[t, x, "CO2"], 1 / 20)
            ssf(column.interphase_mass_transfer[t, x, "H2O"], 1 / 100)
            ssf(column.mass_transfer_driving_force[t, x, "CO2"], 1 / 200 * 20)
            ssf(column.mass_transfer_driving_force[t, x, "H2O"], 1e-4 * 20)
            
            ssf(column.liquid_phase.heat[t, x], 1e-4)

            if column.config.surrogate_enhancement_factor_model is None:
                ssf(column.conc_CO2_bulk[t, x], 3)
                if x != column.liquid_phase.length_domain.last():
                    cst(column.conc_CO2_equil_bulk_eqn[t, x], 10)

        for x in column.vapor_phase.length_domain:
            ssf(column.heat_transfer_coeff[t, x], 1 / 3e6)
            ssf(column.vapor_phase.heat[t, x], 1e-4)

def strip_statevar_bounds_for_initialization(m):
    
    # Strip variable bounds 
    for t in m.fs.time:    
        for x in m.fs.absorber.liquid_phase.length_domain:
            
            for j in m.fs.absorber.config.liquid_phase.property_package.true_phase_component_set:
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setlb(None)
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setub(None)
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].domain = Reals
                
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_phase_comp_true[j].setlb(None)
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_phase_comp_true[j].setub(None)
            
            for j in m.fs.absorber.config.liquid_phase.property_package.apparent_species_set:
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_comp[j].setlb(None)
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_comp[j].setub(None)
                
            for j in m.fs.absorber.config.liquid_phase.property_package.apparent_phase_component_set:
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_phase_comp.setlb(None)
                m.fs.absorber.liquid_phase.properties[t, x].mole_frac_phase_comp.setub(None)
    
    # for t in m.fs.time:    
    #     for x in m.fs.absorber.vapor_phase.length_domain:
    #         for j in m.fs.absorber.config.vapor_phase.property_package.component_list:
    #             m.fs.absorber.vapor_phase.properties[t, x].mole_frac_comp[j].setlb(None)
    #             m.fs.absorber.vapor_phase.properties[t, x].mole_frac_comp[j].setub(None)
                
    # for t in m.fs.time:    
    #     for x in m.fs.absorber.vapor_phase.length_domain:
    #         for j in m.fs.absorber.config.vapor_phase.property_package._phase_component_set:
    #             m.fs.absorber.vapor_phase.properties[t, x].mole_frac_phase_comp[j].setlb(None)
    #             m.fs.absorber.vapor_phase.properties[t, x].mole_frac_phase_comp[j].setub(None)
     
    # for t in m.fs.time:    
    #     for x in m.fs.absorber.vapor_phase.length_domain:
            
    #         # Pressure
    #         m.fs.absorber.liquid_phase.properties[t, x].temperature.setlb(None)
    #         m.fs.absorber.liquid_phase.properties[t, x].temperature.setub(None)
    #         m.fs.absorber.liquid_phase.properties[t, x].temperature.domain = Reals
    #         m.fs.absorber.vapor_phase.properties[t, x].temperature.setlb(None)
    #         m.fs.absorber.vapor_phase.properties[t, x].temperature.setub(None)
    #         m.fs.absorber.vapor_phase.properties[t, x].temperature.domain = Reals
            
    return m


def strip_some_statevar_bounds(m):
    
    for t in m.fs.time:    
        for x in m.fs.absorber.vapor_phase.length_domain:
            
            # Pressure
            m.fs.absorber.liquid_phase.properties[t, x].pressure.setlb(None)
            m.fs.absorber.liquid_phase.properties[t, x].pressure.setub(None)
            m.fs.absorber.liquid_phase.properties[t, x].pressure.domain = Reals
            m.fs.absorber.vapor_phase.properties[t, x].pressure.setlb(None)
            m.fs.absorber.vapor_phase.properties[t, x].pressure.setub(None)
            m.fs.absorber.vapor_phase.properties[t, x].pressure.domain = Reals
    
            # Molar flowrate
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol.setlb(None)
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol.setub(None)
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol.domain = Reals
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol.setlb(None)
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol.setub(None)
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol.domain = Reals
            
            # flow_mol_phase
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase['Liq'].setlb(None)
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase['Liq'].setub(None)
            m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase['Liq'].domain = Reals
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol_phase['Vap'].setlb(None)
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol_phase['Vap'].setub(None)
            m.fs.absorber.vapor_phase.properties[t, x].flow_mol_phase['Vap'].domain = Reals
            
            # velocity_liq
            m.fs.absorber.velocity_liq.setlb(None)
            m.fs.absorber.velocity_liq.domain = Reals
            
            # sqrt_conc_interface_MEA
            m.fs.absorber.sqrt_conc_interface_MEA.setlb(None)
            m.fs.absorber.sqrt_conc_interface_MEA.setub(None)
            
            # conc_interface_MEA
            m.fs.absorber.conc_interface_MEA.setlb(None)
            m.fs.absorber.conc_interface_MEA.setub(None)
            
            # Enhancement factor
            m.fs.absorber.enhancement_factor.setlb(None)
            m.fs.absorber.enhancement_factor.setub(None)
                 
            for j in m.fs.absorber.config.liquid_phase.property_package.true_phase_component_set:
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setlb(None)
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].setub(None)
                m.fs.absorber.liquid_phase.properties[t, x].flow_mol_phase_comp_true[j].domain = Reals
                
                m.fs.absorber.liquid_phase.properties[t, x].log_conc_mol_phase_comp_true[j].setlb(None)
                
    return m
    

def define_column_design_parameters(m):
    m.fs.absorber.co2_capture = Var(m.fs.time,
                                initialize=98,
                                doc='''CO2 Capture Rate [%]''')
    
    @m.fs.absorber.Constraint(m.fs.time,
                      doc='''Correlation for CO2 Capture Rate''')

    def co2_capture_eqn(b, t):
        return b.co2_capture[t]*(b.vapor_inlet.mole_frac_comp[0, "CO2"]
                                 *b.vapor_inlet.flow_mol[0]) == ( 
                                 (b.vapor_inlet.mole_frac_comp[0, "CO2"]*b.vapor_inlet.flow_mol[0] -
                                  b.vapor_outlet.mole_frac_comp[0, "CO2"]*b.vapor_outlet.flow_mol[0])*100)
    
    m.fs.HDratio = Expression(
        expr=m.fs.absorber.length_column / m.fs.absorber.diameter_column,
        doc="Column height to column diameter ratio (-)"
        )
    m.fs.LGratio = Expression(
        expr=m.fs.absorber.liquid_inlet.flow_mol[0] / m.fs.absorber.vapor_inlet.flow_mol[0],
        doc="Inlet liquid (solvent) flowrate to inlet gas flowrate ratio (-)"
        )
    m.fs.volume_column = Expression(
        expr=math.pi * m.fs.absorber.length_column * (m.fs.absorber.diameter_column / 2) ** 2,
        doc="Volume of column (cubic m)"
        )
    m.fs.volume_column_withheads = Expression(
        expr=(
            math.pi * (
                m.fs.absorber.diameter_column ** 2 * m.fs.absorber.length_column
                ) / 4
            + math.pi / 3 * m.fs.absorber.diameter_column ** 3
            ),
        doc="Volume of column with heads (cubic m)",
    )
    
    return m   


def check_conservation():
    vap_in = m.fs.absorber.vapor_phase.properties[0, 0]
    vap_out = m.fs.absorber.vapor_phase.properties[0, 1]
    liq_in = m.fs.absorber.liquid_phase.properties[0, 1]
    liq_out = m.fs.absorber.liquid_phase.properties[0, 0]
    
    # Material conservation
    for j in ["CO2", "H2O"]:
        mass = abs(value(
            vap_in.get_material_flow_terms("Vap", j) +
            liq_in.get_material_flow_terms("Liq", j) -
            vap_out.get_material_flow_terms("Vap", j) -
            liq_out.get_material_flow_terms("Liq", j)))

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
        liq_out.get_enthalpy_flow_terms("Liq")))
    
    
def model_results():
    # Variables to be tested
    print('Liquid phase temperature')
    for x in m.fs.absorber.vapor_phase.length_domain:
        print(value(m.fs.absorber.liquid_phase.properties[0,x].temperature))
        
    print('Vapor phase temperature')
    for x in m.fs.absorber.vapor_phase.length_domain:
        print(value(m.fs.absorber.vapor_phase.properties[0,x].temperature))
    
    print('Vapor phase partial pressure of CO2')
    for x in m.fs.absorber.vapor_phase.length_domain:
        print(value(m.fs.absorber.vapor_phase.properties[0,x].pressure*m.fs.absorber.vapor_phase.properties[0,x].mole_frac_comp['CO2']))
        
    print('Henrys constant of CO2 in liquid phase')
    for x in m.fs.absorber.vapor_phase.length_domain:
        print(value(m.fs.absorber.liquid_phase.properties[0,x].henry['Liq','CO2']))
        
    print('Percent CO2 capture')   
    print(value(m.fs.absorber.co2_capture[0]))
    
    print('Flood fraction')
    for x in m.fs.absorber.vapor_phase.length_domain:
        print(value(m.fs.absorber.flood_fraction[0,x]))


def profiles_plot(m, CO2_loading, lean_flowrate):
    import matplotlib.pyplot as plt    
                    
    # Computed temperature profile(s)
    Tliq = []
    Tvap = []
    for x in m.fs.absorber.vapor_phase.length_domain:
        Tliq.append(value(m.fs.absorber.liquid_phase.properties[0,x].temperature - 273.15))
        Tvap.append(value(m.fs.absorber.vapor_phase.properties[0,x].temperature - 273.15))
     
    mass_transfer_vap_CO2 = []
    mass_transfer_vap_H2O = []
    for x in m.fs.absorber.vapor_phase.length_domain:
        mass_transfer_vap_CO2.append(value(m.fs.absorber.vapor_phase.mass_transfer_term[0,x,"Vap","CO2"]))
        mass_transfer_vap_H2O.append(value(m.fs.absorber.vapor_phase.mass_transfer_term[0,x,"Vap","H2O"]))
        
    mass_transfer_liq_CO2 = []
    mass_transfer_liq_H2O = []
    for x in m.fs.absorber.liquid_phase.length_domain:
        mass_transfer_liq_CO2.append(value(m.fs.absorber.liquid_phase.mass_transfer_term[0,x,"Liq","CO2"]))
        mass_transfer_liq_H2O.append(value(m.fs.absorber.liquid_phase.mass_transfer_term[0,x,"Liq","H2O"]))

    plt.figure()

    plt.plot(m.fs.absorber.vapor_phase.length_domain, Tliq,
             label='computed, $T_{liq}$', 
             linestyle='-',
             color='tab:blue')
    plt.plot(m.fs.absorber.vapor_phase.length_domain, Tvap,
             label='computed, $T_{vap}$', 
             linestyle='--',
             color='tab:orange')
    plt.legend(loc='best',ncol=1)
    plt.grid()
    # plt.ylim([40, 80])
    # plt.xlim([0, 1])
    title = (str(CO2_loading) + " molCO2/molMEA" + 
             ", " + str(lean_flow_rate[l]) + " mol/s")    
    plt.title(title)
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("Temperature [°C]") 
    
    plt.show()
    

def print_column_design_parameters(m):    
    print("\n ******* Printing some results *******")
    print("\nColumn diameter: ", value(m.fs.absorber.diameter_column), "m")
    print("Column height: ", value(m.fs.absorber.length_column), "m")
    print("Column volume: ", value(m.fs.volume_column), "m3")
    print("Column volume with heads: ", value(m.fs.volume_column_withheads), "m3")
    print("Column height to diameter ratio: ", value(m.fs.HDratio))
    print("\nSolvent inlet molar flowrate: ", value(m.fs.absorber.liquid_inlet.flow_mol[0]), "mol/s")
    print("L/G ratio: ", value(m.fs.LGratio))
        
    print("\nCO2 capture: ", value(m.fs.absorber.co2_capture[0]), "%")


if __name__ == "__main__":

    optarg = {
        'nlp_scaling_method': 'user-scaling',
        'linear_solver': 'ma57',
        'OF_ma57_automatic_scaling': 'yes',
        'max_iter': 300,
        'tol': 1e-8,
    }

    ts = time.time()
    
    nfe = 40
    use_surrogate = False
    x_nfe_list = get_uniform_grid(nfe)
    # x_nfe_list = get_custom_grid_dx([0.3, 0.8], [0.005, 0.1, 0.005])
    # x_nfe_list = get_custom_grid_nfe([0.3, 0.8], [50, 10, 40])    
    
    # Create dataframes to save results
    df_out_myresults  = pd.DataFrame()

    dfs = []
    sheetnames = []
    
    H2O_loading = 7.878958 #7.9116 #H2O:MEA ratio on apparent basis
    CO2_loading_list = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3] 
    #[0.224555]  #CO2:MEA ratio on apparent basis
    lean_flow_rate = [20000, 21000, 22000, 23000, 24000, 25000,
						26000, 27000, 28000, 29000, 30000] #[25500]
    vapor_flow_rate = 12000
    lean_flow_rate = np.array([1.5, 2.25, 4])*vapor_flow_rate
    CO2_loading_list = [0.14, 0.19, 0.24]
    inlet_Tl = [310, 315, 320]
    for Tl in inlet_Tl:
        for k in range(0, len(CO2_loading_list), 3):
            for l in range(0, len(lean_flow_rate), 3):

                ts_loopstart = time.time()

                m = build_column_model(x_nfe_list, use_surrogate=use_surrogate)
                print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))

                set_inputs(CO2_loading_list[k], H2O_loading, lean_flow_rate[l])

                print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))

                scale_mea_liquid_params(m.fs.liquid_properties, ions=True, scaling_factor_flow_mol=3e-4)
                scale_mea_vapor_params(m.fs.vapor_properties, scaling_factor_flow_mol=3e-4)

                define_column_design_parameters(m)

                switch_liquid_to_parmest_params(m.fs.liquid_properties, ions=True)
                scale_absorber(m.fs.absorber)
                # strip_statevar_bounds_for_initialization(m)
                initialize_inherent_reactions(m.fs.absorber.liquid_phase)

                iscale.calculate_scaling_factors(m)

                m.fs.absorber.initialize(
                    outlvl=idaeslog.DEBUG,
                    optarg=optarg
                )

                # check_scaling(m)

                # # Strip all bounds - works only for simulations (square problems)
                xfrm = TransformationFactory('contrib.strip_var_bounds')
                xfrm.apply_to(m, reversible=True)
                # strip_some_statevar_bounds(m)

                bounded_vars = 0
                for v in m.component_data_objects(Var, descend_into=True):
                    if v.lb != None or v.ub != None:
                        # print(v.name, v.lb, v.ub)
                        bounded_vars = bounded_vars + 1
                print('Number of bounded variables: ', value(bounded_vars))

                # Fix unused variables
                unused_vars_set = unused_variables_set(m)

                numberof_unused_vars = 0
                for unused_vars in unused_vars_set:
                    unused_vars.fix()
                    numberof_unused_vars = numberof_unused_vars + 1
                print('Number of fixed unused variables: ', value(numberof_unused_vars))
                print("\nSolve model, variable bounds stripped ...")
                print("\n")

                # Solve model
                solver.options = optarg
                try:
                    res = solver.solve(m, tee=True)
                    # Check whether mass & energy balances close
                    check_conservation(m)

                except:
                    print("An exception occurred")
                pyomo.opt.assert_optimal_termination(res)

                # constrviol = large_residuals_set(m.fs.absorber)
                # print(constrviol)

                # assert check_optimal_termination(res)

                # print('degrees_of_freedom = {}'.format(degrees_of_freedom(m)))
                # check_conservation()
                # model_results()
                # profiles_plot(m, CO2_loading_list[k], lean_flow_rate[l])
                # model_robustness_stats()

                print("\n-------- Simulation Results --------")
                print_column_design_parameters(m)

                import importlib
                import save_run_profiles

                importlib.reload(save_run_profiles)

                df = save_run_profiles.save_run_profiles(m)

                dfs.append(df)
                sheetnames.append(f'Tl={Tl},alpha={CO2_loading_list[k]},L_G ={lean_flow_rate[l]}')
            
    # Check if results folder exists (create if it does not exist), save results as .csv
    # Directory name    
    import os  
    directory = 'Results_varying_leanflowrate_CO2loading'
    terminal_condition = os.path.join(directory, 'results_terminal_condition.csv')
    
    try:
        os.mkdir(directory)
        df_out_myresults.to_csv(terminal_condition) 
    except:
        df_out_myresults.to_csv(terminal_condition) 

    print("\n")
    print("----------------------------------------------------------")
    print('Total simulation time: ', value(time.time() - ts), " s")
    print("----------------------------------------------------------")

