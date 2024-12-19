# Import Python libraries
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Pyomo libraries
import pyomo.opt
from pyomo.environ import (
    Block,
    ConcreteModel,
    value,
    Var,
    Reals,
    NonNegativeReals,
    Param,
    TransformationFactory,
    Constraint,
    Expression,
    Objective,
    SolverStatus,
    TerminationCondition,
    check_optimal_termination,
    assert_optimal_termination,
    exp,
    log,
    sqrt,
    units as pyunits,
    Set,
    Reference
)
from pyomo.common.collections import ComponentSet, ComponentMap

from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.common.config import ConfigValue, Bool

# Import IDAES Libraries
from idaes.core.util.constants import Constants as CONST
from idaes.models_extra.column_models.solvent_column import PackedColumnData
from idaes.models_extra.column_models.MEAsolvent_column import _fix_vars, _restore_fixedness

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import declare_process_block_class, FlowsheetBlock, StateBlock
from idaes.core.util.exceptions import InitializationError
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog

from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.core.util.scaling as iscale
from pyomo.util.subsystems import (
    create_subsystem_block,
)
from idaes.core.solvers.petsc import (
    _sub_problem_scaling_suffix,
)

from flowsheets.mea_properties import (
    MEALiquidParameterBlock,
    FlueGasParameterBlock,
    scale_mea_liquid_params,
    scale_mea_vapor_params,
    switch_liquid_to_parmest_params,
    initialize_inherent_reactions
)

from enhancement_factor_model_new import make_enhancement_factor_model, initialize_enhancement_factor_model

__author__ = "Douglas Allan"

logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)



def make_fake_column_model(flowsheet, properties_liquid):
    blk = flowsheet.column = Block()

    def return_flowsheet():
        return flowsheet
    
    blk.flowsheet = return_flowsheet
    # blk.config = Block()
    # blk.config.liquid_phase = Block()
    # blk.config.liquid_phase.property_package = Reference(m.fs.liq_params)

    blk.liquid_phase = Block()
    blk.liquid_phase.length_domain = Set(ordered=True, initialize=[0, 1])
    blk.liquid_phase.liquid_domain = Set(ordered=True, initialize=[0,])
    blk.liquid_phase.properties = properties_liquid.build_state_block(
        blk.flowsheet().time,
        blk.liquid_phase.liquid_domain,
        has_phase_equilibrium=False,
        defined_state=True,
    )

    lunits = properties_liquid.get_metadata().get_derived_units

    blk.mass_transfer_coeff_liq = Var(
        blk.flowsheet().time,
        blk.liquid_phase.liquid_domain,
        ["CO2"],
        initialize=1e-4,
        units=lunits("velocity")
    )


    # The only vapor phase property that enters into enhancement factor calculations is partial
    # pressure of CO2. Therefore just create a normal Pyomo block instead of an entire StateBlock
    blk.vapor_phase = Block()
    blk.vapor_phase.length_domain = Set(ordered=True, initialize=[0, 1])
    blk.vapor_phase.vapor_domain = Set(ordered=True, initialize=[1,])
    blk.vapor_phase.properties = Block(blk.flowsheet().time, blk.vapor_phase.vapor_domain)
    # The model will look for both mole frac comp of CO2 and pressure, but only their product is relevant
    # to the output. Therefore, fix the mole fraction of CO2 at 1 and vary partial pressure by varying
    # the actual pressure
    for t in blk.flowsheet().time:
        for x in blk.vapor_phase.vapor_domain:
            blk.vapor_phase.properties[t, x].mole_frac_comp = Param(["CO2"], initialize=1)
            blk.vapor_phase.properties[t, x].pressure = Var(initialize = 5000, units=pyunits.Pa)

    blk.mass_transfer_coeff_vap = Var(
        blk.flowsheet().time,
        blk.vapor_phase.vapor_domain,
        ["CO2"],
        initialize=1e-5,
        units=(
            lunits("amount")
            / lunits("pressure")
            / lunits("length") ** 2
            / lunits("time")
        )
    )
    log_diffus_liq_comp_list = blk.log_diffus_liq_comp_list = [
            "CO2",
            "MEA",
        ]  # Can add ions if we want them
    solute_comp_list = blk.solute_comp_list = ["CO2"]

    blk.log_diffus_liq_comp = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        blk.log_diffus_liq_comp_list,
        bounds=(None, 100),
        initialize=1,
        units=pyunits.dimensionless,
        doc="""Logarithm of the liquid phase diffusivity of a species""",
    )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        blk.log_diffus_liq_comp_list,
        doc="Defines log variable for liquid phase diffusivity",
    )
    def log_diffus_liq_comp_eqn(b, t, x, j):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            return exp(b.log_diffus_liq_comp[t, x, j]) * lunits(
                "diffusivity"
            ) == (b.liquid_phase.properties[t, x].diffus_phase_comp["Liq", j])

    blk.log_mass_transfer_coeff_liq = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        solute_comp_list,
        initialize=-9,
        units=pyunits.dimensionless,
        doc="""Logarithm of the liquid mass transfer coeff""",
    )

    @blk.Constraint(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        solute_comp_list,
        doc="""Defines log variable for the liquid mass transfer coeff""",
    )
    def log_mass_transfer_coeff_liq_eqn(b, t, x, j):
        if x == b.liquid_phase.length_domain.last():
            return Constraint.Skip
        else:
            return (
                exp(b.log_mass_transfer_coeff_liq[t, x, j]) * lunits("velocity")
                == b.mass_transfer_coeff_liq[t, x, j]
            )

    blk.log_enhancement_factor = Var(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        units=pyunits.dimensionless,
        bounds=(None, 100), #  100 is about where we start getting AMPL evaluation errors due to overflow
        initialize=5,
        doc="Natural logarithm of the enhancement factor",
    )

    @blk.Expression(
        blk.flowsheet().time,
        blk.liquid_phase.length_domain,
        doc="Enhancement factor",
    )
    def enhancement_factor(b, t, x):
        if x == b.liquid_phase.length_domain.last():
            return Expression.Skip
        else:
            return exp(b.log_enhancement_factor[t, x])

    @blk.Expression(
        blk.flowsheet().time,
        blk.vapor_phase.length_domain,
        doc="Intermediate for calculating CO2 mass transfer driving force",
    )
    def psi(b, t, x):
        if x == b.vapor_phase.length_domain.first():
            return Expression.Skip
        else:
            zb = blk.liquid_phase.length_domain.prev(x)
            return (
                b.enhancement_factor[t, zb]
                * b.mass_transfer_coeff_liq[t, zb, "CO2"]
                / b.mass_transfer_coeff_vap[t, x, "CO2"]
            )
    make_enhancement_factor_model(blk)
    


def initialize_model(
        blk,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        optarg=None,
        solver=None,
    ):
    initialize_inherent_reactions(blk.liquid_phase.properties)
    blk.liquid_phase.properties.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=False,
            state_args=state_args,
        )
    for t in blk.flowsheet().time:
        for x in blk.liquid_phase.length_domain:
            if x == blk.liquid_phase.length_domain.last():
                continue
            zf = blk.liquid_phase.length_domain.next(x)
            for j in blk.log_diffus_liq_comp_list:
                blk.log_diffus_liq_comp[t, x, j].value = log(value(blk.liquid_phase.properties[t, x].diffus_phase_comp["Liq", j]))
            # CO2 is the only index
            blk.log_mass_transfer_coeff_liq[t, x, "CO2"].value = log(value(blk.mass_transfer_coeff_liq[t, x, "CO2"]))
    initialize_enhancement_factor_model(
        blk, 
        outlvl=outlvl,
        optarg=optarg,
        solver=solver,
    )
    
def calculate_scaling_factors(blk):
    for sub_blk in blk.liquid_phase.properties.values():
        sub_blk.calculate_scaling_factors()

    def gsf(var):
        return iscale.get_scaling_factor(var, default=1, warning=True)

    def ssf(var, s):
        iscale.set_scaling_factor(var, s, overwrite=False)

    def cst(con, s):
        iscale.constraint_scaling_transform(con, s, overwrite=False)

    for t in blk.flowsheet().time:
        for x_liq in blk.liquid_phase.length_domain:
            if x_liq == blk.liquid_phase.length_domain.last():
                continue
            x_vap = blk.vapor_phase.length_domain.next(x_liq)

            ssf(blk.vapor_phase.properties[t, x_vap].pressure, 1e-5)

            for j in blk.log_diffus_liq_comp_list:
                sf_diffus_liq_comp = iscale.get_scaling_factor(
                    blk.liquid_phase.properties[t, x_liq].diffus_phase_comp[
                        "Liq", j
                    ],
                    default=2e8,
                    warning=False,
                )
                cst(blk.log_diffus_liq_comp_eqn[t, x_liq, j], sf_diffus_liq_comp)

            for j in blk.solute_comp_list:
                sf = iscale.get_scaling_factor(
                    blk.mass_transfer_coeff_liq[t, x_liq, j],
                    default=1e4,
                    warning=False,
                )
                ssf(blk.mass_transfer_coeff_liq[t, x_liq, j], sf)
                cst(blk.log_mass_transfer_coeff_liq_eqn[t, x_liq, j], sf)


            for j in ["CO2"]:
                sf = iscale.get_scaling_factor(
                    blk.mass_transfer_coeff_vap[t, x_vap, j],
                    default=25000,
                    warning=False,
                )
                ssf(blk.mass_transfer_coeff_vap[t, x_vap, j], sf)


    # TODO bring this into new form later
    for t in blk.flowsheet().time:
        for x in blk.liquid_phase.length_domain:
            iscale.set_scaling_factor(blk.conc_CO2_bulk[t, x], 3)
            iscale.set_scaling_factor(blk.conc_CO2_equil_bulk[t, x], 3)
    
    for (t, x), con in blk.conc_CO2_bulk_eqn.items():
        zf = blk.liquid_phase.length_domain.next(x)
        sf_C = gsf(blk.conc_CO2_bulk[t, x])
        cst(con, sf_C)
            
def strip_true_comp_lower_bounds(column):
    for blk in column.liquid_phase.properties.values():
        for idx in blk.flow_mol_phase_comp_true.index_set():
            blk.flow_mol_phase_comp_true[idx].domain = Reals
            blk.flow_mol_phase_comp_true[idx].bounds = (None, None)
        
        for idx in blk.mole_frac_phase_comp_true.index_set():
            blk.mole_frac_phase_comp_true[idx].domain = Reals
            blk.mole_frac_phase_comp_true[idx].bounds = (None, None)

def check_model(column):
    model_okay = True

    for blk in column.liquid_phase.properties.values():
        for idx in blk.flow_mol_phase_comp_true.index_set():
            if not blk.flow_mol_phase_comp_true[idx].value > 0:
                model_okay = False
        
        for idx in blk.mole_frac_phase_comp_true.index_set():
            if not blk.mole_frac_phase_comp_true[idx].value > 0:
                model_okay = False
    
    # This variable should always be negative
    if hasattr(column, "log_singular_MEA_CO2_ratio"):
        for var in column.log_singular_MEA_CO2_ratio.values():
            if not value(var) < 0:
                model_okay = False

    # If this variable goes to zero, some equations lose meaning
    for var in column.conc_CO2_bulk.values():
        if not abs(value(var)) > 1e-6:
            model_okay=False

    return model_okay

if __name__ == "__main__":
    m = ConcreteModel()

    mode = "stripper"
    assert mode in {"stripper", "absorber"}

    nCO2 = 11
    nH2O = 11
    nPCO2 = 6
    nT = 11
    nk_liq = 6
    nk_vap = 6

    # nCO2 = 6
    # nH2O = 6
    # nPCO2 = 3
    # nT = 6
    # nk_liq = 3
    # nk_vap = 3

    CO2_loading_bounds = [0.05, 0.6]
    H2O_loading_bounds = [6.5, 8.5]

    if mode == "stripper":
        pCO2_range = [7500, 75000]
        T_range = [360, 405]
    elif mode == "absorber":
        pCO2_range = [50, 7500]
        T_range = [305, 345]

    k_liq_range = [5e-5, 5e-4]
    k_vap_range = [3e-5, 5.5e-5]

    n_time = nCO2*nH2O
    mea_array = np.zeros(n_time)
    co2_array = np.zeros(n_time)
    h2o_array = np.zeros(n_time)

    m.fs = FlowsheetBlock(dynamic=False, time_set=list(range(n_time)))

    m.fs.liq_params = MEALiquidParameterBlock()

    make_fake_column_model(m.fs, m.fs.liq_params)

    scale_mea_liquid_params(m.fs.liq_params, ions=True, scaling_factor_flow_mol=3e-4)
    calculate_scaling_factors(m.fs.column)
    strip_true_comp_lower_bounds(m.fs.column)
    
    m.fs.column.vapor_phase.properties[:, :].pressure.fix(0.042 * 105000)
    m.fs.column.liquid_phase.properties[:, :].flow_mol.fix(25000)
    m.fs.column.liquid_phase.properties[:, :].temperature.fix(332.495) #313.15)
    m.fs.column.liquid_phase.properties[:, :].pressure.fix(258700) #105000)
    m.fs.column.mass_transfer_coeff_liq.fix(0.00010814704194056394)
    m.fs.column.mass_transfer_coeff_vap.fix(4.383300498413155e-05)

    for i, CO2_loading in enumerate(np.linspace(CO2_loading_bounds[0], CO2_loading_bounds[1], nCO2)):
        for j, H2O_loading in enumerate(np.linspace(H2O_loading_bounds[0], H2O_loading_bounds[1], nH2O)):
            n = nCO2*i+j
            xMEA = 1/(CO2_loading + H2O_loading + 1)
            xCO2 = CO2_loading/(CO2_loading + H2O_loading + 1)
            xH2O = H2O_loading/(CO2_loading + H2O_loading + 1)

            mea_array[n] = xMEA
            co2_array[n] = xCO2
            h2o_array[n] = xH2O
            m.fs.column.liquid_phase.properties[n, :].mole_frac_comp["MEA"].fix(xMEA)
            m.fs.column.liquid_phase.properties[n, :].mole_frac_comp["CO2"].fix(xCO2)
            m.fs.column.liquid_phase.properties[n, :].mole_frac_comp["H2O"].fix(xH2O)

    # plt.scatter(h2o_array, co2_array)
    # plt.xlabel("x_H2O")
    # plt.ylabel("x_CO2")

    # plt.show()

    print(degrees_of_freedom(m))

    initialize_model(
        m.fs.column,
        outlvl=idaeslog.DEBUG,
        optarg={
            # 'bound_push' : 1e-22,
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
            # 'mu_strategy': 'monotone',
        }
    )
    assert check_model(m.fs.column)

    solver = get_solver(
        "ipopt", 
        {
            # 'bound_push' : 1e-22,
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
            # 'mu_strategy': 'monotone',
        }
        )

    # df_data  = pd.DataFrame() 
    data = []

    for pCO2 in np.logspace(log(pCO2_range[0]), log(pCO2_range[1]), nPCO2, base=exp(1)):
        print(pCO2)
        for T in np.linspace(T_range[0], T_range[1], nT):
            print(T)
            for k_liq in np.logspace(log(k_liq_range[0]), log(k_liq_range[1]), nk_liq, base=exp(1)):
                for k_vap in np.linspace(k_vap_range[0], k_vap_range[1], nk_vap):
                    m.fs.column.vapor_phase.properties[:, :].pressure.fix(pCO2)
                    m.fs.column.liquid_phase.properties[:, :].temperature.fix(T)
                    m.fs.column.mass_transfer_coeff_liq.fix(k_liq)
                    m.fs.column.mass_transfer_coeff_vap.fix(k_vap)

                    results = solver.solve(m, tee=False, load_solutions=False)
                    if not check_optimal_termination(results):
                        initialize_model(
                                m.fs.column,
                                outlvl=idaeslog.CRITICAL,
                                optarg={
                                    # 'bound_push' : 1e-22,
                                    'nlp_scaling_method': 'user-scaling',
                                    'linear_solver': 'ma57',
                                    'OF_ma57_automatic_scaling': 'yes',
                                    'max_iter': 300,
                                    'tol': 1e-8,
                                    'halt_on_ampl_error': 'no',
                                    # 'mu_strategy': 'monotone',
                                }
                        )
                        results = solver.solve(m, tee=False, load_solutions=False)
                    if check_optimal_termination(results):
                        m.solutions.load_from(results)

                        if check_model(m.fs.column):
                            for t in m.fs.time:
                                my_results = {
                                    'CO2_loading': value(m.fs.column.liquid_phase.properties[t, 0].mole_frac_comp["CO2"]/m.fs.column.liquid_phase.properties[t, 0].mole_frac_comp["MEA"]),
                                    'H2O_loading': value(m.fs.column.liquid_phase.properties[t, 0].mole_frac_comp["H2O"]/m.fs.column.liquid_phase.properties[t, 0].mole_frac_comp["MEA"]),
                                    'log_pCO2': log(pCO2),
                                    'T': T,
                                    'k_liq': k_liq,
                                    'k_vap': k_vap,
                                    'log_enhancement_factor': value(m.fs.column.log_enhancement_factor[t, 0])
                                    # 'IPOPT_Terminal_Condition': results.solver.termination_condition,
                                }   
            
                                data.append(my_results)
                        else:
                            print(f"Solver returned nonphysical solutions: pCO2: {pCO2}, T: {T}, k_liq: {k_liq}, k_vap: {k_vap}")
                    else:
                        print(f"Solver failure: pCO2: {pCO2}, T: {T}, k_liq: {k_liq}, k_vap: {k_vap}")


    df_data = pd.DataFrame(data)
    df_data.to_csv("surrogate_training_data.csv")


    # enhancement_factor_array = np.zeros(n_time)

    # for t in m.fs.time:
    #     enhancement_factor_array[int(t)] = value(m.fs.column.enhancement_factor[t, 0])
    
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from scipy.interpolate import griddata

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter(mea_array, co2_array, enhancement_factor_array)
    # ax.set_xlabel("x_mea")
    # ax.set_ylabel("x_co2")
    # ax.set_zlabel("enhancement factor")
    # plt.show()
    
