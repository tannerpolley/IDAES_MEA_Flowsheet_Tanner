#%%
from pyomo.environ import value, units as pyunits, log
from idaes.models.properties.modular_properties.eos.ideal import Ideal
import numpy as np
import pandas as pd
import xlwings as xw
from idaes.models_extra.column_models.properties.MEA_vapor import visc_d_comp


def save_run_profiles(m):
    n = len(m.fs.unit.vapor_phase.length_domain)
    H = value(m.fs.unit.length_column)
    stages = np.linspace(0, H, n)
    z_range = np.linspace(0, 1.0, n)
    dz = z_range[1] - z_range[0]

    dfs = []
    sheetnames = []

    def make_df(keys, array, name, return_df=False):

        d = {}
        for k, v in zip(keys, array.T):
            # Reverses
            d[k] = v[::-1]

        df = pd.DataFrame(d)
        df.index = stages[::-1]
        df.index.name = 'Position'
        dfs.append(df)
        sheetnames.append(name)
        if return_df:
            return df

    R = 8.314

    Fl_arr = np.empty((n, 11))
    Fv_arr = np.empty((n, 5))
    x_arr = np.empty((n, 9))
    y_arr = np.empty((n, 4))
    Cl_arr = np.empty((n, 9))
    Cv_arr = np.empty((n, 4))
    ql_arr = np.empty((n, 11))
    qv_arr = np.empty((n, 12))
    T_arr = np.empty((n, 2))
    transport_arr = np.empty((n, 20))
    CO2_arr = np.empty((n, 9))
    H2O_arr = np.empty((n, 11))
    enhance_arr = np.empty((n, 7))
    ChEq_arr = np.empty((n, 4))
    liq_prop_arr = np.empty((n, 15))
    vap_prop_arr = np.empty((n, 14))
    enhance_2_arr = np.empty((n, 3))

    liquid = ['CO2', 'MEA', 'H2O']
    liquid_w_ions = ['CO2', 'MEA', 'H2O', 'MEA_+', 'MEACOO_-', 'HCO3_-']
    vapor = ['CO2', 'H2O', 'N2', 'O2']
    flux_species = ['CO2', 'H2O']

    for i, zl in enumerate(m.fs.unit.vapor_phase.length_domain):
        blk = m.fs.unit
        lunits = (
            blk.config.liquid_phase.property_package.get_metadata().get_derived_units
        )

        A_col = value(blk.area_column)

        lp = blk.liquid_phase.properties[0, zl]

        if i < 100:
            zv = blk.vapor_phase.length_domain[i+1]
        else:
            zv = zl

        vp = blk.vapor_phase.properties[0, zv]


        # States

        P = value(pyunits.convert(vp.pressure, to_units=lunits("pressure")))
        T_arr[i] = [value(lp.temperature), value(vp.temperature)]
        T_string = ['Tl', 'Tv']

        Fl_arr[i] = [value(lp.flow_mol_phase_comp['Liq', j]) for j in liquid] + \
                    [value(lp.flow_mol_phase['Liq'])] + \
                    [value(lp.flow_mol_phase_comp_true['Liq', j]) for j in liquid_w_ions] + \
                    [sum([value(lp.flow_mol_phase_comp_true['Liq', j]) for j in liquid_w_ions])]

        Fl_string = ['Fl_CO2', 'Fl_MEA', 'Fl_H2O', 'Fl',
                     'Fl_CO2_true', 'Fl_MEA_true', 'Fl_H2O_true', 'Fl_MEAH_true', 'Fl_MEACOO_true', 'Fl_HCO3_true',
                     'Fl_true']

        Fv_arr[i] = [value(vp.flow_mol_phase_comp['Vap', j]) for j in vapor] + \
                    [value(vp.flow_mol_phase['Vap'])]

        Fv_string = ['Fv_CO2', 'Fv_H2O', 'Fv_N2', 'Fv_O2', 'Fv']

        x_arr[i] = [value(lp.mole_frac_comp[j]) for j in liquid] + \
                   [value(lp.mole_frac_phase_comp_true['Liq', j]) for j in liquid_w_ions]
        x_string = ['x_CO2', 'x_MEA', 'x_H2O',
                    'x_CO2_true', 'x_MEA_true', 'x_H2O_true', 'x_MEAH_true', 'x_MEACOO_true', 'x_HCO3_true']
        y_arr[i] = [value(vp.mole_frac_comp[j]) for j in vapor]
        y_string = ['y_CO2', 'y_H2O', 'y_N2', 'y_O2']

        Cl_arr[i] = [value(lp.conc_mol_comp[j]) for j in liquid] + \
                    [value(lp.conc_mol_phase_comp_true['Liq', j]) for j in liquid_w_ions]
        Cl_string = ['Cl_CO2', 'Cl_MEA', 'Cl_H2O',
                     'Cl_CO2_true', 'Cl_MEA_true', 'Cl_H2O_true', 'Cl_MEAH_true', 'Cl_MEACOO_true', 'Cl_HCO3_true']

        Cv_arr[i] = [P / R / value(vp.temperature) * value(vp.mole_frac_comp[j]) for j in vapor]
        Cv_string = ['Cv_CO2', 'Cv_H2O', 'Cv_N2', 'Cv_O2']


        # Thermodynamics
        H_CO2_mix = value(lp.henry['Liq', 'CO2'])
        Pv_CO2 = P * y_arr[i][0]
        Pl_CO2 = H_CO2_mix * Cl_arr[i][3]
        DF_CO2 = value(blk.mass_transfer_driving_force[0, zv, "CO2"])

        Pv_H2O = P * y_arr[i][1]
        Psat_H2O = value(lp.pressure_sat_comp['H2O'])
        Pl_H2O = x_arr[i][5] * Psat_H2O
        DF_H2O = value(blk.mass_transfer_driving_force[0, zv, "H2O"])

        Nl_CO2 = value(blk.liquid_phase.mass_transfer_term[0, zl, "Liq", 'CO2'])  # mol/m^3-s
        Nl_H2O = value(blk.liquid_phase.mass_transfer_term[0, zl, "Liq", 'H2O'])  # mol/m^3-s

        Nv_CO2 = value(blk.vapor_phase.mass_transfer_term[0, zv, "Vap", 'CO2'])  # mol/m^3-s
        Nv_H2O = value(blk.vapor_phase.mass_transfer_term[0, zv, "Vap", 'H2O'])  # mol/m^3-s

        if zv == 0:
            psi = 0
        else:
            psi = value(blk.psi[0, zv])

        a_e = value(blk.area_interfacial[0, zv])
        a_eA = a_e*A_col
        kv_CO2 = value(blk.mass_transfer_coeff_vap[0, zv, "CO2"])
        kv_H2O = value(blk.mass_transfer_coeff_vap[0, zv, "H2O"])

        CO2_arr[i] = [Nl_CO2, Nv_CO2, kv_CO2, a_eA, DF_CO2, Pv_CO2, Pl_CO2, psi, H_CO2_mix]
        CO2_string = ['Nl_CO2', 'Nv_CO2', 'kv_CO2', 'a_eA', 'DF_CO2', 'Pv_CO2', 'Pl_CO2', 'Psi', 'H_CO2_mix']
        y_H2O = value(vp.mole_frac_comp['H2O'])
        x_H2O_true = value(lp.mole_frac_phase_comp_true['Liq', 'H2O'])
        H2O_arr[i] = [Nl_H2O, Nv_H2O, kv_H2O, a_eA, DF_H2O, Pv_H2O, Pl_H2O, y_H2O, P, x_H2O_true, Psat_H2O]
        H2O_string = ['Nl_H2O', 'Nv_H2O', 'kv_CO2', 'a_eA', 'DF_H2O', 'Pv_H2O', 'Pl_H2O', 'y_H2O', 'P', 'x_H2O_true', 'Psat_H2O']

        # Energy
        ql_arr[i] = ([value(lp.temperature)] +
                     [value(lp.enth_mol_phase_comp['Liq', j]) for j in liquid] +
                     [value(lp.enth_mol_phase['Liq'])] +
                     [value(lp.enth_mol_phase_comp['Liq', j]) * value(
                         blk.liquid_phase.mass_transfer_term[0, zl, "Liq", j]) for j in flux_species] +
                     [value(blk.liquid_phase.enthalpy_transfer[0, zl])] +
                     [value(blk.liquid_phase.heat[0, zl])] +
                     [blk.liquid_phase.enthalpy_flow_dx[0, zl, 'Liq'].value] +
                     [value(lp.get_enthalpy_flow_terms('Liq'))]
                     )
        ql_arr[i][-2] = ql_arr[i][-2]/6.0

        ql_string = ['Tl', 'Hl_CO2', 'Hl_MEA', 'Hl_H2O', 'Hl', 'Hlt_CO2', 'Hlt_H2O', 'Hlt', 'q_trn', 'dHldz', 'Hlf']

        qv_arr[i] = ([value(vp.temperature)] +
                     [value(vp.enth_mol_phase_comp['Vap', j]) for j in vapor] +
                     [value(vp.enth_mol_phase['Vap'])] +
                     [value(vp.enth_mol_phase_comp['Vap', j]) * value(
                         blk.vapor_phase.mass_transfer_term[0, zv, "Vap", j]) for j in flux_species] +
                     [value(blk.vapor_phase.enthalpy_transfer[0, zv])] +
                     [value(blk.vapor_phase.heat[0, zv])] +
                     [blk.vapor_phase.enthalpy_flow_dx[0, zv, 'Vap'].value] +
                     [value(vp.get_enthalpy_flow_terms('Vap'))]
                     )

        qv_arr[i][-2] = qv_arr[i][-2] / 6.0
        qv_string = ['Tv', 'Hv_CO2', 'Hv_H2O', 'Hv_N2', 'Hv_O2', 'Hv', 'Hvt_CO2', 'Hvt_H2O', 'Hvt', 'q_trn', 'dHvdz', 'Hvf']

        # Transport
        kl_CO2 = value(blk.mass_transfer_coeff_liq[0, zl, "CO2"])
        kv_CO2 = value(blk.mass_transfer_coeff_vap[0, zv, "CO2"])
        kv_H2O = value(blk.mass_transfer_coeff_vap[0, zv, "H2O"])
        a_e = value(blk.area_interfacial[0, zv])
        ul = value(blk.velocity_liq[0, zl])
        uv = value(blk.velocity_vap[0, zv])
        h_L = value(blk.holdup_liq[0, zl])
        if zv == 0.0:
            h_V = 1
        else:
            h_V = value(blk.holdup_vap[0, zl])
        uv_Fl = value(blk.gas_velocity_fld[0, zv])
        flood_fraction = value(blk.flood_fraction[0, zv])
        UT_base = value(blk.heat_transfer_coeff_base[0, zv])
        UT = value(blk.heat_transfer_coeff[0, zv])
        a_p = value(blk.packing_specific_area)
        eps = value(blk.eps_ref)
        Clp = value(blk.Cl_ref)
        Cvp = value(blk.Cv_ref)
        Lp = A_col*a_p/eps
        d_h = 4*eps/a_p

        transport_arr[i] = [kl_CO2, kv_CO2, kv_H2O, ul, uv, uv_Fl, flood_fraction, h_L, h_V,a_e, UT_base, UT, P,
                            Clp, Cvp, eps, a_p, A_col, Lp, d_h]
        transport_string = ['kl_CO2', 'kv_CO2', 'kv_H2O', 'ul', 'uv', 'uv_Fl', 'fld_frac', 'h_L', 'h_V', 'a_e', 'UT_base', 'UT',
                            'P', 'Clp', 'Cvp', 'eps', 'a_p', 'A', 'Lp', 'd_h']

        # Enhancement Factor
        if zl < 1.0:
            k2_rate = np.exp(value(blk.log_rate_constant[0, zl]))
        else:
            k2_rate = 0
        Ha = np.exp(value(blk.log_hatta_number[0, zl]))
        E = np.exp(value(blk.log_enhancement_factor[0, zl]))
        if zv == 0:
            psi = 0
        else:
            psi = value(blk.psi[0, zv])

        Dl_CO2 = value(lp.diffus_phase_comp["Liq", 'CO2'])
        Dl_MEA = value(lp.diffus_phase_comp["Liq", 'MEA'])
        Cl_MEA_true = value(lp.conc_mol_phase_comp_true['Liq', 'MEA'])
        enhance_arr[i] = [k2_rate, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, psi]
        enhance_string = ['k2_rate', 'Cl_MEA_true', 'Dl_CO2', 'kl_CO2', 'Ha', 'E', 'psi']
        enhance_2_arr[i] = [E, Pv_CO2, DF_CO2]
        enhance_2_string = ['E', 'P_CO2', 'P_equil']

        # Chemical Equilibrium
        K_eq_car = np.exp(value(lp.log_k_eq['carbamate']))
        K_eq_bic = np.exp(value(lp.log_k_eq['bicarbonate']))

        ext_1 = value(lp.apparent_inherent_reaction_extent['carbamate'])
        ext_2 = value(lp.apparent_inherent_reaction_extent['bicarbonate'])

        ChEq_arr[i] = [K_eq_car, K_eq_bic, ext_1, ext_2]
        ChEq_string = ['K_eq_car', 'K_eq_bic', 'ext_1', 'ext_2']

        # Physical Properties
        rho_mol_l = value(lp.dens_mol_phase['Liq'])
        rho_mol_v = value(vp.dens_mol_phase['Vap'])
        rho_mass_l = value(lp.dens_mass_phase['Liq'])
        rho_mass_v = value(vp.dens_mass_phase['Vap'])
        mul = value(lp.visc_d_phase["Liq"])
        muv = value(vp.visc_d_phase["Vap"])

        muv_CO2 = value(visc_d_comp(vp, blk.vapor_phase.properties.params.Vap, 'CO2'))
        muv_H2O = value(visc_d_comp(vp, blk.vapor_phase.properties.params.Vap, 'H2O'))
        muv_N2 = value(visc_d_comp(vp, blk.vapor_phase.properties.params.Vap, 'N2'))
        muv_O2 = value(visc_d_comp(vp, blk.vapor_phase.properties.params.Vap, 'O2'))
        sigma_l = value(lp.surf_tens_phase["Liq"])
        Dv_CO2 = value(vp.diffus_phase_comp["Vap", 'CO2'])
        Dv_H2O = value(vp.diffus_phase_comp["Vap", 'H2O'])
        Dl_CO2 = value(lp.diffus_phase_comp["Liq", 'CO2'])
        Dl_MEA = value(lp.diffus_phase_comp_true["Liq","MEA"])
        Dl_MEAH = value(lp.diffus_phase_comp_true["Liq","MEA_+"])
        Dl_MEACOO = value(lp.diffus_phase_comp_true["Liq","MEACOO_-"])
        V = value(lp.vol_mol_phase["Liq"])
        V_CO2 = value(Ideal.get_vol_mol_pure(lp, 'liq', 'CO2', lp.temperature))
        V_MEA = value(Ideal.get_vol_mol_pure(lp, 'liq', 'MEA', lp.temperature))
        V_H2O = value(Ideal.get_vol_mol_pure(lp, 'liq', 'H2O', lp.temperature))
        kt_vap = np.exp(value(blk.log_therm_cond_vap[0, zv]))

        liq_prop_arr[i] = [rho_mol_l, rho_mass_l, V, V_CO2, V_MEA, V_H2O, mul, sigma_l, Dl_CO2, Dl_MEA, Dl_MEAH, Dl_MEACOO] + [
            value(lp.cp_mol_phase_comp['Liq', j]) for j in liquid]
        liq_prop_string = ['rho_mol', 'rho_mass', 'V', 'V_CO2', 'V_MEA', 'V_H2O', 'mu', 'sigma',
                           'Dl_CO2', 'Dl_MEA', 'Dl_MEAH', 'Dl_MEACOO', 'Cpl_CO2',
                           'Cpl_MEA', 'Cpl_H2O']
        vap_prop_arr[i] = [rho_mol_v, rho_mass_v, muv_CO2, muv_H2O, muv_N2, muv_O2, muv, Dv_CO2, Dv_H2O] + [value(vp.cp_mol_phase_comp['Vap', j]) for j in
                                                                          vapor] + [kt_vap]
        vap_prop_string = ['rho_mol', 'rho_mass', 'muv_CO2', 'muv_H2O', 'muv_N2', 'muv_O2', 'mu', 'Dv_CO2', 'Dv_H2O', 'Cpv_CO2', 'Cpv_H2O', 'Cpv_N2', 'Cpv_O2', 'kt_vap']

    make_df(Fl_string, Fl_arr, 'Fl')
    make_df(Fv_string, Fv_arr, 'Fv')
    make_df(x_string, x_arr, 'x')
    make_df(y_string, y_arr, 'y')
    make_df(Cl_string, Cl_arr, 'Cl')
    make_df(Cv_string, Cv_arr, 'Cv')
    make_df(T_string, T_arr, 'T')
    make_df(CO2_string, CO2_arr, 'CO2')
    make_df(H2O_string, H2O_arr, 'H2O')
    make_df(ql_string, ql_arr, 'ql')
    make_df(qv_string, qv_arr, 'qv')
    make_df(transport_string, transport_arr, 'transport')
    make_df(ChEq_string, ChEq_arr, 'ChEq')
    make_df(enhance_string, enhance_arr, 'enhance')
    make_df(liq_prop_string, liq_prop_arr, 'liq_prop')
    make_df(vap_prop_string, vap_prop_arr, 'vap_prop')

    df = make_df(enhance_2_string, enhance_2_arr, 'enhance', return_df=True)

    # wb = xw.Book('Simulation_Results/Profiles_IDAES.xlsx', read_only=False)
    # i = 0
    # for sheetname, df in zip(sheetnames, dfs):
    #     try:
    #         wb.sheets[sheetname].clear()
    #     except:
    #         wb.sheets.add(sheetname)
    #     wb.sheets[sheetname].range("A1").value = df
    #
    #     # wb.sheets[sheetname].activate()
    #     # wb.sheets[sheetname].api.Application.ActiveWindow.SplitRow = 1
    #     # wb.sheets[sheetname].api.Application.ActiveWindow.SplitColumn = 0
    #     # wb.sheets[sheetname].api.Application.ActiveWindow.FreezePanes = True
    #
    # # for i, sheet_name in enumerate(sheetnames):
    # #     sheet = wb.sheets[sheet_name]
    # #     sheet.api.Move(Before=wb.sheets[i].api)
    #
    # for sheet in wb.sheets:
    #     if sheet.name not in sheetnames:
    #         sheet.delete()
    # wb.save(path=r'Simulation_Results\Profiles_IDAES.xlsx')

    # Tl_sim = T_arr.T[0]
    # i_inters = list(Tl_sim).index(max(Tl_sim))
    # z_bulge = list(m.fs.unit.vapor_phase.length_domain)[i_inters]
    # z_bulge = z_bulge * 6

    return df

if __name__ == '__main__':
    save_run_profiles(m)
