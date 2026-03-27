import numpy as np


def convert_data(df, run=0, type='mole'):
    X = df.iloc[run-1, :].to_numpy()

    MWs_l = np.array([.04401, .06108, .01802])  # kg/mol
    MWs_v = np.array([.04401, .01802, .02801, .032])  # kg/mol


    L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds = X

    # Molecular Weights
    MW_CO2 = MWs_l[0]
    MW_MEA = MWs_l[1]
    MW_H2O = MWs_l[2]
    MW_N2 = MWs_v[2]
    MW_O2 = MWs_v[3]

    alpha_O2_N2 = 0.08485753604
    alpha_H2O_CO2 = 0.9626010166

    # Liquid Calculations
    Fl_T = L_G * Fv_T

    x_MEA_unloaded = w_MEA_unloaded / (MW_MEA / MW_H2O + w_MEA_unloaded * (1 - MW_MEA / MW_H2O))
    x_H2O_unloaded = 1 - x_MEA_unloaded

    Fl_MEA_b = Fl_T * x_MEA_unloaded
    Fl_H2O_b = Fl_T * x_H2O_unloaded

    Fl_CO2_b = Fl_MEA_b * alpha
    Fl = [Fl_CO2_b, Fl_MEA_b, Fl_H2O_b]
    Fl_T = sum(Fl)

    x = [Fl_CO2_b/Fl_T, Fl_MEA_b/Fl_T, Fl_H2O_b/Fl_T]
    x_CO2, x_MEA, x_H2O = x

    # Vapor Calculations

    # Find Vapor Mole Fractions
    y_H2O = y_CO2 * alpha_H2O_CO2
    y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
    y_O2 = y_N2 * alpha_O2_N2

    y = [y_CO2, y_H2O, y_N2, y_O2]

    H = 6
    D = .64

    Tv = Tv_0
    Tl = Tl_z
    nested_dic = {}

    dic_2 = {'diameter': D, 'length': H,
             'vapor_inlet': {'flow_mol': Fv_T,
                             'temperature': Tv,
                             'pressure': 109180,
                             'mole_frac_comp': {'CO2': y_CO2,
                                                'H2O': y_H2O,
                                                'N2': y_N2,
                                                'O2': y_O2}},
             'liquid_inlet': {'flow_mol': Fl_T,
                              'temperature': Tl,
                              'pressure': 109180,
                              'mole_frac_comp': {'CO2': x_CO2,
                                                 'H2O': x_H2O,
                                                 'MEA': x_MEA}}}
    nested_dic[str(run)] = dic_2

    return nested_dic

