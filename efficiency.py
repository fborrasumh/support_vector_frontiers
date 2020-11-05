import numpy as np
from docplex.mp.model import Model
import svf_multi_comp

#Calcular tabla de eficiencias
def get_efficiency(data_simulation, sol_w, best_d,best_eps):
    df_est= svf_multi_comp.dataframe_est(data_simulation, best_d, sol_w)
    df_eff = data_simulation.copy()
    df_eff['ef_fdh'] = np.nan
    df_eff['ef_dea'] = np.nan
    df_eff['ef_svf'] = np.nan
    df_eff['ef_csvf'] = np.nan
    df_eff['svf_eps_insen'] = ''
    df_eff['csvf_eps_insen'] = ''
    for i in range(0, len(data_simulation)):
        df_eff['ef_fdh'].values[i] = round(efficiency_fdh(data_simulation, i),3)
        df_eff['ef_dea'] .values[i] = round(efficiency_dea(data_simulation, i),3)
        df_eff['ef_svf'].values[i] = round(efficiency_svf(data_simulation,df_est, i),3)
        df_eff['ef_csvf'].values[i] = round(efficiency_csvf(data_simulation, df_est, i),3)
        df_eff['svf_eps_insen'].values[i] = get_svf_epsilon_insensible_efficiency(data_simulation, df_est, i,best_eps)
        df_eff['csvf_eps_insen'].values[i] = get_csvf_epsilon_insensible_efficiency(data_simulation, df_est, i,best_eps)
    return df_eff

#Calcular la eficiencia FDH
def efficiency_fdh(data, i):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("FDH Multioutput")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

#Calcular la eficiencia DEA
def efficiency_dea(data, i):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("DEA Multioutput")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

#Calcular la eficiencia SVF
def efficiency_svf(data, df_est, i):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF efficiency multioutput")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    mdl.solve()
    eff = mdl.solution["phi"]
    return eff

#Ver si la DMU es epsilon insensible para SVF
def get_svf_epsilon_insensible_efficiency(data, df_est, i,eps):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("SVF epsilon insensible")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.binary_var_dict(name_landa, name="landa")
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * (y_est[k][r]-eps) for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["phi"]
    else:
        eff=0
    if(eff>1):
        return 'no'
    else:
        return 'yes'

#Calcular la eficiencia CSVF
def efficiency_csvf(data, df_est, i):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF efficiency multioutput")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * y_est[k][r] for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["phi"]
    else:
        eff=0
    return eff

#Ver si la DMU es epsilon insensible para CSVF
def get_csvf_epsilon_insensible_efficiency(data, df_est, i,eps):
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    x = X.values.tolist()
    X_est = df_est.filter(regex=("x.*"))
    x_est = X_est.values.tolist()
    # Número de dimensiones X del problema
    n_dim_x = len(X_est.columns)
    # Datos de las variables Y
    Y = data.filter(regex=("y.*"))
    y = Y.values.tolist()
    Y_est = df_est.filter(regex=("y.*"))
    y_est = Y_est.values.tolist()
    # Número de dimensiones y del problema
    n_dim_y = len(Y.columns)
    # Número de observaciones del problema
    n_obs = len(Y_est)
    # Variable landa
    name_landa = range(0, n_obs)
    mdl = Model("CSVF epsilon insensible")
    # Variables
    ##Variable phi
    phi = mdl.continuous_var(name="phi", ub=1e33, lb=0)
    ##Variable landa
    landa_var = mdl.continuous_var_dict(name_landa, name="landa", ub=1e33, lb=0)
    # Función objetivo
    mdl.maximize(phi)
    # Restricciones
    for j in range(n_dim_x):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * x_est[k][j] for k in range(n_obs)) <= x[i][j]
        )
    for r in range(n_dim_y):
        mdl.add_constraint(
            mdl.sum(landa_var[k] * (y_est[k][r]-eps) for k in range(n_obs)) >= phi * y[i][r]
        )
    mdl.add_constraint(mdl.sum(landa_var[k] for k in range(n_obs)) == 1)
    msol=mdl.solve()
    if msol is not None:
        eff = mdl.solution["phi"]
    else:
        eff=0
    if(eff>1):
        return 'no'
    else:
        return 'yes'