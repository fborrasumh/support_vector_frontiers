import pandas as pd
from sklearn.model_selection import KFold
import itertools  # Necesaria para el producto cartesiano
import numpy as np
from itertools import zip_longest  # Necesario para la traspuesta del vector de vectores
import cplex

#Función de transformación B(z)
def transformacion(x_i, t_k):
    z = x_i - t_k
    if z < 0:
        return -1
    elif z == 0:
        return 0
    else:
        return 1

#Función para crear la matriz del grid
def create_matrix_t_equi(X, d):
    # Número de columnas x
    n_dim = len(X.columns)
    # Lista de listas de ts
    t = list()
    # Lista de indices (posiciones) para crear el vector de subind
    t_ind = list()
    for col in range(0, n_dim):
        # Ts de la dimension col
        ts = list()
        t_max = X.iloc[:, col].max()
        t_min = X.iloc[:, col].min()
        amplitud = (t_max - t_min) / (d)
        for i in range(0, d + 1):
            t_i = t_min + i * amplitud
            ts.append(t_i)
        t.append(ts)
        t_ind.append(np.arange(0, len(ts)))
    return t, t_ind

#Función para calcular posición en el grid de una x
def calculate_pos_phi(x, t, r):
    p = []
    for l in range(0, len(t)):
        for m in range(0, len(t[l])):
            trans = transformacion(x[l], r[m][l])
            if trans < 0:
                p.append(m - 1)
                break
            if trans == 0:
                p.append(m)
                break
            if trans > 0 and m == len(t[l]) - 1:
                p.append(m)
                break
    return p

#Función para la transformación phi de una x dada su posición en el grid
def calculate_value_phi(vector_subind, p):
    phi = []
    n_dim = len(p)
    for i in range(0, len(vector_subind)):
        for j in range(0, n_dim):
            r = 0
            if p[j] >= vector_subind[i][j]:
                r = 1
            else:
                r = 0
                break
        phi.append(r)
    return phi

#Función para calcular la matrix phi de todas las observaciones
def calculate_matriz_phi(X, t, vector_subind):
    x_list = X.values.tolist()
    # transpuesta de t para calcular el vector de posiciones de las observaciones
    r = list(zip_longest(*t))
    M = []
    for x in x_list:
        p = calculate_pos_phi(x, t, r)
        phi = calculate_value_phi(vector_subind, p)
        M.append(phi)
    return M

# Función objetivo = n_var + n_obs
def create_obj(n_var, n_obs, n_dim_y, C):
    obj_w = [float(1)] * n_var * n_dim_y
    obj_xi = [float(C)] * n_obs * n_dim_y
    obj = np.append(obj_w, obj_xi)
    return obj

# Transformación de las restricciones de u+v a u-v
def matriz_to_abs(m):
    n_obs = len(m)
    n_var = len(m[0])
    mat = list()
    for i in range(0, n_obs):
        for j in range(0, n_var):
            if m[i][j] == 1:
                mat.append(float(-1))
                mat.append(float(1))
            else:
                mat.append(float(0))
                mat.append(float(0))
    mat = np.array(mat).reshape(n_obs, n_var * 2)
    return mat

# Matriz de restricciones (1) y (2)
def create_rest(M, n_dim_y):
    # Primera restricción: y-w*phi(x)<=0
    n_obs = len(M)
    m_abs = matriz_to_abs(M)
    len_rest = len(m_abs[0])
    m_rest_w = []
    m_rest_w_neg = []
    for j in range(0, n_dim_y):
        for f in range(0, n_obs):
            f_rest_1 = list()
            f_rest_2 = list()
            rest = m_abs[f]
            for k in range(0, n_dim_y):
                for l in range(0, len_rest):
                    if j == k:
                        f_rest_1.append(rest[l])
                        f_rest_2.append(-rest[l])
                    else:
                        f_rest_1.append(0)
                        f_rest_2.append(0)
            m_rest_w.append(f_rest_1)
            m_rest_w_neg.append(f_rest_2)
    mat_zero = np.zeros((n_obs * n_dim_y, n_obs * n_dim_y))
    m_rest1 = np.concatenate((m_rest_w, mat_zero), axis=1)
    mat_identity = np.identity(n_obs * n_dim_y)
    m_rest2 = np.concatenate((m_rest_w_neg, -mat_identity), axis=1)
    m_rest = np.concatenate((m_rest1, m_rest2), axis=0)
    ind = [[[x for x in range(len(m_rest[0]))]]]
    ind = ind * len(m_rest)
    m_rest = m_rest.tolist()
    for i in range(len(ind)):
        m_rest[i] = ind[i] + [m_rest[i]]
    return m_rest

# Matriz de restricciones (3)
def create_rest3(t, vector_subind, model, n_dim_y):
    grid_points = list()
    r = list(zip_longest(*t))
    for combination in itertools.product(*t):
        grid_points.append(combination)
    cont = 0
    for point in grid_points:
        cont += 1
        p = calculate_pos_phi(point, t, r)
        left_side = calculate_value_phi(vector_subind, p)
        for t_dim in range(0, len(t)):
            k = p[t_dim] - 1
            if k >= 0:
                p[t_dim] = k
                right_side = calculate_value_phi(vector_subind, p)
                p[t_dim] = k + 1
                rest = np.asarray(left_side, dtype=np.float32) - np.asarray(
                    right_side, dtype=np.float32
                )
                rest = rest_to_abs(rest)
                rest_multioutput(rest, n_dim_y, model)
    return model

# Pasar las restricciones (3) a formato multioutput
def rest_multioutput(rest, n_dim_y, model):
    len_rest = len(rest)
    for i in range(0, n_dim_y):
        f_rest = list()
        for j in range(0, n_dim_y):
            for k in range(0, len_rest):
                if i == j:
                    f_rest.append(rest[k])
                else:
                    f_rest.append(0)
        ind = [[x for x in range(len(f_rest))]]
        f_rest = ind + [f_rest]
        f_rest = [f_rest]
        rhs = [float(0)]
        senses = "G"
        model.linear_constraints.add(rhs=rhs, senses=senses, lin_expr=f_rest)
    return model

# Pasar una restricción a formato u-v
def rest_to_abs(rest):
    n_var = len(rest)
    mat = []
    for i in range(0, n_var):
        if rest[i] == 1:
            mat.append(float(1))
            mat.append(float(-1))
        else:
            mat.append(float(0))
            mat.append(float(0))
    return mat

# Lado derecho de las restricciones
def create_rhs(Y, eps):
    Y = Y.values.tolist()
    rhs = list()
    for col in range(0, len(Y[0])):
        for fila in range(0, len(Y)):
            rhs.append(-Y[fila][col])
    for col in range(0, len(Y[0])):
        for fila in range(0, len(Y)):
            rhs.append(Y[fila][col] + eps)
    return rhs

# Modificar el rhs de un modelo ya creado
def modify_rhs(rhs, eps, n_obs, n_dim_y):
    n_obs_rest_1 = n_obs * n_dim_y
    rhs_rest_1 = [-x for x in rhs[-n_obs_rest_1:]]
    rhs_rest_2 = [x + eps for x in rhs[-n_obs_rest_1:]]
    n_obs_rest3 = n_obs_rest_1 * 2
    rhs_rest_3 = rhs[:-n_obs_rest3]
    rhs = np.concatenate((rhs_rest_3, rhs_rest_1), axis=0)
    rhs = np.concatenate((rhs, rhs_rest_2), axis=0).tolist()
    return rhs

#Función para generar el modelo svf
def generate_model_svf(data, c, eps, d):
    #######################################################################
    # Datos de las variables distintas de Y
    X = data.filter(regex=("x.*"))
    # Datos de las variables distintas de X
    Y = data.filter(regex=("y.*"))
    n_dim_y = len(Y.columns)
    n_obs = len(Y)
    #######################################################################

    #######################################################################
    # Matriz de t y de indices de ts
    t, t_ind = create_matrix_t_equi(X, d)
    # vector de subindices de w
    vector_subind = list()
    for combination in itertools.product(*t_ind):
        vector_subind.append(combination)
    M = calculate_matriz_phi(X, t, vector_subind)
    # Número de variables u-v del problema
    n_var = len(M[0]) * 2
    #######################################################################

    ################Problema de optimización:############################
    ##Crear el problema de optimización
    p = cplex.Cplex()
    p.set_log_stream(None)
    p.set_error_stream(None)
    p.set_warning_stream(None)
    p.set_results_stream(None)
    p.parameters.threads.set(1)

    ##Función objetivo
    ###Sentido
    p.objective.set_sense(p.objective.sense.minimize)
    ###Componente lineal
    obj = create_obj(n_var, n_obs, n_dim_y, c)
    # Número de variables u-v + xi del problema
    n_var = len(obj)

    ##Variables
    ###Límite superior de las variables
    ub = [float(1e33)] * n_var
    ###Límite inferior de las variables
    lb = [float(0)] * n_var
    p.variables.add(obj=obj, ub=ub, lb=lb)

    # Restricciones
    p = create_rest3(t, vector_subind, p, n_dim_y)
    rest = create_rest(M, n_dim_y)
    len_rest = len(rest)
    senses = "L" * len_rest
    rhs = create_rhs(Y, eps)
    p.linear_constraints.add(rhs=rhs, senses=senses, lin_expr=rest)

    return p

#Función para modificar un modelo SVF cambiando C y eps
def modify_solve_model_svf(model, c, eps, n_obs, n_dim_y):
    # Recuperar la información del problema del primer paso
    ##Variables
    ###Número de variables
    n_var = model.variables.get_num()
    ###Límite inferior
    lb = model.variables.get_lower_bounds()
    ###Límite superior
    ub = model.variables.get_upper_bounds()
    ###Cogemos todas las restricciones del problema
    rest = model.linear_constraints.get_rows()
    senses = model.linear_constraints.get_senses()
    rhs = model.linear_constraints.get_rhs()
    rhs = modify_rhs(rhs, eps, n_obs, n_dim_y)
    n_var_w = n_var - n_obs * n_dim_y
    obj_w = [float(1)] * n_var_w
    obj_xi = [float(c)] * n_obs * n_dim_y
    obj = np.append(obj_w, obj_xi)

    ################Problema de optimización:############################
    ##Crear el problema de optimización
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    model.parameters.threads.set(1)

    ##Función objetivo
    ###Sentido
    model.objective.set_sense(model.objective.sense.minimize)

    ##Variables
    model.variables.add(obj=obj, ub=ub, lb=lb)
    ##Restricciones
    model.linear_constraints.add(rhs=rhs, senses=senses, lin_expr=rest)
    # SOLUCIONAR EL PROBLEMA
    # model.solve()
    return model

# Función par obtener las soluciones de un modelo ya resuelto
def get_solutions(model, n_obs, n_dim_y):
    model.solve()
    uv_index = n_obs * n_dim_y
    # VALORES DE LA SOLUCIÓN
    s = model.solution.get_values()
    # Array de u-v
    uv = s[:-uv_index]
    # Array de xi
    # xi = s[-n_obs:]
    v = list()
    u = list()
    for i in range(0, len(uv)):
        if i % 2 == 0:
            u.append(uv[i])
        else:
            v.append(uv[i])
    # Calcular valor de las W
    w = list()
    for i in range(len(v)):
        w.append(u[i] - v[i])
    # Numero de ws por dimensión
    n_w_dim = int(len(w) / n_dim_y)
    mat_w = [[] for i in range(0, n_dim_y)]
    cont = 0
    for i in range(0, n_dim_y):
        for j in range(0, n_w_dim):
            mat_w[i].append(w[cont])
            cont += 1
    return mat_w


# Calcula el error cuadrático medio de un modelo entrenado
def calculate_cv_mse(data_train, data_test, w, d):
    data_test_X = data_test.filter(regex=("x.*"))
    data_test_Y = data_test.filter(regex=("y.*"))
    n_dim_y = len(data_test_Y.columns)
    data_train_X = data_train.filter(regex=("x.*"))
    t, t_ind = create_matrix_t_equi(data_train_X, d)
    error = 0
    for i in range(len(data_test_X)):
        x = data_test_X.iloc[i]
        for j in range(n_dim_y):
            y_est = estimacion(w[j], x, t, t_ind)
            y = data_test_Y.iloc[i, j]
            error = error + (y - y_est) ** 2
    return error

# Estimación del modelo SVR para un valor X
def estimacion(w, x, t, t_ind):
    r = list(zip_longest(*t))
    p = calculate_pos_phi(x, t, r)
    vector_subind = list()
    for combination in itertools.product(*t_ind):
        vector_subind.append(combination)
    phi = calculate_value_phi(vector_subind, p)
    suma = 0
    for i in range(0, len(w)):
        suma = suma + w[i] * phi[i]
    return suma

#Calcular el número de particiones de cada dimensión de x en el grid
def calculate_d(data):
    n_obs = len(data.index)
    d = list()
    for i in range(1, 11):
        n = int(round(0.1 * i * n_obs, 0))
        if n > 0:
            d.append(n)
    return d

# Función de validación cruzada. Devuelve un dataframe con el error de cada combinación [C,eps,d]
def cross_validation(data, C, eps, d, n_folds, seed):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = 0
    error_folds = pd.DataFrame(columns=["C","eps","d", "error"])
    n_obs_data = len(data.values)
    for train_index, test_index in kf.split(data):
        fold += 1
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        n_obs_train = len(data_train.values)
        Y = data.filter(regex=("y.*"))
        n_dim_y = len(Y.columns)
        for val_d in d:
            model_d = generate_model_svf(data_train, 0, 0, val_d)
            for c in C:
                for e in eps:
                    model = modify_solve_model_svf(model_d, c, e, n_obs_train, n_dim_y)
                    w = get_solutions(model, n_obs_train, n_dim_y)
                    error_bruto = calculate_cv_mse(data_train, data_test, w, val_d)
                    error_folds = error_folds.append(
                        {
                            "C": c,
                            "eps": e,
                            "d": val_d,
                            "error": error_bruto,
                        },
                        ignore_index=True,
                    )
    error_folds = error_folds.groupby(['C','eps','d']).sum() / n_obs_data
    #Esto se pone para que en caso de empates se coja el que mayor C-eps-d tiene
    error_folds = error_folds.sort_index(ascending=False)
    return error_folds

#Dataframe con los valores inciciales de x y su estimación y_sfv
def dataframe_est(data, d, w):
    X = data.filter(regex=("x.*"))
    Y = data.filter(regex=("y.*"))
    n_dim_y = len(Y.columns)
    t, t_ind = create_matrix_t_equi(X, d)
    vector_ts = list()
    for combination in itertools.product(*t):
        vector_ts.append(combination)
    df_est = pd.DataFrame(X)
    num_columns = len(df_est.columns)
    name_columns = ["x" + str(i + 1) for i in range(num_columns)]
    df_est.columns = name_columns
    df_y = [[] for i in range(0, n_dim_y)]
    for i in range(len(df_est)):
        x = df_est.iloc[i]
        for j in range(n_dim_y):
            y_est = estimacion(w[j], x, t, t_ind)
            df_y[j].append(round(y_est, 6))
    name_columns = ["y" + str(i + 1) for i in range(n_dim_y)]
    df_y_empty = pd.DataFrame(columns=name_columns)
    df_est = pd.concat((df_est, df_y_empty), axis=1)
    for j in range(n_dim_y):
        df_est["y" + str(j + 1)] = df_y[j]
    return df_est
