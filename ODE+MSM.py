# ODE's for Population using MSM & Sindy (population-only)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import pysindy as ps

# 1) Combining data of all the patients
def load_patient_one(patient_id, filepath="HUPA{:04d}P.csv"):
    try:
        df = pd.read_csv(filepath.format(patient_id), sep=';')
    except Exception as e:
       return None

    df = df.copy()
    df['patient_id'] = patient_id

    # creating lag variables 
    for col in ['steps', 'carb_input', 'glucose', 'basal_rate', 'heart_rate','bolus_volume_delivered']:
        df[f'{col}_lag1'] = df[col].shift(1)

    df = df.dropna().reset_index(drop=True)
    for col in df.columns:
        if col != 'time':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

def load_patients_all(no_patients=28, filepath="HUPA{:04d}P.csv"):
    dfs = []
    for pid in range(1, no_patients+1):
        dfp = load_patient_one(pid, filepath)
        if dfp is not None:
            dfs.append(dfp)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# 2) Defining GPS with TWO continuous treatments- both basal and bolus are treatments 
def iptw_gps_multi(df, treatments=('basal_rate','bolus_volume_delivered'),
                   confounders=('glucose_lag1','carb_input_lag1','heart_rate_lag1','steps_lag1'),
                   eps=1e-3):
   
    d = df.copy()

    # Log-transform to avoid zeros
    Y = np.log(np.clip(d[list(treatments)].values, 0, None) + eps)

    X = sm.add_constant(d[list(confounders)], has_constant='add')
    ols = [sm.OLS(Y[:, j], X).fit() for j in range(Y.shape[1])]
    mu_cond = np.column_stack([m.predict(X) for m in ols])

    # residuals covariance
    resid = Y - mu_cond
    Sigma_cond = np.cov(resid.T)
    Sigma_cond += 1e-6 * np.eye(Sigma_cond.shape[0])  # stabilizer

    f_cond = multivariate_normal.pdf(Y, mean=Y.mean(axis=0), cov=Sigma_cond)
    f_marg = multivariate_normal.pdf(Y, mean=Y.mean(axis=0), cov=np.cov(Y.T))

    w = f_marg / f_cond
    d['iptw'] = np.clip(w, 0.01, 20.0)
    return d

# 3) ODE of a Population using MSM & Sindy
def population_msm_sindy(df, dt=5.0):
    confounders = ['glucose_lag1','carb_input_lag1','heart_rate_lag1','steps_lag1']  
    df = iptw_gps_multi(df, treatments=('basal_rate','bolus_volume_delivered'),
                        confounders=confounders)

    predictors = ['basal_rate','bolus_volume_delivered','carb_input',
                  'glucose_lag1','heart_rate_lag1','steps_lag1']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[predictors])
    X_msm = sm.add_constant(pd.DataFrame(X_scaled, columns=predictors, index=df.index),
                            has_constant='add')

    msm = sm.WLS(df['glucose'], X_msm, weights=df['iptw']).fit() 
    #glucose values adjusted for confounding
    df['glucose_adjusted'] = df['glucose'] - msm.predict(X_msm) + df['glucose'].mean()
    
    #here we take glucose adjusted through msm and not the original glucose value
    ode_features = ['glucose_adjusted','carb_input','basal_rate','bolus_volume_delivered']
    X_ode = StandardScaler().fit_transform(df[ode_features].to_numpy(dtype=float))

    #again sindy is used for getting ode
    opt = ps.STLSQ(threshold=1e-4)
    lib = ps.PolynomialLibrary(degree=2)
    sindy = ps.SINDy(optimizer=opt, feature_library=lib,
                     feature_names=['G','c','ba','bo'])
    sindy.fit(X_ode, t=dt)

    return sindy.equations()[0]  # return only the ODE for G

# 4) Print only the population ODE
if __name__ == "__main__":
    df_all = load_patients_all(no_patients=28)
    if df_all is not None:
        ode_eq = population_msm_sindy(df_all, dt=5.0)
        print("Population ODE with MSM")
        print(ode_eq)
    else:
        print("No patient data available.")
