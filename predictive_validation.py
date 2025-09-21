# glucose_msm_sindy_metrics_min.py
# Load MSM-GPS -> SINDy -> evaluate (RMSE/MAE/change in sign) 
# Plotting Bar graph - campare ODE+MSM vs Only ODE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import pysindy as ps

# configurations 
CSV_PATTERN   = "HUPA{:04d}P.csv"   # HUPA0001P.csv, ...
NUM_PATIENTS  = 28
CSV_SEP       = ";"
DT_MIN        = 5.0                 # minutes per sample
HORIZON_STEPS = (3, 6, 12)          # 15, 30, 60 min
STRIDE_STEPS  = 12                  # start windows every 60 min
MEAL_THR      = 5.0                 # g carbs at window start
BOLUS_THR     = 0.5                 # U insulin at window start
RMSE_FIG_PATH = "rmse_panels.png"

# 1) combining all the  data 
def load_all(n=NUM_PATIENTS, pattern=CSV_PATTERN, sep=CSV_SEP):
    frames = []
    for pid in range(1, n + 1):
        try:
            f = pd.read_csv(pattern.format(pid), sep=sep)
            f["patient_id"] = pid
            frames.append(f)
        except Exception:
            pass
    if not frames:
        raise RuntimeError("No patient files found.")
    return pd.concat(frames, ignore_index=True)

def prep(df, dt=DT_MIN):
    need = ["glucose","carb_input","basal_rate","bolus_volume_delivered",
            "steps","heart_rate","patient_id"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")

    chunks = []
    for _, g in df.sort_values("patient_id").groupby("patient_id", sort=False):
        g = g.reset_index(drop=True).copy()
        g["time"] = np.arange(0, dt * len(g), dt); g = g.set_index("time")
        for c in ["glucose","carb_input","basal_rate","bolus_volume_delivered"]:
            if len(g[c]) >= 7 and g[c].notna().sum() >= 7:
                g[c] = savgol_filter(g[c].interpolate(limit_direction="both"),
                                     window_length=7, polyorder=2, mode="interp")
        for c in ["glucose","carb_input","steps","heart_rate","basal_rate","bolus_volume_delivered"]:
            g[f"{c}_lag1"] = g[c].shift(1)
        chunks.append(g.dropna())
    return pd.concat(chunks).reset_index()

# 2) MSM/GPS (two continuous doses: basal, bolus) 
#    GPS- since treatment (basal& bolus is continuous)    
def gps_weights(df,
                treatments=("basal_rate","bolus_volume_delivered"),
                confs=("glucose_lag1","carb_input_lag1","heart_rate_lag1","steps_lag1"),
                eps=1e-3):
    d  = df.copy()
    Y  = np.log(np.clip(d[list(treatments)].to_numpy(), 0, None) + eps)
    X  = sm.add_constant(d[list(confs)], has_constant="add")
    mu = np.column_stack([sm.OLS(Y[:, j], X).fit().predict(X) for j in range(Y.shape[1])])
    S_c = np.cov((Y - mu).T) + 1e-6*np.eye(Y.shape[1])
    mu_m, S_m = Y.mean(axis=0), np.cov(Y.T) + 1e-6*np.eye(Y.shape[1])
    den = np.array([multivariate_normal.pdf(Y[i], mean=mu[i],  cov=S_c) for i in range(len(Y))])
    num = multivariate_normal.pdf(Y, mean=mu_m, cov=S_m)
    d["iptw"] = np.clip(num / den, 0.01, 20.0)
    return d

# 3) MSM working model → deconfounded glucose 
def residualize(df_w):
    X = sm.add_constant(df_w[["basal_rate","bolus_volume_delivered","carb_input",
                              "glucose_lag1","heart_rate_lag1","steps_lag1"]], has_constant="add")
    fit = sm.WLS(df_w["glucose"].to_numpy(), X, weights=df_w["iptw"].to_numpy()).fit()
    df_w["Ghat"] = fit.predict(X)
    df_w["G_adj"] = df_w["glucose"] - df_w["Ghat"] + df_w["glucose"].mean()
    return fit, df_w

# 4) SINDy to calculate ODE 
def sindy_fit(df, cols, names, dt=DT_MIN):
    X = df[cols].to_numpy()
    m = ps.SINDy(ps.STLSQ(threshold=1e-4), ps.PolynomialLibrary(degree=2), feature_names=names)
    m.fit(X, t=dt)
    return m

# 5) rollouts + metrics 
def rollout(model, y0, window, dt=DT_MIN):
    y = float(y0)
    for c,ba,bo in window:
        y += dt * model.predict(np.array([[y,c,ba,bo]]))[0,0]
    return y

def evaluate(df, direct, msm,
             horizons=HORIZON_STEPS, stride=STRIDE_STEPS,
             dt=DT_MIN, meal_thr=MEAL_THR, bolus_thr=BOLUS_THR):
    per = {h: {k:[] for k in
               ["rmse_direct","rmse_msm","mae_direct","mae_msm","sign_meal_msm","sign_bolus_msm"]}
           for h in horizons}

    for _, g in df.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True)
        G, Gadj, Ghat = g["glucose"].to_numpy(), g["G_adj"].to_numpy(), g["Ghat"].to_numpy()
        c, ba, bo = g["carb_input"].to_numpy(), g["basal_rate"].to_numpy(), g["bolus_volume_delivered"].to_numpy()
        starts = np.arange(0, len(g) - max(horizons), stride)

        for h in horizons:
            d_obs=d_dir=d_msm=[]; meal_o=meal_p=[]; bol_o=bol_p=[]
            d_obs, d_dir, d_msm, meal_o, meal_p, bol_o, bol_p = [],[],[],[],[],[],[]

            for t0 in starts:
                t1 = t0 + h
                W  = np.column_stack([c[t0:t1], ba[t0:t1], bo[t0:t1]])
                if len(W) < h: continue
                Gd = rollout(direct, G[t0],  W, dt)
                Ga = rollout(msm,    Gadj[t0], W, dt)

                d_obs.append(G[t1]-G[t0])
                d_dir.append(Gd - G[t0])
                d_msm.append((Ga - Gadj[t0]) + (Ghat[t1]-Ghat[t0]))

                if c[t0]  >= meal_thr:  meal_o.append(d_obs[-1]);  meal_p.append(d_msm[-1])
                if bo[t0] >= bolus_thr: bol_o.append(d_obs[-1]);   bol_p.append(d_msm[-1])

            if not d_obs: continue
            d_obs, d_dir, d_msm = map(np.array, (d_obs, d_dir, d_msm))
            rmse = lambda a,b: np.sqrt(np.mean((a-b)**2))
            mae  = lambda a,b: np.mean(np.abs(a-b))
            sacc = lambda p,o: np.mean(np.sign(p)==np.sign(o)) if len(o)>0 else np.nan
            per[h]["rmse_direct"].append(rmse(d_dir, d_obs))
            per[h]["rmse_msm"].append(rmse(d_msm, d_obs))
            per[h]["mae_direct"].append(mae(d_dir, d_obs))
            per[h]["mae_msm"].append(mae(d_msm, d_obs))
            if meal_o: per[h]["sign_meal_msm"].append(sacc(meal_p, meal_o))
            if bol_o:  per[h]["sign_bolus_msm"].append(sacc(bol_p,  bol_o))

    def mean_ci(xs):
        xs = np.array(xs, float); xs = xs[~np.isnan(xs)]
        if len(xs)==0: return (np.nan, np.nan, 0)
        m = xs.mean(); se = xs.std(ddof=1)/np.sqrt(len(xs))
        return (m, 1.96*se, len(xs))
    return {h:{k:mean_ci(v) for k,v in per[h].items()} for h in horizons}

def print_summary(summary):
    for h, row in summary.items():
        print(f"\n=== Horizon {h} steps (~{int(h*DT_MIN)} min) ===")
        for k in ["rmse_direct","rmse_msm","mae_direct","mae_msm","sign_meal_msm","sign_bolus_msm"]:
            m, ci, n = row[k]
            label = k.replace("_"," ").upper()
            if np.isnan(m): print(f"  {label}: n=0")
            else:
                unit = " (mg/dL)" if "RMSE" in label or "MAE" in label else ""
                print(f"  {label}: {m:.3f} ± {ci:.3f}{unit}  [n={n}]")

def plot_rmse(summary, savepath=RMSE_FIG_PATH):
    hs = sorted(summary.keys())
    d  = [summary[h]["rmse_direct"][0] for h in hs]
    m  = [summary[h]["rmse_msm"][0]   for h in hs]
    cd = [summary[h]["rmse_direct"][1] for h in hs]
    cm = [summary[h]["rmse_msm"][1]   for h in hs]
    fig, ax = plt.subplots(1, 3, figsize=(11,3.4), sharey=True)
    for a, title, dv, mv, civ, cim in zip(ax, [f"{int(h*DT_MIN)} min" for h in hs], d, m, cd, cm):
        a.bar(["Direct","MSM+ODE"], [dv, mv], yerr=[civ, cim], capsize=4)
        a.set_title(title); a.set_ylabel("RMSE on ΔG (mg/dL)"); a.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(savepath, dpi=200, bbox_inches="tight"); plt.show()

# main - print 
if __name__ == "__main__":
    raw = load_all()
    df  = prep(raw, dt=DT_MIN)

    df_w = gps_weights(df)
    _, df_w = residualize(df_w)
    print("\n[IPTW] weights summary:\n", df_w["iptw"].describe())

    direct = sindy_fit(df_w, ["glucose","carb_input","basal_rate","bolus_volume_delivered"],
                       ["G","c","ba","bo"], dt=DT_MIN)
    msm    = sindy_fit(df_w, ["G_adj","carb_input","basal_rate","bolus_volume_delivered"],
                       ["G_adj","c","ba","bo"], dt=DT_MIN)

    summary = evaluate(df_w, direct, msm,
                       horizons=HORIZON_STEPS, stride=STRIDE_STEPS,
                       dt=DT_MIN, meal_thr=MEAL_THR, bolus_thr=BOLUS_THR)
    print_summary(summary)
    plot_rmse(summary, savepath=RMSE_FIG_PATH)
