
# Short-horizon validation for Direct SINDy vs MSM+GPS→SINDy (ΔG)

# Assumes CSV columns:
#   'glucose','carb_input','basal_rate','bolus_volume_delivered',
#   'steps','heart_rate'


from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import Dict, Iterable, Tuple

from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal

import statsmodels.api as sm
import pysindy as ps



# Config (edit as needed)

CSV_PATTERN = "HUPA{:04d}P.csv"
NUM_PATIENTS = 28
CSV_SEP = ";"
DT_MIN = 5.0                 # minutes per grid step
HORIZON_STEPS = (3, 6, 12)   # 15, 30, 60 min
STRIDE_STEPS = 12            # 60 min between starts (non-overlapping)
MEAL_THR = 5.0               # grams at window start considered a "meal" event
BOLUS_THR = 0.5              # units at window start considered a "bolus" event
RMSE_FIG_PATH = "rmse_panels.png"


# 0) Load & basic preprocessing

def load_all_patients(n: int = NUM_PATIENTS,
                      pattern: str = CSV_PATTERN,
                      sep: str = CSV_SEP) -> pd.DataFrame:
    """Load all patient CSVs into a single DataFrame with a patient_id column."""
    frames = []
    for pid in range(1, n + 1):
        try:
            df = pd.read_csv(pattern.format(pid), sep=sep)
            df["patient_id"] = pid
            frames.append(df)
        except Exception:
            # Skip missing/bad files silently to keep run smooth
            continue
    if not frames:
        raise RuntimeError("No patient files loaded. Check path/pattern.")
    return pd.concat(frames, ignore_index=True)


def preprocess(df_raw: pd.DataFrame, dt_min: float = DT_MIN) -> pd.DataFrame:
    """
    - Sort per patient
    - Reset a uniform 5-min time index per patient
    - Smooth key signals (Savitzky–Golay)
    - Add 1-step lags (5-min lags)
    """
    need_cols = [
        "glucose", "carb_input", "basal_rate", "bolus_volume_delivered",
        "steps", "heart_rate", "patient_id"
    ]
    missing = [c for c in need_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in CSVs: {missing}")

    df_sorted = df_raw.sort_values(["patient_id"]).reset_index(drop=True)

    processed = []
    for pid, g in df_sorted.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True).copy()

        # Build a uniform 5-min time grid for this patient
        g["time"] = np.arange(0, dt_min * len(g), dt_min)
        g = g.set_index("time")

        # Smooth a few key continuous series
        for col in ["glucose", "carb_input", "basal_rate", "bolus_volume_delivered"]:
            if len(g[col]) >= 7 and g[col].notna().sum() >= 7:
                g[col] = savgol_filter(
                    g[col].interpolate(limit_direction="both"),
                    window_length=7, polyorder=2, mode="interp"
                )

        # Add 1-step lags (5-min lag)
        for col in ["glucose", "carb_input", "steps", "heart_rate",
                    "basal_rate", "bolus_volume_delivered"]:
            g[f"{col}_lag1"] = g[col].shift(1)

        g = g.dropna()
        processed.append(g)

    dfp = pd.concat(processed).reset_index()  # keep 'time' as a column
    return dfp


#-----------------------------------------------------------
# 1) GPS stabilized weights for TWO continuous doses (ba, bo)
#-----------------------------------------------------------
def pdf_rowwise(Y: np.ndarray, mean_rows: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Row-wise MVN pdf: each Y[i] uses its own mean mean_rows[i]."""
    return np.array([multivariate_normal.pdf(Y[i], mean=mean_rows[i], cov=cov)
                     for i in range(Y.shape[0])])


def iptw_gps_multi(
    df: pd.DataFrame,
    treatments: Tuple[str, str] = ("basal_rate", "bolus_volume_delivered"),
    confounders: Tuple[str, ...] = ("glucose_lag1", "carb_input_lag1", "heart_rate_lag1", "steps_lag1"),
    eps: float = 1e-3,
    debug: bool = False
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Build stabilized inverse-probability weights for two continuous treatments
    using a pragmatic multivariate normal GPS on log-doses.
    """
    d = df.copy()

    # Log-doses to handle zeros / skew
    Y = np.log(np.clip(d[list(treatments)].values, 0, None) + eps)  # (n, 2)

    # Conditional mean for each dose component via OLS on confounders
    X = sm.add_constant(d[list(confounders)], has_constant='add')    # (n, k)
    ols = [sm.OLS(Y[:, j], X).fit() for j in range(Y.shape[1])]
    mu_cond = np.column_stack([m.predict(X) for m in ols])           # (n, 2)

    # Residual covariance (denominator)
    resid = Y - mu_cond
    Sigma_cond = np.cov(resid.T) + 1e-6 * np.eye(resid.shape[1])

    # Marginal mean & covariance (numerator)
    mu_marg = Y.mean(axis=0)                                        # (2,)
    Sigma_marg = np.cov(Y.T) + 1e-6 * np.eye(Y.shape[1])

    # Row-wise densities
    den = pdf_rowwise(Y, mu_cond, Sigma_cond)                       # (n,)
    num = multivariate_normal.pdf(Y, mean=mu_marg, cov=Sigma_marg)  # (n,)

    # Stabilized weights (clipped)
    w = np.clip(num / den, 0.01, 20.0)
    d["iptw"] = w

    if debug:
        print(f"[GPS] Y: {Y.shape}, mu_cond: {mu_cond.shape}")
        print(f"[GPS] Sigma_cond: {Sigma_cond.shape}, mu_marg: {mu_marg.shape}, Sigma_marg: {Sigma_marg.shape}")

    gps_fits = dict(mu_cond=mu_cond, Sigma_cond=Sigma_cond, mu_marg=mu_marg, Sigma_marg=Sigma_marg)
    return d, gps_fits


# ---------------------------------------------------------------
# 2) MSM working model & residualization
# ---------------------------------------------------------------
def residualize_glucose(df_w: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Weighted working model: G_t ~ (ba, bo, c, G_{t-1}, HR_{t-1}, S_{t-1}) with IPTW.
    Adds 'Ghat' and 'G_adj' to df_w.
    """
    X = sm.add_constant(df_w[[
        "basal_rate", "bolus_volume_delivered", "carb_input",
        "glucose_lag1", "heart_rate_lag1", "steps_lag1"
    ]], has_constant='add')
    y = df_w["glucose"].values
    w = df_w["iptw"].values

    wls = sm.WLS(y, X, weights=w).fit()
    df_w["Ghat"] = wls.predict(X)
    df_w["G_adj"] = df_w["glucose"] - df_w["Ghat"] + df_w["glucose"].mean()
    return wls, df_w


# ---------------------------------------------------------------
# 3) Fit SINDy models (Direct on G, MSM on G_adj)
# ---------------------------------------------------------------
def fit_sindy_direct(df: pd.DataFrame, dt: float = DT_MIN) -> ps.SINDy:
    X = df[["glucose", "carb_input", "basal_rate", "bolus_volume_delivered"]].to_numpy()
    lib = ps.PolynomialLibrary(degree=2)
    opt = ps.STLSQ(threshold=1e-4)
    model = ps.SINDy(feature_library=lib, optimizer=opt,
                     feature_names=["G", "c", "ba", "bo"])
    model.fit(X, t=dt)
    return model


def fit_sindy_msm(df: pd.DataFrame, dt: float = DT_MIN) -> ps.SINDy:
    X = df[["G_adj", "carb_input", "basal_rate", "bolus_volume_delivered"]].to_numpy()
    lib = ps.PolynomialLibrary(degree=2)
    opt = ps.STLSQ(threshold=1e-4)
    model = ps.SINDy(feature_library=lib, optimizer=opt,
                     feature_names=["G_adj", "c", "ba", "bo"])
    model.fit(X, t=dt)
    return model


# ----------------------------------------------------------------
# 4) Rollouts & evaluation
# ----------------------------------------------------------------
def euler_rollout_sindy(model: ps.SINDy,
                        y0: float,
                        inputs_window: np.ndarray,
                        dt: float = DT_MIN,
                        var_index: int = 0) -> float:
    """
    One-step Euler integration over a short window using a fitted SINDy model.
    y0: initial glucose state (G or G_adj)
    inputs_window: array (h, 3) with columns (c, ba, bo)
    """
    y = float(y0)
    for (c, ba, bo) in inputs_window:
        x = np.array([[y, c, ba, bo]])
        dy_dt = model.predict(x)[0, var_index]  # derivative for the first state variable
        y = y + dt * dy_dt
    return y


def evaluate_combined(
    df: pd.DataFrame,
    model_direct: ps.SINDy,
    model_msm: ps.SINDy,
    horizons: Iterable[int] = HORIZON_STEPS,
    stride: int = STRIDE_STEPS,
    dt: float = DT_MIN,
    meal_thr: float = MEAL_THR,
    bolus_thr: float = BOLUS_THR
) -> Dict[int, Dict[str, Tuple[float, float, int]]]:
    """
    Compute per-patient metrics then aggregate across patients (mean ± 95% CI).
    Metrics (per horizon): RMSE/MAE for Direct & MSM, plus sign accuracy after meal/bolus.
    """
    per_patient = {
        h: {
            "rmse_direct": [], "rmse_msm": [],
            "mae_direct": [],  "mae_msm": [],
            "sign_meal_msm": [], "sign_bolus_msm": []
        } for h in horizons
    }

    for pid, g in df.groupby("patient_id", sort=False):
        g = g.copy().reset_index(drop=True)

        G = g["glucose"].to_numpy()
        Gadj = g["G_adj"].to_numpy()
        Ghat = g["Ghat"].to_numpy()
        c_arr = g["carb_input"].to_numpy()
        ba_arr = g["basal_rate"].to_numpy()
        bo_arr = g["bolus_volume_delivered"].to_numpy()

        max_h = max(horizons)
        starts = np.arange(0, len(g) - max_h, stride)  # e.g., every 60min

        for h in horizons:
            dG_obs_list, dG_dir_list, dG_msm_list = [], [], []
            meal_obs, meal_pred = [], []
            bolus_obs, bolus_pred = [], []

            for t0 in starts:
                t1 = t0 + h
                window = np.column_stack([c_arr[t0:t1], ba_arr[t0:t1], bo_arr[t0:t1]])
                if window.shape[0] < h:
                    continue

                # Direct model on raw G
                G0 = G[t0]
                G_end_pred = euler_rollout_sindy(model_direct, G0, window, dt=dt, var_index=0)

                # MSM model on G_adj, then map back to raw G:
                # ΔG = ΔG_adj + (Ghat_{t+h} - Ghat_t)
                Gadj0 = Gadj[t0]
                Gadj_end_pred = euler_rollout_sindy(model_msm, Gadj0, window, dt=dt, var_index=0)

                dG_obs = G[t1] - G0
                dG_dir = G_end_pred - G0
                dG_msm = (Gadj_end_pred - Gadj0) + (Ghat[t1] - Ghat[t0])

                dG_obs_list.append(dG_obs)
                dG_dir_list.append(dG_dir)
                dG_msm_list.append(dG_msm)

                # Event-based sign checks at window start
                if c_arr[t0] >= meal_thr:
                    meal_obs.append(dG_obs)
                    meal_pred.append(dG_msm)
                if bo_arr[t0] >= bolus_thr:
                    bolus_obs.append(dG_obs)
                    bolus_pred.append(dG_msm)

            # Per-patient metrics for this horizon
            if not dG_obs_list:
                continue

            dG_obs = np.array(dG_obs_list)
            d_dir = np.array(dG_dir_list)
            d_msm = np.array(dG_msm_list)

            rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
            mae = lambda a, b: np.mean(np.abs(a - b))
            sacc = lambda a, b: np.mean(np.sign(a) == np.sign(b)) if len(a) > 0 else np.nan

            per_patient[h]["rmse_direct"].append(rmse(d_dir, dG_obs))
            per_patient[h]["rmse_msm"].append(rmse(d_msm, dG_obs))
            per_patient[h]["mae_direct"].append(mae(d_dir, dG_obs))
            per_patient[h]["mae_msm"].append(mae(d_msm, dG_obs))

            if meal_obs:
                per_patient[h]["sign_meal_msm"].append(sacc(np.array(meal_pred), np.array(meal_obs)))
            if bolus_obs:
                per_patient[h]["sign_bolus_msm"].append(sacc(np.array(bolus_pred), np.array(bolus_obs)))

    # Aggregate across patients: mean ± 95% CI
    def mean_ci(xs: Iterable[float]) -> Tuple[float, float]:
        xs = np.array(xs, dtype=float)
        xs = xs[~np.isnan(xs)]
        if len(xs) == 0:
            return (np.nan, np.nan)
        m = xs.mean()
        se = xs.std(ddof=1) / np.sqrt(len(xs))
        return (m, 1.96 * se)

    summary: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    for h in horizons:
        row = {}
        for metric, vals in per_patient[h].items():
            m, ci = mean_ci(vals)
            row[metric] = (m, ci, len(vals))
        summary[h] = row
    return summary


def print_summary(summary: Dict[int, Dict[str, Tuple[float, float, int]]]) -> None:
    """Nicely print mean ± 95% CI and sample sizes per horizon."""
    for h, row in summary.items():
        print(f"\n=== Horizon {h} steps (~{int(h * DT_MIN)} min) ===")
        for metric in ["rmse_direct", "rmse_msm", "mae_direct", "mae_msm",
                       "sign_meal_msm", "sign_bolus_msm"]:
            m, ci, n = row[metric]
            label = metric.replace("_", " ").upper()
            if np.isnan(m):
                print(f"  {label}: n=0")
            else:
                unit = " (mg/dL)" if "RMSE" in label or "MAE" in label else ""
                print(f"  {label}: {m:.3f} ± {ci:.3f}{unit}   [n={n}]")


def plot_rmse_bars(summary: Dict[int, Dict[str, Tuple[float, float, int]]],
                   savepath: str | None = RMSE_FIG_PATH) -> None:
    """3-panel RMSE bar chart with 95% CIs for Direct vs MSM+ODE at 15/30/60 min."""
    horizons = sorted(summary.keys())
    direct = [summary[h]["rmse_direct"][0] for h in horizons]
    msm = [summary[h]["rmse_msm"][0] for h in horizons]
    ci_d = [summary[h]["rmse_direct"][1] for h in horizons]
    ci_m = [summary[h]["rmse_msm"][1] for h in horizons]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    titles = [f"{int(h * DT_MIN)} min" for h in horizons]

    for ax, title, d, m, cid, cim in zip(axes, titles, direct, msm, ci_d, ci_m):
        ax.bar(["Direct", "MSM+ODE"], [d, m], yerr=[cid, cim], capsize=4)
        ax.set_title(title)
        ax.set_ylabel("RMSE on ΔG (mg/dL)")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load & preprocess
    df_raw = load_all_patients(n=NUM_PATIENTS, pattern=CSV_PATTERN, sep=CSV_SEP)
    df = preprocess(df_raw, dt_min=DT_MIN)

    # 2) GPS → MSM residualization
    df_w, gps_fits = iptw_gps_multi(df, debug=True)
    wls, df_w = residualize_glucose(df_w)
    print("\n[Weights] IPTW summary:\n", df_w["iptw"].describe())

    # 3) Fit population-level SINDy models
    model_direct = fit_sindy_direct(df_w, dt=DT_MIN)
    model_msm = fit_sindy_msm(df_w, dt=DT_MIN)

    # Optional: print learned equations
    # print("\n--- Direct SINDy (G) ---\n", model_direct.equations()[0])
    # print("\n--- MSM+GPS→SINDy (G_adj) ---\n", model_msm.equations()[0])

    # 4) Evaluate short-horizon rollouts
    summary = evaluate_combined(
        df_w, model_direct, model_msm,
        horizons=HORIZON_STEPS, stride=STRIDE_STEPS, dt=DT_MIN,
        meal_thr=MEAL_THR, bolus_thr=BOLUS_THR
    )
    print_summary(summary)
    plot_rmse_bars(summary, savepath=RMSE_FIG_PATH)
