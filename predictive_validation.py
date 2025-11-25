"""
glucose_msm_sindy_metrics_min.py

Compare two pipelines on HUPA data:
  1) Direct SINDy ODE on raw glucose.
  2) MSM + GPS + residualization + SINDy (our method).

Outputs:
  - Text summary of RMSE/MAE and sign metrics.
  - A 3-panel bar plot (15/30/60 min horizons) saved as rmse_panels.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import pysindy as ps

# --------------------------- configuration ---------------------------

CSV_PATTERN   = "HUPA{:04d}P.csv"   # HUPA0001P.csv, ...
NUM_PATIENTS  = 28
CSV_SEP       = ";"

DT_MIN        = 5.0                 # minutes per sample
HORIZON_STEPS = (3, 6, 12)          # 15, 30, 60 min
STRIDE_STEPS  = 12                  # start windows every 60 min

MEAL_THR      = 5.0                 # g carbs at window start
BOLUS_THR     = 0.5                 # U insulin at window start
RMSE_FIG_PATH = "rmse_panels.png"

METRIC_KEYS = [
    "rmse_direct", "rmse_msm",
    "mae_direct",  "mae_msm",
    "sign_meal_msm", "sign_bolus_msm",
]

# --------------------------- helpers ---------------------------

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))

def mae(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.mean(np.abs(a - b))

def sign_accuracy(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    if len(obs) == 0:
        return np.nan
    return np.mean(np.sign(pred) == np.sign(obs))

def mean_ci(xs):
    """Return (mean, 95% CI half-width, n) ignoring NaNs."""
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return (np.nan, np.nan, 0)
    m  = xs.mean()
    se = xs.std(ddof=1) / np.sqrt(len(xs))
    return (m, 1.96 * se, len(xs))

# --------------------------- data prep ---------------------------

def load_all(n=NUM_PATIENTS, pattern=CSV_PATTERN, sep=CSV_SEP):
    """Load all patient CSVs into one DataFrame."""
    frames = []
    for pid in range(1, n + 1):
        try:
            f = pd.read_csv(pattern.format(pid), sep=sep)
            f["patient_id"] = pid
            frames.append(f)
        except Exception:
            # some IDs may be missing; skip them
            pass
    if not frames:
        raise RuntimeError("No patient files found.")
    return pd.concat(frames, ignore_index=True)

def prep(df, dt=DT_MIN):
    """Sort by patient, smooth key signals, and create 1-step lags."""
    required = [
        "glucose","carb_input","basal_rate","bolus_volume_delivered",
        "steps","heart_rate","patient_id"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    chunks = []
    for _, g in df.sort_values("patient_id").groupby("patient_id", sort=False):
        g = g.reset_index(drop=True).copy()
        g["time"] = np.arange(0, dt * len(g), dt)
        g = g.set_index("time")

        # smooth a few noisy channels
        for c in ["glucose","carb_input","basal_rate","bolus_volume_delivered"]:
            if len(g[c]) >= 7 and g[c].notna().sum() >= 7:
                g[c] = savgol_filter(
                    g[c].interpolate(limit_direction="both"),
                    window_length=7, polyorder=2, mode="interp"
                )

        # one-step lags for glucose, carbs, steps, HR, and insulin
        for c in ["glucose","carb_input","steps","heart_rate",
                  "basal_rate","bolus_volume_delivered"]:
            g[f"{c}_lag1"] = g[c].shift(1)

        chunks.append(g.dropna())

    return pd.concat(chunks).reset_index()

# --------------------------- MSM / GPS ---------------------------

def gps_weights(df,
                treatments=("basal_rate","bolus_volume_delivered"),
                confs=("glucose_lag1","carb_input_lag1","heart_rate_lag1","steps_lag1"),
                eps=1e-3):
    """
    Compute stabilized multivariate GPS weights for (basal, bolus).
    Treatment is log-transformed and modelled as bivariate normal.
    """
    d  = df.copy()
    Y  = np.log(np.clip(d[list(treatments)].to_numpy(), 0, None) + eps)
    X  = sm.add_constant(d[list(confs)], has_constant="add")

    # conditional means μ_i(H)
    mu = np.column_stack([
        sm.OLS(Y[:, j], X).fit().predict(X)
        for j in range(Y.shape[1])
    ])

    # conditional and marginal covariances
    S_c = np.cov((Y - mu).T) + 1e-6*np.eye(Y.shape[1])
    mu_m, S_m = Y.mean(axis=0), np.cov(Y.T) + 1e-6*np.eye(Y.shape[1])

    den = np.array([
        multivariate_normal.pdf(Y[i], mean=mu[i],  cov=S_c)
        for i in range(len(Y))
    ])
    num = multivariate_normal.pdf(Y, mean=mu_m, cov=S_m)

    d["iptw"] = np.clip(num / den, 0.01, 20.0)
    return d

def residualize(df_w):
    """
    Fit a weighted working model G_t ~ (A_t, carbs, lags) and build
    deconfounded glucose G_adj.
    """
    X = sm.add_constant(
        df_w[[
            "basal_rate","bolus_volume_delivered","carb_input",
            "glucose_lag1","heart_rate_lag1","steps_lag1"
        ]],
        has_constant="add"
    )
    fit = sm.WLS(
        df_w["glucose"].to_numpy(),
        X,
        weights=df_w["iptw"].to_numpy()
    ).fit()

    df_w["Ghat"] = fit.predict(X)
    df_w["G_adj"] = df_w["glucose"] - df_w["Ghat"] + df_w["glucose"].mean()
    return fit, df_w

# --------------------------- SINDy / rollouts ---------------------------

def sindy_fit(df, cols, names, dt=DT_MIN):
    """Fit a SINDy model with degree-2 polynomial library."""
    X = df[cols].to_numpy()
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=1e-4),
        feature_library=ps.PolynomialLibrary(degree=2),
        feature_names=names,
    )
    model.fit(X, t=dt)
    return model

def rollout(model, y0, window, dt=DT_MIN):
    """Euler rollout of dG/dt = f(G, carbs, basal, bolus)."""
    y = float(y0)
    for c, ba, bo in window:
        y += dt * model.predict(np.array([[y, c, ba, bo]]))[0, 0]
    return y

# --------------------------- evaluation ---------------------------

def evaluate(df, direct, msm,
             horizons=HORIZON_STEPS, stride=STRIDE_STEPS,
             dt=DT_MIN, meal_thr=MEAL_THR, bolus_thr=BOLUS_THR):
    """
    For each patient and horizon:
      - simulate ΔG with direct and MSM+ODE models,
      - compute RMSE/MAE,
      - compute sign accuracy after meals and boluses.
    Returns a dict: summary[h][metric] = (mean, CI, n).
    """
    per = {h: {k: [] for k in METRIC_KEYS} for h in horizons}

    for _, g in df.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True)
        G     = g["glucose"].to_numpy()
        Gadj  = g["G_adj"].to_numpy()
        Ghat  = g["Ghat"].to_numpy()
        carbs = g["carb_input"].to_numpy()
        basal = g["basal_rate"].to_numpy()
        bolus = g["bolus_volume_delivered"].to_numpy()

        starts = np.arange(0, len(g) - max(horizons), stride)

        for h in horizons:
            d_obs = []
            d_dir = []
            d_msm = []
            meal_o, meal_p = [], []
            bol_o,  bol_p  = [], []

            for t0 in starts:
                t1 = t0 + h
                W = np.column_stack([
                    carbs[t0:t1],
                    basal[t0:t1],
                    bolus[t0:t1],
                ])
                if len(W) < h:
                    continue

                # roll forward
                Gd = rollout(direct, G[t0],    W, dt)
                Ga = rollout(msm,    Gadj[t0], W, dt)

                d_true = G[t1] - G[t0]
                d_d    = Gd - G[t0]
                d_m    = (Ga - Gadj[t0]) + (Ghat[t1] - Ghat[t0])

                d_obs.append(d_true)
                d_dir.append(d_d)
                d_msm.append(d_m)

                if carbs[t0] >= meal_thr:
                    meal_o.append(d_true)
                    meal_p.append(d_m)
                if bolus[t0] >= bolus_thr:
                    bol_o.append(d_true)
                    bol_p.append(d_m)

            if not d_obs:
                continue

            per[h]["rmse_direct"].append(rmse(d_dir, d_obs))
            per[h]["rmse_msm"].append(rmse(d_msm, d_obs))
            per[h]["mae_direct"].append(mae(d_dir, d_obs))
            per[h]["mae_msm"].append(mae(d_msm, d_obs))
            if meal_o:
                per[h]["sign_meal_msm"].append(sign_accuracy(meal_p, meal_o))
            if bol_o:
                per[h]["sign_bolus_msm"].append(sign_accuracy(bol_p, bol_o))

    return {h: {k: mean_ci(v) for k, v in per[h].items()} for h in horizons}

def print_summary(summary):
    """Pretty print metrics for each horizon."""
    for h, row in summary.items():
        print(f"\n=== Horizon {h} steps (~{int(h * DT_MIN)} min) ===")
        for k in METRIC_KEYS:
            m, ci, n = row[k]
            label = k.replace("_", " ").upper()
            if np.isnan(m):
                print(f"  {label}: n=0")
            else:
                unit = " (mg/dL)" if "RMSE" in label or "MAE" in label else ""
                print(f"  {label}: {m:.3f} ± {ci:.3f}{unit}  [n={n}]")

# --------------------------- plotting ---------------------------

def plot_rmse(summary, savepath=RMSE_FIG_PATH):
    """
    Draw a 3-panel bar chart of RMSE (Direct vs MSM+ODE) with:
      - shared y-axis across horizons,
      - enough headroom so the direct 60-min bar is not truncated.
    """
    hs = sorted(summary.keys())
    if not hs:
        print("No horizons to plot.")
        return

    rmse_direct_mean = [summary[h]["rmse_direct"][0] for h in hs]
    rmse_msm_mean    = [summary[h]["rmse_msm"][0]    for h in hs]
    rmse_direct_ci   = [summary[h]["rmse_direct"][1] for h in hs]
    rmse_msm_ci      = [summary[h]["rmse_msm"][1]    for h in hs]

    all_tops = []
    for dv, mv, civ, cim in zip(rmse_direct_mean, rmse_msm_mean,
                                rmse_direct_ci, rmse_msm_ci):
        if not np.isnan(dv) and not np.isnan(civ):
            all_tops.append(dv + civ)
        if not np.isnan(mv) and not np.isnan(cim):
            all_tops.append(mv + cim)

    if not all_tops:
        print("RMSE values are NaN; skipping plot.")
        return

    ymax = max(all_tops) * 1.15
    ymin = 0.0

    fig, axes = plt.subplots(1, len(hs), figsize=(11, 3.4), sharey=True)
    if len(hs) == 1:
        axes = [axes]

    panel_labels   = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    horizon_labels = [f"{int(h * DT_MIN)} min" for h in hs]

    for i, (ax, h, dv, mv, civ, cim) in enumerate(
            zip(axes, hs, rmse_direct_mean, rmse_msm_mean,
                rmse_direct_ci, rmse_msm_ci)):

        ax.bar(["Direct", "MSM+ODE"], [dv, mv], yerr=[civ, cim], capsize=4)
        ax.set_ylim(ymin, ymax)

        label = panel_labels[i] if i < len(panel_labels) else ""
        ax.set_title(f"{label} {horizon_labels[i]}")
        if i == 0:
            ax.set_ylabel("RMSE on ΔG (mg/dL)")

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved RMSE panel figure to {savepath}")

# --------------------------- main ---------------------------

if __name__ == "__main__":
    print("Loading and preparing data...")
    raw = load_all()
    df  = prep(raw, dt=DT_MIN)

    print("Estimating GPS weights and building deconfounded glucose...")
    df_w = gps_weights(df)
    _, df_w = residualize(df_w)
    print("\n[IPTW] weights summary:\n", df_w["iptw"].describe())

    print("\nFitting SINDy models (direct and MSM-adjusted)...")
    direct = sindy_fit(
        df_w,
        ["glucose","carb_input","basal_rate","bolus_volume_delivered"],
        ["G","c","ba","bo"],
        dt=DT_MIN,
    )
    msm = sindy_fit(
        df_w,
        ["G_adj","carb_input","basal_rate","bolus_volume_delivered"],
        ["G_adj","c","ba","bo"],
        dt=DT_MIN,
    )

    print("\nEvaluating horizons...")
    summary = evaluate(
        df_w, direct, msm,
        horizons=HORIZON_STEPS, stride=STRIDE_STEPS,
        dt=DT_MIN, meal_thr=MEAL_THR, bolus_thr=BOLUS_THR,
    )

    print_summary(summary)
    plot_rmse(summary, savepath=RMSE_FIG_PATH)
