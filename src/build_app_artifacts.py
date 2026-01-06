# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:12:06 2026

@author: agath
"""

import pandas as pd
import os


#%%

BASE_DIR = r"C:/Users/agath/Desktop/retail project profolio/app_data"
OUT_DIR = os.path.join(BASE_DIR, "app_data")
os.makedirs(OUT_DIR, exist_ok=True)

#%%

# load results

baseline_series = pd.read_csv(os.path.join(BASE_DIR, "baseline_series_results.csv"))
global_series   = pd.read_csv(os.path.join(BASE_DIR, "global_series_results.csv"))
tft_series      = pd.read_csv(os.path.join(BASE_DIR, "tft_series_scores_loaded.csv"))

baseline_overall = pd.read_csv(os.path.join(BASE_DIR, "baseline_overall_results.csv"))
global_overall   = pd.read_csv(os.path.join(BASE_DIR, "global_overall_results.csv"))
tft_overall      = pd.read_csv(os.path.join(BASE_DIR, "tft_fold_summaries_loaded.csv"))


#%%

# algin series id

baseline_series["series_id"] = baseline_series["series_id"].astype(str)

global_series["series_id"] = global_series["series_id"].astype(str)

tft_series["series_id"] = tft_series["series_id"].astype(str)

#%%
# keep only top 500 series

top_series = set(tft_series["series_id"].unique())

baseline_series = baseline_series[baseline_series["series_id"].isin(top_series)]
global_series   = global_series[global_series["series_id"].isin(top_series)]


#%% 
# for tft series, there are results for 10 foles for each series
# we can use the average of the 10 folds and do the compaison


tft_series_agg = (
    tft_series
    .groupby("series_id", as_index=False)
    .agg(
        tft_rmsse=("rmsse", "mean"),
        tft_rmsse_std=("rmsse", "std"),
        n_folds=("rmsse", "count"),
    )
)

# tft_series_agg.to_csv(
#     r"C:\Users\agath\Desktop\retail project profolio\tft_results\tft_series_agg.csv",
#     index=False
# )



#%%


series_compare = (
    baseline_series[["series_id", "rmsse"]].rename(columns={"rmsse": "baseline_rmsse"})
    .merge(
        global_series[["series_id", "rmsse"]].rename(columns={"rmsse": "global_rmsse"}),
        on="series_id", how="inner"
    )
    .merge(
        tft_series_agg[["series_id", "tft_rmsse"]],   # <- 這裡改成聚合後的
        on="series_id", how="inner"
    )
)

series_compare["tft_vs_global_gain"] = series_compare["global_rmsse"] - series_compare["tft_rmsse"]

# series_compare.to_csv(
#     r"C:\Users\agath\Desktop\retail project profolio\tft_results\series_compare.csv",
#     index=False
# )


series_compare.to_csv(os.path.join(OUT_DIR, "series_model_comparison.csv"), index=False)

baseline_overall.to_csv(os.path.join(OUT_DIR, "baseline_overall.csv"), index=False)
global_overall.to_csv(os.path.join(OUT_DIR, "global_overall.csv"), index=False)
tft_overall.to_csv(os.path.join(OUT_DIR, "tft_overall.csv"), index=False)

#%%

# buid app-level overall metrics


baseline_overall_app = baseline_overall.copy()
baseline_overall_app["model"] = "Baseline"

# 保險：確保欄位命名一致
baseline_overall_app = baseline_overall_app.rename(
    columns={
        "mean_rmsse_avg_over_folds": "mean_rmsse",
        "weighted_rmsse_avg_over_folds": "weighted_rmsse",
    }
)

# baseline_overall_app["n_folds"] = baseline_overall_app.get("num_folds", 1)

# --- global ---
global_overall_app = global_overall.copy()
global_overall_app["model"] = "Global"

global_overall_app = global_overall_app.rename(
    columns={
        "mean_rmsse_avg_over_folds": "mean_rmsse",
        "weighted_rmsse_avg_over_folds": "weighted_rmsse",
    }
)

# global_overall_app["n_folds"] = global_overall_app.get("num_folds", 1)

# --- TFT (aggregate folds) ---
tft_overall_app = (
    tft_overall
    .agg(
        mean_rmsse=("mean_rmsse", "mean"),
        weighted_rmsse=("weighted_rmsse", "mean"),
        # n_folds=("mean_rmsse", "count"),
    )
    .T
)

tft_overall_app["model"] = "TFT"

# --- concat all ---
overall_app = pd.concat(
    [
        baseline_overall_app[["model", "mean_rmsse", "weighted_rmsse"]],
        global_overall_app[["model", "mean_rmsse", "weighted_rmsse"]],
        tft_overall_app[["model", "mean_rmsse", "weighted_rmsse"]],
    ],
    ignore_index=True
)

# save
overall_app.to_csv(os.path.join(OUT_DIR, "app_overall_metrics.csv"), index=False)


#%%

# Optional: add helper columns for app
# =========================

series_compare["best_model"] = series_compare[
    ["baseline_rmsse", "global_rmsse", "tft_rmsse"]
].idxmin(axis=1)

series_compare["best_model"] = series_compare["best_model"].replace({
    "baseline_rmsse": "Baseline",
    "global_rmsse": "Global",
    "tft_rmsse": "TFT",
})

series_compare["tft_beats_global"] = series_compare["tft_vs_global_gain"] > 0

# overwrite (app-ready)
series_compare.to_csv(
    os.path.join(OUT_DIR, "series_model_comparison.csv"),
    index=False
)
