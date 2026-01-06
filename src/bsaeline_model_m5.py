# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 09:50:25 2025

@author: agath
"""


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

#%%


#%%
# this is my merge dataset -- calendar, sales, and prices
df = pd.read_parquet("C:/Users/agath/Desktop/retail project profolio/m5_model_ready.parquet")

#%%
print(df.columns)

#%%

sample_df=df.head(50000)

#%%
# start with baseline model first

# RMSE (均方根誤差): 計算預測值與實際值之間差異的標準差，反映誤差的大小
# Scaling (縮放): 將計算出的 RMSE 除以同一個時間序列在訓練期間的「一步預測」的 RMSE，這樣就消除了數據的尺度影響
# RMSSE: 結果表示該預測模型相較於最簡單的「猜測」（即前一個時間點的值就是下一個預測值）的誤差有多大，如果 RMSSE 小於 1，表示模型優於簡單預測
# 比較不同序列： 由於消除了尺度問題，RMSSE 允許在不同規模和特性的時間序列數據中比較模型的表現
# rmsse = sqaure root of (forecast error/ scale) --> 我的預測錯誤除以這個 SKU 平常自己上下波動的程度

def rmsse(y_true, y_pred, y_train, eps=1e-8):
    """
    y_train: history used to define scale (denominator)
    y_true : actual values in forecast window
    y_pred : predicted values in forecast window
    """
    diffs = np.diff(y_train)
    scale = np.mean(diffs**2)

    # if series is flat / too sparse -> avoid division by 0
    if scale < eps:
        return np.nan

    return np.sqrt(np.mean((y_true - y_pred)**2) / scale)

#%%
# start at 52 weeks ago (same week last year and forecast next 4 weeks)

def seasonal_naive(y_train, season_length=52, horizon=4):
    """
    Forecast next horizon steps using values from the same season last year:
    y_hat[t+h] = y[t+h-season_length]
    """
    start = len(y_train) - season_length
    return y_train[start:start + horizon]


# rolling - origin evaluation (one series)
# evalue the baseline on one series across many folds

# min_train_weeks → 最少訓練歷史
# season_length → 季節回看距離
# horizon → 一次預測多遠
# step → 隔多久做一次「假裝今天是現在」的實驗
# for each fold compute 1 RMSSE, and take the mean across all folds

def rolling_backtest_series(y, horizon=4, season_length=52, min_train_weeks=104, step=4):

    if len(y) < min_train_weeks + season_length + horizon:
        return np.nan

    errors = []

    # cutoff is the "origin" where forecast starts
    for cutoff in range(min_train_weeks, len(y) - horizon, step):
        train = y[:cutoff]
        test = y[cutoff:cutoff + horizon]

        # seasonal naive prediction
        pred = seasonal_naive(train, season_length=season_length, horizon=horizon)

        err = rmsse(test, pred, train)
        if not np.isnan(err):
            errors.append(err)

    return np.mean(errors) if errors else np.nan




# weighte means
# values = per-series RMSSE
# weights (sales volume or revune)
# high sales items should matter more in the metric
# excludes series with 0 sales across all weeks (weights > 0)

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    if mask.sum() == 0:
        return np.nan

    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])



def run_seasonal_naive_baseline(
    df,
    keys=("item_id", "store_id"),
    time_col="wm_yr_wk",
    target_col="sales_wk",
    price_col="sell_price",
    horizon=4,
    season_length=52,
    min_train_weeks=104,
    step=4,
    weight_type="volume"  # "volume" or "revenue"
):
    results = []

    # Ensure correct sort (important!)
    df_sorted = df.sort_values(list(keys) + [time_col])

    for key_vals, g in df_sorted.groupby(list(keys)):
        y = g[target_col].values.astype(float)

        # series-level backtest score
        score = rolling_backtest_series(
            y,
            horizon=horizon,
            season_length=season_length,
            min_train_weeks=min_train_weeks,
            step=step
        )

        # weight choice
        if weight_type == "volume":
            w = np.nansum(y)
        elif weight_type == "revenue":
            # revenue weight = sum(sales * price)
            if price_col not in g.columns:
                raise ValueError(f"price_col='{price_col}' not found in df.")
            w = np.nansum(g[target_col].values.astype(float) * g[price_col].values.astype(float))
        else:
            raise ValueError("weight_type must be 'volume' or 'revenue'")

        # unpack key values
        row = {k: v for k, v in zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,))}
        row.update({"rmsse": score, "weight": w, "n_weeks": len(y)})
        results.append(row)

    res = pd.DataFrame(results)

    summary = {
        "mean_rmsse": res["rmsse"].mean(),
        "weighted_rmsse": weighted_mean(res["rmsse"], res["weight"]),
        "num_series": len(res),
        "num_scored_series": int(res["rmsse"].notna().sum())
    }

    return res, summary



baseline_results, baseline_summary = run_seasonal_naive_baseline(
    df,
    keys=("item_id", "store_id"),
    time_col="wm_yr_wk",
    target_col="sales_wk",
    price_col="sell_price",
    horizon=4,
    season_length=52,
    min_train_weeks=104,
    step=4,              # set to 4 if you want fewer folds/faster
    weight_type="volume"  # or "volume"
)

#%%
# graphs for the baseline model

# right tail -- many series have hight RMSSE
baseline_results["volume_bucket"] = pd.qcut(
    baseline_results["weight"],
    q=5,
    labels=["Very Low", "Low", "Mid", "High", "Very High"]
)

baseline_results.groupby("volume_bucket")["rmsse"].mean()



# negative correlation, -0.038 
# higher weights --> smaller rmsse 
# seasonal patters are more stable for high-volume items

baseline_results[["rmsse", "weight"]].corr(method="spearman")

plt.figure()
plt.hist(
    baseline_results["rmsse"].dropna(),
    bins=50,
    edgecolor="black"
)
plt.axvline(baseline_results["rmsse"].mean(), linestyle="--")
plt.xlabel("RMSSE")
plt.ylabel("Number of series")
plt.title("Distribution of RMSSE (Seasonal Naïve Baseline)")
plt.show()


mean_rmsse = baseline_results["rmsse"].mean()
weighted_rmsse = weighted_mean(
    baseline_results["rmsse"],
    baseline_results["weight"]
)

plt.figure()
plt.bar(
    ["Mean RMSSE", "Weighted RMSSE"],
    [mean_rmsse, weighted_rmsse]
)
plt.ylabel("RMSSE")
plt.title("Baseline Performance Summary")
plt.show()


df_plot = baseline_results.copy()

df_plot["weight_bin"] = pd.qcut(
    df_plot["weight"],
    q=5,
    labels=["Very Low", "Low", "Mid", "High", "Very High"]
)

plt.figure()
df_plot.boxplot(column="rmsse", by="weight_bin")
plt.xlabel("Series Importance (Quantile)")
plt.ylabel("RMSSE")
plt.title("RMSSE by Sales Importance")
plt.suptitle("")
plt.show()

#%%
# global model

# define multiple "pretend today" time points
# if today is week 104, train on the past 2 years (103 weeks) and forecast weeks 105-108
# for step 4, we are moving the cutoff forward by 4 weeks

# fold 2 is to train on index 107 weeks (108 totals) and test on index 108 to 111 weeks
# train always start at the begining and ends later and later
# test is always the next 4 weeks and cuto-off moves every 4 weeks

def make_folds(unique_weeks, horizon=4, min_train_weeks=104, step=4, max_folds=None):
    unique_weeks = np.array(sorted(unique_weeks))
    folds = []
    for idx in range(min_train_weeks - 1, len(unique_weeks) - horizon, step):
        train_end_week = unique_weeks[idx]
        test_weeks = unique_weeks[idx + 1: idx + 1 + horizon]
        folds.append((train_end_week, test_weeks))

    if max_folds is not None and len(folds) > max_folds:
        folds = folds[-max_folds:]  # keep most recent folds (faster, very reasonable)

    return folds


#%%
# this function decides how we learn the model
# modeling funrcion for one fold
# train_df = all rows where wm_yr_wk < - train end week 
# test_df = all rows where wm_yr_wk is in the next 4 weeks
# treate category columsn as categories

def train_predict_global_lgb_fold(
    df,
    train_end_week,
    test_weeks,
    features,
    categorical_features,
    target_col="sales_wk",
    time_col="wm_yr_wk",
    lgb_params=None,
    num_boost_round=300
):
    if lgb_params is None:
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
        }

    train_df = df[df[time_col] <= train_end_week].copy()
    test_df  = df[df[time_col].isin(test_weeks)].copy()

    # LightGBM categorical handling
    for c in categorical_features:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")
            test_df[c]  = test_df[c].astype("category")

    dtrain = lgb.Dataset(
        train_df[features],
        label=train_df[target_col],
        categorical_feature=categorical_features,
        free_raw_data=False
    )

# train one global model on train_df, our label is sales_wk

    model = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round)

# test is making the predictionso n the test_df
    test_df["pred"] = model.predict(test_df[features])
    return train_df, test_df




#%%
# compute per-series RMSSE for one fold

# for each fold, one RMSSE per (item, store) series
# plus the series weight

def score_fold_per_series(
    train_df,
    test_pred_df,
    keys=("item_id", "store_id"),
    target_col="sales_wk",
    price_col="sell_price",
    weight_type="volume"
):
    # series weights from training history (consistent / leakage-safe)
    if weight_type == "volume":
        w = train_df.groupby(list(keys))[target_col].sum().rename("weight")
    elif weight_type == "revenue":
        w = (train_df[target_col] * train_df[price_col]).groupby(
            [train_df[keys[0]], train_df[keys[1]]]
        ).sum().rename("weight")
    else:
        raise ValueError("weight_type must be 'volume' or 'revenue'")

    # y_train arrays (for RMSSE denominator)
    y_train_map = train_df.groupby(list(keys))[target_col].apply(lambda s: s.values).rename("y_train")

    # fold y_true / y_pred arrays
    g = test_pred_df.groupby(list(keys))
    y_true_map = g[target_col].apply(lambda s: s.values).rename("y_true")
    y_pred_map = g["pred"].apply(lambda s: s.values).rename("y_pred")

    merged = pd.concat([y_train_map, y_true_map, y_pred_map, w], axis=1)

    # compute rmsse per series
    merged["rmsse"] = merged.apply(lambda r: rmsse(r["y_true"], r["y_pred"], r["y_train"]), axis=1)

    out = merged.reset_index()[list(keys) + ["rmsse", "weight"]]
    return out

#%%
# run all folds and summarize

def run_global_lgb_rolling(
    df,
    features,
    categorical_features,
    horizon=4,
    min_train_weeks=104,
    step=4,
    max_folds=10,              # start small for speed
    weight_type="volume",
    target_col="sales_wk",
    time_col="wm_yr_wk",
    price_col="sell_price",
    lgb_params=None,
    num_boost_round=300
):
    df = df.sort_values([time_col, "item_id", "store_id"]).copy()

    folds = make_folds(df[time_col].unique(), horizon, min_train_weeks, step, max_folds)

    fold_summaries = []
    all_series_scores = []

    for train_end_week, test_weeks in folds:
        train_df, test_pred_df = train_predict_global_lgb_fold(
            df=df,
            train_end_week=train_end_week,
            test_weeks=test_weeks,
            features=features,
            categorical_features=categorical_features,
            target_col=target_col,
            time_col=time_col,
            lgb_params=lgb_params,
            num_boost_round=num_boost_round
        )

        series_scores = score_fold_per_series(
            train_df=train_df,
            test_pred_df=test_pred_df,
            keys=("item_id", "store_id"),
            target_col=target_col,
            price_col=price_col,
            weight_type=weight_type
        )

        fold_mean = series_scores["rmsse"].mean()
        fold_weighted = weighted_mean(series_scores["rmsse"], series_scores["weight"])

        fold_summaries.append({
            "train_end_week": int(train_end_week),
            "mean_rmsse": float(fold_mean),
            "weighted_rmsse": float(fold_weighted),
            "num_series_scored": int(series_scores["rmsse"].notna().sum())
        })

        series_scores["train_end_week"] = int(train_end_week)
        all_series_scores.append(series_scores)

    fold_summaries_df = pd.DataFrame(fold_summaries)
    all_series_scores_df = pd.concat(all_series_scores, ignore_index=True) if all_series_scores else pd.DataFrame()

    overall = {
        "mean_rmsse_avg_over_folds": float(fold_summaries_df["mean_rmsse"].mean()),
        "weighted_rmsse_avg_over_folds": float(fold_summaries_df["weighted_rmsse"].mean()),
        "num_folds": int(len(fold_summaries_df))
    }

    return fold_summaries_df, all_series_scores_df, overall


#%%

FEATURES = [
    "item_id", "store_id", "cat_id", "dept_id", "state_id",
    "week_sin", "week_cos", "year",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_4", "roll_mean_8", "roll_mean_12",
    "roll_std_4", "roll_std_8", "roll_std_12",
    "cv_4wk", "cv_8wk", "cv_12wk",
    "sell_price", "price_drop_flag", "promo_depth",
    "is_event", "snap_active", "promo_x_event", "promo_x_snap"
]

CATEGORICALS = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]


fold_summaries_df, series_scores_df, overall = run_global_lgb_rolling(
    df=df,
    features=FEATURES,
    categorical_features=CATEGORICALS,
    horizon=4,
    min_train_weeks=104,
    step=4,
    max_folds=10,
    weight_type="volume",   # keep consistent with baseline first
    num_boost_round=300
)
#%%
# algign global model outputs

# 1) global_overall_df: one-row summary (same style as tft_overall / baseline_overall)
global_overall_df = pd.DataFrame([{
    "model": "Global",
    "mean_rmsse_avg_over_folds": overall["mean_rmsse_avg_over_folds"],
    "weighted_rmsse_avg_over_folds": overall["weighted_rmsse_avg_over_folds"],
    "num_folds": overall["num_folds"]
}])

# 2) global_series_df: one row per (item_id, store_id), average RMSSE across folds
global_series_df = (
    series_scores_df
    .groupby(["item_id", "store_id"], as_index=False)
    .agg(
        rmsse=("rmsse", "mean"),
        # weight is computed from each fold's training history; keep mean (or last) for stability
        weight=("weight", "mean"),
        n_folds=("train_end_week", "nunique"),
    )
)

# 3) add series_id (so you can match TFT’s series_id if needed)
global_series_df["series_id"] = (
    global_series_df["item_id"].astype(str) + "_" + global_series_df["store_id"].astype(str)
)

global_series_df["model"] = "Global"

# (optional) reorder columns to match your baseline_series_df style
global_series_df = global_series_df[
    ["series_id", "item_id", "store_id", "rmsse", "weight", "n_folds", "model"]
]

print(global_overall_df.head())
print(global_series_df.head(), global_series_df.shape)



# 4) save
out_dir = r"C:\Users\agath\Desktop\retail project profolio\baseline_results"
os.makedirs(out_dir, exist_ok=True)

global_overall_df.to_csv(os.path.join(out_dir, "global_overall_results.csv"), index=False)
global_series_df.to_csv(os.path.join(out_dir, "global_series_results.csv"), index=False)


#%%

# Metrics
baseline_mean = baseline_summary["mean_rmsse"]
baseline_weighted = baseline_summary["weighted_rmsse"]

global_mean = fold_summaries_df["mean_rmsse"].mean()
global_weighted = fold_summaries_df["weighted_rmsse"].mean()

labels = [
    "Mean RMSSE\n(Baseline)",
    "Mean RMSSE\n(Global)",
    "Weighted RMSSE\n(Baseline)",
    "Weighted RMSSE\n(Global)"
]

values = [
    baseline_mean,
    global_mean,
    baseline_weighted,
    global_weighted
]

colors = [
    "#4C72B0",  # Baseline - blue
    "#DD8452",  # Global - orange
    "#4C72B0",  # Baseline - blue
    "#DD8452"   # Global - orange
]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=colors)

plt.ylabel("RMSSE")
plt.title("Baseline vs Global Model Performance")

# Rotate x labels
plt.xticks(rotation=20, ha="right")

plt.tight_layout()
plt.show()


#%%

global_series_rmsse = (
    series_scores_df
    .groupby(["item_id", "store_id"])["rmsse"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

plt.hist(
    baseline_results["rmsse"].dropna(),
    bins=50,
    alpha=0.6,
    label="Seasonal Naïve"
)

plt.hist(
    global_series_rmsse["rmsse"].dropna(),
    bins=50,
    alpha=0.6,
    label="Global LightGBM"
)

plt.xlabel("RMSSE")
plt.ylabel("Number of series")
plt.title("RMSSE Distribution: Baseline vs Global Model")
plt.legend()
plt.show()

#%%
# --- 0) Define orders (make sure these exist BEFORE use)
weight_order = ["Very Low", "Low", "Mid", "High", "Very High"]
model_order = ["Baseline", "Global"]

# --- 1) Baseline series table (already has rmsse + weight)
baseline_plot = baseline_results[["item_id", "store_id", "rmsse", "weight"]].copy()
baseline_plot["model"] = "Baseline"

# --- 2) Global series table: must have one rmsse per (item_id, store_id)
# global_series_rmsse should have columns: item_id, store_id, rmsse
global_plot = global_series_rmsse.merge(
    baseline_results[["item_id", "store_id", "weight"]],
    on=["item_id", "store_id"],
    how="left",
    validate="one_to_one"  # helps catch duplicates silently causing issues
)
global_plot = global_plot[["item_id", "store_id", "rmsse", "weight"]].copy()
global_plot["model"] = "Global"

# --- 3) Combine
plot_df = pd.concat([baseline_plot, global_plot], ignore_index=True)

# --- 4) Build bins using BASELINE weights only (consistent interpretation)
# (This is the key fix!)
bin_source = baseline_plot["weight"].replace([np.inf, -np.inf], np.nan).dropna()

plot_df["weight_bin"] = pd.cut(
    plot_df["weight"],
    bins=pd.qcut(bin_source, q=5, retbins=True, duplicates="drop")[1],
    labels=weight_order,
    include_lowest=True
)

# Drop rows with missing bins or rmsse
plot_df = plot_df.dropna(subset=["weight_bin", "rmsse"])

# --- 5) Enforce ordering
plot_df["weight_bin"] = pd.Categorical(plot_df["weight_bin"], categories=weight_order, ordered=True)
plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)

# --- 6) Create positions + data, skipping empty groups safely
positions, data, labels = [], [], []
pos, gap = 1, 1

for wb in weight_order:
    for m in model_order:
        vals = plot_df.loc[
            (plot_df["weight_bin"] == wb) & (plot_df["model"] == m),
            "rmsse"
        ].dropna().values

        if len(vals) == 0:
            # keep spacing but skip empty box to avoid matplotlib errors
            pos += 1
            continue

        data.append(vals)
        positions.append(pos)
        labels.append(f"{wb}\n{m}")
        pos += 1
    pos += gap

# --- 7) Plot
plt.figure(figsize=(12, 6))
bp = plt.boxplot(
    data,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    showfliers=False
)

# Color: alternate baseline/global
colors = {"Baseline": "#4C72B0", "Global": "#DD8452"}
for i, box in enumerate(bp["boxes"]):
    # because we append in order wb -> Baseline then Global, alternate by i
    box.set_facecolor(colors[model_order[i % 2]])

plt.xticks(positions, labels, rotation=25, ha="right")
plt.xlabel("Sales Importance (from baseline weights) & Model")
plt.ylabel("RMSSE")
plt.title("RMSSE by Sales Importance: Baseline vs Global Model")


plt.tight_layout()
plt.show()

#%%

baseline_series_df = baseline_results.copy()

# 確保欄位命名一致（和 TFT 對齊）
baseline_series_df = baseline_series_df.rename(columns={
    "rmsse": "rmsse",
    "series_id": "series_id"
})

baseline_series_df["model"] = "baseline"

print("baseline_series_df shape:", baseline_series_df.shape)
baseline_series_df.head()

print(baseline_series_df.columns)
#%%

baseline_series_df["series_id"] = (
    baseline_series_df["item_id"].astype(str)
    + "_"
    + baseline_series_df["store_id"].astype(str)
)

baseline_overall_df = pd.DataFrame([{
    "model": "baseline",
    "mean_rmsse": baseline_series_df["rmsse"].mean(),
    "weighted_rmsse": (
        (baseline_series_df["rmsse"] * baseline_series_df["weight"]).sum()
        / baseline_series_df["weight"].sum()
    ),
    "num_series": len(baseline_series_df)
}])


#%%


OUT_DIR = r"C:\Users\agath\Desktop\retail project profolio\baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)

baseline_series_df.to_csv(
    os.path.join(OUT_DIR, "baseline_series_results.csv"),
    index=False
)

baseline_overall_df.to_csv(
    os.path.join(OUT_DIR, "baseline_overall_results.csv"),
    index=False
)

print("Baseline results saved to:", OUT_DIR)