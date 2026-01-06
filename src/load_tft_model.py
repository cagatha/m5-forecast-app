# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 09:51:48 2025

@author: agath
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import pytorch_forecasting.data.encoders as pf_enc

#%%
#%%

import numpy as np, torch
np.set_printoptions(edgeitems=2, threshold=10, suppress=True)
torch.set_printoptions(edgeitems=2, threshold=10)

#%%


torch.serialization.add_safe_globals([pf_enc.GroupNormalizer])


#%%

# -----------------------------
# 0) Paths
# -----------------------------
DATA_PATH = r"C:\Users\agath\Desktop\retail project profolio\m5_model_ready.parquet"
CKPT_PATH = r"C:\Users\agath\Desktop\retail project profolio\tft_checkpoints\manual_final.ckpt"




#%%
# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_parquet(DATA_PATH)

# time_idx
all_weeks = np.array(sorted(df["wm_yr_wk"].unique()))
week_to_idx = {w: i for i, w in enumerate(all_weeks)}
df["time_idx"] = df["wm_yr_wk"].map(week_to_idx).astype(int)
df = df.sort_values(["item_id", "store_id", "time_idx"]).reset_index(drop=True)

# categoricals + target
cat_cols = ["item_id", "store_id", "cat_id", "dept_id", "state_id",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")
df["sales_wk"] = df["sales_wk"].astype(float)

# series_id
df["series_id"] = (df["item_id"].astype(str) + "_" + df["store_id"].astype(str)).astype("category")

# -----------------------------
# 2) Feature lists (same as training)
# -----------------------------
static_categoricals = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]

time_varying_known_categoricals = [
    "event_name_1", "event_type_1", "event_name_2", "event_type_2"
]

time_varying_known_reals = [
    "time_idx",
    "year", "month", "weekofyear",
    "week_sin", "week_cos",
    "sell_price",
    "price_vs_item_mean", "price_vs_item_store_mean",
    "price_drop_flag", "promo_depth",
    "is_event", "snap_active",
    "promo_x_event", "promo_x_snap",
    "snap_CA", "snap_TX", "snap_WI",
]

def keep_existing(cols, _df):
    return [c for c in cols if c in _df.columns]

static_categoricals = keep_existing(static_categoricals, df)
time_varying_known_categoricals = keep_existing(time_varying_known_categoricals, df)
time_varying_known_reals = keep_existing(time_varying_known_reals, df)

# -----------------------------
# 3) Select top 500 series  (defines df_tft)
# -----------------------------
top_series = (
    df.groupby("series_id")["sales_wk"]
      .sum()
      .sort_values(ascending=False)
      .head(500)
      .index
)

df_tft = df[df["series_id"].isin(top_series)].copy()
df_tft = df_tft.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
df_tft["series_id"] = df_tft["series_id"].astype("category")

# categorical cleanup (strings + NA)
cat_fix_cols = list(set(["series_id"] + static_categoricals + time_varying_known_categoricals))
for c in cat_fix_cols:
    if c in df_tft.columns:
        df_tft[c] = df_tft[c].astype(str).replace("nan", "NA").fillna("NA").astype("category")

# missing values in known reals (same logic as training)
def add_if_missing(lst, col):
    if col not in lst:
        lst.append(col)

if "sell_price" in df_tft.columns:
    df_tft["sell_price_missing"] = df_tft["sell_price"].isna().astype(int)
    df_tft["sell_price"] = df_tft.groupby("series_id")["sell_price"].ffill().fillna(0.0)
    add_if_missing(time_varying_known_reals, "sell_price_missing")

for col in ["price_vs_item_mean", "price_vs_item_store_mean"]:
    if col in df_tft.columns:
        miss_col = f"{col}_missing"
        df_tft[miss_col] = df_tft[col].isna().astype(int)
        df_tft[col] = df_tft[col].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        add_if_missing(time_varying_known_reals, miss_col)

for c in list(time_varying_known_reals):
    if c in df_tft.columns:
        df_tft[c] = df_tft[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

time_varying_known_reals = sorted(set(time_varying_known_reals))

# -----------------------------
# 4) Rebuild training dataset (encoders)
# -----------------------------
max_prediction_length = 4
max_encoder_length = 52

max_time = df_tft["time_idx"].max()
train_cutoff = max_time - max_prediction_length

training = TimeSeriesDataSet(
    df_tft[df_tft["time_idx"] <= train_cutoff],
    time_idx="time_idx",
    target="sales_wk",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=["sales_wk"],
    target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# -----------------------------
# 5) Load model checkpoint
# -----------------------------
tft_loaded = TemporalFusionTransformer.load_from_checkpoint(
    CKPT_PATH,
    weights_only=False   # <-- important
)
tft_loaded.eval()
print("Loaded TFT checkpoint:", CKPT_PATH)

#%%
print(df_tft.columns)

#%%


#%% =============================
# TFT interpretability (safe + Spyder-friendly)
# =============================
# =============================
# TFT interpretability (Spyder-safe, version-robust)
# =============================
import os
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TimeSeriesDataSet

plt.close("all")
plt.ion()  # interactive on (helps Spyder)

OUT_DIR = r"C:\Users\agath\Desktop\retail project profolio\tft_interpret"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Build a predict dataset from the SAME training dataset (keeps encoders identical)
validation = TimeSeriesDataSet.from_dataset(
    training,
    df_tft,
    predict=True,
    stop_randomization=True
)

val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)

# 2) Raw prediction + return_x
out = tft_loaded.predict(val_loader, mode="raw", return_x=True)

# 3) Unpack predict output robustly (tuple/list or dict-like)
if isinstance(out, (list, tuple)):
    raw, x = out[0], out[1]
elif isinstance(out, dict):
    raw, x = out["output"], out["x"]
else:
    raise TypeError(f"Unexpected predict() return type: {type(out)}")

# 4) Ensure raw is a plain dict
# Some versions return an Output(...) object with attributes like prediction, attention, variables, etc.
if not isinstance(raw, dict):
    if hasattr(raw, "_asdict"):          # namedtuple-like
        raw = raw._asdict()
    elif hasattr(raw, "__dict__"):       # Output object
        raw = raw.__dict__
    else:
        raise TypeError(f"raw output is not dict-like: {type(raw)}")

print("raw type:", type(raw))
print("raw keys:", list(raw.keys()))

# 5) Make sure tensors are on CPU for plotting (optional but often helps)
def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj

raw = {k: _to_cpu(v) for k, v in raw.items()}

# 6) Interpret using KEYWORD arguments to avoid argument-misbinding in some versions
# Try the most common signature first, then fallbacks.
interpretation = None
errors = []

try:
    interpretation = tft_loaded.interpret_output(output=raw, x=x, reduction="sum")
except Exception as e:
    errors.append(("output=raw,x=x", repr(e)))

if interpretation is None:
    try:
        interpretation = tft_loaded.interpret_output(raw_predictions=raw, x=x, reduction="sum")
    except Exception as e:
        errors.append(("raw_predictions=raw,x=x", repr(e)))

if interpretation is None:
    try:
        # some versions don't need x explicitly
        interpretation = tft_loaded.interpret_output(raw, reduction="sum")
    except Exception as e:
        errors.append(("raw only", repr(e)))

if interpretation is None:
    print("interpret_output failed for all known signatures:")
    for where, err in errors:
        print(" -", where, "->", err)
    raise RuntimeError("interpret_output() failed. See printed errors above.")

print("interpretation type:", type(interpretation))
if isinstance(interpretation, dict):
    print("interpretation keys:", list(interpretation.keys()))
else:
    print("interpretation preview:", str(interpretation)[:300])

# 7) Plot interpretation (works when interpretation is dict)
plt.close("all")
tft_loaded.plot_interpretation(interpretation)

# 8) Save all open figures
fig_nums = plt.get_fignums()
print("Matplotlib figures open:", fig_nums)

for i, fnum in enumerate(fig_nums, start=1):
    fig = plt.figure(fnum)
    fig_path = os.path.join(OUT_DIR, f"tft_interpret_{i}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print("Saved:", fig_path)

plt.show(block=False)
plt.pause(0.1)
plt.close("all")



#%%
# -----------------------------
# 6) Metrics + folds + evaluation
# -----------------------------
def rmsse(y_true, y_pred, y_train, eps=1e-8):
    y_train = np.asarray(y_train, dtype=float)
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    diffs = np.diff(y_train)
    scale = np.mean(diffs**2)
    if scale < eps:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred)**2) / scale)

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])

def make_folds_time_idx(unique_time_idx, horizon=4, min_train=104, step=4, max_folds=10):
    t = np.array(sorted(unique_time_idx))
    folds = []
    for i in range(min_train - 1, len(t) - horizon, step):
        train_end = t[i]
        test_times = t[i+1:i+1+horizon]
        folds.append((int(train_end), test_times.astype(int)))
    if (max_folds is not None) and (len(folds) > max_folds):
        folds = folds[-max_folds:]
    return folds

#%%

def predict_fold_long(
    tft_model,
    base_training_dataset,
    df_all,
    train_end,
    test_times,
    batch_size=256,
):
    fold_df = df_all[(df_all["time_idx"] <= train_end) | (df_all["time_idx"].isin(test_times))].copy()
    fold_df = fold_df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

    fold_pred_ds = TimeSeriesDataSet.from_dataset(
        base_training_dataset,
        fold_df,
        predict=True,
        stop_randomization=True
    )
    fold_loader = fold_pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # ---- version-robust predict() unpacking ----
    out = tft_model.predict(fold_loader, return_x=True)
    if isinstance(out, (list, tuple)):
        pred = out[0]
        x = out[1]
    else:
        raise TypeError(f"Unexpected predict() return type: {type(out)}")

    pred_np = pred.detach().cpu().numpy()
    decoder_time_idx = x["decoder_time_idx"].detach().cpu().numpy()
    group_enc = x["groups"][:, 0].detach().cpu().numpy()

    # encoder mapping
    enc = fold_pred_ds._categorical_encoders.get("series_id", None)
    if enc is None:
        enc = base_training_dataset._categorical_encoders["series_id"]
    series_ids = enc.inverse_transform(group_enc)

    rows = []
    N, H = pred_np.shape
    for i in range(N):
        sid = str(series_ids[i])
        for h in range(H):
            rows.append((sid, int(decoder_time_idx[i, h]), float(pred_np[i, h]), int(train_end)))

    return pd.DataFrame(rows, columns=["series_id", "time_idx", "y_pred", "train_end"])

#%%

def score_fold_rmsse(df_all, pred_long, train_end, target_col="sales_wk"):
    actual = df_all[["series_id", "time_idx", target_col]].rename(columns={target_col: "y_true"})
    merged = pred_long.merge(actual, on=["series_id", "time_idx"], how="left")

    train_hist = df_all[df_all["time_idx"] <= train_end].sort_values(["series_id", "time_idx"])
    y_train_map = train_hist.groupby("series_id")["sales_wk"].apply(lambda s: s.values)
    w_map = train_hist.groupby("series_id")["sales_wk"].sum()

    g = merged.sort_values(["series_id", "time_idx"]).groupby("series_id")
    y_true_map = g["y_true"].apply(lambda s: s.values)
    y_pred_map = g["y_pred"].apply(lambda s: s.values)

    out = pd.DataFrame({
        "y_train": y_train_map,
        "y_true": y_true_map,
        "y_pred": y_pred_map,
        "weight": w_map
    }).dropna(subset=["y_true", "y_pred", "y_train"])

    out["rmsse"] = out.apply(lambda r: rmsse(r["y_true"], r["y_pred"], r["y_train"]), axis=1)
    out = out.reset_index().rename(columns={"index": "series_id"})
    out["train_end"] = int(train_end)
    return out[["series_id", "train_end", "rmsse", "weight"]]

# folds
folds = make_folds_time_idx(
    df_tft["time_idx"].unique(),
    horizon=max_prediction_length,
    min_train=104,
    step=4,
    max_folds=10
)
print("num folds:", len(folds), "example:", folds[0])

# evaluate
batch_size = 256
all_fold_scores = []
fold_summaries = []
all_pred_long = []

for train_end, test_times in folds:
    pred_long = predict_fold_long(
        tft_model=tft_loaded,
        base_training_dataset=training,
        df_all=df_tft,
        train_end=train_end,
        test_times=test_times,
        batch_size=batch_size
    )
    
    all_pred_long.append(pred_long)
    fold_scores = score_fold_rmsse(df_tft, pred_long, train_end)
    fold_mean = fold_scores["rmsse"].mean()
    fold_weighted = weighted_mean(fold_scores["rmsse"], fold_scores["weight"])

    fold_summaries.append({
        "train_end_time_idx": int(train_end),
        "mean_rmsse": float(fold_mean),
        "weighted_rmsse": float(fold_weighted),
        "num_series_scored": int(fold_scores["rmsse"].notna().sum())
    })
    all_fold_scores.append(fold_scores)

tft_fold_summaries_df = pd.DataFrame(fold_summaries)
tft_series_scores_df = pd.concat(all_fold_scores, ignore_index=True) if all_fold_scores else pd.DataFrame()
tft_predictions_long = pd.concat(all_pred_long, ignore_index=True)

tft_overall = {
    "mean_rmsse_avg_over_folds": float(tft_fold_summaries_df["mean_rmsse"].mean()),
    "weighted_rmsse_avg_over_folds": float(tft_fold_summaries_df["weighted_rmsse"].mean()),
    "num_folds": int(len(tft_fold_summaries_df)),
}
print("\nTFT fold summaries (loaded):")
print(tft_fold_summaries_df)
print("\nTFT overall (loaded):")
print(tft_overall)


# add true values to my predictions

# --- add TRUE values (and optional calendar columns) from df_tft ---
actual_cols = ["series_id", "time_idx", "sales_wk"]
extra_cols = []
for c in ["wm_yr_wk", "date"]:   # optional, only if exist
    if c in df_tft.columns:
        extra_cols.append(c)

actual_long = (
    df_tft[actual_cols + extra_cols]
      .rename(columns={"sales_wk": "y_true"})
      .copy()
)

tft_predictions_long = tft_predictions_long.merge(
    actual_long,
    on=["series_id", "time_idx"],
    how="left"
)

# --- add horizon index within each fold (0..H-1) for plotting ---
tft_predictions_long = tft_predictions_long.sort_values(
    ["series_id", "train_end", "time_idx"]
).reset_index(drop=True)

tft_predictions_long["horizon"] = tft_predictions_long.groupby(
    ["series_id", "train_end"]
).cumcount()

# tft_predictions_long.to_csv(
#     r"C:\Users\agath\Desktop\retail project profolio\tft_results\tft_predictions_long_with_true.csv",
#     index=False
# )

#%%

# combinging the prediction results with promo details

df_tft["series_id"] = df_tft["series_id"].astype(str)
feature_cols = [
    "series_id",
    "time_idx",
    "wm_yr_wk",
    "date",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "promo_depth",
    "price_drop_flag",
    "sell_price",
    "snap_active",
    "is_event"
]

features = df_tft[feature_cols].copy()

preds_enriched = tft_predictions_long.merge(
    features,
    on=["series_id", "time_idx"],
    how="left"
)

preds_enriched.to_csv(
    "app_data/tft_predictions_enriched.csv",
    index=False
)

#%%
# optional: save outputs
out_dir = r"C:\Users\agath\Desktop\retail project profolio\tft_results"
os.makedirs(out_dir, exist_ok=True)
tft_fold_summaries_df.to_csv(os.path.join(out_dir, "tft_fold_summaries_loaded.csv"), index=False)
tft_series_scores_df.to_csv(os.path.join(out_dir, "tft_series_scores_loaded.csv"), index=False)
print("Saved CSVs to:", out_dir)

#%%

