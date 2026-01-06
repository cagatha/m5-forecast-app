# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 12:26:48 2025

@author: agath
"""

import torch
import lightning as L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# this is my merge dataset -- calendar, sales, and prices
df = pd.read_parquet("C:/Users/agath/Desktop/retail project profolio/m5_model_ready.parquet")

sample_df=df.head(50000)


print(df.columns)

#%%
# prepare data

all_weeks = np.array(sorted(df["wm_yr_wk"].unique()))
week_to_idx = {w: i for i, w in enumerate(all_weeks)}

# create time idx
df["time_idx"] = df["wm_yr_wk"].map(week_to_idx).astype(int)
df = df.sort_values(["item_id", "store_id", "time_idx"]).reset_index(drop=True)


# 
cat_cols = ["item_id", "store_id", "cat_id", "dept_id", "state_id",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

# 5) Ensure target is float
df["sales_wk"] = df["sales_wk"].astype(float)



#%%

# define TFT features

# static features - features don't change over time
static_categoricals = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]


# known feuture features

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



def keep_existing(cols, df):
    return [c for c in cols if c in df.columns]

static_categoricals = keep_existing(static_categoricals, df)
time_varying_known_categoricals = keep_existing(time_varying_known_categoricals, df)
time_varying_known_reals = keep_existing(time_varying_known_reals, df)

#%%

# create a combined id : series id + group id
df["series_id"] = (df["item_id"].astype(str) + "_" + df["store_id"].astype(str)).astype("category")


#%%
# define fold cut points
# horizon is 4 --> predicting the next 4 weeks
# use 1 year of history (Weeks = 52)
# we can train up to 273-4=269 weeks

# encoder -- how much past history the model look at
# decoder -- how many future steps it predicts (4 weeks)

# for weekly retail demand it may not need full year to learn seasonality
# recent 6 months may contain enough for promotions, price changes, and demand regime shifts

# we select the top 500 series otherwise the model will take forever to run
# the top 500 is the 500 series with the top sales
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

#%%
# we have missing values in sell prices, we create a column that indicates the missingness of the prices
# fill the sell prices with the previous week price, if not historical price then set 0

# fox categorical strings
cat_fix_cols = list(set(["series_id"] + static_categoricals + time_varying_known_categoricals))

for c in cat_fix_cols:
    if c in df_tft.columns:
        # Make everything string and replace nan-string with NA
        df_tft[c] = df_tft[c].astype(str).replace("nan", "NA").fillna("NA").astype("category")


#%%
# convert categoricals to strings and replace nan --> NA
# deal with missing values
# add missing value indicator + forward fill price by earlier week and fall back to 0
def add_if_missing(lst, col):
    if col not in lst:
        lst.append(col)
        
if "sell_price" in df_tft.columns:
    df_tft["sell_price_missing"] = df_tft["sell_price"].isna().astype(int)
    df_tft["sell_price"] = (
        df_tft.groupby("series_id")["sell_price"]
              .ffill()
              .fillna(0.0)
    )
    if "sell_price_missing" not in time_varying_known_reals:
        add_if_missing(time_varying_known_reals, "sell_price_missing")

for col in ["price_vs_item_mean", "price_vs_item_store_mean"]:
    if col in df_tft.columns:
        df_tft[f"{col}_missing"] = df_tft[col].isna().astype(int)
        df_tft[col] = df_tft[col].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        miss_col = f"{col}_missing"
        if miss_col not in time_varying_known_reals:
            add_if_missing(time_varying_known_reals, f"{col}_missing")
            
for c in list(time_varying_known_reals):
    if c in df_tft.columns:
        df_tft[c] = df_tft[c].replace([np.inf, -np.inf], np.nan)


bad = {}
for c in time_varying_known_reals:
    if c in df_tft.columns:
        bad[c] = (float(df_tft[c].isna().mean()), float(np.isinf(df_tft[c]).mean()))
bad = {k: v for k, v in bad.items() if v[0] > 0 or v[1] > 0}
print("Non-finite known reals (should be empty):", bad)

if len(bad) > 0:
    for c in bad.keys():
        df_tft[c] = df_tft[c].fillna(0.0)
    print("Filled remaining NaNs in:", list(bad.keys()))
#%%
# creating the training dataset
# we can only train up to 269 weeks
# our target is sales_wk
# group id is item id + store id
# sales_wk is our target -- it is excluded from the unknown reals because it's already the target

# tft does not need featured engineering to simulate memory
# it directly ingests the raw sequence
max_prediction_length = 4
max_encoder_length = 52

max_time = df_tft["time_idx"].max()
train_cutoff = max_time - max_prediction_length

#%%

time_varying_known_reals = sorted(set(time_varying_known_reals))
#%%
# time_varying_unknown_reals are time -dependent variables that are not known at forecast time but are observed in the past
# target (sales_wk) is the unknown varaible trying to predict

# tft learns from 
# 1) past values of the target
# 2) known future features (calendar, promo, price)
# 3) static features

# Group Normalizer -- normalizer per series 
# softplus enforces positive outputs, which make sense for demand

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
    time_varying_unknown_reals=['sales_wk'],  # target is already specified by `target=...`
    target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

#%%

# --- validation set: "predict next horizon" windows
validation = TimeSeriesDataSet.from_dataset(
    training,
    df_tft,
    predict=True,
    stop_randomization=True
)

batch_size =  256 
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader   = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

print("train batches:", len(train_dataloader))
print("val batches:", len(val_dataloader))

# quick sanity check
x, y = next(iter(train_dataloader))
print("x keys:", x.keys())

#%%

checkpoint_callback = ModelCheckpoint(
    dirpath="tft_checkpoints",          # folder to save
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,                       # keep best model only
    save_last=True                     # ALSO save last epoch
)


#%%
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=8,               # try 16 first; 32 later
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),          # strong default for TFT
    optimizer="adam",
    reduce_on_plateau_patience=3,
)

print(tft.size())  # model parameter count


#%%



early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5,
    verbose=True,
    mode="min",
)

lr_monitor = LearningRateMonitor()
logger = CSVLogger("tft_logs", name="m5_tft")

trainer = L.Trainer(
    max_epochs=20,                 # start small; increase later
    accelerator="auto",            # uses GPU if available
    devices="auto",
    gradient_clip_val=0.1,
    log_every_n_steps=50,
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    logger=logger,
    enable_checkpointing=True,
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

#%%

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


# -----------------------------
# 0) Basic checks
# -----------------------------

print("Num series in df_tft:", df_tft["series_id"].nunique())
print("Rows in df_tft:", len(df_tft))
print("Training samples (windows):", len(training))
print("Train batches (before limit):", len(train_dataloader))

#%%

def make_folds_time_idx(unique_time_idx, horizon=4, min_train=104, step=4, max_folds=10):
    t = np.array(sorted(unique_time_idx))
    folds = []
    for i in range(min_train - 1, len(t) - horizon, step):
        train_end = t[i]
        test_times = t[i+1:i+1+horizon]
        folds.append((int(train_end), test_times.astype(int)))
    if (max_folds is not None) and (len(folds) > max_folds):
        folds = folds[-max_folds:]  # keep most recent folds (compute-aware)
    return folds

horizon = max_prediction_length
step = 4
min_train = 104
max_folds = 10

folds = make_folds_time_idx(
    df_tft["time_idx"].unique(),
    horizon=horizon,
    min_train=min_train,
    step=step,
    max_folds=max_folds
)

print("num folds:", len(folds))
print("example fold:", folds[0])

#%%

# -----------------------------
# 2) Helper: fold dataframe = history up to train_end + next horizon weeks
# -----------------------------
def build_fold_df(df_all, train_end, test_times, group_col="series_id", time_col="time_idx"):
    # keep only rows that are needed for the fold
    fold_df = df_all[(df_all[time_col] <= train_end) | (df_all[time_col].isin(test_times))].copy()
    fold_df = fold_df.sort_values([group_col, time_col]).reset_index(drop=True)
    return fold_df


#%%

# -----------------------------
# 3) Predict one fold (robust to PF return formats)
# -----------------------------
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

# -----------------------------
# 4) Score one fold: per-series RMSSE + weights (from training history only)
# -----------------------------
def score_fold_rmsse(df_all, pred_long, train_end, target_col="sales_wk"):
    # merge true values for forecast weeks
    actual = df_all[["series_id", "time_idx", target_col]].rename(columns={target_col: "y_true"})
    merged = pred_long.merge(actual, on=["series_id", "time_idx"], how="left")

    # training history (denominator + weights), leakage-safe
    train_hist = df_all[df_all["time_idx"] <= train_end].sort_values(["series_id", "time_idx"])
    y_train_map = train_hist.groupby("series_id")["sales_wk"].apply(lambda s: s.values)
    w_map = train_hist.groupby("series_id")["sales_wk"].sum()

    # y_true / y_pred arrays for horizon
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


#%%
# -----------------------------
# 5) Run rolling evaluation (no retraining)
# -----------------------------
all_fold_scores = []
fold_summaries = []

for train_end, test_times in folds:
    pred_long = predict_fold_long(
        tft_model=tft,
        base_training_dataset=training,
        df_all=df_tft,
        train_end=train_end,
        test_times=test_times,
        batch_size=batch_size
    )

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

tft_overall = {
    "mean_rmsse_avg_over_folds": float(tft_fold_summaries_df["mean_rmsse"].mean()),
    "weighted_rmsse_avg_over_folds": float(tft_fold_summaries_df["weighted_rmsse"].mean()),
    "num_folds": int(len(tft_fold_summaries_df))
}

print("\nTFT fold summaries:")
print(tft_fold_summaries_df)
print("\nTFT overall:")
print(tft_overall)

#%%
# load the model

ckpt_path = r"C:\Users\agath\Desktop\retail project profolio\tft_checkpoints\manual_final.ckpt"
tft_loaded = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
tft_loaded.eval()


#%%




