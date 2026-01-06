# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:18:45 2026

@author: agath
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# Prefer plotly for interactivity; fallback to matplotlib if missing
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
    import matplotlib.pyplot as plt
    


#%%

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Walmart M5 商品預測作品集 (Baseline vs Global vs TFT)",
    layout="wide",
)

st.title("Walmart M5 商品預測作品集 — 模型比較 & TFT 預測")

# -----------------------------
# Paths (cloud-friendly: read from repo artifacts/)
# -----------------------------
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = str(REPO_ROOT / "artifacts")

FILES = {
    "series_compare": "series_model_comparison.csv",
    "baseline_overall": "baseline_overall.csv",
    "global_overall": "global_overall.csv",
    "tft_overall": "tft_overall.csv",
    "tft_pred_long_true": "tft_predictions_enriched.csv",
}

def _safe_path(name: str) -> str:
    return str(REPO_ROOT / "artifacts" / FILES[name])

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _require_cols(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")

def _kpi(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)
    
#%%
    
# -----------------------------
# Load data
# -----------------------------
with st.sidebar:
    st.subheader("📦 Data status")
    loaded = {}
    for k in ["series_compare", "baseline_overall", "global_overall", "tft_overall"]:
        p = _safe_path(k)
        ok = os.path.exists(p)
        st.write(("✅" if ok else "❌") + f" {FILES[k]}")
    pred_path = _safe_path("tft_pred_long_true")
    st.write(("✅" if os.path.exists(pred_path) else "⚠️") + f" {FILES['tft_pred_long_true']} (Tab 2)")

# Load mandatory artifacts
series_compare = load_csv(_safe_path("series_compare"))
baseline_overall = load_csv(_safe_path("baseline_overall"))
global_overall = load_csv(_safe_path("global_overall"))
tft_overall = load_csv(_safe_path("tft_overall"))

# Basic expected columns
_require_cols(series_compare, ["series_id", "baseline_rmsse", "global_rmsse", "tft_rmsse"], "series_model_comparison")

# add gain column if not present
if "tft_vs_global_gain" not in series_compare.columns:
    series_compare["tft_vs_global_gain"] = series_compare["global_rmsse"] - series_compare["tft_rmsse"]
    
    
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["① Series-level Performance Explorer", "② TFT Forecast Explorer (Actual vs Predicted)"])

# =========================================================
# Tab 1: Series-level Performance Explorer
# =========================================================
with tab1:
    st.subheader("① Series-level Performance Explorer")

    # Sidebar filters (inside tab)
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔎 Filters (Tab 1)")
        rmsse_cap = st.slider("Clip RMSSE at (for visualization)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        show_top_n = st.number_input("Show top N by TFT improvement", min_value=10, max_value=500, value=30, step=10)

    df = series_compare.copy()

    # Optional clipping for plots only
    df_plot = df.copy()
    for c in ["baseline_rmsse", "global_rmsse", "tft_rmsse"]:
        df_plot[c] = df_plot[c].clip(upper=rmsse_cap)

    # KPIs
    n = len(df)
    tft_wins = (df["tft_rmsse"] < df["global_rmsse"]).sum()
    global_wins = (df["global_rmsse"] <= df["tft_rmsse"]).sum()
    avg_tft = df["tft_rmsse"].mean()
    avg_global = df["global_rmsse"].mean()
    avg_base = df["baseline_rmsse"].mean()
    avg_gain = (df["global_rmsse"] - df["tft_rmsse"]).mean()
    median_gain = (df["global_rmsse"] - df["tft_rmsse"]).median()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi("Series count", f"{n:,}")
    with c2:
        _kpi("TFT wins vs Global", f"{tft_wins:,} ({tft_wins/n:.1%})")
    with c3:
        _kpi("Global wins vs TFT", f"{global_wins:,} ({global_wins/n:.1%})")
    with c4:
        _kpi("Avg RMSSE (Global → TFT)", f"{avg_global:.3f} → {avg_tft:.3f}")
    with c5:
        _kpi("Avg gain (Global - TFT)", f"{avg_gain:.4f}", help_text="Positive means TFT better on average.")

    st.markdown(
        """
**RMSSE 是什麼？（Root Mean Squared Scaled Error）**  
- 用「與基準模型（通常是 seasonal naive）相比」的方式把誤差做尺度化，方便不同商品線彼此比較。  
- **RMSSE 越小越好**；若 **RMSSE < 1**，代表比基準模型更好。  

**整體解讀（為什麼平均上 TFT 可能更好）**  
TFT 可能不是每條商品線的預測成果都贏 Global 模型，但它在某些商品線上帶來較大的改善幅度，而輸的那些 series 輸得比較小，  
因此整體的平均（mean）錯誤率更低。
        """
    )

    # Charts
    left, right = st.columns([1, 1])

    with left:
        st.write("### A) RMSSE distribution (clipped for display)")
        if PLOTLY_OK:
            melt = df_plot.melt(id_vars=["series_id"], value_vars=["baseline_rmsse", "global_rmsse", "tft_rmsse"],
                                var_name="model", value_name="rmsse")
            fig = px.box(melt, x="model", y="rmsse", points="outliers")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 圖 A 解讀：RMSSE 分佈越往下代表整體誤差越小。你可以用這張圖快速比較 Baseline / Global / TFT 在『多數商品線』上的整體表現差異（已做 clipping 方便視覺化）。")
        else:
            fig, ax = plt.subplots()
            ax.boxplot([df_plot["baseline_rmsse"], df_plot["global_rmsse"], df_plot["tft_rmsse"]], labels=["baseline", "global", "tft"])
            ax.set_ylabel("RMSSE (clipped)")
            st.pyplot(fig)
            st.caption("📌 圖 A 解讀：RMSSE 分佈越往下代表整體誤差越小。你可以用這張圖快速比較 Baseline / Global / TFT 在『多數商品線』上的整體表現差異（已做 clipping 方便視覺化）。")

    with right:
        st.write("### B) Global - TFT gain distribution")
        gain = df["global_rmsse"] - df["tft_rmsse"]
        if PLOTLY_OK:
            fig = px.histogram(gain, nbins=60, labels={"value": "Global - TFT (positive = TFT better)"})
            fig.update_traces(marker_color="mediumpurple")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 圖 A 解讀：RMSSE 分佈越往下代表整體誤差越小。你可以用這張圖快速比較 Baseline / Global / TFT 在『多數商品線』上的整體表現差異（已做 clipping 方便視覺化）。")
        else:
            fig, ax = plt.subplots()
            ax.hist(gain, bins=60)
            ax.set_xlabel("Global - TFT (positive = TFT better)")
            ax.set_ylabel("count")
            st.pyplot(fig)

    st.write("### C) TFT 改善最多的商品線（Top 改善清單）")
    top_improve = df.sort_values("tft_vs_global_gain", ascending=False).head(int(show_top_n))
    st.dataframe(top_improve, use_container_width=True)
    st.caption("📌 表 C 解讀：這裡列出『Global − TFT』改善幅度最大的商品線。數值越大表示 TFT 相對 Global 進步越多，適合拿來做案例解釋：TFT 在哪些商品線上帶來明顯提升。")

    st.write("### D) 每條商品線：Global vs TFT 散點圖")
    if PLOTLY_OK:
        fig = px.scatter(
            df_plot,
            x="global_rmsse",
            y="tft_rmsse",
            hover_data=["series_id", "baseline_rmsse"],
            labels={"global_rmsse": "Global RMSSE (clipped)", "tft_rmsse": "TFT RMSSE (clipped)"},
        )
        fig.update_traces(marker=dict(color="teal", size=7, opacity=0.8))
        # add diagonal line y=x
        fig.add_shape(type="line", x0=0, y0=0, x1=rmsse_cap, y1=rmsse_cap, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 圖 D 解讀：每個點是一條商品線。虛線是 y=x；點落在虛線下方代表 TFT RMSSE < Global RMSSE（TFT 較好），落在上方則代表 Global 較好。越靠近左下角代表兩者誤差都更小。")
    else:
        st.info("Install plotly to see interactive scatter: pip install plotly")

    # Overall table quick view
    with st.expander("Show overall CSVs (baseline/global/tft)"):
        st.write("baseline_overall")
        st.dataframe(baseline_overall, use_container_width=True)
        st.write("global_overall")
        st.dataframe(global_overall, use_container_width=True)
        st.write("tft_overall (fold summaries)")
        st.dataframe(tft_overall, use_container_width=True)



# =========================================================
# Tab 2: TFT Forecast Explorer — Actual vs Predicted (+ Promo + Off flags)
# =========================================================
with tab2:
    st.subheader("② TFT Forecast Explorer — Actual vs Predicted")

    st.markdown("**APE（Absolute Percentage Error，絕對百分比誤差）**：\n\n- 公式：`APE = |實際 - 預測| / |實際| × 100%`\n- 數值越小越好；我們用它來標記『預測偏差較大』的時間點，方便對照是否與促銷（promo_depth）相關。")

    # ---------- Load prediction data ----------
    tft_pred = load_csv(pred_path)
    tft_pred["series_id"] = tft_pred["series_id"].astype(str)

    # ---------- FORCE x-axis to be DATE ----------
    date_col = None
    if "date" in tft_pred.columns:
        date_col = "date"
    elif "date_x" in tft_pred.columns:
        date_col = "date_x"
    elif "date_y" in tft_pred.columns:
        date_col = "date_y"

    if date_col is None:
        st.error("Tab 2 requires a date column (date / date_x / date_y) for business-friendly x-axis.")
        st.stop()

    tft_pred["plot_date"] = pd.to_datetime(tft_pred[date_col], errors="coerce")
    tft_pred = tft_pred.dropna(subset=["plot_date"]).copy()
    tft_pred = tft_pred.sort_values(["series_id", "plot_date"])
    x_label = "Date"

    # ---------- Quick column checks ----------
    required_cols = ["series_id", "plot_date", "y_true", "y_pred"]
    missing = [c for c in required_cols if c not in tft_pred.columns]
    if missing:
        st.error(f"Tab 2 missing required columns: {missing}")
        st.stop()

    # =====================================================
    # Controls ON TOP (no horizon selector)
    # =====================================================
    with st.container():
        c1, c2, c3 = st.columns([3, 2, 3])

        with c1:
            series_options = sorted(tft_pred["series_id"].unique())
            selected_series = st.multiselect(
                "選擇商品線（可多選）",
                options=series_options,
                default=series_options[:1],
            )

        with c2:
            show_promo = ("promo_depth" in tft_pred.columns) and st.checkbox(
                "顯示促銷強度（promo_depth）", value=True
            )

        with c3:
            show_off = st.checkbox('顯示「預測偏差」標記（APE）', value=True)
            off_threshold = st.slider(
                "當 APE >（%）時標記",
                min_value=0.0,
                max_value=200.0,
                value=30.0,
                step=5.0,
            )

    if not selected_series:
        st.info("Please select at least one series.")
        st.stop()

    # ---------- Filter ----------
    df = tft_pred[tft_pred["series_id"].isin(selected_series)].copy()
    df = df.sort_values(["series_id", "plot_date"])

    # ---------- Plot ----------
    if not PLOTLY_OK:
        st.info("Install plotly to see interactive charts: pip install plotly")
        st.stop()

    import plotly.graph_objects as go

    actual_color = "red"
    pred_color = "blue"
    promo_color = "orange"

    def make_one_series_fig(g: pd.DataFrame, title: str | None = None) -> go.Figure:
        g = g.sort_values("plot_date").copy()

        fig = go.Figure()

        # 1) Promo bars FIRST (behind lines)
        if show_promo and "promo_depth" in g.columns:
            promo_vals = g["promo_depth"].fillna(0)
            if promo_vals.max() > 0:
                fig.add_trace(
                    go.Bar(
                        x=g["plot_date"],
                        y=promo_vals,
                        name="Promo (depth)",
                        marker_color=promo_color,
                        opacity=0.18,
                        yaxis="y2",
                        hovertemplate="Date=%{x}<br>Promo depth=%{y}<extra></extra>",
                    )
                )

        # 2) Actual line
        fig.add_trace(
            go.Scatter(
                x=g["plot_date"],
                y=g["y_true"],
                mode="lines",
                name="Actual",
                line=dict(color=actual_color, width=3),
                hovertemplate="Date=%{x}<br>Actual=%{y}<extra></extra>",
            )
        )

        # 3) Predicted line
        fig.add_trace(
            go.Scatter(
                x=g["plot_date"],
                y=g["y_pred"],
                mode="lines",
                name="Predicted",
                line=dict(color=pred_color, width=3),
                hovertemplate="Date=%{x}<br>Predicted=%{y}<extra></extra>",
            )
        )

        # 4) "Prediction Off" flags (Percent Error / APE)
        if show_off:
            tmp = g.copy()
            denom = tmp["y_true"].abs().replace(0, np.nan)
            tmp["ape"] = (tmp["y_true"] - tmp["y_pred"]).abs() / denom * 100.0

            off_pts = tmp[tmp["ape"] > off_threshold].dropna(subset=["ape"]).copy()

            if not off_pts.empty:
                promo_for_hover = (
                    off_pts["promo_depth"].fillna(0).to_numpy()
                    if "promo_depth" in off_pts.columns
                    else np.zeros(len(off_pts))
                )

                fig.add_trace(
                    go.Scatter(
                        x=off_pts["plot_date"],
                        y=off_pts["y_true"],  # anchor at actual value
                        mode="markers",
                        name="Prediction Off (APE)",
                        marker=dict(symbol="x", size=10, color="black"),
                        customdata=np.stack(
                            [
                                off_pts["y_pred"].to_numpy(),
                                off_pts["ape"].to_numpy(),
                                promo_for_hover,
                            ],
                            axis=1,
                        ),
                        hovertemplate=(
                            "Date=%{x}"
                            "<br>Actual=%{y}"
                            "<br>Predicted=%{customdata[0]:.2f}"
                            "<br>APE=%{customdata[1]:.1f}%"
                            "<br>Promo depth=%{customdata[2]}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            height=520 if title is None else 380,
            margin=dict(l=20, r=60, t=50 if title else 40, b=20),
            legend_title_text="",
            barmode="overlay",
        )

        fig.update_xaxes(title=x_label)

        # Secondary axis for promo (right side), visually quiet
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                title="Promo",
                rangemode="tozero",
            )
        )

        return fig

    # Single series: one big chart
    if len(selected_series) == 1:
        sid = selected_series[0]
        g = df[df["series_id"] == sid].copy()
        fig = make_one_series_fig(g, title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Multiple series: one chart per series (readable for stakeholders)
        for sid in selected_series:
            g = df[df["series_id"] == sid].copy()
            fig = make_one_series_fig(g, title=str(sid))
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "APE（Absolute Percentage Error，絕對百分比誤差）= |實際 - 預測| / |實際| × 100%。"
        "\n橘色柱狀 = 促銷強度（promo_depth）；黑色 X = 預測偏差（APE 超過門檻）。"
        "\n用途：快速檢查『預測誤差較大』是否集中發生在促銷期間。"
    )

