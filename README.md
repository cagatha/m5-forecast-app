## Live Demo
- https://m5-forecast-app-agatha-chang.streamlit.app

# M5 Forecasting Demo (Streamlit)

這是一個可互動的 Streamlit 作品集 Demo，用來展示：

- **Tab 1：** Baseline / Global / TFT 的 **series-level** 表現比較（RMSSE）
- **Tab 2：** Actual vs Predicted 視覺化，並用 **Promo depth**（促銷強度）與 **APE prediction-off flags**（預測偏差標記）協助解讀模型在促銷期間的穩定性

## Repo Structure
```
m5-forecast-app/
├─ app/
│  └─ streamlit_app.py         
│
├─ artifacts/
│  ├─ series_model_comparison.csv
│  ├─ baseline_overall.csv
│  ├─ global_overall.csv
│  ├─ tft_overall.csv
│  └─ tft_predictions_enriched.csv
│
├─ src/                         
│  ├─ baseline_model_m5.py
│  ├─ advanced_model_m5.py
│  ├─ load_tft_model.py
│  └─ build_app_artifacts.py
│
├─ requirements.txt
├─ README.md
└─ .gitignore

```



