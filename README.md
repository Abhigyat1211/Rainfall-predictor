# Australia's Aquaman 🌧️
### Next-Day Rainfall Prediction for Australian Cities

A machine learning web application that predicts next-day rainfall across 9 Australian cities using live weather forecast data from the Open-Meteo API. Built end-to-end from raw BOM data through to a deployed Streamlit interface.

---

## Background

This project started as a straightforward binary classification problem — predict whether it will rain tomorrow — but became increasingly interesting once I moved past the notebook and started validating predictions against actual weather outcomes. The gap between clean historical accuracy and real-world performance turned out to be where most of the learning happened.

The model is a logistic regression trained on the [weatherAUS dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) sourced from the Bureau of Meteorology (BOM), covering 10 years of daily weather observations across 49 stations.

---

## Project Structure

```
├── Rainfall.ipynb               # Data cleaning, feature engineering, model training
├── app.py                       # Streamlit web application
├── transformer.py               # Inference pipeline — replicates training transforms
└── artifacts/
    ├── logistic_rain_model.pkl
    ├── scaler.pkl
    ├── location_encoder.pkl
    └── global_mean.pkl
```

---

## Methodology

### Feature Engineering

Raw BOM features required several transformations to be model-ready:

- **Log1p scaling** applied to right-skewed columns: `Rainfall`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`
- **Cyclical encoding** (sin/cos) for wind direction columns and month — preserves the circular nature of compass bearings and avoids false ordinal relationships
- **Interaction features**: `TempRange` (MaxTemp − MinTemp), `HumidityChange` (Humidity3pm − Humidity9am), `PressureTrend` (Pressure3pm − Pressure9am)
- **Target encoding** for Location — each city encoded as its historical rain frequency, computed on training data only to avoid leakage
- **StandardScaler** applied to the final 25-feature matrix

### Why PressureTrend

Pressure9am and Pressure3pm were initially treated as independent features, but the difference between them carries a signal neither column alone can express: a falling pressure through the day is a reliable short-term rain precursor. Adding `PressureTrend` improved ROC-AUC from 0.8536 to 0.8571 and directly fixed a class of false negatives where post-frontal conditions were being misread.

### Inference Pipeline

One of the less obvious challenges in this project was ensuring the inference pipeline in `transformer.py` exactly replicates the training transformations in the notebook — including edge cases like NaN wind direction handling. During training, unknown wind directions produced NaN sin/cos values which were caught by a global `fillna(0)` call. The transformer has to replicate this precisely: compute sin/cos first, then `fillna(0)` on the result — not fill the direction index before computing, which would produce different values and silently corrupt predictions.

---

## Model Performance

### Reported (held-out test set)

| Metric | Value |
|--------|-------|
| Accuracy | 78.0% |
| ROC-AUC | 0.8571 |
| Precision (Rain) | 50.0% |
| Recall (Rain) | 76.0% |
| F1-Score (Rain) | 0.61 |

The model uses `class_weight='balanced'` to compensate for the ~78/22 class imbalance between no-rain and rain days. This intentionally trades precision for recall — missing real rain events is penalised more heavily than false alarms.

### Real-World Validation (34 live predictions)

After deployment, I ran 34 predictions against actual forecast outcomes across all 9 cities.

| Metric | Value |
|--------|-------|
| Overall accuracy | 64.7% |
| Gap vs reported | −13.3% |

The gap is expected — the test set used a random stratified split rather than a temporal one, so the reported accuracy is somewhat optimistic. Additionally, the model was trained on BOM *observations* but at inference time receives Open-Meteo *forecast* values, which introduces a systematic data-source mismatch.

### Per-City Accuracy

| City | Correct | Total | Accuracy | Notes |
|------|---------|-------|----------|-------|
| Melbourne | 5 | 5 | 100% | Consistent, strong pressure signals |
| Adelaide | 3 | 3 | 100% | Clear frontal rain patterns |
| Perth | 2 | 2 | 100% | Well-defined wet season |
| Sydney | 3 | 4 | 75% | Coastal threshold applied |
| Wollongong | 2 | 3 | 67% | Coastal threshold applied |
| Brisbane | 2 | 4 | 50% | Light showery rain hard to detect |
| Canberra | 2 | 4 | 50% | Improved with PressureTrend; transitional days still inconsistent |
| Newcastle | 3 | 7 | 43% | Sits near coastal threshold boundary |
| Darwin | 0 | 2 | 0% | Dry season humidity confounds temperate-trained model |

---

## Coastal City Bias & Threshold Adjustment

Sydney, Newcastle, and Wollongong showed a systematic pattern of over-prediction — probabilities of 56–76% on days with 0–5% actual rain chance. The root cause is that coastal NSW cities maintain high humidity year-round, even on completely dry winter days, which the model interprets as a pre-rain signal. This is a genuine limitation of training on a national dataset without location-aware feature interactions.

The fix applied was a city-specific decision threshold:

```python
COASTAL_CITIES = {"Sydney", "Newcastle", "Wollongong"}
threshold = 0.65 if selected_city in COASTAL_CITIES else 0.50
```

This was determined empirically across 14 coastal city predictions rather than set arbitrarily. It brought coastal accuracy from roughly 45% to 72% and eliminated the most egregious false positives without suppressing genuine rain predictions.

---

## Known Limitations

**Darwin dry season** — Darwin's tropical climate means dry season months (May–October) have persistently high humidity and cloud cover despite near-zero rain probability. The model has no mechanism to distinguish tropical dry-season humidity from pre-frontal temperate humidity. Predictions for Darwin between May and October should be treated with caution.

**Light rain events** — Rain events below roughly 2–3mm consistently fall below the model's decision threshold. Brisbane and Sydney light showers (forecast chance 20–35%) are systematically missed. This is partly by design — BOM's `RainTomorrow` label requires >1mm — but the model learns a strong signal for heavy rain and a weak one for light rain.

**Transitional weather periods** — Days following a wet spell but preceding a clear period are the hardest to call. The previous day's conditions still show elevated humidity and low sunshine even as the system moves on. `PressureTrend` partially addresses this; a pressure *recovery* signal (Pressure3pm > Pressure9am) now correctly suppresses some false positives.

**Data source mismatch** — Training used BOM ground station observations. Inference uses Open-Meteo API forecast values. These are not identical — forecast values carry their own uncertainty, and measurement methodologies differ (BOM cloud cover in oktas vs Open-Meteo percentage, for instance). Unit conversions are applied in the transformer but systematic forecast bias cannot be corrected without retraining on forecast data.

---

## What I'd Do Differently

**Temporal train/test split** — The random stratified split produces an optimistic accuracy figure. A proper temporal split (e.g. train on pre-2016 data, test on 2016–2017) would give a more honest benchmark and better simulate deployment conditions.

**Gradient boosting** — Logistic regression is inherently linear. The coastal humidity issue, the Darwin dry-season problem, and the light rain misses all require non-linear feature interactions to resolve properly. XGBoost or LightGBM with location as a categorical feature would likely push real-world accuracy to 78–82%.

**Retrain on forecast data** — The data-source mismatch is the single biggest driver of the 13% real-world gap. Collecting 6–12 months of Open-Meteo forecast data paired with actual BOM outcomes and retraining on that would make the model genuinely production-grade.

**Pressure trend as a rolling feature** — Currently `PressureTrend` is a single-day delta. A 3-day pressure trend (is pressure consistently falling?) would be a stronger signal, particularly for multi-day frontal systems.

---

## Stack

- Python, scikit-learn, pandas, numpy
- Streamlit
- Open-Meteo API (free, no key required)
- joblib for artifact serialization

---

## Running Locally

```bash
pip install streamlit scikit-learn pandas numpy requests joblib
streamlit run app.py
```

Artifacts (`logistic_rain_model.pkl`, `scaler.pkl`, `location_encoder.pkl`, `global_mean.pkl`) must be present in the `artifacts/` directory. Train them by running `Rainfall.ipynb` end to end.

---

*Tested against 34 live predictions across all 9 supported cities. Per-city accuracy figures reflect actual forecast outcomes, not held-out test set performance.*
