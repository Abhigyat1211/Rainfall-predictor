import numpy as np
import pandas as pd

# Wind direction lookup — 16 compass points, indices 0–15
WIND_DIRECTIONS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
WIND_MAP = {val: i for i, val in enumerate(WIND_DIRECTIONS)}


def process_realtime_input(live_data_dict, scaler, location_encoder, global_mean):
    """
    Transforms raw API dictionary telemetry values into the exact
    scaled numeric array structure the model was trained on.
    """
    df_live = pd.DataFrame([live_data_dict])

    # 1. Re-calculate engineered interaction features
    df_live['TempRange']      = df_live['MaxTemp']      - df_live['MinTemp']
    df_live['HumidityChange'] = df_live['Humidity3pm']  - df_live['Humidity9am']
    df_live['PressureTrend']  = df_live['Pressure3pm']  - df_live['Pressure9am']  # falling = rain signal

    # 2. Extract Month and build its cyclical sine/cosine transformations
    if 'Date' in df_live.columns:
        df_live['Month'] = pd.to_datetime(df_live['Date']).dt.month
    else:
        df_live['Month'] = 6  # Default fallback to June if missing

    df_live['Month_sin'] = np.sin(2 * np.pi * df_live['Month'] / 12)
    df_live['Month_cos'] = np.cos(2 * np.pi * df_live['Month'] / 12)

    # 3. Apply log1p scaling to right-skewed columns
    skewed_cols = ['Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
    for col in skewed_cols:
        if col in df_live.columns:
            df_live[col] = np.log1p(df_live[col].astype(float))

    # 4. Handle target encoding for Location
    if 'Location' in df_live.columns:
        df_live['Location_encoded'] = df_live['Location'].map(location_encoder).fillna(global_mean)
    else:
        df_live['Location_encoded'] = global_mean

    # 5. Cyclical sine/cosine mapping for wind compass directions
    #
    # IMPORTANT: During training, unknown/NaN wind directions produced NaN sin/cos
    # values, which were then caught by the global fillna(0) in cell 43 of the notebook.
    # This means the training convention for a missing wind direction is sin=0, cos=0
    # (not a real compass point, but that is what the scaler was fitted on).
    # We must replicate that exact behaviour here — compute sin/cos first, THEN fillna(0)
    # on the resulting sin/cos columns, NOT on the index before computing them.
    wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

    for col in wind_cols:
        if col in df_live.columns:
            idx_series = df_live[col].map(WIND_MAP)   # unknown direction -> NaN
        else:
            idx_series = pd.Series([np.nan] * len(df_live))

        sin_vals = np.sin(2 * np.pi * idx_series / 16)  # NaN input -> NaN output
        cos_vals = np.cos(2 * np.pi * idx_series / 16)

        # Match training: NaN sin/cos -> 0 (replicates notebook's global fillna(0))
        df_live[f'{col}_sin'] = sin_vals.fillna(0).values
        df_live[f'{col}_cos'] = cos_vals.fillna(0).values

    # 6. Binary encode RainToday
    if 'RainToday' in df_live.columns:
        df_live['RainToday_encoded'] = df_live['RainToday'].map({'No': 0, 'Yes': 1}).fillna(0)
    else:
        df_live['RainToday_encoded'] = 0

    # 7. Enforce exact column order matching X_train_final
    # Pressure3pm and PressureTrend added after notebook retrain with PressureTrend feature
    expected_order = [
        'Rainfall', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 'Temp3pm',
        'Month', 'Month_sin', 'Month_cos', 'TempRange', 'HumidityChange', 'PressureTrend',
        'Location_encoded', 'WindGustDir_sin', 'WindGustDir_cos', 'WindDir9am_sin',
        'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos', 'RainToday_encoded'
    ]

    df_features = df_live[expected_order]

    # 8. Standardize using the saved scaler
    return scaler.transform(df_features)