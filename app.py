import streamlit as st
import joblib
import datetime
import requests
import pandas as pd
from transformer import process_realtime_input

st.set_page_config(page_title="Geospatial Rain Engine", page_icon="🌧️", layout="wide")

CITY_COORDINATES = {
    "Adelaide": {"lat": -34.9285, "lon": 138.6007},
    "Brisbane": {"lat": -27.4698, "lon": 153.0251},
    "Canberra": {"lat": -35.2809, "lon": 149.1300},
    "Darwin": {"lat": -12.4634, "lon": 130.8456},
    "Melbourne": {"lat": -37.8136, "lon": 144.9631},
    "Newcastle": {"lat": -32.9284, "lon": 151.7817},
    "Perth": {"lat": -31.9505, "lon": 115.8605},
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
    "Wollongong": {"lat": -34.4278, "lon": 150.8931},
}

# Coastal cities have characteristically high humidity even on dry winter days,
# which causes the model to systematically over-predict rain. A higher decision
# threshold corrects for this without retraining.
COASTAL_CITIES = {"Sydney", "Newcastle", "Wollongong"}
COASTAL_THRESHOLD = 0.65
DEFAULT_THRESHOLD = 0.50

# How far ahead the date picker allows the user to select a target date.
MAX_DAYS_AHEAD = 14

# Open-Meteo's forecast endpoint defaults to 7 days and never includes
# yesterday unless `past_days` is set. Since we always look up
# (target_date - 1 day), and target_date can be up to MAX_DAYS_AHEAD days
# out, we need:
#   - past_days=1        -> so "yesterday relative to today" is present
#                            (needed when target_date == today)
#   - forecast_days=16    -> Open-Meteo's max, so "target_date - 1" is
#                            present even when target_date == today + 14
# This guarantees every selectable target date maps to a *distinct* day
# in the API response, instead of silently collapsing onto today's data.
API_PAST_DAYS = 1
API_FORECAST_DAYS = 16


def degrees_to_compass(degrees):
    """Convert a bearing in degrees to the nearest 16-point compass label.
    Returns None for missing input so the transformer can handle it correctly."""
    if degrees is None:
        return None
    directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    return directions[round(degrees / 22.5) % 16]


def safe_hourly(hourly, key, hour_index, default):
    """Extract a value from the hourly dict at a given absolute hour index.
    Returns `default` only when the value is missing/None — genuine 0s are preserved."""
    values = hourly.get(key, [])
    if len(values) > hour_index:
        val = values[hour_index]
        return val if val is not None else default
    return default


def safe_daily(daily, key, idx, default):
    """Extract a value from the daily dict at a given day index.
    Returns `default` only when the value is missing/None — genuine 0s are preserved."""
    values = daily.get(key, [])
    if len(values) > idx:
        val = values[idx]
        return val if val is not None else default
    return default


@st.cache_resource
def load_production_pipeline():
    return (
        joblib.load('artifacts/logistic_rain_model.pkl'),
        joblib.load('artifacts/scaler.pkl'),
        joblib.load('artifacts/location_encoder.pkl'),
        joblib.load('artifacts/global_mean.pkl'),
    )


model, scaler, location_encoder, global_mean = load_production_pipeline()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Evaluation Metrics")
    st.metric(label="Model Accuracy", value="78.00%")
    st.metric(label="ROC-AUC Score", value="0.8571")  # updated from 0.8536
    col1, col2 = st.columns(2)
    col1.metric(label="Precision (Rain)", value="50.00%")
    col2.metric(label="Recall (Rain)", value="76.00%")
    st.markdown("---")
    st.caption(
        "ℹ️ Coastal cities (Sydney, Newcastle, Wollongong) use a higher decision "
        "threshold (0.65) to correct for systematic humidity bias."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🌧️ Australia's Aquaman")
st.write("Next-day rainfall prediction for Australian cities using weather forecast data.")

with st.container():
    st.subheader("🌐 Input Parameters & Station Location")
    col_input, col_map = st.columns([1, 1])

    with col_input:
        selected_city = st.selectbox(
            "Select Target Station City:", sorted(list(CITY_COORDINATES.keys()))
        )

        today = datetime.date.today()
        max_date = today + datetime.timedelta(days=MAX_DAYS_AHEAD)
        target_date = st.date_input(
            "Select Prediction Date (Target Day):",
            today,
            min_value=today,
            max_value=max_date,
        )

        st.info(
            "ℹ️ The model analyzes weather conditions of **(Target Date - 1)** "
            "to predict rain on the selected date."
        )

        run_forecast = st.button("Predict", type="primary")

    with col_map:
        station_lat = CITY_COORDINATES[selected_city]["lat"]
        station_lon = CITY_COORDINATES[selected_city]["lon"]
        map_data = pd.DataFrame({'lat': [station_lat], 'lon': [station_lon]})
        st.map(map_data, zoom=10, size=25)

# ── Prediction ────────────────────────────────────────────────────────────────
if run_forecast:
    base_date = target_date - datetime.timedelta(days=1)

    # Pick the correct decision threshold for this city
    threshold = COASTAL_THRESHOLD if selected_city in COASTAL_CITIES else DEFAULT_THRESHOLD

    with st.spinner(
        f"Fetching forecast data for {base_date.strftime('%B %d, %Y')} (Target Date - 1)..."
    ):
        api_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={station_lat}&longitude={station_lon}"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,sunshine_duration,"
            f"precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant"
            f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,cloud_cover,"
            f"wind_speed_10m,wind_direction_10m"
            f"&past_days={API_PAST_DAYS}"
            f"&forecast_days={API_FORECAST_DAYS}"
            f"&timezone=auto"
        )

        try:
            response = requests.get(api_url, timeout=10).json()

            if 'daily' not in response:
                st.error("⚠️ Weather forecast data unavailable at the moment.")
            else:
                daily = response['daily']
                hourly = response.get('hourly', {})

                # Locate the correct day index inside the API response.
                # With past_days=1 and forecast_days=16 above, base_date should
                # always be present for any target_date the user could have
                # picked (today .. today+14). If it's ever missing, that's a
                # real problem (e.g. API contract changed) — surface it instead
                # of silently reusing today's data, which is what caused
                # identical probabilities across different target dates before.
                date_list = daily.get('time', [])
                if base_date.strftime("%Y-%m-%d") not in date_list:
                    st.error(
                        f"⚠️ Forecast data for {base_date.strftime('%B %d, %Y')} "
                        f"was not returned by the weather API (got range "
                        f"{date_list[0] if date_list else 'N/A'} to "
                        f"{date_list[-1] if date_list else 'N/A'}). "
                        f"Please try again or pick a different date."
                    )
                    st.stop()

                day_idx = date_list.index(base_date.strftime("%Y-%m-%d"))

                # Hourly index = day offset × 24 + hour-of-day.
                # timezone=auto means the list is already in local time so
                # index 9 = 09:00 local and index 15 = 15:00 local.
                day_hour_offset = day_idx * 24
                hour_9am = day_hour_offset + 9
                hour_3pm = day_hour_offset + 15

                # Sunshine: API returns seconds; training data uses hours.
                # safe_daily preserves genuine 0 (fully overcast day) — critical
                # because sunshine=0 is a strong positive rain signal.
                raw_sunshine_sec = safe_daily(daily, 'sunshine_duration', day_idx, default=None)
                sunshine_hours = (raw_sunshine_sec / 3600.0) if raw_sunshine_sec is not None else 7.6

                # Cloud cover: API returns percent (0–100); training data uses oktas (0–8).
                raw_cloud_pct = safe_hourly(hourly, 'cloud_cover', hour_3pm, default=45)
                cloud_oktas = raw_cloud_pct / 12.5

                payload = {
                    'Location': selected_city,
                    'Date': base_date.strftime("%Y-%m-%d"),
                    'Rainfall': safe_daily(daily, 'precipitation_sum', day_idx, default=0),
                    'WindGustSpeed': safe_daily(daily, 'wind_gusts_10m_max', day_idx, default=30),
                    'WindSpeed9am': safe_hourly(hourly, 'wind_speed_10m', hour_9am, default=13),
                    'WindSpeed3pm': safe_hourly(hourly, 'wind_speed_10m', hour_3pm, default=19),
                    'Humidity9am': safe_hourly(hourly, 'relative_humidity_2m', hour_9am, default=65),
                    'Humidity3pm': safe_hourly(hourly, 'relative_humidity_2m', hour_3pm, default=50),
                    'Pressure9am': safe_hourly(hourly, 'pressure_msl', hour_9am, default=1016),
                    'Pressure3pm': safe_hourly(hourly, 'pressure_msl', hour_3pm, default=1015),
                    'Cloud3pm': cloud_oktas,
                    'MaxTemp': safe_daily(daily, 'temperature_2m_max', day_idx, default=22),
                    'MinTemp': safe_daily(daily, 'temperature_2m_min', day_idx, default=12),
                    'Temp3pm': safe_hourly(hourly, 'temperature_2m', hour_3pm, default=20),
                    'Sunshine': sunshine_hours,
                    # degrees_to_compass returns None for missing data; the transformer
                    # then correctly maps None -> NaN -> sin=0, cos=0 (matches training).
                    'WindGustDir': degrees_to_compass(safe_daily(daily, 'wind_direction_10m_dominant', day_idx, default=None)),
                    'WindDir9am': degrees_to_compass(safe_hourly(hourly, 'wind_direction_10m', hour_9am, default=None)),
                    'WindDir3pm': degrees_to_compass(safe_hourly(hourly, 'wind_direction_10m', hour_3pm, default=None)),
                }
                payload['RainToday'] = 'Yes' if payload['Rainfall'] > 1.0 else 'No'

                processed_vector = process_realtime_input(
                    payload, scaler, location_encoder, global_mean
                )

                # Do NOT use model.predict() — it always uses a hard 0.5 threshold
                # internally and cannot be overridden. Derive the binary label ourselves
                # from predict_proba so the coastal-city threshold is actually honoured.
                probability = model.predict_proba(processed_vector)[0][1]
                rain_predicted = probability >= threshold

                # ── Results ───────────────────────────────────────────────────
                st.markdown("---")
                st.subheader(
                    f"🔮 Prediction for {selected_city} on {target_date.strftime('%B %d, %Y')}"
                )
                st.info(
                    f"📋 This prediction is based on forecasted weather conditions of "
                    f"**{base_date.strftime('%B %d, %Y')}** (Target Date - 1)"
                )

                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Rain Probability", f"{probability:.1%}")
                if selected_city in COASTAL_CITIES:
                    res_col1.caption(f"Coastal threshold: {COASTAL_THRESHOLD:.0%}")

                if rain_predicted:
                    res_col2.error("🌧️ Rain is Forecasted")
                    st.warning("Atmospheric conditions suggest higher chance of rainfall.")
                else:
                    res_col2.success("☀️ Clear Skies Expected")
                    st.info("Conditions look stable with low chance of rain.")

        except Exception as e:
            st.error(f"Failed to fetch forecast data: {e}")
