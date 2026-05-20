import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime
import io
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prophet optional import (consolidated)
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    plot_plotly = None
    plot_components_plotly = None
    PROPHET_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Sample - Superstore.csv"
MODEL_PATH = BASE_DIR / "models" / "forecasting_model.pkl"
PROPHET_MODEL_PATH = BASE_DIR / "models" / "prophet_model.pkl"


@st.cache_data
def load_data():
    # Try utf-8 first, fall back to latin1 / cp1252 for files with special characters
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"], dayfirst=False, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to read CSV with utf-8/latin1/cp1252 encodings")
    # Normalize column names and strip non-breaking spaces
    df.columns = [c.replace("\xa0", " ").strip() if isinstance(c, str) else c for c in df.columns]
    df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c, inplace=True)
    return df


def prepare_monthly(df):
    df = df.copy()
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Order Month")[['Sales', 'Profit']].sum().reset_index()
    monthly = monthly.sort_values("Order Month")
    monthly["month_index"] = np.arange(len(monthly))
    return monthly



def train_and_save_model(monthly):
    X = monthly[["month_index"]].values
    y = monthly["Sales"].values
    model = LinearRegression()
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def load_or_train_model(monthly):
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_and_save_model(monthly)
    return model


def train_prophet(monthly):
    # monthly: DataFrame with 'Order Month' and 'Sales'
    m = monthly[["Order Month", "Sales"]].rename(columns={"Order Month": "ds", "Sales": "y"}).copy()
    # Prophet expects ds as datetime
    m["ds"] = pd.to_datetime(m["ds"])
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    # Add monthly seasonality explicitly (approx 30.5 days)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(m)
    return model


def load_or_train_prophet(monthly):
    """Load a saved Prophet model if present, otherwise train and save a new one."""
    if PROPHET_AVAILABLE and PROPHET_MODEL_PATH.exists():
        try:
            with open(PROPHET_MODEL_PATH, 'rb') as f:
                pm = pickle.load(f)
                return pm
        except Exception:
            pass
    # train and save
    pm = train_prophet(monthly)
    PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(PROPHET_MODEL_PATH, 'wb') as f:
            pickle.dump(pm, f)
    except Exception:
        # ignore save errors but return model
        pass
    return pm


def save_prophet_model(model):
    try:
        PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROPHET_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception:
        return False


def prophet_forecast(prophet_model, monthly, periods=12):
    # Build future dataframe
    last = monthly["Order Month"].max()
    future = prophet_model.make_future_dataframe(periods=periods, freq="MS")
    forecast = prophet_model.predict(future)
    # Keep relevant columns and rename
    fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly"]].copy()
    return fc


def plot_prophet_forecast(monthly, fc, show_ci=True):
    fig = go.Figure()
    # Historical actuals
    fig.add_trace(go.Scatter(x=monthly["Order Month"], y=monthly["Sales"], mode="lines+markers", name="Actual", line=dict(color="#00cc96")))
    # Predicted (yhat) for entire range
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Prophet: yhat", line=dict(color="#ab63fa")))
    if show_ci and "yhat_lower" in fc and "yhat_upper" in fc:
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines", name="Upper", line=dict(color="rgba(171,99,250,0.0)"), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines", name="Lower", line=dict(color="rgba(171,99,250,0.0)"), fill='tonexty', fillcolor='rgba(171,99,250,0.15)', showlegend=False))
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    return fig


def plot_prophet_components(fc, prophet_model):
    # Use components returned in fc (trend, yearly) and compute monthly from ds
    comps = []
    # Trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=fc["ds"], y=fc["trend"], mode="lines", name="Trend", line=dict(color="#636efa")))
    fig_trend.update_layout(template="plotly_dark", title="Trend")
    comps.append(("Trend", fig_trend))

    # Yearly seasonality: show as mean by day-of-year
    if "yearly" in fc.columns:
        yearly = fc[["ds", "yearly"]].copy()
        yearly["doy"] = yearly["ds"].dt.dayofyear
        yearly_agg = yearly.groupby("doy")["yearly"].mean().reset_index()
        fig_year = go.Figure()
        fig_year.add_trace(go.Scatter(x=yearly_agg["doy"], y=yearly_agg["yearly"], mode="lines", name="Yearly"))
        fig_year.update_layout(template="plotly_dark", title="Yearly Seasonality (avg by day of year)")
        comps.append(("Yearly", fig_year))

    # Monthly effect: approximate by grouping by month
    monthly_effect = fc[["ds", "yhat"]].copy()
    monthly_effect["month"] = monthly_effect["ds"].dt.month
    mon_agg = monthly_effect.groupby("month")["yhat"].mean().reset_index()
    fig_mon = go.Figure()
    fig_mon.add_trace(go.Bar(x=mon_agg["month"], y=mon_agg["yhat"], name="Monthly Avg Forecast", marker_color="#00cc96"))
    fig_mon.update_layout(template="plotly_dark", title="Monthly Forecast Pattern")
    comps.append(("Monthly", fig_mon))

    return comps


def forecast(model, monthly, periods=12):
    last_index = monthly["month_index"].max()
    future_idx = np.arange(last_index + 1, last_index + 1 + periods)
    preds = model.predict(future_idx.reshape(-1, 1))
    future_dates = pd.date_range(start=monthly["Order Month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    future_df = pd.DataFrame({"Order Month": future_dates, "Sales": preds, "month_index": future_idx})
    return future_df


def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2


def main():
    st.set_page_config(page_title="AI-Powered Retail Forecasting Intelligence Platform", layout="wide")

    # Custom CSS for dark glassmorphism style
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #cfd8dc }
        .card { background: rgba(255,255,255,0.03); border-radius:12px; padding:14px; box-shadow: 0 6px 18px rgba(0,0,0,0.6);}
        .kpi { font-size:20px; color:#e6eef3 }
        .kpi-value { font-size:28px; font-weight:700; color:#fff }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    monthly = prepare_monthly(df)
    model = load_or_train_model(monthly)

    # Sidebar filters
    st.sidebar.title("Filters")
    regions = df["Region"].dropna().unique().tolist()
    categories = df["Category"].dropna().unique().tolist()
    sel_regions = st.sidebar.multiselect("Region", options=regions, default=regions)
    sel_categories = st.sidebar.multiselect("Category", options=categories, default=categories)
    min_date = df["Order Date"].min()
    max_date = df["Order Date"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    forecast_periods = st.sidebar.slider("Forecast months", min_value=1, max_value=24, value=12)
    # Model selector
    model_option = st.sidebar.selectbox("Forecast model", options=["Linear Regression", "Prophet Forecasting"])
    show_conf = st.sidebar.checkbox("Show confidence intervals (Prophet)", value=True)

    # Filtered dataframe for KPIs and region/category panels
    mask = df["Region"].isin(sel_regions) & df["Category"].isin(sel_categories) & (df["Order Date"] >= pd.to_datetime(date_range[0])) & (df["Order Date"] <= pd.to_datetime(date_range[1]))
    df_f = df.loc[mask]
    monthly_f = prepare_monthly(df_f)

    # Header
    # Branding / Hero
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px'>
        <div>
            <h1 style='margin:0;padding:0;color:#ffffff'>AI-Powered Retail Forecasting Intelligence Platform</h1>
            <div style='color:#9fb3c8'>Enterprise-grade forecasting & analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Premium analytics · Interactive forecasting · Business insights")

    # Executive overview KPIs
    total_sales = df_f["Sales"].sum()
    total_profit = df_f["Profit"].sum()
    best_region = df_f.groupby("Region")["Sales"].sum().idxmax() if not df_f.empty else "-"
    best_category = df_f.groupby("Category")["Sales"].sum().idxmax() if not df_f.empty else "-"

    col1, col2, col3, col4 = st.columns([1.5,1.5,1.5,1.5])
    with col1:
        st.markdown(f"<div class='card'><div class='kpi'>Total Sales</div><div class='kpi-value'>${total_sales:,.0f} 💰</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='kpi'>Total Profit</div><div class='kpi-value'>${total_profit:,.0f} 📈</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><div class='kpi'>Best Region</div><div class='kpi-value'>{best_region} 🌍</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='card'><div class='kpi'>Top Category</div><div class='kpi-value'>{best_category} 🏷️</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Sales Trend Analysis
    st.subheader("Sales Trend Analysis")
    if monthly_f.empty:
        st.info("No data for selected filters/date range.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_f["Order Month"], y=monthly_f["Sales"], mode="lines+markers", name="Monthly Sales", line=dict(color="#00cc96")))
        # Rolling average
        monthly_f["rolling6"] = monthly_f["Sales"].rolling(6, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=monthly_f["Order Month"], y=monthly_f["rolling6"], mode="lines", name="6-mo MA", line=dict(color="#636efa", dash="dash")))
        fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(t=20))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AI Sales Forecasting
    st.subheader("AI Sales Forecasting")
    # Choose dataset for model training: filtered monthly if present, else full monthly
    monthly_model = monthly_f if (not monthly_f.empty) else monthly

    # Safety check for insufficient data
    if len(monthly_model) < 6:
        st.warning("Insufficient monthly data for reliable forecasting. Please broaden the date range or filters.")
    else:
        if model_option == "Linear Regression":
            # Linear Regression path (existing)
            X_hist = monthly_model[["month_index"]].values
            preds_hist = model.predict(X_hist)
            mae, rmse, r2 = calc_metrics(monthly_model["Sales"].values, preds_hist)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=monthly_model["Order Month"], y=monthly_model["Sales"], mode="lines+markers", name="Actual", line=dict(color="#00cc96")))
            fig2.add_trace(go.Scatter(x=monthly_model["Order Month"], y=preds_hist, mode="lines", name="Predicted", line=dict(color="#ef553b", dash="dot")))
            # Future forecast
            future_df = forecast(model, monthly_model, periods=forecast_periods)
            fig2.add_trace(go.Scatter(x=future_df["Order Month"], y=future_df["Sales"], mode="lines", name="Forecast", line=dict(color="#ab63fa")))
            fig2.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("MAE", f"${mae:,.2f}")
            col_b.metric("RMSE", f"${rmse:,.2f}")
            col_c.metric("R²", f"{r2:.3f}")

        else:
            # Prophet path
            if not PROPHET_AVAILABLE:
                st.error("Prophet is not installed. Install the 'prophet' package and restart the app.")
            else:
                # Prepare dataframe for Prophet
                dfp = monthly_model[["Order Month", "Sales"]].rename(columns={"Order Month": "ds", "Sales": "y"}).copy()
                # Train prophet with interval width based on toggle
                interval = 0.95 if show_conf else 0.80
                try:
                    # Attempt to load existing Prophet model or train a new one
                    with st.spinner('Loading or training Prophet model...'):
                        m = load_or_train_prophet(monthly_model)

                    # Allow user to retrain and save a new model
                    if st.button('Retrain & Save Prophet Model'):
                        with st.spinner('Retraining Prophet model...'):
                            m_new = Prophet(interval_width=interval, yearly_seasonality=True)
                            m_new.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                            m_new.fit(dfp)
                            saved = save_prophet_model(m_new)
                            if saved:
                                st.success('Prophet model retrained and saved to models/prophet_model.pkl')
                                m = m_new
                            else:
                                st.warning('Model retrained but failed to save to disk.')

                    future = m.make_future_dataframe(periods=forecast_periods, freq='MS')
                    forecast_df = m.predict(future)

                    # In-sample predictions for metrics
                    hist_pred = m.predict(dfp)
                    mae, rmse, r2 = calc_metrics(dfp['y'].values, hist_pred['yhat'].values)

                    # Plot actual vs forecast
                    figp = go.Figure()
                    figp.add_trace(go.Scatter(x=dfp['ds'], y=dfp['y'], mode='markers+lines', name='Actual', line=dict(color='#00cc96')))
                    figp.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='#ab63fa')))
                    if show_conf:
                        figp.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper', line=dict(color='rgba(171,99,250,0.15)'), showlegend=False))
                        figp.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower', line=dict(color='rgba(171,99,250,0.15)'), fill='tonexty', fillcolor='rgba(171,99,250,0.1)', showlegend=False))
                    figp.update_layout(template='plotly_dark', hovermode='x unified')
                    st.plotly_chart(figp, use_container_width=True)

                    # Download buttons: forecast CSV and components
                    try:
                        csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download Prophet forecast (CSV)', data=csv_bytes, file_name='prophet_forecast.csv', mime='text/csv')
                    except Exception:
                        st.warning('Unable to prepare CSV download for forecast.')

                    try:
                        comps_cols = ['ds'] + [c for c in forecast_df.columns if c not in ['ds','yhat','yhat_lower','yhat_upper']]
                        comps_df = forecast_df[comps_cols]
                        comps_bytes = comps_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download Prophet components (CSV)', data=comps_bytes, file_name='prophet_components.csv', mime='text/csv')
                    except Exception:
                        pass

                    # Optional: export main forecast plot as PNG (requires kaleido)
                    try:
                        img = figp.to_image(format='png')
                        st.download_button('Download Forecast PNG', data=img, file_name='prophet_forecast.png', mime='image/png')
                    except Exception:
                        # silently ignore if to_image not available
                        pass

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric('MAE', f"${mae:,.2f}")
                    col_b.metric('RMSE', f"${rmse:,.2f}")
                    col_c.metric('R²', f"{r2:.3f}")

                    # Forecast components
                    st.markdown('**Forecast Components (Prophet)**')
                    try:
                        if plot_components_plotly is not None:
                            comp_fig = plot_components_plotly(m, forecast_df)
                            st.plotly_chart(comp_fig, use_container_width=True)
                        else:
                            # Fallback: show trend and seasonal plots manually
                            if 'trend' in forecast_df.columns:
                                ftrend = go.Figure()
                                ftrend.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['trend'], name='Trend', line=dict(color='#36a2eb')))
                                ftrend.update_layout(template='plotly_dark')
                                st.plotly_chart(ftrend, use_container_width=True)
                    except Exception:
                        st.warning('Unable to render Prophet components plot.')

                except Exception as e:
                    st.error(f"Prophet training/loading failed: {e}")

    st.markdown("---")

    # Regional Performance
    st.subheader("Regional Performance")
    reg = df_f.groupby("Region")[['Sales', 'Profit']].sum().reset_index()
    if not reg.empty:
        fig3 = px.bar(reg, x="Region", y=["Sales", "Profit"], barmode="group", template="plotly_dark", color_discrete_sequence=["#00cc96", "#ffa15a"]) 
        st.plotly_chart(fig3, use_container_width=True)

    # Category Insights
    st.subheader("Product Category Insights")
    cat = df_f.groupby("Category")["Sales"].sum().reset_index()
    if not cat.empty:
        fig4 = px.pie(cat, values="Sales", names="Category", hole=0.45, template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # AI Business Insights (simple rule-based)
    st.subheader("AI Business Insights")
    insights = []
    if not df_f.empty:
        top_cat = cat.sort_values("Sales", ascending=False).iloc[0]["Category"]
        insights.append(f"Top revenue category: {top_cat}.")
        growth = (monthly["Sales"].iloc[-1] - monthly["Sales"].iloc[0]) / monthly["Sales"].iloc[0]
        if growth > 0.05:
            insights.append("Sales show positive growth over the selected period.")
        else:
            insights.append("Sales are flat or declining — consider promotions or product reviews.")
        strong_region = best_region
        insights.append(f"Strongest region by sales: {strong_region}.")

    for ins in insights:
        st.markdown(f"- {ins}")

    st.sidebar.markdown("---")
    st.sidebar.write("Model path: ", MODEL_PATH)


if __name__ == "__main__":
    main()
