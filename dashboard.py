import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(page_title="Wearable Health Dashboard", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Try different paths
    possible_paths = [
        "Wearable_data/data_processing/apple_health_raw.csv",
        "Wearable_data/apple_health_raw.csv",
        "data_processing/apple_health_raw.csv",
        "apple_health_raw.csv"
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
            break
        except FileNotFoundError:
            continue
            
    if df is None:
        st.error("Could not find apple_health_raw.csv. Please ensure the data file exists.")
        return None

    # Date parsing
    df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
    df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')
    
    # Value parsing
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    return df

def process_metrics(df):
    # Filter for relevant metrics
    metrics = [
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "HKQuantityTypeIdentifierVO2Max",
        "HKQuantityTypeIdentifierStepCount",
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierAppleStandTime",
        "HKQuantityTypeIdentifierRespiratoryRate",
        "HKQuantityTypeIdentifierOxygenSaturation",
        "HKQuantityTypeIdentifierWalkingSpeed"
    ]
    
    # Filter for Watch data only for steps to avoid double counting
    # For other metrics, it's usually fine, but consistency is good.
    # However, some data might come from other sources. 
    # The instruction said: ALWAYS filter by `sourceName` containing "Watch" when analyzing steps.
    
    df_steps = df[
        (df['type'] == 'HKQuantityTypeIdentifierStepCount') & 
        (df['sourceName'].str.contains('Watch', case=False, na=False))
    ].copy()
    
    df_other = df[
        (df['type'].isin(metrics)) & 
        (df['type'] != 'HKQuantityTypeIdentifierStepCount')
    ].copy()
    
    df_metrics = pd.concat([df_steps, df_other])
    
    # Pivot/Resample to daily
    # We need to aggregate differently: Steps = Sum, RHR/HRV/VO2 = Avg (or specific logic)
    
    df_metrics['date'] = df_metrics['startDate'].dt.date
    df_metrics['date'] = pd.to_datetime(df_metrics['date'])
    
    # Steps
    daily_steps = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierStepCount'].groupby('date')['value'].sum()
    
    # RHR
    daily_rhr = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierRestingHeartRate'].groupby('date')['value'].mean()
    
    # HRV
    daily_hrv = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN'].groupby('date')['value'].mean()
    
    daily_vo2 = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierVO2Max'].groupby('date')['value'].mean()

    # Active Energy (Sum)
    daily_energy = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierActiveEnergyBurned'].groupby('date')['value'].sum()

    # Stand Time (Sum)
    daily_stand = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierAppleStandTime'].groupby('date')['value'].sum()

    # Respiratory Rate (Mean)
    daily_resp = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierRespiratoryRate'].groupby('date')['value'].mean()

    # SpO2 (Mean)
    daily_spo2 = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierOxygenSaturation'].groupby('date')['value'].mean()

    # Walking Speed (Mean)
    daily_speed = df_metrics[df_metrics['type'] == 'HKQuantityTypeIdentifierWalkingSpeed'].groupby('date')['value'].mean()
    
    # --- Recovery Score Calculation ---
    # Formula: Recovery % based on HRV, RHR, Respiratory Rate vs 30-day baseline
    
    # 1. Calculate Baselines (30-day rolling mean, shifted by 1 day to not include today in baseline)
    rhr_baseline = daily_rhr.rolling(window=30).mean().shift(1)
    hrv_baseline = daily_hrv.rolling(window=30).mean().shift(1)
    resp_baseline = daily_resp.rolling(window=30).mean().shift(1)
    
    # 2. Calculate Deviation %
    # HRV: Higher is better
    hrv_score = (daily_hrv - hrv_baseline) / hrv_baseline
    
    # RHR: Lower is better -> Negate the change
    rhr_score = (rhr_baseline - daily_rhr) / rhr_baseline
    
    # Respiratory Rate: Lower is generally better (calm), but stable is key. 
    # For simple recovery, we'll treat Lower as better.
    resp_score = (resp_baseline - daily_resp) / resp_baseline
    
    # 3. Combine Scores (Equal Weighting)
    # Fill NAs with 0 (neutral) for days missing some metrics, or just let them be NaN
    # We averaging the available scores
    raw_recovery = pd.DataFrame({
        'hrv': hrv_score,
        'rhr': rhr_score,
        'resp': resp_score
    }).mean(axis=1)
    
    # 4. Map to 0-100 Scale
    # Assuming the raw_recovery varies roughly between -0.2 (-20%) to +0.2 (+20%) usually
    # Map 0.0 (baseline) to 50.
    # Map +0.2 to 100? Map -0.2 to 0?
    # Let's apply a sigmoid or simple linear clamp.
    # Linear: 50 + (raw_recovery * 250) -> 0.2 becomes +50 (total 100)
    
    recovery_score = 50 + (raw_recovery * 250)
    recovery_score = recovery_score.clip(lower=0, upper=100)

    # Combine into one dataframe
    daily_df = pd.DataFrame({
        'Steps': daily_steps,
        'RHR': daily_rhr,
        'HRV': daily_hrv,
        'VO2Max': daily_vo2,
        'ActiveEnergy': daily_energy,
        'StandTime': daily_stand,
        'RespiratoryRate': daily_resp,
        'SpO2': daily_spo2,
        'WalkingSpeed': daily_speed,
        'RecoveryScore': recovery_score
    })
    
    return daily_df

def process_sleep(df):
    # Filter for sleep analysis
    sleep_df = df[df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis'].copy()
    
    if sleep_df.empty:
        return None
        
    # Calculate duration in hours
    sleep_df['duration'] = (sleep_df['endDate'] - sleep_df['startDate']).dt.total_seconds() / 3600
    
    # We want to sum duration per "night". 
    # A simple heuristic is to shift the date back by a few hours (e.g. 4 hours) so late night sleep counts for the previous day's "night",
    # or just group by the date of the startDate. 
    # Let's group by the date of the startDate for simplicity, but maybe shift 12 hours to align "night" to the date the sleep started mostly?
    # Actually, usually sleep is credited to the day it ends or starts. Let's stick to startDate.date for now.
    
    sleep_df['date'] = sleep_df['startDate'].dt.date
    sleep_df['date'] = pd.to_datetime(sleep_df['date'])
    
    # Filter for "Asleep" values if possible. The value column usually contains strings like "HKCategoryValueSleepAnalysisAsleep"
    # But sometimes it's just "InBed". Let's try to find "Asleep" or "Core" or "Deep" or "REM".
    # If we only have "InBed", we use that.
    
    # Check unique values in 'value' column for sleep
    # For now, let's just sum all sleep records per day to get "Total Sleep"
    
    daily_sleep = sleep_df.groupby('date')['duration'].sum()
    
    return daily_sleep

# --- Main App ---
st.title("🏃‍♂️ Wearable Health Dashboard")

raw_df = load_data()

if raw_df is not None:
    # Process Data
    daily_metrics = process_metrics(raw_df)
    daily_sleep = process_sleep(raw_df)
    
    if daily_sleep is not None:
        daily_metrics['SleepDuration'] = daily_sleep
    
    # --- Sidebar ---
    st.sidebar.header("Settings")
    
    # Date Range
    min_date = daily_metrics.index.min()
    max_date = daily_metrics.index.max()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter Data
    mask = (daily_metrics.index >= pd.Timestamp(start_date)) & (daily_metrics.index <= pd.Timestamp(end_date))
    filtered_df = daily_metrics.loc[mask]
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlations", "Sleep Analysis", "Raw Data"])
    
    with tab1:
        st.header("Overview")
        
        # Metrics Summary
        cols = st.columns(len(filtered_df.columns))
        for i, col_name in enumerate(filtered_df.columns):
            avg_val = filtered_df[col_name].mean()
            current_val = filtered_df[col_name].iloc[-1] if not filtered_df[col_name].isna().all() else 0
            
            with cols[i]:
                st.metric(
                    label=col_name,
                    value=f"{current_val:.1f}",
                    delta=f"{current_val - avg_val:.1f} vs Avg"
                )
        
        # Trends
        st.subheader("Metric Trends")
        
        # Metric Selector
        metric_options = filtered_df.columns.tolist()
        # defaulting to steps or RHR if available
        default_index = 0
        if "RHR" in metric_options:
            default_index = metric_options.index("RHR")
            
        selected_metric = st.selectbox(
            "Select Metric to Visualize", 
            options=metric_options,
            index=default_index
        )
        
        # View Mode Selector
        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            view_mode = st.radio(
                "View Mode",
                ["7-Day Average", "Raw Daily", "Both"],
                horizontal=True
            )
        with col_opts2:
            show_trend = st.checkbox("Show Linear Trend", value=False)
        
        # Prepare data for plotting
        if selected_metric:
            chart_data = pd.DataFrame()
            chart_colors = []
            
            # 1. Rolling Average
            if view_mode in ["7-Day Average", "Both"]:
                rolling_series = filtered_df[selected_metric].rolling(window=7).mean()
                chart_data['7-Day Average'] = rolling_series
                chart_colors.append("#0068C9") # Standard Blue
            
            # 2. Raw Data
            if view_mode in ["Raw Daily", "Both"]:
                chart_data['Daily Raw'] = filtered_df[selected_metric]
                chart_colors.append("#83C9FF") # Light Blue
                
            # 3. Linear Trend Line
            slope_val = None
            if show_trend:
                # Remove NaNs for calculation
                y = filtered_df[selected_metric].dropna()
                if len(y) > 1:
                    x = np.arange(len(y))
                    z = np.polyfit(x, y, 1)
                    slope_val = z[0] # Slope
                    p = np.poly1d(z)
                    
                    # Create a series relative to original dates
                    trend_series = pd.Series(p(x), index=y.index)
                    # Reindex to full timeframe to align with chart
                    chart_data['Linear Trend'] = trend_series
                    chart_colors.append("#FF0000") # Red
            
            st.line_chart(chart_data, color=chart_colors)
            
            # Optional: Show stats for the selected metric
            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Current (Daily)", f"{filtered_df[selected_metric].iloc[-1]:.1f}")
            with stats_cols[1]:
                if "7-Day Average" in chart_data.columns:
                    val = chart_data['7-Day Average'].iloc[-1]
                    st.metric("7-Day Avg", f"{val:.1f}")
                else:
                    st.metric("7-Day Avg", "N/A")
            with stats_cols[2]:
                if slope_val is not None:
                    st.metric("Trend Slope", f"{slope_val:.3f} / day")
                else:
                    st.metric("All-time Avg", f"{filtered_df[selected_metric].mean():.1f}")
        
    with tab2:
        st.header("Correlation Analysis")
        st.write("Heatmap showing relationships between different health metrics.")
        
        if not filtered_df.empty:
            corr = filtered_df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
            st.pyplot(fig)
            
            st.info("""
            **How to read this:**
            - **1.0 (Red)**: Perfect positive correlation (e.g. as one goes up, the other goes up).
            - **-1.0 (Blue)**: Perfect negative correlation (e.g. as one goes up, the other goes down).
            - **0.0 (White)**: No correlation.
            
            *Look for negative correlations between Stress (low HRV) and other factors.*
            """)
        else:
            st.warning("Not enough data for correlation analysis.")

    with tab3:
        st.header("Sleep Analysis")
        
        if 'SleepDuration' in filtered_df.columns:
            st.subheader("Sleep Duration vs Recovery")
            
            # Dual axis plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Sleep Duration (hours)', color=color)
            ax1.bar(filtered_df.index, filtered_df['SleepDuration'], color=color, alpha=0.6, label='Sleep')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
            color = 'tab:red'
            ax2.set_ylabel('HRV (ms)', color=color)  # we already handled the x-label with ax1
            ax2.plot(filtered_df.index, filtered_df['HRV'], color=color, linewidth=2, label='HRV')
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            st.pyplot(fig)
            
            st.write("### Scatter Plot: Sleep vs HRV")
            fig2, ax = plt.subplots()
            sns.regplot(data=filtered_df, x='SleepDuration', y='HRV', ax=ax)
            st.pyplot(fig2)
            
        else:
            st.warning("No Sleep Data Available")

    with tab4:
        st.header("Raw Data")
        st.dataframe(filtered_df)
