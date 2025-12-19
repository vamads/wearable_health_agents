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
