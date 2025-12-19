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
