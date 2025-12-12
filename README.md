# Wearable Data Health Agent

## Project Overview
This project aims to leverage personal wearables data to provide deep health insights. The ultimate goal is to build a system that combines:
1.  **Data Analysis**: Visualizing and tracking key health metrics (Heart Rate, HRV, VO2 Max, Steps).
2.  **Machine Learning**: Using ML to uncover patterns, anomalies, and predictive insights from the data.
3.  **Agentic Intelligence**: Using AI agents to retrieve expert scientific information and contextualize the user's data against medical research and benchmarks.

## Current Status
- **Dashboard**: A Jupyter Notebook (`wearbale_playground.ipynb`) visualizes key metrics.
- **Benchmarks**: Data is compared against benchmarks for user (e.g., age, height, etc).
- **Data Pipeline**: Raw Apple Health export (`apple_health_raw.csv`) is processed to extract relevant metrics.
