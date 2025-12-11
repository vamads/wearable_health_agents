# Wearable Data Health Agent

## Project Overview
This project aims to leverage personal wearables data to provide deep health insights. The ultimate goal is to build a system that combines:
1.  **Data Analysis**: Visualizing and tracking key health metrics (Heart Rate, HRV, VO2 Max, Steps).
2.  **Machine Learning**: Using ML to uncover patterns, anomalies, and predictive insights from the data.
3.  **Agentic Intelligence**: Using AI agents to retrieve expert scientific information and contextualize the user's data against medical research and benchmarks.

## Current Status
- **Dashboard**: A Jupyter Notebook (`wearbale_playground.ipynb`) visualizes key metrics.
- **Benchmarks**: Data is compared against benchmarks for a 26-year-old male.
- **Data Pipeline**: Raw Apple Health export (`apple_health_raw.csv`) is processed to extract relevant metrics.

## Instructions for Coding Agents
If you are an AI agent assisting with this project, please follow these guidelines:

### 1. Context & Goals
- The user wants to understand *their* health, not just see numbers. Always try to provide context (e.g., "Is this good?", "What does science say?").
- Future work involves integrating ML models (e.g., forecasting, anomaly detection) and external knowledge retrieval (e.g., searching PubMed).

### 2. Data Handling
- **Source**: `data_processing/apple_health_raw.csv` (or root `apple_health_raw.csv`).
- **Caveats**:
    - **Step Counts**: The raw CSV contains duplicate step counts from both iPhone and Apple Watch. **ALWAYS filter by `sourceName` containing "Watch"** when analyzing steps to avoid double-counting.
    - **Dates**: Ensure proper timezone handling when parsing `startDate`.

### 3. Tech Stack
- **Language**: Python
- **Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebooks are preferred for exploration and visualization.

### 4. Workflow
- When creating new visualizations or analyses, prefer updating the `wearbale_playground.ipynb` or creating a new notebook.
- Use `data_processing/update_notebook.py` as a reference for how to programmatically generate/update the notebook if needed.
