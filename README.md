# 📊 Data Analysis and Visualization Dashboard

## Overview
This is an **interactive data analysis dashboard** built with Python and Streamlit.  
Upload your CSV files, explore data, generate visualizations, handle missing values, and download summary reports — all in a user-friendly interface.

---

## Features
- Upload and preview CSV datasets
- Descriptive statistics for numeric and categorical data
- Histograms, bar charts, boxplots, scatter plots, correlation heatmaps, and pairplots
- Missing data inspection and handling
- Download descriptive summary report
- Fully interactive Streamlit app

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/data_analysis_dashboard.git
cd data_analysis_dashboard

2. Install dependencies

pip install -r requirements.txt


3. Run the dashboard
streamlit run data_analysis_dashboard.py

Usage

Go to 📂 Upload Dataset and upload a CSV file.

Explore data in 📊 Data Summary.

Visualize columns using 📈 Visualization.

Inspect and handle missing data in 🧩 Missing Data Handling.

Download a summary report in 📥 Download Report.

Notes

Only CSV files are supported.

Original uploaded data is preserved in memory; all transformations apply to the in-memory copy.

Ensure CSVs have headers in the first row and consistent column types.

The app is designed to handle invalid inputs gracefully and will display helpful error messages.

Author

Muhammad Sharique
BS Software Engineering, University of Sindh

License

This project is open-source and free to use for educational purposes.
