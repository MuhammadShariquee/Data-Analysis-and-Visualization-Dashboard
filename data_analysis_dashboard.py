# data_analysis_dashboard.py
# -------------------------
# Imports (only allowed libs)
# -------------------------
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as imp_err:
    # If imports fail, print message (Streamlit will show this in logs)
    print(f"Error importing required libraries: {imp_err}")
    raise

# -------------------------
# Global settings & theme
# -------------------------
sns.set_theme(style="whitegrid")  # consistent Seaborn styling
plt.rcParams["figure.dpi"] = 110

# -------------------------
# Helper functions
# -------------------------
def safe_read_csv(uploaded_file):
    """
    Safely read a CSV uploaded via Streamlit file_uploader.
    Returns a pandas DataFrame or None on failure.
    """
    if uploaded_file is None:
        return None
    try:
        # Validate file size (guard against empty file)
        try:
            size = uploaded_file.size
        except Exception:
            size = None
        if size == 0:
            st.error("Invalid or empty file. The uploaded file has zero bytes.")
            return None

        # Attempt reading CSV
        df_local = pd.read_csv(uploaded_file)
        # If dataframe empty after reading, warn
        if df_local is None or (isinstance(df_local, pd.DataFrame) and df_local.empty):
            st.error("Invalid or empty file. No data found after reading CSV.")
            return None
        return df_local
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def get_numeric_columns(df):
    """Return list of numeric column names."""
    try:
        return df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        return []


def get_categorical_columns(df):
    """Return list of non-numeric column names."""
    try:
        return df.select_dtypes(exclude=[np.number]).columns.tolist()
    except Exception:
        return []


def show_dataframe_preview(df):
    """Show preview (first rows) of dataframe safely."""
    try:
        if df is None:
            return
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Unable to display dataframe preview: {e}")


def compute_numeric_summary(df):
    """Compute mean, median, std for numeric columns."""
    try:
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            return None
        means = df[numeric_cols].apply(np.mean, axis=0, skipna=True)
        medians = df[numeric_cols].apply(np.median, axis=0)
        stds = df[numeric_cols].apply(np.std, axis=0, ddof=0)
        summary_df = pd.DataFrame({
            "mean": means,
            "median": medians,
            "std": stds
        })
        return summary_df
    except Exception:
        return None


def plot_histogram(df, column):
    """Plot histogram for a numeric column."""
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=30)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {column}")
    plt.tight_layout()
    return fig


def plot_bar(df, column):
    """Plot bar chart for a categorical or discrete column."""
    fig, ax = plt.subplots()
    vc = df[column].value_counts(dropna=False).head(50)
    vc.plot(kind="bar", ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"Bar Chart of {column}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_box(df, column):
    """Plot boxplot for a numeric column."""
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Box Plot of {column}")
    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    """Scatter plot for two numeric columns."""
    fig, ax = plt.subplots()
    ax.scatter(df[x_col].dropna(), df[y_col].dropna(), alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.tight_layout()
    return fig


def plot_corr_heatmap(df):
    """Correlation heatmap for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, cmap="vlag", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_pairplot(df, cols):
    """Seaborn pairplot (returns fig via the returned object)."""
    # Seaborn pairplot returns a PairGrid object with .fig attribute
    g = sns.pairplot(df[cols].dropna())
    plt.tight_layout()
    return g.fig


def missing_data_table(df):
    """Return missing count & percentage table."""
    try:
        total = df.shape[0]
        miss_count = df.isnull().sum()
        miss_percent = (miss_count / total * 100).round(2)
        miss_table = pd.DataFrame({
            "missing_count": miss_count,
            "missing_percent": miss_percent
        }).sort_values(by="missing_count", ascending=False)
        return miss_table
    except Exception:
        return None


def safe_fill_mean(df):
    """Fill numeric columns with their mean. Returns modified copy."""
    try:
        df_copy = df.copy()
        numeric_cols = get_numeric_columns(df_copy)
        for col in numeric_cols:
            try:
                mean_val = df_copy[col].mean()
                df_copy[col].fillna(mean_val, inplace=True)
            except Exception:
                # If fill fails for a column, continue
                continue
        return df_copy
    except Exception:
        return df


# -------------------------
# Streamlit app layout start
# -------------------------
def main():
    st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
    st.title("📊 Data Analysis and Visualization Dashboard")
    st.caption("Upload a CSV and explore your data. Built with Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit.")
    st.markdown("---")

    # Initialize session state variables to persist dataframe and messages
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = None  # keep a copy of original upload
    if "last_action" not in st.session_state:
        st.session_state["last_action"] = ""

    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "📂 Upload Dataset",
        "📊 Data Summary",
        "📈 Visualization",
        "🧩 Missing Data Handling",
        "📥 Download Report",
        "ℹ️ About"
    ]
    choice = st.sidebar.selectbox("Go to", pages)

    # -------------------------
    # Upload Dataset Section
    # -------------------------
    if choice == "📂 Upload Dataset":
        st.header("📂 Upload Dataset")
        st.caption("Upload a CSV file to begin. The app validates and reads the file safely.")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)
        # If user uploads a new file, attempt to read
        if uploaded_file is not None:
            try:
                df = safe_read_csv(uploaded_file)
                if df is not None:
                    # store both original and working copy
                    st.session_state["df"] = df.copy()
                    st.session_state["original_df"] = df.copy()
                    st.success("File uploaded and loaded successfully ✅")
                else:
                    # safe_read_csv already displayed error
                    st.session_state["df"] = None
            except Exception as e:
                st.error(f"Unexpected error while loading file: {e}")
                st.session_state["df"] = None
        else:
            st.warning("Please upload a CSV file. 📁")

        # Show preview if available
        if st.session_state["df"] is not None:
            try:
                st.subheader("Dataset Preview")
                show_dataframe_preview(st.session_state["df"])
                st.write("**Shape:**", st.session_state["df"].shape)
                with st.expander("Show data types"):
                    st.table(st.session_state["df"].dtypes.astype(str))
            except Exception as e:
                st.error(f"Error displaying dataset information: {e}")

    # -------------------------
    # Data Summary Section
    # -------------------------
    elif choice == "📊 Data Summary":
        st.header("📊 Data Summary")
        st.caption("View descriptive statistics for numeric and categorical columns.")
        df = st.session_state.get("df", None)
        if df is None:
            st.warning("No dataset loaded. Please upload a CSV in 'Upload Dataset' first.")
        else:
            try:
                if df.empty:
                    st.error("No data available for summary.")
                else:
                    # Numeric summary using describe + custom stats
                    numeric_cols = get_numeric_columns(df)
                    if numeric_cols:
                        st.subheader("Numerical Summary")
                        try:
                            desc = df[numeric_cols].describe().T
                            # Add mean/median/std computed with numpy to ensure consistency
                            numeric_summary = compute_numeric_summary(df)
                            if numeric_summary is not None:
                                merged = desc.join(numeric_summary)
                                st.table(merged)
                            else:
                                st.table(desc)
                            st.caption("Statistics computed: count, mean, std, min, 25%, 50%, 75%, max. Additional: mean, median, std.")
                        except Exception as e:
                            st.error(f"Error computing numeric summary: {e}")
                    else:
                        st.warning("No numeric columns found.")

                    # Categorical columns summary
                    cat_cols = get_categorical_columns(df)
                    if cat_cols:
                        st.subheader("Categorical Columns (Top 5 values)")
                        try:
                            for col in cat_cols:
                                st.markdown(f"**{col}**")
                                vc = df[col].value_counts(dropna=False).head(5)
                                st.table(vc)
                        except Exception as e:
                            st.error(f"Error computing categorical summaries: {e}")
                    else:
                        st.info("No categorical columns found (all columns are numeric).")
            except Exception as e:
                st.error(f"Unexpected error in Data Summary: {e}")

    # -------------------------
    # Visualization Section
    # -------------------------
    elif choice == "📈 Visualization":
        st.header("📈 Visualization")
        st.caption("Choose a plot type and select appropriate columns. All plotting is validated.")
        df = st.session_state.get("df", None)
        if df is None:
            st.warning("No dataset loaded. Please upload a CSV in 'Upload Dataset' first.")
        else:
            if df.empty:
                st.error("Dataset is empty. Nothing to visualize.")
            else:
                try:
                    viz_type = st.sidebar.selectbox("Select visualization", [
                        "Histogram",
                        "Bar Chart",
                        "Box Plot",
                        "Scatter Plot",
                        "Correlation Heatmap",
                        "Pairplot"
                    ])
                    numeric_cols = get_numeric_columns(df)
                    cat_cols = get_categorical_columns(df)

                    # Histogram
                    if viz_type == "Histogram":
                        st.subheader("Histogram")
                        if not numeric_cols:
                            st.warning("No numeric columns available for a histogram.")
                        else:
                            col = st.selectbox("Select numeric column", numeric_cols)
                            if col:
                                try:
                                    fig = plot_histogram(df, col)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting histogram: {e}")

                    # Bar Chart
                    elif viz_type == "Bar Chart":
                        st.subheader("Bar Chart")
                        # Allow either categorical or numeric discrete columns
                        possible_cols = cat_cols + numeric_cols
                        if not possible_cols:
                            st.warning("No columns available for bar chart.")
                        else:
                            col = st.selectbox("Select column for bar chart", possible_cols)
                            if col:
                                try:
                                    fig = plot_bar(df, col)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting bar chart: {e}")

                    # Box Plot
                    elif viz_type == "Box Plot":
                        st.subheader("Box Plot")
                        if not numeric_cols:
                            st.warning("No numeric columns available for box plot.")
                        else:
                            col = st.selectbox("Select numeric column", numeric_cols)
                            if col:
                                try:
                                    fig = plot_box(df, col)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting box plot: {e}")

                    # Scatter Plot
                    elif viz_type == "Scatter Plot":
                        st.subheader("Scatter Plot")
                        if len(numeric_cols) < 2:
                            st.warning("Need at least two numeric columns for a scatter plot.")
                        else:
                            x_col = st.selectbox("X-axis (numeric)", numeric_cols, index=0)
                            y_col = st.selectbox("Y-axis (numeric)", numeric_cols, index=min(1, len(numeric_cols)-1))
                            if x_col and y_col:
                                try:
                                    fig = plot_scatter(df, x_col, y_col)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting scatter plot: {e}")

                    # Correlation Heatmap
                    elif viz_type == "Correlation Heatmap":
                        st.subheader("Correlation Heatmap")
                        if len(numeric_cols) < 2:
                            st.warning("Not enough numeric data to plot a correlation heatmap.")
                        else:
                            try:
                                fig = plot_corr_heatmap(df)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error plotting correlation heatmap: {e}")

                    # Pairplot
                    elif viz_type == "Pairplot":
                        st.subheader("Pairplot")
                        if len(numeric_cols) < 2:
                            st.warning("Not enough numeric data to create a pairplot.")
                        else:
                            # let user choose subset of numeric columns for pairplot
                            default_selection = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                            cols = st.multiselect("Select numeric columns for pairplot (2-6 recommended)", numeric_cols, default=default_selection)
                            if not cols or len(cols) < 2:
                                st.info("Select at least two numeric columns to generate a pairplot.")
                            else:
                                try:
                                    fig = plot_pairplot(df, cols)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error plotting pairplot: {e}")

                except Exception as e:
                    st.error(f"Unexpected error in Visualization: {e}")

    # -------------------------
    # Missing Data Handling Section
    # -------------------------
    elif choice == "🧩 Missing Data Handling":
        st.header("🧩 Missing Data Handling")
        st.caption("Inspect missing values and apply simple strategies (drop rows or fill numeric columns with mean).")
        df = st.session_state.get("df", None)
        if df is None:
            st.warning("No dataset loaded. Please upload a CSV in 'Upload Dataset' first.")
        else:
            try:
                if df.empty:
                    st.error("Dataset is empty. Nothing to inspect.")
                else:
                    st.subheader("Missing Values Overview")
                    miss_table = missing_data_table(df)
                    if miss_table is None:
                        st.error("Unable to compute missing data statistics.")
                    else:
                        st.table(miss_table)

                    st.subheader("Missing Data Visualization")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
                        ax.set_title("Missing Data Heatmap (rows vs columns)")
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error visualizing missing data: {e}")

                    st.markdown("---")
                    st.subheader("Choose an action")
                    action = st.radio("Select missing data action", (
                        "No action",
                        "Drop rows with missing data",
                        "Fill numeric columns with mean"
                    ))

                    apply_btn = st.button("Apply Changes")
                    if apply_btn:
                        try:
                            if action == "No action":
                                st.info("No changes applied to the dataset.")
                            elif action == "Drop rows with missing data":
                                before_rows = st.session_state["df"].shape[0]
                                st.session_state["df"] = st.session_state["df"].dropna().reset_index(drop=True)
                                after_rows = st.session_state["df"].shape[0]
                                st.success(f"Dropped rows with missing data. Rows before: {before_rows}, after: {after_rows}.")
                                st.session_state["last_action"] = "dropna"
                            elif action == "Fill numeric columns with mean":
                                before_na = st.session_state["df"].isnull().sum().sum()
                                st.session_state["df"] = safe_fill_mean(st.session_state["df"])
                                after_na = st.session_state["df"].isnull().sum().sum()
                                st.success(f"Filled numeric columns with mean. Missing values before: {before_na}, after: {after_na}.")
                                st.session_state["last_action"] = "fill_mean"
                            else:
                                st.warning("Unknown action selected.")
                        except Exception as e:
                            st.error(f"Error applying missing data action: {e}")

                    st.caption("Actions are applied to the current in-memory dataset. Original upload is preserved in session_state['original_df'].")
            except Exception as e:
                st.error(f"Unexpected error in Missing Data Handling: {e}")

    # -------------------------
    # Download Report Section
    # -------------------------
    elif choice == "📥 Download Report":
        st.header("📥 Download Report")
        st.caption("Generate and download descriptive summary report (CSV).")
        df = st.session_state.get("df", None)
        if df is None:
            st.warning("No dataset to generate report. Upload a CSV in 'Upload Dataset' first.")
        else:
            try:
                if df.empty:
                    st.error("No data available to generate report.")
                else:
                    # Create descriptive summary using describe()
                    try:
                        summary = df.describe(include="all").T  # include both numeric & categorical
                        # convert to CSV bytes
                        csv = summary.to_csv().encode('utf-8')
                        st.subheader("Summary Preview")
                        st.dataframe(summary, use_container_width=True)
                        st.download_button("Download Summary Report", data=csv, file_name="summary_report.csv", mime="text/csv")
                        st.success("Summary report generated. Click the download button to save it as CSV.")
                    except Exception as e:
                        st.error(f"Error generating or preparing report: {e}")
            except Exception as e:
                st.error(f"Unexpected error in Download Report: {e}")

    # -------------------------
    # About Section
    # -------------------------
    elif choice == "ℹ️ About":
        st.header("ℹ️ About")
        try:
            st.markdown("""
            **Purpose:**  
            This dashboard lets you upload a CSV file, explore its structure, compute descriptive statistics, visualize data, handle missing values, and download a summary report — all within a user-friendly Streamlit interface. 📚

            **Tools Used:**  
            - Python  
            - Pandas  
            - NumPy  
            - Matplotlib  
            - Seaborn  
            - Streamlit

            **Author:**  
            Developed by **Muhammad Sharique**, BS Software Engineering, University of Sindh.

            **Notes & Tips:**  
            - Always upload CSV files only.  
            - The original uploaded data is preserved in memory; transformations are applied to the in-memory copy.  
            - If you encounter any errors, check the file format, encoding, and whether the CSV has headers.  
            """)
            st.caption("Thank you for using the dashboard! ✨")
        except Exception as e:
            st.error(f"Error displaying About section: {e}")

    # -------------------------
    # Footer & small debug/help
    # -------------------------
    st.markdown("---")
    st.caption("For best results, ensure your CSV has headers in the first row and consistent column types. The app is designed to handle invalid inputs gracefully and will display helpful messages if something goes wrong.")

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as app_e:
        # Prevent Streamlit from stopping — display error message in-app where possible
        try:
            st.error(f"An unexpected error occurred while running the app: {app_e}")
        except Exception:
            print(f"An unexpected error occurred while running the app: {app_e}")

