# ============================================================
# BrainScope EDA
# Part 1 - Upload & Basic Exploration
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

import sweetviz as sv
from ydata_profiling import ProfileReport

import tempfile
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="BrainScope EDA",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 BrainScope EDA")
st.write("Upload a dataset and perform exploratory data analysis.")

# ============================================================
# Upload Dataset
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ============================================================
# Load Dataset
# ============================================================

@st.cache_data
def load_data(file):

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)

try:

    df = load_data(uploaded_file)

except Exception as e:

    st.error(f"Error loading dataset:\n{e}")
    st.stop()

# ============================================================
# Dataset Overview
# ============================================================

st.header("Dataset Overview")

rows, cols = df.shape

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", rows)
col2.metric("Columns", cols)
col3.metric("Numeric Columns", len(numeric_cols))
col4.metric("Categorical Columns", len(categorical_cols))

st.divider()

# ============================================================
# Dataset Preview
# ============================================================

st.subheader("Dataset Preview")

preview = st.selectbox(
    "Preview",
    ["Head", "Tail", "Random Sample"]
)

if preview == "Head":

    st.dataframe(df.head(), use_container_width=True)

elif preview == "Tail":

    st.dataframe(df.tail(), use_container_width=True)

else:

    sample_size = min(10, len(df))

    st.dataframe(
        df.sample(sample_size),
        use_container_width=True
    )

st.divider()

# ============================================================
# Column Information
# ============================================================

st.subheader("Column Information")

info_df = pd.DataFrame({

    "Column": df.columns,

    "Data Type": df.dtypes.astype(str),

    "Non-Null Count": df.count().values,

    "Missing Values": df.isnull().sum().values,

    "Unique Values": df.nunique().values

})

st.dataframe(
    info_df,
    use_container_width=True
)

st.divider()

# ============================================================
# Summary Statistics
# ============================================================

st.subheader("Summary Statistics")

stats_option = st.radio(
    "Statistics",
    ["Numeric Only", "All Columns"],
    horizontal=True
)

if stats_option == "Numeric Only":

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

else:

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

st.divider()

# ============================================================
# Missing Value Summary
# ============================================================

st.subheader("Missing Values")

missing = pd.DataFrame({

    "Missing Count": df.isnull().sum(),

    "Percentage (%)":
        (df.isnull().mean() * 100).round(2)

})

missing = missing[
    missing["Missing Count"] > 0
]

if missing.empty:

    st.success("No missing values detected.")

else:

    st.dataframe(
        missing,
        use_container_width=True
    )

st.divider()

# ============================================================
# Duplicate Rows
# ============================================================

st.subheader("Duplicate Rows")

duplicates = df.duplicated().sum()

if duplicates == 0:

    st.success("No duplicate rows detected.")

else:

    st.warning(f"{duplicates} duplicate rows found.")

st.divider()

# ============================================================
# Chunk 2 - Visualisations
# ============================================================

st.header("Visualisations")

analysis = st.selectbox(
    "Choose a visualisation",
    [
        "Correlation Heatmap",
        "Histogram",
        "Scatter Plot",
        "Box Plot",
        "Count Plot",
        "QQ Plot"
    ]
)

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

if analysis == "Correlation Heatmap":

    if len(numeric_cols) < 2:
        st.warning("At least two numeric columns are required.")
    else:

        method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman", "kendall"]
        )

        corr = df[numeric_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        st.pyplot(fig)

# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------

elif analysis == "Histogram":

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available.")

    else:

        column = st.selectbox(
            "Select Column",
            numeric_cols
        )

        bins = st.slider(
            "Bins",
            5,
            100,
            30
        )

        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=f"Histogram of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ------------------------------------------------------------
# Scatter Plot
# ------------------------------------------------------------

elif analysis == "Scatter Plot":

    if len(numeric_cols) < 2:
        st.warning("At least two numeric columns are required.")

    else:

        x = st.selectbox(
            "X-axis",
            numeric_cols
        )

        y = st.selectbox(
            "Y-axis",
            numeric_cols,
            index=1
        )

        colour = st.selectbox(
            "Colour (optional)",
            ["None"] + list(df.columns)
        )

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=None if colour == "None" else colour,
            title=f"{y} vs {x}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ------------------------------------------------------------
# Box Plot
# ------------------------------------------------------------

elif analysis == "Box Plot":

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available.")

    else:

        column = st.selectbox(
            "Numeric Column",
            numeric_cols
        )

        fig = px.box(
            df,
            y=column,
            title=f"Box Plot of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ------------------------------------------------------------
# Count Plot
# ------------------------------------------------------------

elif analysis == "Count Plot":

    if len(categorical_cols) == 0:
        st.warning("No categorical columns available.")

    else:

        column = st.selectbox(
            "Categorical Column",
            categorical_cols
        )

        fig = px.histogram(
            df,
            x=column,
            title=f"Count Plot of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ------------------------------------------------------------
# QQ Plot
# ------------------------------------------------------------

elif analysis == "QQ Plot":

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available.")

    else:

        column = st.selectbox(
            "Numeric Column",
            numeric_cols
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        stats.probplot(
            df[column].dropna(),
            dist="norm",
            plot=ax
        )

        ax.set_title(f"QQ Plot of {column}")

        st.pyplot(fig)

st.divider()

# ============================================================
# Chunk 3 - Reports
# ============================================================

st.header("Reports")

report = st.selectbox(
    "Choose a report",
    [
        "None",
        "Sweetviz",
        "YData Profiling"
    ]
)

# ------------------------------------------------------------
# Sweetviz
# ------------------------------------------------------------

if report == "Sweetviz":

    if st.button("Generate Sweetviz Report"):

        with st.spinner("Generating report..."):

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".html"
            )

            report = sv.analyze(df)

            report.show_html(
                temp_file.name,
                open_browser=False
            )

            with open(temp_file.name, "rb") as f:

                st.download_button(
                    "Download Sweetviz Report",
                    f,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )

# ------------------------------------------------------------
# YData Profiling
# ------------------------------------------------------------

elif report == "YData Profiling":

    if st.button("Generate Profiling Report"):

        with st.spinner("Generating report..."):

            profile = ProfileReport(
                df,
                explorative=True
            )

            html = profile.to_html()

            st.download_button(
                "Download Profiling Report",
                html,
                file_name="profiling_report.html",
                mime="text/html"
            )

st.divider()

# ============================================================
# Dataset Download
# ============================================================

st.header("Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Clean Dataset",
    csv,
    file_name="dataset.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "BrainScope EDA | Built with Streamlit"
)

