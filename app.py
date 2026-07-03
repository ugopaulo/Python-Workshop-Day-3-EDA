"""
Brainy EDA — A Streamlit app for quick exploratory data analysis.

Upload a CSV or Excel file and get instant access to summary stats,
missing-value diagnostics, histograms, boxplots, correlation heatmaps,
scatter plots, and a pairplot — all in one scrolling page.
"""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page config — set this first, before any other Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brainy EDA",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧠 Brainy EDA")
st.caption("Upload a dataset and let the brain do the exploring.")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_all_columns(df: pd.DataFrame) -> list[str]:
    return df.columns.tolist()


# ----------------------------------------------------------------------------
# 1. Upload
# ----------------------------------------------------------------------------
st.header("1. Upload your data")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Works with .csv, .xlsx, and .xls files.",
)

if uploaded_file is None:
    st.info("👆 Upload a file to unlock the exploration tools below.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file has no rows. Please try a different file.")
    st.stop()

st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Preview data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

numeric_cols = get_numeric_columns(df)
all_cols = get_all_columns(df)

# ----------------------------------------------------------------------------
# 2. Overview & summary stats
# ----------------------------------------------------------------------------
st.header("2. Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Numeric columns", len(numeric_cols))
col4.metric("Missing values", int(df.isna().sum().sum()))

st.subheader("Summary statistics")
if numeric_cols:
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
else:
    st.info("No numeric columns found for summary statistics.")

st.subheader("Column info")
info_df = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "non-null count": df.notna().sum(),
        "null count": df.isna().sum(),
        "null %": (df.isna().mean() * 100).round(2),
        "unique values": df.nunique(),
    }
)
st.dataframe(info_df, use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Missing values
# ----------------------------------------------------------------------------
st.header("3. Missing values")

if df.isna().sum().sum() == 0:
    st.success("No missing values found. Nice and clean! ✨")
else:
    missing_fig = px.imshow(
        df.isna(),
        color_continuous_scale=["#EDEDED", "#5B2A86"],
        aspect="auto",
        labels=dict(color="Missing"),
        title="Missing value map (light = present, purple = missing)",
    )
    missing_fig.update_coloraxes(showscale=False)
    st.plotly_chart(missing_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# 4. Histograms
# ----------------------------------------------------------------------------
st.header("4. Histograms")

if numeric_cols:
    hist_col = st.selectbox("Choose a numeric column", numeric_cols, key="hist_col")
    bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, key="hist_bins")
    hist_fig = px.histogram(df, x=hist_col, nbins=bins, marginal="box", title=f"Distribution of {hist_col}")
    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.info("No numeric columns available for histograms.")

# ----------------------------------------------------------------------------
# 5. Boxplots
# ----------------------------------------------------------------------------
st.header("5. Boxplots")

if numeric_cols:
    box_cols = st.multiselect(
        "Choose numeric column(s)",
        numeric_cols,
        default=numeric_cols[: min(3, len(numeric_cols))],
        key="box_cols",
    )
    if box_cols:
        box_fig = px.box(df, y=box_cols, points="outliers", title="Boxplot(s) for outlier detection")
        st.plotly_chart(box_fig, use_container_width=True)
    else:
        st.info("Select at least one column to see a boxplot.")
else:
    st.info("No numeric columns available for boxplots.")

# ----------------------------------------------------------------------------
# 6. Correlation heatmap
# ----------------------------------------------------------------------------
st.header("6. Correlation heatmap")

if len(numeric_cols) >= 2:
    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
    corr = df[numeric_cols].corr(method=corr_method)
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"{corr_method.title()} correlation",
    )
    st.plotly_chart(corr_fig, use_container_width=True)
else:
    st.info("Need at least two numeric columns for a correlation heatmap.")

# ----------------------------------------------------------------------------
# 7. Scatter plot
# ----------------------------------------------------------------------------
st.header("7. Scatter plot")

if len(numeric_cols) >= 2:
    scatter_col1, scatter_col2, scatter_col3 = st.columns(3)
    x_col = scatter_col1.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
    y_col = scatter_col2.selectbox(
        "Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y"
    )
    color_options = ["None"] + all_cols
    color_col = scatter_col3.selectbox("Color by (optional)", color_options, key="scatter_color")
    color_arg = None if color_col == "None" else color_col
    scatter_fig = px.scatter(
        df, x=x_col, y=y_col, color=color_arg, title=f"{y_col} vs {x_col}", trendline=None
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
else:
    st.info("Need at least two numeric columns for a scatter plot.")

# ----------------------------------------------------------------------------
# 8. Pairplot / scatter matrix
# ----------------------------------------------------------------------------
st.header("8. Pairplot")

if len(numeric_cols) >= 2:
    default_pair_cols = numeric_cols[: min(4, len(numeric_cols))]
    pair_cols = st.multiselect(
        "Choose numeric columns (2–6 recommended for readability)",
        numeric_cols,
        default=default_pair_cols,
        key="pair_cols",
    )
    if len(pair_cols) >= 2:
        pair_fig = px.scatter_matrix(df, dimensions=pair_cols, title="Scatter matrix")
        pair_fig.update_traces(diagonal_visible=False)
        st.plotly_chart(pair_fig, use_container_width=True)
    else:
        st.info("Select at least two columns to see a pairplot.")
else:
    st.info("Need at least two numeric columns for a pairplot.")

# ----------------------------------------------------------------------------
# 9. Export cleaned data
# ----------------------------------------------------------------------------
st.header("9. Export")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="data")
st.download_button(
    "Download data as Excel (.xlsx)",
    data=excel_buffer.getvalue(),
    file_name="data_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.caption("🧠 Brainy EDA — built with Streamlit")
