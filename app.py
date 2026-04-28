import os
import logging
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from huggingface_hub import InferenceClient

# ================= CONFIG =================
st.set_page_config(page_title="AI-Automated Analyst Dashboard", layout="wide")

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)

def log_event(msg: str) -> None:
    logging.info(msg)

# ================= MODEL CONFIG =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

# ================= LOAD HF CLIENT =================
@st.cache_resource
def load_hf_client():
    if not HF_TOKEN:
        raise ValueError("No Hugging Face token found.")
    return InferenceClient(api_key=HF_TOKEN)

# ================= DATA LOADING =================
@st.cache_data
def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(
        uploaded_file,
        na_values=["?", "NA", "N/A", "null", "NULL", "nan", "NaN", ""]
    )
    return df

# ================= PREPROCESS =================
def get_numeric_df(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes(include=["number"]).dropna()

# ================= MODEL: CLUSTERING =================
def run_clustering(data: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    result = data.copy()
    result["Cluster"] = clusters
    return result

# ================= LLM CONTEXT =================
def build_context(df: pd.DataFrame) -> str:
    rows, cols = df.shape

    column_info = "\n".join(
        [f"- {col}: {dtype}" for col, dtype in df.dtypes.items()]
    )

    sample = df.head(10).to_string(index=False)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_summary = ""

    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().round(2).to_string()

    context = f"""
Dataset shape: {rows} rows x {cols} columns

Columns and dtypes:
{column_info}

Sample rows:
{sample}

Numeric summary:
{numeric_summary}
"""
    return context.strip()

# ================= LLM =================
def ask_llm_api(question: str, df: pd.DataFrame, client: InferenceClient) -> str:
    context = build_context(df)

    messages = [
        {
            "role": "system",
            "content": "You are a data analyst. Answer using only the dataset context."
        },
        {
            "role": "user",
            "content": f"{context}\n\nQuestion:\n{question}"
        }
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.3
    )

    return completion.choices[0].message.content.strip()

# ================= MISSING VALUES =================
def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df)) * 100

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing %": missing_percent.round(2).values
    })

    return summary.sort_values(by="Missing Count", ascending=False)

# ================= UI =================
st.title("📊 AI-Automated Analyst Dashboard")

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = load_uploaded_data(uploaded_file)

column = st.sidebar.selectbox("Select Column", df.columns)
is_numeric = pd.api.types.is_numeric_dtype(df[column])

# ================= FILTER LOGIC =================
if is_numeric:
    min_val = float(df[column].min())
    max_val = float(df[column].max())

    min_range, max_range = st.sidebar.slider(
        "Filter Range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

    filtered_df = df[
        (df[column] >= min_range) &
        (df[column] <= max_range)
    ].copy()

else:
    unique_vals = df[column].dropna().unique().tolist()

    selected_vals = st.sidebar.multiselect(
        "Select Categories",
        options=unique_vals,
        default=unique_vals[:min(3, len(unique_vals))]
    )

    filtered_df = (
        df[df[column].isin(selected_vals)].copy()
        if selected_vals else df.copy()
    )

numeric_df = get_numeric_df(filtered_df)

# ================= METRICS =================
st.write(filtered_df.head())

# ================= VISUAL =================
if is_numeric:
    fig = px.histogram(filtered_df, x=column)
else:
    vc = filtered_df[column].value_counts().reset_index()
    vc.columns = [column, "Count"]
    fig = px.bar(vc, x=column, y="Count")

st.plotly_chart(fig)
