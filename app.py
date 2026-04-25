import os
import logging
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from huggingface_hub import InferenceClient

# ================= CONFIG =================
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)

def log_event(msg: str) -> None:
    logging.info(msg)

# ================= MODEL CONFIG =================
# Use an API-served instruct model instead of loading locally
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Hugging Face token
# Prefer environment variable locally: export HF_TOKEN="your_token_here"
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

# ================= LOAD HF CLIENT =================
@st.cache_resource
def load_hf_client():
    """
    Create a Hugging Face InferenceClient once and reuse it.
    """
    if not HF_TOKEN:
        raise ValueError(
            "No Hugging Face token found. Set HF_TOKEN as an environment variable "
            "or add it to Streamlit secrets."
        )

    client = InferenceClient(
        api_key=HF_TOKEN
    )
    return client

# ================= DATA LOADING =================
@st.cache_data
def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

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
    column_info = "\n".join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])
    sample = df.head(10).to_string(index=False)

    numeric_summary = ""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
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

# ================= LLM (API VERSION) =================
def ask_llm_api(question: str, df: pd.DataFrame, client: InferenceClient) -> str:
    context = build_context(df)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial and business data analyst. "
                "Answer only from the dataset context provided. "
                "Do not invent values or conclusions not supported by the data shown. "
                "If the answer cannot be supported by the dataset context, say that clearly. "
                "Do not just describe the columns. "
                "Instead, identify actual patterns, unusual values, possible relationships, missing data concerns, and business-relevant observations. "
                "Be specific and reference values or categories when possible."
            ),
        },
        {
            "role": "user",
            "content": f"""
Here is the dataset context:

{context}

Question:
{question}

Instructions:
- Do not give a generic dataset description.
- Focus on actual patterns in the sample and summary provided.
- Mention missing values or skew only if supported by the data.
- If the question asks for trends, give at least 3 specific observations.
- If the dataset context is insufficient, say exactly what is missing.

Format your answer as:
1. Direct answer
2. Key insights
3. Data limitations
""".strip(),
        },
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.3
    )

    return completion.choices[0].message.content.strip()

# ================= UI =================
st.title("📊 AI-Powered Finance Analytics Dashboard")

st.markdown("""
### 🚀 Capabilities
- Upload any CSV dataset
- Interactive filtering
- AI insights with Hugging Face API
- Customer / financial segmentation
- Interactive visualizations
""")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.stop()

try:
    df = load_uploaded_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded dataset is empty.")
    st.stop()

st.sidebar.success(f"Loaded: {uploaded_file.name}")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

column = st.sidebar.selectbox("Select Column", df.columns)
is_numeric = pd.api.types.is_numeric_dtype(df[column])

st.sidebar.write(f"Detected Type: {'Numeric' if is_numeric else 'Categorical'}")

# ================= FILTER LOGIC =================
if is_numeric:
    min_val = float(df[column].min())
    max_val = float(df[column].max())
    default_val = float(df[column].mean())

    threshold = st.sidebar.slider(
        "Filter Threshold",
        min_value=min_val,
        max_value=max_val,
        value=default_val
    )

    filtered_df = df[df[column] >= threshold].copy()
else:
    unique_vals = df[column].dropna().unique().tolist()

    selected_vals = st.sidebar.multiselect(
        "Select Categories",
        options=unique_vals,
        default=unique_vals[:min(3, len(unique_vals))]
    )

    filtered_df = df[df[column].isin(selected_vals)].copy() if selected_vals else df.copy()

numeric_df = get_numeric_df(filtered_df)

# ================= METRICS =================
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(filtered_df))
c2.metric("Columns", len(df.columns))

if is_numeric and not filtered_df.empty:
    c3.metric("Mean", round(filtered_df[column].mean(), 2))
else:
    c3.metric("Unique Values", filtered_df[column].nunique())

# ================= LOAD API CLIENT =================
with st.spinner("Connecting to Hugging Face API..."):
    try:
        hf_client = load_hf_client()
        api_loaded = True
    except Exception as e:
        st.error(f"API client failed to load: {e}")
        hf_client = None
        api_loaded = False

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Visuals",
    "🤖 AI Insights",
    "🧠 Clustering"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.subheader("Summary Stats")
    st.write(filtered_df.describe(include="all"))

    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Missing Count": filtered_df.isnull().sum().values,
        "Missing %": (filtered_df.isnull().mean() * 100).round(2).values
    })
    st.dataframe(missing_df, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("Interactive Visualizations")

    if is_numeric and not filtered_df.empty:
        fig = px.histogram(filtered_df, x=column, title=f"{column} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        value_counts = filtered_df[column].value_counts().reset_index()
        value_counts.columns = [column, "Count"]
        fig = px.bar(value_counts, x=column, y="Count", title=f"{column} Category Counts")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=False,
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("Ask AI About Your Uploaded Data")
    question = st.text_input("Enter your question about the uploaded dataset")

    if question:
        log_event(f"User Question: {question}")

        if not api_loaded:
            st.error("The Hugging Face API client is not ready. Please check your token.")
        else:
            with st.spinner("Generating insight..."):
                try:
                    response = ask_llm_api(question, filtered_df, hf_client)
                    st.success(response)
                except Exception as e:
                    st.error(f"API error: {e}")

# ================= TAB 4 =================
with tab4:
    st.subheader("Customer / Financial Segmentation")

    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
    else:
        k = st.slider("Select Number of Clusters", 2, 6, 3)

        if st.button("Run Clustering"):
            try:
                clustered = run_clustering(numeric_df, k)

                st.write("Clustered Data Preview")
                st.dataframe(clustered.head(), use_container_width=True)

                x_col = clustered.columns[0]
                y_col = clustered.columns[1]

                fig = px.scatter(
                    clustered,
                    x=x_col,
                    y=y_col,
                    color="Cluster",
                    title="Cluster Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Clustering error: {e}")

# ================= DOWNLOAD =================
st.markdown("---")
st.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# ================= SYSTEM HEALTH =================
st.sidebar.markdown("### 🩺 System Health")
st.sidebar.success("Running")
st.sidebar.write("Logging: Active")
st.sidebar.write(f"Model: {MODEL_NAME}")
st.sidebar.write("Inference: Hugging Face API")
st.sidebar.write("Clustering: KMeans Ready")
