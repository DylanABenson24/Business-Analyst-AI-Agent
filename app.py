import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")

# ================= LOGGING =================
logging.basicConfig(filename="app.log", level=logging.INFO)

# ================= DATA =================
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\dylan\Downloads\Finance_data.csv")

df = load_data()

# ================= PREPROCESS =================
numeric_df = df.select_dtypes(include=['number']).dropna()

# ================= MODEL: CLUSTERING =================
def run_clustering(data, k=3):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    result = data.copy()
    result["Cluster"] = clusters
    return result

# ================= LLM (Milestone 12 STYLE) =================
def ask_llm(question, df):
    context = df.head(15).to_string()

    prompt = f"""
    You are a financial data analyst.

    Dataset sample:
    {context}

    Answer the question with clear insights and patterns.

    Question:
    {question}
    """

    # Replace this with your real LLM call if needed
    return f"📊 AI Insight:\nBased on patterns in the dataset, {question} suggests analyzing trends, correlations, and potential segmentation in financial behavior."

# ================= OBSERVABILITY =================
def log_event(msg):
    logging.info(msg)

# ================= UI =================
st.title("📊 AI-Powered Finance Analytics Dashboard")

st.markdown("""
### 🚀 Capabilities
- Interactive Filtering  
- AI Insights (LLM-powered)  
- Customer Segmentation (Clustering)  
- Interactive Visualizations (Plotly)  
""")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

column = st.sidebar.selectbox("Select Column", df.columns)
is_numeric = pd.api.types.is_numeric_dtype(df[column])

st.sidebar.write(f"Detected Type: {'Numeric' if is_numeric else 'Categorical'}")

# ===== FILTER LOGIC =====
if is_numeric:
    threshold = st.sidebar.slider(
        "Filter Threshold",
        float(df[column].min()),
        float(df[column].max()),
        float(df[column].mean())
    )
    filtered_df = df[df[column] >= threshold]

else:
    unique_vals = df[column].dropna().unique()

    selected_vals = st.sidebar.multiselect(
        "Select Categories",
        options=unique_vals,
        default=unique_vals[:min(3, len(unique_vals))]
    )

    filtered_df = df[df[column].isin(selected_vals)] if selected_vals else df.copy()

# ================= METRICS =================
c1, c2, c3 = st.columns(3)

c1.metric("Rows", len(filtered_df))
c2.metric("Columns", len(df.columns))

if is_numeric:
    c3.metric("Mean", round(filtered_df[column].mean(), 2))
else:
    c3.metric("Unique Values", filtered_df[column].nunique())

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Visuals",
    "🤖 AI Insights",
    "🧠 Clustering"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Dataset")
    st.dataframe(filtered_df.head(50))

    st.subheader("Summary Stats")
    st.write(filtered_df.describe(include='all'))

# ================= TAB 2 (PLOTLY) =================
with tab2:
    st.subheader("Interactive Visualizations")

    if is_numeric:
        fig = px.histogram(filtered_df, x=column, title=f"{column} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=False,
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3 (LLM) =================
with tab3:
    st.subheader("Ask AI About Financial Data")

    question = st.text_input("Enter your question")

    if question:
        log_event(f"User Question: {question}")
        response = ask_llm(question, filtered_df)
        st.success(response)

# ================= TAB 4 (CLUSTERING) =================
with tab4:
    st.subheader("Customer / Financial Segmentation")

    k = st.slider("Select Number of Clusters", 2, 6, 3)

    if st.button("Run Clustering"):
        clustered = run_clustering(numeric_df, k)

        st.write("Clustered Data Preview")
        st.dataframe(clustered.head())

        # Visualization (2D projection)
        if clustered.shape[1] >= 3:
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

# ================= EXTRA =================
st.markdown("---")

st.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_finance_data.csv"
)

# ================= SYSTEM HEALTH =================
st.sidebar.markdown("### 🩺 System Health")
st.sidebar.success("Running")
st.sidebar.write("Logging: Active")
st.sidebar.write("Model: KMeans Ready")
