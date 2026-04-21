import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional Groq import (won't crash if missing)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title='AI Finance Dashboard', layout='wide')

# ---------------- DATA UPLOAD ----------------
st.sidebar.title('Upload Data')

uploaded_file = st.sidebar.file_uploader(
    'Upload CSV',
    type=['csv']
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    st.title('AI Finance Dashboard')
    st.info('Upload a CSV file in the sidebar to begin.')
    st.stop()

df = load_data(uploaded_file)

# ---------------- OPTIONAL LLM ----------------
client = None

if GROQ_AVAILABLE:
    try:
        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )
    except Exception:
        client = None

def ask_llm(question, data):

    if client is None:
        return "LLM unavailable (missing groq package or API key). Analytics features still work."

    context = data.head(15).to_string()

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role":"user",
                "content":f"Analyze this data:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    return completion.choices[0].message.content

# ---------------- CLUSTERING ----------------
def run_clustering(data, k=3):

    if len(data.columns) < 2:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(scaled)

    result = data.copy()
    result["Cluster"] = labels

    return result

# ---------------- FILTERS ----------------
column = st.sidebar.selectbox(
    'Select Column',
    df.columns
)

is_numeric = pd.api.types.is_numeric_dtype(df[column])

if is_numeric:

    threshold = st.sidebar.slider(
        'Filter Threshold',
        float(df[column].min()),
        float(df[column].max()),
        float(df[column].mean())
    )

    filtered_df = df[
        df[column] >= threshold
    ]

else:

    vals = df[column].dropna().unique()

    chosen = st.sidebar.multiselect(
        'Select Categories',
        vals,
        default=vals[:min(3,len(vals))]
    )

    filtered_df = (
        df[df[column].isin(chosen)]
        if len(chosen)>0
        else df.copy()
    )

# ---------------- UI ----------------
st.title('AI-Powered Finance Analytics Dashboard')

c1,c2,c3 = st.columns(3)

c1.metric('Rows',len(filtered_df))
c2.metric('Columns',len(df.columns))

if is_numeric:
    c3.metric(
        'Mean',
        round(filtered_df[column].mean(),2)
    )
else:
    c3.metric(
        'Unique Values',
        filtered_df[column].nunique()
    )

# ---------------- TABS ----------------
t1,t2,t3,t4 = st.tabs([
    'Overview',
    'Visuals',
    'AI Insights',
    'Clustering'
])

# Overview
with t1:

    st.dataframe(
        filtered_df.head(50)
    )

    st.write(
        filtered_df.describe(
            include='all'
        )
    )

# Visuals
with t2:

    if is_numeric:

        fig = px.histogram(
            filtered_df,
            x=column
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    nums = filtered_df.select_dtypes(
        include=['number']
    )

    if len(nums.columns) > 1:

        fig = px.imshow(
            nums.corr(),
            title='Correlation Heatmap'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# AI Chat
with t3:

    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    for m in st.session_state.messages:

        with st.chat_message(
            m['role']
        ):
            st.write(
                m['content']
            )

    q = st.chat_input(
        'Ask about your uploaded data...'
    )

    if q:

        st.session_state.messages.append({
            'role':'user',
            'content':q
        })

        answer = ask_llm(
            q,
            filtered_df
        )

        st.session_state.messages.append({
            'role':'assistant',
            'content':answer
        })

        st.rerun()

# Clustering
with t4:

    k = st.slider(
        'Clusters',
        2,
        6,
        3
    )

    if st.button(
        'Run Clustering'
    ):

        num_data = filtered_df.select_dtypes(
            include=['number']
        ).dropna()

        clustered = run_clustering(
            num_data,
            k
        )

        if clustered is None:

            st.warning(
                'Need at least two numeric columns.'
            )

        else:

            st.dataframe(
                clustered.head()
            )

            fig = px.scatter(
                clustered,
                x=clustered.columns[0],
                y=clustered.columns[1],
                color='Cluster'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ---------------- DOWNLOAD ----------------
st.download_button(
    'Download Filtered CSV',
    filtered_df.to_csv(index=False),
    file_name='filtered_data.csv'
)
