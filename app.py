import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ----------------------------
# Settings
# ----------------------------
st.set_page_config(layout="wide", page_title="Industrial HR Geo-Visualization")

# ----------------------------
# Load Dataset
# ----------------------------
def find_dataset():
    candidates = [
        "merged_workers.csv",
        "merged_workers_nlp.csv",
        "merged_workers_preprocessed.csv",
        "data/merged_workers.csv",
        "data/merged_workers_nlp.csv",
        "data/merged_workers_preprocessed.csv"
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

DATA_PATH = find_dataset()
if DATA_PATH is None:
    st.error("No dataset found! Place a CSV in project root or data/ folder.")
    st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # ensure all expected columns
    cols = [
        "indiastates", "nic_name", "industry_category",
        "main_workers__total___persons", "main_workers__total__males", "main_workers__total__females",
        "marginal_workers__total___persons", "marginal_workers__total__males", "marginal_workers__total__females"
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = 0 if "workers" in col else ""
    # clean text
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.replace("`", "").str.strip()
    # add combined totals
    df["total_workers_persons"] = df["main_workers__total___persons"] + df["marginal_workers__total___persons"]
    df["total_workers_males"] = df["main_workers__total__males"] + df["marginal_workers__total__males"]
    df["total_workers_females"] = df["main_workers__total__females"] + df["marginal_workers__total__females"]
    # fill missing industry_category for filter
    df['industry_category'] = df['industry_category'].fillna("Unknown").astype(str).str.strip()
    return df

df = load_data(DATA_PATH)

# ----------------------------
# Load Models (optional)
# ----------------------------
@st.cache_resource
def load_models():
    artifacts = {}
    tfidf_path = Path("tfidf_vectorizer.pkl")
    le_path = Path("label_encoder.pkl")
    model_path = Path("linear_svc_industry.pkl")
    try:
        artifacts["tfidf"] = joblib.load(tfidf_path) if tfidf_path.exists() else None
        artifacts["le"] = joblib.load(le_path) if le_path.exists() else None
        artifacts["model"] = joblib.load(model_path) if model_path.exists() else None
    except:
        artifacts = {"tfidf": None, "le": None, "model": None}
    return artifacts

artifacts = load_models()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.title("Industrial HR Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Geo Map", "Industry Prediction", "Data Download", "About"])

# Worker Type Filter
worker_type = st.sidebar.selectbox("Worker Type", ["All", "Main Workers", "Marginal Workers"])
# Sex Filter
sex = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])
# State Filter
state_list = sorted(df["indiastates"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", ["All"] + state_list)
# Industry Category Filter (fixed options)
industry_options = ["All", "Others", "Agriculture", "Chemicals", "Construction",
                    "Manufacturing", "Poultry", "Retail", "Transport"]
industry_category = st.sidebar.selectbox("Industry Category", industry_options)

# ----------------------------
# Apply filters
# ----------------------------
df_filtered = df.copy()

# Worker Type filter
if worker_type != "All":
    if worker_type == "Main Workers":
        df_filtered["total_workers_persons"] = df_filtered["main_workers__total___persons"]
        df_filtered["total_workers_males"] = df_filtered["main_workers__total__males"]
        df_filtered["total_workers_females"] = df_filtered["main_workers__total__females"]
    elif worker_type == "Marginal Workers":
        df_filtered["total_workers_persons"] = df_filtered["marginal_workers__total___persons"]
        df_filtered["total_workers_males"] = df_filtered["marginal_workers__total__males"]
        df_filtered["total_workers_females"] = df_filtered["marginal_workers__total__females"]

# Sex filter
if sex != "All":
    if sex == "Male":
        df_filtered["total_workers_persons"] = df_filtered["total_workers_males"]
    else:
        df_filtered["total_workers_persons"] = df_filtered["total_workers_females"]

# State filter
if selected_state != "All":
    df_filtered = df_filtered[df_filtered["indiastates"] == selected_state]

# Industry Category filter
if industry_category != "All":
    df_filtered = df_filtered[df_filtered["industry_category"].astype(str).str.strip().str.lower() == industry_category.lower()]

# ----------------------------
# PAGE: OVERVIEW
# ----------------------------
if page == "Overview":
    st.title("Industrial Human Resource Geo-Visualization — Overview")
    st.markdown("**Aim:** Update industrial classification info of main & marginal workers by sex and by section/division/class for policy making.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Main Workers", f"{int(df_filtered['main_workers__total___persons'].sum()):,}")
    col2.metric("Marginal Workers", f"{int(df_filtered['marginal_workers__total___persons'].sum()):,}")
    col3.metric("Total Workers", f"{int(df_filtered['total_workers_persons'].sum()):,}")
    col4.metric("Number of States", f"{df_filtered['indiastates'].nunique():,}")
    col5.metric("Unique NIC activities", f"{df_filtered['nic_name'].nunique():,}")

    st.markdown("---")
    st.subheader("Industry Category Distribution")
    cat_counts = df_filtered["industry_category"].value_counts().reset_index()
    cat_counts.columns = ["industry_category","count"]
    if len(cat_counts) > 0:
        fig = px.bar(cat_counts, x="industry_category", y="count", title="Workers by Industry Category")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("State-wise Workers (Top 20)")
    state_counts = df_filtered.groupby("indiastates")["total_workers_persons"].sum().reset_index().sort_values(by="total_workers_persons", ascending=False).head(20)
    fig2 = px.bar(state_counts, x="indiastates", y="total_workers_persons", title="Top 20 States by Total Workers", labels={"indiastates":"State","total_workers_persons":"Workers"})
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top NIC Activities (Main + Marginal)")
    top_n = st.slider("Top N NIC", 5, 30, 10)
    top_nic_total = df_filtered.groupby("nic_name")["total_workers_persons"].sum().sort_values(ascending=False).head(top_n).reset_index()
    fig3 = px.bar(top_nic_total, x="total_workers_persons", y="nic_name", orientation="h", title=f"Top {top_n} NIC Activities")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# PAGE: GEO MAP
# ----------------------------
elif page == "Geo Map":
    st.title("Geo Map of Workers")
    geojson_path = Path("data/india_states.geojson")
    if not geojson_path.exists():
        geojson_path = Path("india_states.geojson")
    if geojson_path.exists():
        with open(geojson_path,"r",encoding="utf-8") as f:
            india_geo = json.load(f)
        state_counts = df_filtered.groupby("indiastates")["total_workers_persons"].sum().reset_index()
        fig_map = px.choropleth_mapbox(state_counts, geojson=india_geo, locations="indiastates",
                                       featureidkey="properties.ST_NM", color="total_workers_persons",
                                       mapbox_style="carto-positron", zoom=3.5,
                                       center={"lat":22.5,"lon":78.9}, opacity=0.7,
                                       title="State-wise Workers")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No geojson found. Showing bar chart instead.")
        state_counts = df_filtered.groupby("indiastates")["total_workers_persons"].sum().reset_index()
        fig = px.bar(state_counts.sort_values("total_workers_persons",ascending=False), x="indiastates", y="total_workers_persons", title="State-wise Workers")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Industry Prediction Page
# ----------------------------
elif page == "Industry Prediction":
    st.title("Industry Category Prediction (NLP)")
    text_input = st.text_area("Enter NIC name / description", placeholder="e.g. Manufacture of plastics, Retail store")
    if artifacts["model"] is None:
        st.warning("Model not found. Prediction unavailable.")
    else:
        if st.button("Predict"):
            if text_input.strip() != "":
                X = artifacts["tfidf"].transform([text_input])
                pred = artifacts["model"].predict(X)
                label = artifacts["le"].inverse_transform(pred)[0]
                st.success(f"Predicted Industry Category: {label}")
            else:
                st.error("Enter text to predict.")
    st.subheader("Sample Predictions from Dataset")
    st.table(df_filtered.sample(6)[["nic_name","industry_category"]].reset_index(drop=True))

# ----------------------------
# Data Download Page
# ----------------------------
elif page == "Data Download":
    st.title("Download Filtered Data")
    st.dataframe(df_filtered.head(100))
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_workers.csv", "text/csv")

# ----------------------------
# About Page
# ----------------------------
else:
    st.title("About")
    st.markdown("""
    **Industrial Human Resource Geo-Visualization**  
    - Dataset: Indian census / NIC merged CSV  
    - Features: EDA, NLP industry grouping, interactive visualization  
    - Filters: Worker Type, Sex, State, Industry Category  
    - Pages: Overview, Geo Map, Industry Prediction, Data Download, About
    """)
