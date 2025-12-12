🏭 Industrial Human Resource Geo-Visualization Dashboard
A Data Analysis + NLP + Streamlit Application
📌 Project Overview

This project aims to update, analyze, and visualize India’s industrial workforce distribution using modern data science techniques.
It focuses on:

Main vs Marginal workers

Male vs Female workforce distribution

Industry classification using both NIC data and a custom industry categorization

NLP-based prediction of industry category from job description text

Geographical visualization of worker patterns across states

The final output is a fully interactive Streamlit dashboard with filters, insights, and an NLP prediction tool.

🎯 Objectives

Modernize outdated worker industrial classification

Provide clear, visual insights for policymaking

Build an interactive analytics dashboard

Integrate NLP to classify job descriptions into industries

Visualize state-wise workforce using GeoJSON maps

🗂️ Dataset

The custom merged dataset includes:

State / Region

Main Workers

Marginal Workers

Male / Female Workers

NIC (Section → Division → Class)

Industrial Activities

Updated Industry Categories:

Manufacturing

Construction

Retail

Agriculture

Poultry

Chemicals

Transport

Others

Multiple CSVs were merged, cleaned, and processed into a single dataset:
merged_workers.csv

🛠️ Tech Stack
Programming Languages

Python

Libraries & Frameworks

Pandas, NumPy — Data processing

Plotly, Matplotlib — Visualization

Streamlit — Interactive dashboard

Scikit-Learn — NLP + ML model

Joblib — Model loading

GeoJSON — Map rendering

📊 Features of the Dashboard
✔ Overview Metrics

Total Workers

Main vs Marginal count

Male vs Female count

States covered

NIC activities count

✔ Geo-Visualization

Choropleth map of India

State-wise worker density

Interactive tooltips & filters

✔ Filters Provided

State

Worker Type

Gender

Industry Category

✔ NLP-Based Industry Prediction

Input: Job description text
Output: Predicted industry category
Model: Linear SVC + TF-IDF

🤖 Machine Learning Model

Three serialized model files are used:

File	Purpose
tfidf_vectorizer.pkl	Converts text → TF-IDF matrix
label_encoder.pkl	Encodes/decodes industry labels
nlp_model.pkl	Predicts industry category

The model predicts categories like:

Manufacturing

Agriculture

Retail

Poultry

Construction

Transport

Chemicals

Others

🚀 How to Run the App
1️⃣ Clone the Repository
git clone https://github.com/your-username/industrial-hr-geo-visualization.git
cd industrial-hr-geo-visualization

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

📁 Project Structure
├── app.py                     # Streamlit app
├── merged_workers.csv         # Final dataset
├── india_states.geojson       # GeoJSON file for mapping
├── nlp_model.pkl              # Trained SVC model
├── tfidf_vectorizer.pkl       # TF-IDF transformer
├── label_encoder.pkl          # Label encoder for industries
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🧠 Key Insights From Analysis

Main workers dominate across most states

Female workforce stronger in retail and poultry

Manufacturing & construction remain widespread

Chemicals and poultry industries concentrated in specific states

NIC classification reveals regional specialization

🏁 Conclusion

This project provides an end-to-end solution for:

Workforce analysis

Modern industry classification

Geo-mapping

NLP-based job classification

Interactive reporting

It serves as a strong foundation for future work such as forecasting, skill analysis, and district-level mapping.

📌 Future Enhancements

Add district-level workforce visualization

Integrate LSTM for workforce forecasting

Auto-generate PDF insights

Add more industry categories

Build a full web-based front-end
