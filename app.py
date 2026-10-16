{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
}import streamlit as st
import duckdb
import pandas as pd
import google.generativeai as genai
from prophet import Prophet

# 1. Configuration
genai.configure(api_key="AQ.Ab8RN6Jg8DrqDLcK2SsABChH3m7SdPka-0gt3qyrFVeUt1x8yA")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(layout="wide")
st.title("Simplified Data Fabric")

# 2. Data Upload
uploaded_file = st.file_uploader("Upload your CSV sales data", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    duckdb.register('sales_data', df)
    st.write("Data Loaded Successfully.")

    # 3. AI Questioning
    query = st.text_input("Ask a question about your data (e.g., 'What are the total sales per region?')")
    if query:
        prompt = f"Given the table 'sales_data', write a DuckDB SQL query for: {query}. Return only the SQL code."
        sql = model.generate_content(prompt).text.replace("
sql", "").replace("``", "")
st.code(sql)
st.write(duckdb.query(sql).df())
                                      if st.button("Unveil Future Sales"):
df['ds'] = pd.to_datetime(df['date']) # Ensure your CSV has 'date' and 'revenue' columns
df['y'] = df['revenue']
m = Prophet()
m.fit(df[['ds', 'y']])
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
st.line_chart(forecast.set_index('ds')[['yhat']])
                                                        
