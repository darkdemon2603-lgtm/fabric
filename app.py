import streamlit as st
import duckdb
import pandas as pd
import google.generativeai as genai
from prophet import Prophet
# Use Streamlit Secrets for the API key to keep it safe
api_key = st.secrets["AQ.Ab8RN6Jg8DrqDLcK2SsABChH3m7SdPka-0gt3qyrFVeUt1x8yA"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(layout="wide")
st.title("Simplified Data Fabric")

uploaded_file = st.file_uploader("Upload your CSV sales data", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    duckdb.register('sales_data', df)
    st.write("Data Loaded Successfully.")

    query = st.text_input("Ask a question about your data:")
    if query:
        prompt = f"Given the table 'sales_data', write a DuckDB SQL query for: {query}. Return only the SQL code."
        sql = model.generate_content(prompt).text.replace("
sql", "").replace("``", "")
st.code(sql)
st.write(duckdb.query(sql).df())

if st.button("Unveil Future Sales"):
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['revenue']
m = Prophet()
m.fit(df[['ds', 'y']])
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
st.line_chart(forecast.set_index('ds')[['yhat']])
