import streamlit as st
import duckdb
import pandas as pd
import google.generativeai as genai
from prophet import Prophet
# Use Streamlit Secrets for the API key to keep it safe
genai.configure(api_key=st.secrets["GCP_SERVICE_ACCOUNT_KEY"])
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
        sql = model.generate_content(prompt).text.replace("sql", "").replace("``", "")
st.code(sql)
st.write(duckdb.query(sql).df())
# ----------------------------------------------------------------------
#  Forecasting – runs only when the user clicks the button
# ----------------------------------------------------------------------
if st.button("Unveil Future Sales"):
    # Ensure the date column is a proper datetime type
    df['ds'] = pd.to_datetime(df['date'])

    # Prophet expects the target column to be called 'y'
    df['y'] = df['revenue']

    # Fit the Prophet model
    m = Prophet()
    m.fit(df[['ds', 'y']])

    # Create a future dataframe for the next 30 days
    future = m.make_future_dataframe(periods=30)

    # Generate the forecast
    forecast = m.predict(future)

    # Plot the forecast (yhat = predicted value)
    st.line_chart(forecast.set_index('ds')[['yhat']])
