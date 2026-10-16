
import streamlit as st
import duckdb
import pandas as pd
from google import genai
from prophet import Prophet

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Simplified Data Fabric")

# --------------------------------------------------
# GEMINI CONFIGURATION
# --------------------------------------------------
try:
    api_key = st.secrets["GCP_SERVICE_ACCOUNT_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(
        "Gemini configuration failed. Check that your "
        "Streamlit secret contains a valid Gemini API key."
    )
    st.stop()

MODEL_NAME = "gemini-3.6-flash"

# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV sales data",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV contains no data.")
            st.stop()

        # Remove surrounding spaces from column names
        df.columns = df.columns.str.strip()

        st.success("Data loaded successfully.")

        with st.expander("Preview your sales data"):
            st.dataframe(df.head(10), use_container_width=True)

        st.write("**Available columns:**")
        st.write(", ".join(df.columns.astype(str)))

        # Create an in-memory DuckDB connection
        con = duckdb.connect(database=":memory:")
        con.register("sales_data", df)

        # --------------------------------------------------
        # ASK QUESTIONS ABOUT SALES DATA
        # --------------------------------------------------
        st.subheader("Ask your data")

        query = st.text_input(
            "Ask a question about your data",
            placeholder=(
                "e.g. What were my top 5 best-selling burgers?"
            ),
            key="sales_question"
        )

        if st.button("Analyse data", type="primary"):
            if not query.strip():
                st.warning("Please enter a question first.")
            else:
                columns = ", ".join(
                    f'"{col}"' for col in df.columns
                )

                prompt = f"""
You are a data analyst writing DuckDB SQL.

The table is called sales_data.
Its columns are: {columns}

User question: {query}

Rules:
- Return one read-only SELECT query.
- A WITH clause is allowed if the final statement is SELECT.
- Use only the sales_data table and its listed columns.
- Do not use INSERT, UPDATE, DELETE, DROP, CREATE,
  ALTER, COPY, ATTACH, DETACH, or other write operations.
- Return only SQL, without Markdown code fences.
"""

                try:
                    with st.spinner("Generating SQL..."):
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=prompt
                        )

                    sql = (response.text or "").strip()

                    # Remove Markdown code fences if present
                    if sql.startswith("```"):
                        lines = sql.splitlines()
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        if lines and lines[0].strip().startswith("```"):
                            lines = lines[1:]
                        sql = "\n".join(lines).strip()

                    # Basic safety checks before executing SQL
                    normalised = sql.lstrip().lower()
                    if not (
                        normalised.startswith("select ")
                        or normalised.startswith("with ")
                        or normalised == "select"
                    ):
                        st.error(
                            "The AI did not return a SELECT query."
                        )
                        st.stop()

                    # Require a single statement
                    if ";" in sql.rstrip().rstrip(";"):
                        st.error(
                            "Multiple SQL statements are not allowed."
                        )
                        st.stop()

                    # Display generated SQL
                    st.subheader("Generated SQL")
                    st.code(sql, language="sql")

                    # Execute the query against the uploaded data
                    result = con.execute(sql).df()

                    st.subheader("Results")
                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.caption(
                        f"{len(result):,} rows returned."
                    )

                except Exception as e:
                    st.error(
                        f"Analysis failed: {type(e).__name__}: {e}"
                    )

        # --------------------------------------------------
        # SALES FORECASTING
        # --------------------------------------------------
        st.divider()
        st.subheader("Sales forecasting")

        if st.button("Unveil Future Sales"):
            # Check required columns
            if "date" not in df.columns:
                st.error(
                    "Forecasting requires a column named 'date'."
                )
            elif "revenue" not in df.columns:
                st.error(
                    "Forecasting requires a column named 'revenue'."
                )
            else:
                try:
                    forecast_df = df[["date", "revenue"]].copy()

                    forecast_df["ds"] = pd.to_datetime(
                        forecast_df["date"],
                        errors="coerce"
                    )
                    forecast_df["y"] = pd.to_numeric(
                        forecast_df["revenue"],
                        errors="coerce"
                    )

                    forecast_df = (
                        forecast_df[["ds", "y"]]
                        .dropna()
                        .groupby("ds", as_index=False)["y"]
                        .sum()
                        .sort_values("ds")
                    )

                    if len(forecast_df) < 2:
                        st.error(
                            "At least two valid dates with revenue "
                            "are needed for forecasting."
                        )
                    else:
                        with st.spinner(
                            "Building your sales forecast..."
                        ):
                            m = Prophet()
                            m.fit(forecast_df)

                            future = m.make_future_dataframe(
                                periods=30,
                                freq="D"
                            )
                            forecast = m.predict(future)

                        st.success("Forecast generated.")

                        st.line_chart(
                            forecast.set_index("ds")[["yhat"]]
                        )

                        st.write("**Next 30 days forecast**")
                        st.dataframe(
                            forecast[
                                ["ds", "yhat", "yhat_lower", "yhat_upper"]
                            ].tail(30),
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(
                        f"Forecast failed: {type(e).__name__}: {e}"
                    )

        con.close()

    except Exception as e:
        st.error(
            f"Could not read the CSV: {type(e).__name__}: {e}"
        )
else:
    st.info("Upload a CSV file to get started.")
