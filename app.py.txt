import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Sample simulated disease data
@st.cache_data
def load_data():
    dates = pd.date_range(start="2024-01-01", end=datetime.today())
    regions = ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad']
    diseases = ['Flu', 'Dengue', 'COVID-19']
    
    data = []
    for region in regions:
        for disease in diseases:
            base = np.random.randint(10, 100)
            trend = np.cumsum(np.random.randn(len(dates)) * 2 + base)
            trend = np.clip(trend, 0, None)
            data.extend(zip([region]*len(dates), [disease]*len(dates), dates, trend))
    
    df = pd.DataFrame(data, columns=['Region', 'Disease', 'Date', 'Cases'])
    return df

df = load_data()

# Streamlit UI 
st.set_page_config(page_title="FluWatch – Real-Time Disease Tracker", layout="wide")

st.title(" FluWatch – Real-Time Disease Trend Tracker & Outbreak Predictor")
st.markdown("Track and predict common disease outbreaks using real-time trends and historical patterns.")

# Sidebar filters 
st.sidebar.header("Filter Criteria")
region = st.sidebar.selectbox("Select Region", df["Region"].unique())
disease = st.sidebar.selectbox("Select Disease", df["Disease"].unique())

# Filtered data
filtered_df = df[(df["Region"] == region) & (df["Disease"] == disease)]

# Line Chart 
st.subheader(f" {disease} Trends in {region}")
fig = px.line(filtered_df, x="Date", y="Cases", title=f"Historical {disease} Cases in {region}", labels={"Cases": "Number of Cases"})
st.plotly_chart(fig, use_container_width=True)

# --- Prediction ---
st.subheader(" Predict Future Outbreak Risk")
days_to_predict = st.slider("Days into the future", min_value=1, max_value=30, value=7)

# Prepare data for modeling
filtered_df = filtered_df.copy()
filtered_df["Days"] = (filtered_df["Date"] - filtered_df["Date"].min()).dt.days
X = filtered_df[["Days"]]
y = filtered_df["Cases"]

model = LinearRegression()
model.fit(X, y)

future_days = np.array([filtered_df["Days"].max() + i for i in range(1, days_to_predict + 1)]).reshape(-1, 1)
predicted_cases = model.predict(future_days)

# Future date range
future_dates = [filtered_df["Date"].max() + timedelta(days=i) for i in range(1, days_to_predict + 1)]
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Cases": predicted_cases})

# Display predictions
fig2 = px.line(pred_df, x="Date", y="Predicted Cases", title=f"{disease} Case Predictions in {region}")
st.plotly_chart(fig2, use_container_width=True)

# Summary
st.markdown("###  Summary")
st.markdown(f"Predicted number of {disease} cases in *{region}* after *{days_to_predict} days: *{int(predicted_cases[-1]):,}**")

# Footer
st.markdown("---")
st.caption("Built with  using Streamlit | Sample data used for demonstration")
