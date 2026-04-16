import streamlit as st

st.title("Wildfire Intensity Prediction Tool")
st.write("SI 465 Final Project - testing inputs for now")

st.header("Region")

col1, col2 = st.columns(2)
with col1:
    west = st.number_input("West longitude", value=-124.0)
    south = st.number_input("South latitude", value=36.0)
with col2:
    east = st.number_input("East longitude", value=-119.0)
    north = st.number_input("North latitude", value=42.0)

st.header("Date Range")

start_date = st.text_input("Start date (YYYY-MM-DD)", value="2024-05-01")
end_date = st.text_input("End date (YYYY-MM-DD)", value="2024-10-31")

st.header("Settings")

resolution = st.selectbox("Grid resolution (degrees)", [0.5, 1.0, 2.0])
max_cloud = st.slider("Max cloud cover % for Landsat", 0, 100, 20)
landsat_limit = st.number_input("Max Landsat scenes to fetch", value=10, min_value=1, max_value=50)

st.divider()

if st.button("Run Pipeline"):
    st.write("### Inputs received:")
    st.write(f"Bounding box: ({west}, {south}, {east}, {north})")
    st.write(f"Date range: {start_date} to {end_date}")
    st.write(f"Resolution: {resolution} degrees")
    st.write(f"Max cloud cover: {max_cloud}%")
    st.write(f"Landsat scenes: up to {landsat_limit}")
    st.success("good)")
