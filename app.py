import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import gdown
import os

# Google Drive File IDs
file_id_1 = '1XrsnjzE_Sp2Pd3EmTOlPj8D5ueirUMeJ'  # Vehicle Data
file_id_2 = '1yKMQ85iLqukoEQWMGhNgcKOqwKqR68ha'  # Geo Data

# File paths
file_1 = 'vehicle_data.csv'
file_2 = 'geo_data.csv'

# Download files if not already present
if not os.path.exists(file_1):
    gdown.download(f'https://drive.google.com/uc?id={file_id_1}', file_1, quiet=False)
if not os.path.exists(file_2):
    gdown.download(f'https://drive.google.com/uc?id={file_id_2}', file_2, quiet=False)

# Load and merge data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(file_1)
    geo = pd.read_csv(file_2)

    df['VehicleSpeedBeforeCrash'] = pd.to_numeric(df['VehicleSpeedBeforeCrash'], errors='coerce')
    df['VehicleSpeedLimit'] = pd.to_numeric(df['VehicleSpeedLimit'], errors='coerce')
    df['SpeedDiff'] = (df['VehicleSpeedBeforeCrash'] - df['VehicleSpeedLimit']).fillna(0)

    def severity_score(row):
        score = 0
        score += 1.5 if row['VehicleDisabled'] == 'Yes' else 0
        score += 1.5 if row['VehicleTowed'] == 'Yes' else 0
        score += 1 if 'Severe' in row['VehicleDamage'] else 0
        score += 2 if row['MostHarmfulEvent_Value'] in ['Overturn (Rollover)', 'Head-On', 'Motor Vehicle In Transport'] else 0
        score += 1 if row['VehicleBodyType'] in ['Motorcycle', 'Truck - Sport Utility Vehicle (SUV)'] else 0
        score += 1.5 if row['SpeedDiff'] > 15 else (1 if row['SpeedDiff'] > 5 else 0)
        return score

    df['RefinedSeverityScore'] = df.apply(severity_score, axis=1)

    geo['Crash_UID_Geo'] = geo['Document Nbr'].astype(str).str[-7:]
    df['Crash_UID_Str'] = df['Crash_UID'].astype(str).str[-7:]

    merged = df.merge(geo, left_on='Crash_UID_Str', right_on='Crash_UID_Geo', how='inner')

    merged['Latitude'] = pd.to_numeric(merged['y'], errors='coerce')
    merged['Longitude'] = pd.to_numeric(merged['x'], errors='coerce')

    merged = merged.dropna(subset=['Latitude', 'Longitude', 'Recommendation'])
    merged['Recommendation'] = merged['Recommendation'].astype(str).str.strip().str.upper()
    merged = merged[~merged['Recommendation'].isin(['N/A', 'NAN', 'NULL', '', 'NONE'])]

    return merged

df = load_and_prepare_data()

# Sidebar filters
st.sidebar.header("🔎 Filters")
min_severity = st.sidebar.slider("Minimum Severity", 0.0, 10.0, 6.5, 0.5)

filtered = df[df['RefinedSeverityScore'] >= min_severity]

def get_color(sev):
    if sev >= 7:
        return 'darkred'
    elif sev >= 6.5:
        return 'orange'
    else:
        return 'blue'

st.title("🚧 Virginia Crash Hotspot Dashboard")
st.markdown("This dashboard maps high-risk crash locations and safety recommendations using DMV and geospatial data.")

if not filtered.empty:
    center_lat = filtered['Latitude'].median()
    center_lon = filtered['Longitude'].median()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered.iterrows():
        popup = f"""
        <b>Crash UID:</b> {row['Crash_UID']}<br>
        <b>Severity:</b> {row['RefinedSeverityScore']}<br>
        <b>Vehicle:</b> {row['VehicleBodyType']}<br>
        <b>Event:</b> {row['MostHarmfulEvent_Value']}<br>
        <b>Recommendation:</b> {row['Recommendation']}
        """
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color=get_color(row['RefinedSeverityScore']), icon='info-sign')
        ).add_to(marker_cluster)

    HeatMap(filtered[['Latitude', 'Longitude']].values.tolist(), radius=10).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px; left: 50px; width: 200px; height: 130px;
        background-color: white; z-index:9999;
        font-size:14px; padding:10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        border-radius: 8px;
    ">
    <b>🚦 Severity Legend</b><br>
    <i class="fa fa-map-marker fa-2x" style="color:darkred"></i> Very High (7+)<br>
    <i class="fa fa-map-marker fa-2x" style="color:orange"></i> High (6.5–7)<br>
    <i class="fa fa-map-marker fa-2x" style="color:blue"></i> Moderate (<6.5)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium_static(m)
else:
    st.warning("No crashes found for the selected filter.")

# Data table
st.subheader("📋 Crash Records")
st.dataframe(filtered[['Crash_UID', 'RefinedSeverityScore', 'MostHarmfulEvent_Value', 'Recommendation']])
