import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import MarkerCluster, HeatMap

# Load Data
@st.cache
def load_data():
    # Load the first dataset: Vehicle Data
    df = pd.read_csv('/path/to/Vehicle Data - Virginia Department of Motor Vehicles.csv')
    
    # Load the second dataset: Geo-enabled Data
    geo_df = pd.read_csv('/path/to/CrashUID_Matches_Vehicle.csv')
    
    # Merge datasets based on Crash_UID
    geo_df['Crash_UID'] = geo_df['Document Nbr'].astype(str).str[-7:]
    df['Crash_UID_Str'] = df['Crash_UID'].astype(str).str[-7:]
    merged_df = df.merge(geo_df, left_on='Crash_UID_Str', right_on='Crash_UID', how='inner')
    
    return df, merged_df

df, merged_df = load_data()

# Streamlit Widgets for User Input (Slicers)
st.title("Virginia DMV Crash Severity Analysis")
st.sidebar.header("Select Options")

# Example Slicer for selecting crash severity
severity_score = st.sidebar.slider("Minimum Crash Severity", min_value=0, max_value=10, value=6, step=1)

# Example Slicer for selecting vehicle body type
vehicle_body_types = df['VehicleBodyType'].unique()
selected_body_type = st.sidebar.selectbox("Select Vehicle Body Type", options=["All"] + list(vehicle_body_types))

# Example Filter for Most Harmful Event
events = df['MostHarmfulEvent_Value'].unique()
selected_event = st.sidebar.selectbox("Select Most Harmful Event", options=["All"] + list(events))

# Data Filtering based on slicer inputs
filtered_df = df
if selected_body_type != "All":
    filtered_df = filtered_df[filtered_df['VehicleBodyType'] == selected_body_type]
if selected_event != "All":
    filtered_df = filtered_df[filtered_df['MostHarmfulEvent_Value'] == selected_event]
filtered_df = filtered_df[filtered_df['RefinedSeverityScore'] >= severity_score]

# Show data in the main panel
st.write(f"Displaying {filtered_df.shape[0]} rows based on selected filters.")
st.dataframe(filtered_df.head())

# Visualize Crash Severity Distribution
st.subheader("Crash Severity Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_df['RefinedSeverityScore'], bins=20, kde=True, ax=ax)
ax.set_title("Crash Severity Score Distribution")
st.pyplot(fig)

# Create the Clustering Visualization
st.subheader("Crash Clusters by Speed and Severity")

cluster_features = filtered_df[['VehicleSpeedBeforeCrash', 'VehicleSpeedLimit', 'RefinedSeverityScore']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)
filtered_df['CrashCluster'] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=cluster_features['VehicleSpeedBeforeCrash'], y=cluster_features['RefinedSeverityScore'],
                hue=filtered_df['CrashCluster'], palette='Set2', ax=ax)
ax.set_title("Crash Clusters by Speed and Severity")
st.pyplot(fig)

# Feature Importance Using RandomForest
st.subheader("Feature Importance for Predicting Crash Severity")

features = ['VehicleSpeedBeforeCrash', 'VehicleSpeedLimit', 'SpeedDiff', 'VehicleBodyType', 
            'VehicleDamage', 'VehicleCondition', 'VehicleTowed', 'VehicleDisabled', 'MostHarmfulEvent_Value']
df_model = df[features + ['RefinedSeverityScore']].copy()

# Encode categoricals
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

X = df_model.drop('RefinedSeverityScore', axis=1)
y = df_model['RefinedSeverityScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
importances.plot(kind='barh', ax=ax)
ax.set_title("Feature Importance for Predicting Crash Severity")
st.pyplot(fig)

# Map for High-Risk Areas
st.subheader("Crash Hotspots on Map")
high_risk_geo = merged_df[
    (merged_df['RefinedSeverityScore'] >= severity_score) &
    (merged_df['Latitude'].notnull()) & 
    (merged_df['Longitude'].notnull())
]

center_lat = high_risk_geo['Latitude'].median()
center_lon = high_risk_geo['Longitude'].median()

risk_map = folium.Map(location=[center_lat, center_lon], zoom_start=9)
marker_cluster = MarkerCluster().add_to(risk_map)

def get_color(sev):
    if sev >= 7:
        return 'darkred'
    elif sev >= 6.5:
        return 'orange'
    else:
        return 'blue'

for _, row in high_risk_geo.iterrows():
    popup = f"""
    <b>Crash UID:</b> {row['Crash_UID_x']}<br>
    <b>Severity:</b> {row['RefinedSeverityScore']}<br>
    <b>Vehicle:</b> {row['VehicleBodyType']}<br>
    <b>Event:</b> {row['MostHarmfulEvent_Value']}
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup, max_width=300),
        icon=folium.Icon(color=get_color(row['RefinedSeverityScore']), icon='info-sign')
    ).add_to(marker_cluster)

st.subheader("Interactive Map of Crash Hotspots")
folium_static(risk_map)

# Display Recommendations
st.subheader("Crash Severity Recommendations")
def generate_recommendation(row):
    base = f"Crash site {row['Crash_UID']} has had {row['CrashCount']} crashes averaging severity {row['AvgSeverity']:.1f}."
    if row['MostHarmfulEvent_Value'] in ['Utility Pole', 'Guard Rail', 'Ditch']:
        infra = "Recommend physical barrier upgrades or object setbacks."
    elif row['VehicleManeuver'] in ['Turning Left', 'Turning Right']:
        infra = "Suggest intersection redesign or turn-slowing measures."
    elif row['AvgMismatch'] > 10:
        infra = "Mismatch with speed and road design — suggest speed calming and signage."
    else:
        infra = "Suggest comprehensive site inspection."
    return base + " " + infra

recommendations = generate_recommendation(filtered_df)
st.write(recommendations)

# To run the Streamlit app:
# In the terminal, run: streamlit run app.py