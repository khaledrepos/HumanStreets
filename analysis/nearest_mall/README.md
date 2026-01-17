# Riyadh Districts – Nearest Mall Analysis

## 📌 Overview
 identify the nearest shopping mall for each district and calculate the distance between them.

## 📊 Outputs
- CSV file containing:
  - District name
  - Nearest mall name
  - Distance (meters & kilometers)
- Visualization charts:
  - Distance to nearest mall for all districts
  - Charts are split into groups for clarity

## 🛠 Tools & Libraries
- Python
- GeoPandas
- OSMnx
- Shapely
- Matplotlib

## 🎯 Objective
For each district in Riyadh:
- Represent each district using a centroid point
- Identify the nearest mall
- Calculate distances accurately using a projected CRS (UTM)
- Handle duplicate districts by keeping the closest mall only

