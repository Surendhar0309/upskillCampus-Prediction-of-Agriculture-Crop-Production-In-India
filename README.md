# Project 4 — Prediction of Agriculture Crop Production in India

**UCT Machine Learning Internship Programme**

---

## Overview
This project uses machine learning to predict crop yield (Quintals/Hectare) across different Indian states using government agricultural data from 2001–2014.

**Best Model:** Random Forest Regressor — **R² = 0.9377**

---

## Repository Structure
```
├── crop_production_prediction.py   ← Main ML script (all steps)
├── data/                           ← Raw CSV datasets (data.gov.in)
│   ├── produce.csv                 ← Production time-series 1993–2014
│   ├── datafile (1).csv            ← Cost & yield by crop/state (PRIMARY)
│   ├── datafile (2).csv            ← Area, production, yield index
│   ├── datafile (3).csv            ← Crop variety & recommended zone
│   └── datafile.csv                ← Price index 2004–2012
├── plots/                          ← 13 EDA & results visualisations
├── models/                         ← Serialised best model (best_model.pkl)
├── outputs/                        ← Model comparison CSV, summary, Report
│   └── Project4_Agriculture_Crop_Production_Report.docx
└── README.md
```

---

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
python crop_production_prediction.py
```

---

## Dataset
Source: [data.gov.in](https://data.gov.in/) — Government of India Open Data (fully licensed)

---

## Results Summary

| Model              | MAE    | RMSE   | R²     | CV R²  |
|--------------------|--------|--------|--------|--------|
| Random Forest      | 31.45  | 75.04  | **0.9377** | 0.458 |
| Gradient Boosting  | 45.78  | 135.68 | 0.7962 | 0.915  |
| Ridge Regression   | 116.38 | 139.20 | 0.7855 | -20.79 |
| Decision Tree      | 60.16  | 179.23 | 0.6444 | 0.561  |

---

## Company
**UCT — Universal Computing Technologies**  
Machine Learning Internship Programme  
