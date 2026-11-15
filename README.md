# Dropout Risk and Counterfactual Recourse Dashboard (thesis prototype, v2)

Khansa Munir (2024-ag-1876), MS Computer Science, University of Agriculture Faisalabad. Supervisor: Ms Nabeela Ashraf.
Thesis: *From Prediction to Prescription: Counterfactual Driven Dropout Risk Intervention Using Stacking Ensemble and Multi-Layer XAI*.

The dashboard displays the saved results of the full-data experiment (4,424 students; SHAP and LIME on all 885 test students; counterfactuals on all 284 flagged students). Nothing is re-trained.

## Tabs
1 Data (overview and preprocessing summary) | 2 Model (colour-labelled metrics, confusion matrix, model comparison, AutoML baseline) | 3 Explainability (global SHAP, SHAP-LIME agreement) | 4 Student risk (risk list, SHAP waterfall, LIME check) | 5 Individual CF (CF1-CF5 table, before/after chart, radar chart, contribution of each change) | 6 Group CF (K-means, Silhouette/Davies-Bouldin/Calinski-Harabasz, group action with per-student feasibility) | 7 What-if (actionable features, rules R1-R4) | 8 Score new data (CSV/Excel upload, live constrained recommendations) | 9 Fairness and calibration | 10 Method quality (validity, feasibility, recourse rate and its search sensitivity) | 11 Export.

## Run locally (Python 3.11 or 3.12)
    pip install -r requirements.txt
    streamlit run app.py
`scikit-learn` and `xgboost` are pinned to the versions used to save `data/final_model.pkl` (see `data/run_info.json`).

## Docker
    docker build -t dropout-dashboard .
    docker run -p 8501:8501 dropout-dashboard

## Deploy (free)
Push this folder to a GitHub repository; on Streamlit Community Cloud choose `app.py` and set Python 3.12 under Advanced settings.

## Limits
Prototype for decision support on an anonymised public dataset; not evaluated with real users. Group counterfactuals use a 600-action grid (search-based, conservative). The recourse rates are lower bounds of the true rate. The fairness table is descriptive only.
