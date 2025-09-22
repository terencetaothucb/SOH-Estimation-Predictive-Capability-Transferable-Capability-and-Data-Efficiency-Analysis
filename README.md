# SOH-Estimation-Predictive-Capability-Transferable-Capability-and-Data-Efficiency-Analysis

This repository contains the datasets, processed features, results, and code used for **State-of-Health (SOH) prediction of lithium-ion batteries** with a focus on **Predictive Capability Transferable Capability and data sufficiency**.  
The work involves two datasets (CALCE and TJU), extracted degradation-related features, and XGBoost-based predictive modeling.
The repository structure is as follows:

---

### 1. Raw Datasets
- **CALCE dataset (University of Maryland, Centre for Advanced Life Cycle Engineering)**  
  - Batteries **CS_35** (source domain, discharged at 1C)  
  - Batteries **CS_33** (target domain, discharged at 0.5C)  
  - Charging protocol: CC at 0.5C until 4.2 V → CV at 4.2 V until current < 0.05 A  
  - Discharge protocol: CC to 2.7 V  

- **TJU dataset (Tongji University)**  
  - 18650-type NCA batteries tested at **25 °C, 35 °C, and 45 °C**  
  - **Source domain:** 19 batteries at 25 °C (charged 0.5C, discharged 1C)  
  - **Target domain:** 3 batteries at 35 °C (charged 0.5C, discharged 1C); the longest-life battery is used as target in our work  

---

### 2. Processed Data

#### CALCE
- `CALCE_source_features.xlsx` → Extracted features of source domain  
- `CALCE_target_features.xlsx` → Extracted features of target domain  
- `CALCE_xgb_feature_addition_results_summary.xlsx` → Data efficiency analysis results  
- `xgb_predicted_target_no_transfer.xlsx` → Predicted SOH without transfer (source model directly applied to target)  
- `xgb_predicted_target_with_transfer.xlsx` → Predicted SOH with transfer using **all features** (`P1_x, P1_y, P2_x, P2_y, P12_Ar`)  
- `xgb_predicted_target_P1_with_transfer.xlsx` → Predicted SOH with transfer using **only P1_x and P1_y**  

#### TJU
- `TJU_CY25_05_1_features_all.xlsx` → Extracted features of 19 source domain batteries  
- `TJU_CY35_05_1_features.xlsx` → Extracted features of target domain battery #1 (longest lifespan)  
- `TJU_xgb_feature_addition_results_summary.xlsx` → Data efficiency analysis results  
- `TJU_xgb_predicted_target_no_transfer.xlsx` → Predicted SOH without transfer  
- `TJU_xgb_predicted_target_with_transfer.xlsx` → Predicted SOH with transfer using **all features**  
- `TJU_xgb_predicted_target_P1_with_transfer.xlsx` → Predicted SOH with transfer using **only P1_x and P1_y**  

#### Additional
- `TJU_CY25_05_1_features_all_1.xlsx` → Extracted features + capacity of 19 source domain batteries  
- `TJU_CY25_025_1_features.xlsx` → Extracted features at **25 °C with 0.25C discharge** (potential future use)  

---

### 3. Code
- `PC TC DE_Figure 1-5.py` → Code for Figures 1–5  
- `PC TC DE_Figure 6-7.py` → Code for Figures 6–7  

---


