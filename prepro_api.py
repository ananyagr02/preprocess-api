# # preprocessing_api.py (Corrected and Final Version)

# import pandas as pd
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import joblib
# import os
# from sklearn.exceptions import NotFittedError # <-- CORRECT: Import added

# # --- CONFIGURATION (Added for clarity and to fix undefined variables) ---
# # Define the columns for zero imputation so the function can access it.
# zero_cols = [
#     "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
#     "fin_flag_cnt", "syn_flag_cnt", "down_up_ratio", "fwd_header_len",
#     "bwd_header_len", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg",
#     "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg"
# ]

# # Define columns for log transformation
# flow_rate_based = [
#     "flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s"
# ]

# # --- 1. Define the Input Data Structure ---
# # This tells FastAPI what kind of JSON data to expect from n8n.
# # The field names here MUST match the names you set in your n8n "Edit Fields" node.
# class Flow(BaseModel):
#     # This list now correctly includes 'src_port' and has no duplicates.
    
#     flow_duration: float
#     tot_fwd_pkts: float
#     tot_bwd_pkts: float
#     totlen_fwd_pkts: float
#     fwd_pkt_len_max: float
#     fwd_pkt_len_std: float
#     bwd_pkt_len_max: float
#     bwd_pkt_len_min: float
#     bwd_pkt_len_mean: float
#     bwd_pkt_len_std: float
#     flow_byts_s: float
#     flow_pkts_s: float
#     flow_iat_mean: float
#     flow_iat_min: float
#     bwd_iat_tot: float
#     bwd_iat_mean: float
#     bwd_iat_min: float
#     fwd_psh_flags: float
#     bwd_psh_flags: float
#     fwd_urg_flags: float
#     bwd_urg_flags: float
#     fwd_header_len: float
#     bwd_header_len: float
#     fwd_pkts_s: float
#     bwd_pkts_s: float
#     fin_flag_cnt: float
#     syn_flag_cnt: float
#     down_up_ratio: float
#     bwd_seg_size_avg: float
#     fwd_byts_b_avg: float
#     fwd_pkts_b_avg: float
#     fwd_blk_rate_avg: float
#     bwd_byts_b_avg: float
#     bwd_pkts_b_avg: float
#     bwd_blk_rate_avg: float
#     init_fwd_win_byts: float
#     init_bwd_win_byts: float
#     fwd_seg_size_min: float

# # --- 2. Load Your Saved Preprocessing Artifacts ---
# try:
#     print("Loading preprocessing artifacts...")
#     median_values = joblib.load('dos_medians.pkl')
#     min_max_scaler = joblib.load('dos_min_max_scaler.pkl')
#     min_max_cols = joblib.load('dos_min_max_cols.pkl')
#     standard_scaler = joblib.load('dos_standard_scaler.pkl')
#     standard_cols = joblib.load('dos_standard_cols.pkl')
#     print("Artifacts loaded successfully.")
# except FileNotFoundError as e:
#     print(f"FATAL ERROR: Preprocessing artifact file not found: {e}.")
#     # Set to None to cause a graceful failure later
#     median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5
# except Exception as e:
#     print(f"An unexpected error occurred loading artifacts: {e}")
#     median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5

# # --- 3. Create the FastAPI Application ---
# app = FastAPI()

# # --- 4. Define the Preprocessing Function ---
# def preprocess_dos_data(df: pd.DataFrame) -> pd.DataFrame: # <-- CORRECT: Renamed for internal clarity
#     """
#     Applies preprocessing steps to the incoming DataFrame using loaded artifacts.
#     """
#     print("Starting data preprocessing for API...")
    
#     if df.empty:
#         print("Warning: Received empty DataFrame for preprocessing.")
#         return pd.DataFrame()

#     # --- Replicate the cleaning steps ---
#     # Step 1: Clean column names (map from pyflowmeter's snake_case)
#     # The API will receive snake_case from n8n. We ensure it matches the training columns.
#     df.columns = [col.strip() for col in df.columns]
#     # No renaming needed if the training artifacts used snake_case.
#     print("Step 1: Cleaned column names.")

#     # Step 2: Handle Infinity values
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     print("Step 2: Replaced Infinity values with NaN.")
    
#     # Step 3: Impute with MEDIAN values (from dos_medians.pkl)
#     # This must happen before scaling.
#     df.fillna(median_values, inplace=True)
#     print("Step 3: Imputed missing values with saved medians.")
    
#     # Step 4: Impute with ZERO
#     zero_cols_to_impute = [col for col in zero_cols if col in df.columns]
#     if zero_cols_to_impute:
#         df[zero_cols_to_impute] = df[zero_cols_to_impute].fillna(0)
#         print(f"Step 4: Imputed {len(zero_cols_to_impute)} columns with zero.")

#     # Step 5: Impute with Derived Formulas
#     # ... (This logic remains the same, but it's now a fallback after median/zero imputation) ...
#     print("Step 5: Applied derived formula imputations as fallback.")

#     # Step 6: Final safety fill for any remaining NaNs
#     df.fillna(0, inplace=True)
#     print("Step 6: Performed final fillna(0) as a safeguard.")

#     # --- Apply Transformations Using Loaded Artifacts ---
#     print("Applying transformations using loaded artifacts...")

#     # Apply Min-Max Scaling
#     if min_max_scaler and min_max_cols:
#         cols_to_transform_mm = [col for col in min_max_cols if col in df.columns]
#         if cols_to_transform_mm:
#             df[cols_to_transform_mm] = min_max_scaler.transform(df[cols_to_transform_mm]) # <-- CORRECT: Using correct variable name
#             print(f"Applied Min-Max scaling to {len(cols_to_transform_mm)} columns.")
#         else:
#             print("No columns found for Min-Max scaling. Skipping.")

#     # Apply Standard Scaling (Z-score)
#     if standard_scaler and standard_cols:
#         cols_to_transform_std = [col for col in standard_cols if col in df.columns]
#         if cols_to_transform_std:
#             df[cols_to_transform_std] = standard_scaler.transform(df[cols_to_transform_std]) # <-- CORRECT: Using correct variable name
#             print(f"Applied Standard scaling to {len(cols_to_transform_std)} columns.")
#         else:
#             print("No columns found for Standard scaling. Skipping.")

#     # Apply Log Transformation
#     flow_rate_cols_to_transform = [col for col in flow_rate_based if col in df.columns]
#     if flow_rate_cols_to_transform:
#         df[flow_rate_cols_to_transform] = np.log1p(df[flow_rate_cols_to_transform])
#         print(f"Applied Log transformation to {len(flow_rate_cols_to_transform)} columns.")

#     # Final check
#     assert df.notna().all().all(), "Error: NaNs still exist after preprocessing!"
#     assert np.isfinite(df.to_numpy()).all(), "Error: Infinite values still exist!"
#     print("✅ Preprocessing checks passed.")

#     return df

# # --- 5. Define the Preprocessing Endpoint ---
# @app.post("/preprocess/dos")
# async def preprocess_dos_data_api(data: List[Flow]): # <-- CORRECT: Renamed for clarity
#     """
#     Receives data from n8n, preprocesses it, and returns the result.
#     """
#     if not data:
#         raise HTTPException(status_code=400, detail="No data provided")

#     if not all([median_values, min_max_scaler, standard_scaler]):
#          raise HTTPException(status_code=503, detail="Server is not ready: Preprocessing artifacts are missing.")

#     try:
#         # Convert the list of Pydantic models into a DataFrame
#         input_df = pd.DataFrame([flow.dict() for flow in data])

#         # Call the internal preprocessing function
#         processed_df = preprocess_dos_data(input_df) # <-- CORRECT: Calls the correct function
        
#         # Convert the final DataFrame back to a list of JSON objects
#         response = processed_df.to_dict(orient='records')
#         return response

#     except Exception as e:
#         print(f"An error occurred during preprocessing: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# # --- 6. Root endpoint ---
# @app.get("/")
# def read_root():
#     return {"status": "DoS Preprocessing API is running"}






# preprocessing_api.py (Corrected Order of Operations)

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

# --- CONFIGURATION ---
zero_cols = [
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fin_flag_cnt", "syn_flag_cnt", "down_up_ratio", "fwd_header_len",
    "bwd_header_len", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg",
    "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg"
]
flow_rate_based = [
    "flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s"
]

# --- 1. Define the Input Data Structure (46 features) ---
class Flow(BaseModel):
    tot_fwd_pkts: float
    tot_bwd_pkts: float
    fin_flag_cnt: float
    syn_flag_cnt: float
    fwd_psh_flags: float
    bwd_psh_flags: float
    fwd_urg_flags: float
    bwd_urg_flags: float
    totlen_fwd_pkts: float
    totlen_bwd_pkts: float
    fwd_pkt_len_max: float
    fwd_pkt_len_min: float
    fwd_pkt_len_mean: float
    fwd_pkt_len_std: float
    bwd_pkt_len_max: float
    bwd_pkt_len_min: float
    bwd_pkt_len_mean: float
    bwd_pkt_len_std: float
    pkt_len_min: float
    pkt_len_max: float
    pkt_len_mean: float
    pkt_len_std: float
    pkt_len_var: float
    fwd_byts_b_avg: float
    fwd_pkts_b_avg: float
    fwd_blk_rate_avg: float
    bwd_byts_b_avg: float
    bwd_pkts_b_avg: float
    bwd_blk_rate_avg: float
    init_fwd_win_byts: float
    init_bwd_win_byts: float
    fwd_seg_size_min: float
    down_up_ratio: float
    bwd_seg_size_avg: float
    fwd_header_len: float
    bwd_header_len: float
    flow_duration: float
    flow_iat_mean: float
    flow_iat_min: float
    bwd_iat_tot: float
    bwd_iat_mean: float
    bwd_iat_min: float
    flow_byts_s: float
    flow_pkts_s: float
    fwd_pkts_s: float
    bwd_pkts_s: float

# --- 2. Load Your Saved Preprocessing Artifacts ---
# IMPORTANT: These artifacts must be regenerated using the corrected artifact generation script.
try:
    print("Loading preprocessing artifacts...")
    median_values = joblib.load('dos_medians.pkl')
    min_max_scaler = joblib.load('dos_min_max_scaler.pkl')
    min_max_cols = joblib.load('dos_min_max_cols.pkl')
    standard_scaler = joblib.load('dos_standard_scaler.pkl')
    standard_cols = joblib.load('dos_standard_cols.pkl')
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading artifacts: {e}")
    median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5

# --- 3. Create the FastAPI Application ---
app = FastAPI()

# --- 4. Define the Preprocessing Function ---
def preprocess_dos_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting data preprocessing for API...")
    if df.empty: return pd.DataFrame()

    # --- START: Imputation and Cleaning ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Step 1: Replaced Infinity values with NaN.")

    df.fillna(median_values, inplace=True)
    print("Step 2: Imputed missing values with saved medians.")

    zero_cols_to_impute = [col for col in zero_cols if col in df.columns]
    if zero_cols_to_impute:
        df[zero_cols_to_impute] = df[zero_cols_to_impute].fillna(0)
        print(f"Step 3: Imputed {len(zero_cols_to_impute)} columns with zero.")

    df.fillna(0, inplace=True)
    print("Step 4: Performed final fillna(0) as a safeguard.")
    # --- END: Imputation and Cleaning ---

    # --- START: Transformations (CORRECTED ORDER) ---
    print("Applying transformations using loaded artifacts...")

    # Step 5: Apply Log Transformation FIRST
    flow_rate_cols_to_transform = [col for col in flow_rate_based if col in df.columns]
    if flow_rate_cols_to_transform:
        df[flow_rate_cols_to_transform] = np.log1p(df[flow_rate_cols_to_transform])
        print(f"Step 5: Applied Log transformation to {len(flow_rate_cols_to_transform)} columns.")

    # Step 6: Apply Min-Max Scaling
    if min_max_scaler and min_max_cols:
        cols_to_transform_mm = [col for col in min_max_cols if col in df.columns]
        if cols_to_transform_mm:
            df[cols_to_transform_mm] = min_max_scaler.transform(df[cols_to_transform_mm])
            print(f"Step 6: Applied Min-Max scaling to {len(cols_to_transform_mm)} columns.")

    # Step 7: Apply Standard Scaling
    if standard_scaler and standard_cols:
        cols_to_transform_std = [col for col in standard_cols if col in df.columns]
        if cols_to_transform_std:
            df[cols_to_transform_std] = standard_scaler.transform(df[cols_to_transform_std])
            print(f"Step 7: Applied Standard scaling to {len(cols_to_transform_std)} columns.")
    # --- END: Transformations ---

    # Final check for data integrity
    assert df.notna().all().all(), "Error: NaNs still exist after preprocessing!"
    assert np.isfinite(df.to_numpy()).all(), "Error: Infinite values still exist!"
    print("✅ Preprocessing checks passed.")
    return df

# --- 5. Define the Preprocessing Endpoint ---
@app.post("/preprocess/dos")
async def preprocess_dos_data_api(data: List[Flow]):
    if not data: raise HTTPException(status_code=400, detail="No data provided")
    if not all([median_values, min_max_scaler, standard_scaler]):
         raise HTTPException(status_code=503, detail="Server is not ready: Preprocessing artifacts are missing.")
    try:
        input_df = pd.DataFrame([flow.dict() for flow in data])
        processed_df = preprocess_dos_data(input_df)
        return processed_df.to_dict(orient='records')
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. Root endpoint ---
@app.get("/")
def read_root(): return {"status": "DoS Preprocessing API (46-feature model) is running"}