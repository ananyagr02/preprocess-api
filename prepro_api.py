# preprocessing_api.py (Corrected Order of Operations)

# import pandas as pd
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import joblib
# import os

# # --- CONFIGURATION ---
# zero_cols = [
#     "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
#     "fin_flag_cnt", "syn_flag_cnt", "down_up_ratio", "fwd_header_len",
#     "bwd_header_len", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg",
#     "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg"
# ]
# flow_rate_based = [
#     "flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s"
# ]

# # --- 1. Define the Input Data Structure (46 features) ---
# class Flow(BaseModel):
#     tot_fwd_pkts: float
#     tot_bwd_pkts: float
#     fin_flag_cnt: float
#     syn_flag_cnt: float
#     fwd_psh_flags: float
#     bwd_psh_flags: float
#     fwd_urg_flags: float
#     bwd_urg_flags: float
#     totlen_fwd_pkts: float
#     totlen_bwd_pkts: float
#     fwd_pkt_len_max: float
#     fwd_pkt_len_min: float
#     fwd_pkt_len_mean: float
#     fwd_pkt_len_std: float
#     bwd_pkt_len_max: float
#     bwd_pkt_len_min: float
#     bwd_pkt_len_mean: float
#     bwd_pkt_len_std: float
#     pkt_len_min: float
#     pkt_len_max: float
#     pkt_len_mean: float
#     pkt_len_std: float
#     pkt_len_var: float
#     fwd_byts_b_avg: float
#     fwd_pkts_b_avg: float
#     fwd_blk_rate_avg: float
#     bwd_byts_b_avg: float
#     bwd_pkts_b_avg: float
#     bwd_blk_rate_avg: float
#     init_fwd_win_byts: float
#     init_bwd_win_byts: float
#     fwd_seg_size_min: float
#     down_up_ratio: float
#     bwd_seg_size_avg: float
#     fwd_header_len: float
#     bwd_header_len: float
#     flow_duration: float
#     flow_iat_mean: float
#     flow_iat_min: float
#     bwd_iat_tot: float
#     bwd_iat_mean: float
#     bwd_iat_min: float
#     flow_byts_s: float
#     flow_pkts_s: float
#     fwd_pkts_s: float
#     bwd_pkts_s: float

# # --- 2. Load Your Saved Preprocessing Artifacts ---
# # IMPORTANT: These artifacts must be regenerated using the corrected artifact generation script.
# try:
#     print("Loading preprocessing artifacts...")
#     median_values = joblib.load('dos_medians.pkl')
#     min_max_scaler = joblib.load('dos_min_max_scaler.pkl')
#     min_max_cols = joblib.load('dos_min_max_cols.pkl')
#     standard_scaler = joblib.load('dos_standard_scaler.pkl')
#     standard_cols = joblib.load('dos_standard_cols.pkl')
#     print("Artifacts loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR loading artifacts: {e}")
#     median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5

# # --- 3. Create the FastAPI Application ---
# app = FastAPI()

# # --- 4. Define the Preprocessing Function ---
# def preprocess_dos_data(df: pd.DataFrame) -> pd.DataFrame:
#     print("Starting data preprocessing for API...")
#     if df.empty: return pd.DataFrame()

#     # --- START: Imputation and Cleaning ---
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     print("Step 1: Replaced Infinity values with NaN.")

#     df.fillna(median_values, inplace=True)
#     print("Step 2: Imputed missing values with saved medians.")

#     zero_cols_to_impute = [col for col in zero_cols if col in df.columns]
#     if zero_cols_to_impute:
#         df[zero_cols_to_impute] = df[zero_cols_to_impute].fillna(0)
#         print(f"Step 3: Imputed {len(zero_cols_to_impute)} columns with zero.")

#     df.fillna(0, inplace=True)
#     print("Step 4: Performed final fillna(0) as a safeguard.")
#     # --- END: Imputation and Cleaning ---

#     # --- START: Transformations (CORRECTED ORDER) ---
#     print("Applying transformations using loaded artifacts...")

#     # Step 5: Apply Log Transformation FIRST
#     flow_rate_cols_to_transform = [col for col in flow_rate_based if col in df.columns]
#     if flow_rate_cols_to_transform:
#         df[flow_rate_cols_to_transform] = np.log1p(df[flow_rate_cols_to_transform])
#         print(f"Step 5: Applied Log transformation to {len(flow_rate_cols_to_transform)} columns.")

#     # Step 6: Apply Min-Max Scaling
#     if min_max_scaler and min_max_cols:
#         cols_to_transform_mm = [col for col in min_max_cols if col in df.columns]
#         if cols_to_transform_mm:
#             df[cols_to_transform_mm] = min_max_scaler.transform(df[cols_to_transform_mm])
#             print(f"Step 6: Applied Min-Max scaling to {len(cols_to_transform_mm)} columns.")

#     # Step 7: Apply Standard Scaling
#     if standard_scaler and standard_cols:
#         cols_to_transform_std = [col for col in standard_cols if col in df.columns]
#         if cols_to_transform_std:
#             df[cols_to_transform_std] = standard_scaler.transform(df[cols_to_transform_std])
#             print(f"Step 7: Applied Standard scaling to {len(cols_to_transform_std)} columns.")
#     # --- END: Transformations ---

#     # Final check for data integrity
#     assert df.notna().all().all(), "Error: NaNs still exist after preprocessing!"
#     assert np.isfinite(df.to_numpy()).all(), "Error: Infinite values still exist!"
#     print("✅ Preprocessing checks passed.")
#     return df

# # --- 5. Define the Preprocessing Endpoint ---
# @app.post("/preprocess/dos")
# async def preprocess_dos_data_api(data: List[Flow]):
#     if not data: raise HTTPException(status_code=400, detail="No data provided")
#     if not all([median_values, min_max_scaler, standard_scaler]):
#          raise HTTPException(status_code=503, detail="Server is not ready: Preprocessing artifacts are missing.")
#     try:
#         input_df = pd.DataFrame([flow.dict() for flow in data])
#         processed_df = preprocess_dos_data(input_df)
#         return processed_df.to_dict(orient='records')
#     except Exception as e:
#         print(f"An error occurred during preprocessing: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # --- 6. Root endpoint ---
# @app.get("/")
# def read_root(): return {"status": "DoS Preprocessing API (46-feature model) is running"}





import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

# --- 1. CONFIGURATION (Must match artifact generation scripts) ---

# --- a. Artifact Filenames ---
MEDIAN_ARTIFACT_PATH = 'dos_medians.pkl'
MM_SCALER_PATH = 'dos_min_max_scaler.pkl'
MM_COLS_PATH = 'dos_min_max_cols.pkl'
STD_SCALER_PATH = 'dos_standard_scaler.pkl'
STD_COLS_PATH = 'dos_standard_cols.pkl'

# --- b. Column Lists for Processing Steps ---
# List for Zero Imputation (from median artifact script)
zero_impute_cols = [
    'protocol', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'ack_flag_cnt',
    'down_up_ratio', 'fwd_header_len', 'bwd_header_len',
    'subflow_fwd_byts', 'subflow_bwd_byts', 'fwd_act_data_pkts'
]
# List for Log Transformation (from scaler artifact script)
flow_rate_based = [
    'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s'
]

# --- 2. Define the Input Data Structure (All 50 features) ---
class Flow(BaseModel):
    protocol: float
    flow_duration: float
    tot_fwd_pkts: float
    tot_bwd_pkts: float
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
    flow_byts_s: float
    flow_pkts_s: float
    flow_iat_mean: float
    flow_iat_std: float
    flow_iat_max: float
    flow_iat_min: float
    fwd_iat_tot: float
    fwd_iat_mean: float
    fwd_iat_std: float
    fwd_iat_max: float
    fwd_iat_min: float
    bwd_iat_tot: float
    bwd_iat_mean: float
    bwd_iat_std: float
    bwd_iat_max: float
    bwd_iat_min: float
    fwd_header_len: float
    bwd_header_len: float
    fwd_pkts_s: float
    bwd_pkts_s: float
    pkt_len_min: float
    pkt_len_max: float
    pkt_len_mean: float
    pkt_len_std: float
    pkt_len_var: float
    fin_flag_cnt: float
    syn_flag_cnt: float
    rst_flag_cnt: float
    ack_flag_cnt: float
    down_up_ratio: float
    pkt_size_avg: float
    fwd_seg_size_avg: float
    bwd_seg_size_avg: float
    subflow_fwd_byts: float
    subflow_bwd_byts: float
    fwd_act_data_pkts: float

# --- 3. Load Your Saved Preprocessing Artifacts ---
try:
    print("Loading preprocessing artifacts for 50-feature model...")
    median_values = joblib.load(MEDIAN_ARTIFACT_PATH)
    min_max_scaler = joblib.load(MM_SCALER_PATH)
    min_max_cols = joblib.load(MM_COLS_PATH)
    standard_scaler = joblib.load(STD_SCALER_PATH)
    standard_cols = joblib.load(STD_COLS_PATH)
    print("✅ Artifacts loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading artifacts: {e}")
    # Set to None so the API will return a 503 error instead of crashing
    median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5

# --- 4. Create the FastAPI Application ---
app = FastAPI()

# --- 5. Define the Preprocessing Function (Identical Logic to Artifact Generation) ---
def preprocess_dos_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Starting Live Data Preprocessing ---")
    if df.empty: return pd.DataFrame()

    # --- START: Multi-Stage Imputation (Replicating the Median Artifact Script) ---
    # Step 1: Handle Infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Step 1: Replaced Infinity values with NaN.")

    # Step 2: Zero Imputation (Specific Columns)
    cols_to_zero_impute = [col for col in zero_impute_cols if col in df.columns]
    if cols_to_zero_impute:
        df[cols_to_zero_impute] = df[cols_to_zero_impute].fillna(0)
        print(f"Step 2: Applied zero-imputation to {len(cols_to_zero_impute)} specific columns.")
    
    # Step 3: Formula Imputation (Specific Columns)
    epsilon = 1e-9
    flow_duration_sec = df['flow_duration'] / 1e6 + epsilon
    df['flow_byts_s'] = df['flow_byts_s'].fillna((df['totlen_fwd_pkts'] + df['totlen_bwd_pkts']) / flow_duration_sec)
    df['flow_pkts_s'] = df['flow_pkts_s'].fillna((df['tot_fwd_pkts'] + df['tot_bwd_pkts']) / flow_duration_sec)
    df['fwd_pkts_s'] = df['fwd_pkts_s'].fillna(df['tot_fwd_pkts'] / flow_duration_sec)
    df['bwd_pkts_s'] = df['bwd_pkts_s'].fillna(df['tot_bwd_pkts'] / flow_duration_sec)
    df['bwd_seg_size_avg'] = df['bwd_seg_size_avg'].fillna(df['totlen_bwd_pkts'] / (df['tot_bwd_pkts'] + epsilon))
    print("Step 3: Applied formula-based imputation to 5 rate/size columns.")

    # Step 4: Median Imputation (Final Imputation for Remaining NaNs)
    df.fillna(median_values, inplace=True)
    print(f"Step 4: Applied median-imputation to {len(median_values)} columns using saved artifact.")

    # Step 5: Safeguard Fill (To catch any column not in the median artifact, if any)
    df.fillna(0, inplace=True)
    print("Step 5: Performed final fillna(0) as a safeguard.")
    # --- END: Imputation ---

    # --- START: Transformations (Replicating the Scaler Artifact Script) ---
    print("\n--- Applying Transformations ---")

    # Step 6: Apply Log Transformation FIRST
    cols_to_log_transform = [col for col in flow_rate_based if col in df.columns]
    if cols_to_log_transform:
        df[cols_to_log_transform] = np.log1p(df[cols_to_log_transform])
        print(f"Step 6: Applied Log transformation to {len(cols_to_log_transform)} columns.")

    # Step 7: Apply Min-Max Scaling
    if min_max_scaler and min_max_cols:
        cols_to_transform_mm = [col for col in min_max_cols if col in df.columns]
        if cols_to_transform_mm:
            df[cols_to_transform_mm] = min_max_scaler.transform(df[cols_to_transform_mm])
            print(f"Step 7: Applied Min-Max scaling to {len(cols_to_transform_mm)} columns.")

    # Step 8: Apply Standard Scaling
    if standard_scaler and standard_cols:
        cols_to_transform_std = [col for col in standard_cols if col in df.columns]
        if cols_to_transform_std:
            df[cols_to_transform_std] = standard_scaler.transform(df[cols_to_transform_std])
            print(f"Step 8: Applied Standard scaling to {len(cols_to_transform_std)} columns.")
    # --- END: Transformations ---

    # Final check for data integrity
    if not df.notna().all().all():
        raise RuntimeError("CRITICAL ERROR: NaNs still exist after preprocessing!")
    if not np.isfinite(df.to_numpy()).all():
        raise RuntimeError("CRITICAL ERROR: Infinite values still exist after preprocessing!")
    print("\n✅ Preprocessing checks passed.")
    return df

# --- 6. Define the Preprocessing Endpoint ---
@app.post("/preprocess/dos")
async def preprocess_dos_data_api(data: List[Flow]):
    if not data: raise HTTPException(status_code=400, detail="No data provided")
    if not all([median_values, min_max_scaler, standard_scaler]):
         raise HTTPException(status_code=503, detail="Server is not ready: Preprocessing artifacts are missing.")
    try:
        # Convert the list of Pydantic models to a DataFrame
        input_df = pd.DataFrame([flow.dict() for flow in data])
        
        # Process the data using our replicated logic
        processed_df = preprocess_dos_data(input_df)
        
        # Return the processed data in the correct format
        return processed_df.to_dict(orient='records')
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. Root endpoint ---
@app.get("/")
def read_root(): return {"status": "DoS Preprocessing API (50-feature model) is running"}