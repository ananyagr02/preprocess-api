# preprocessing_api.py

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os # Needed for checking file existence

# --- 1. Define the Input Data Structure ---
# This tells FastAPI what kind of JSON data to expect from n8n.
# The field names here MUST match the names you set in your n8n "Edit Fields" node
# for the DoS feature set.
class Flow(BaseModel):
    # This list MUST contain ONLY the 41 DoS features you've selected,
    # with the snake_case names.
    dst_port: float
    protocol: float
    flow_duration: float
    tot_fwd_pkts: float
    tot_bwd_pkts: float
    totlen_fwd_pkts: float
    fwd_pkt_len_max: float
    fwd_pkt_len_std: float
    bwd_pkt_len_max: float
    bwd_pkt_len_min: float
    bwd_pkt_len_mean: float
    bwd_pkt_len_std: float
    flow_byts_s: float
    flow_pkts_s: float
    flow_iat_mean: float
    flow_iat_min: float
    bwd_iat_tot: float
    bwd_iat_mean: float
    bwd_iat_min: float
    fwd_psh_flags: float
    bwd_psh_flags: float
    fwd_urg_flags: float
    bwd_urg_flags: float
    fwd_header_len: float
    bwd_header_len: float
    fwd_pkts_s: float # Note: This appears twice in your list, assuming it's a typo and will be handled.
    bwd_pkts_s: float
    fin_flag_cnt: float
    syn_flag_cnt: float
    down_up_ratio: float
    bwd_seg_size_avg: float
    fwd_byts_b_avg: float
    fwd_pkts_b_avg: float
    fwd_blk_rate_avg: float
    bwd_byts_b_avg: float
    bwd_pkts_b_avg: float
    bwd_blk_rate_avg: float
    init_fwd_win_byts: float
    init_bwd_win_byts: float
    fwd_seg_size_min: float

# --- 2. Load Your Saved Preprocessing Artifacts ---
# These files MUST be in the same directory as this script when deployed.
try:
    print("Loading preprocessing artifacts...")
    median_values = joblib.load('dos_medians.pkl')
    min_max_scaler = joblib.load('dos_min_max_scaler.pkl')
    min_max_cols = joblib.load('dos_min_max_cols.pkl')
    standard_scaler = joblib.load('dos_standard_scaler.pkl')
    standard_cols = joblib.load('dos_standard_cols.pkl')
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Preprocessing artifact file not found: {e}. Please ensure all .pkl files are in the same directory as the script.")
    # In a real application, you'd likely want to exit or raise a more specific error.
    # For now, we'll allow the script to proceed but error out later if artifacts are used.
    median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5
except Exception as e:
    print(f"An unexpected error occurred loading artifacts: {e}")
    # Similar to FileNotFoundError, handle critical failures.
    median_values, min_max_scaler, min_max_cols, standard_scaler, standard_cols = [None]*5

# --- 3. Create the FastAPI Application ---
app = FastAPI()

# --- 4. Define the Preprocessing Function ---
def preprocess_dos_data_for_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing steps to the incoming DataFrame using loaded artifacts.
    """
    print("Starting data preprocessing for API...")
    
    if df.empty:
        print("Warning: Received empty DataFrame for preprocessing.")
        return pd.DataFrame() # Return empty if input is empty

    # --- Replicate the cleaning steps EXACTLY as done during artifact generation ---
    # This ensures consistency.

    # Step 1: Strip leading spaces from column names (if any)
    df.columns = [col.strip() for col in df.columns]
    # Handle potential column name issues if they exist after strip (e.g., "Fwd Packets/s")
    df.rename(columns={'bwd_seg_size_avg': 'Avg Bwd Segment Size', 'fwd_seg_size_avg': 'Avg Fwd Segment Size'}, inplace=True)
    print("Step 1: Cleaned and normalized column names.")

    # Step 2: Handle Infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Step 2: Replaced Infinity values with NaN.")

    # Step 3: Drop rows where critical identifier columns have NaN
    # These columns are essential for identifying flows, and their absence would be problematic.
    critical_cols_for_dropna = ["Source Port", "Destination Port", "Protocol"] # Label is not expected in API input
    existing_critical_cols = [col for col in critical_cols_for_dropna if col in df.columns]
    if existing_critical_cols:
        original_rows = len(df)
        df.dropna(subset=existing_critical_cols, inplace=True)
        if original_rows > len(df):
            print(f"Step 3: Dropped {original_rows - len(df)} rows with NaN in critical columns.")
    else:
        print("Step 3: No critical identifier columns found for NaN row removal.")

    # Step 4: Impute with Zero
    zero_cols_to_impute = [col for col in zero_cols if col in df.columns]
    if zero_cols_to_impute:
        df[zero_cols_to_impute].fillna(0, inplace=True)
        print(f"Step 4: Imputed {len(zero_cols_to_impute)} columns with zero.")

    # Step 5: Impute with Derived Formulas
    epsilon = 1e-9 # To prevent division by zero
    
    if "Flow Bytes/s" in df.columns and "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns and "Flow Duration" in df.columns:
        df["Flow Bytes/s"] = df["Flow Bytes/s"].fillna(
            (df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]) / (df["Flow Duration"] + epsilon)
        )
    if "Fwd Packets/s" in df.columns and "Total Fwd Packets" in df.columns and "Flow Duration" in df.columns:
        # Assuming Flow Duration is in microseconds, convert to seconds for rate calculation
        df["Fwd Packets/s"] = df["Fwd Packets/s"].fillna(df["Total Fwd Packets"] / (df["Flow Duration"] / 1e6 + epsilon))
    if "Bwd Packets/s" in df.columns and "Total Backward Packets" in df.columns and "Flow Duration" in df.columns:
        df["Bwd Packets/s"] = df["Bwd Packets/s"].fillna(df["Total Backward Packets"] / (df["Flow Duration"] / 1e6 + epsilon))
    if "Avg Bwd Segment Size" in df.columns and "Total Length of Bwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        df["Avg Bwd Segment Size"] = df["Avg Bwd Segment Size"].fillna(df["Total Length of Bwd Packets"] / (df["Total Backward Packets"] + epsilon))
    print("Step 5: Applied derived formula imputations.")

    # Step 6: Final safety fill for any remaining NaNs
    df.fillna(0, inplace=True)
    print("Step 6: Performed final fillna(0).")

    # --- Apply Transformations Using Loaded Artifacts ---
    print("Applying transformations using loaded artifacts...")

    # Apply Min-Max Scaling
    if min_max_scaler and min_max_cols:
        print(f"Applying Min-Max scaling to {len(min_max_cols)} columns...")
        try:
            # Ensure columns exist in the current data before transforming
            cols_to_transform_mm = [col for col in min_max_cols if col in df.columns]
            if cols_to_transform_mm:
                df[cols_to_transform_mm] = min_max_scaler.transform(df[cols_to_mm_cols])
                print(f"Applied Min-Max scaling to {len(cols_to_transform_mm)} columns.")
            else:
                print("No columns found for Min-Max scaling in the input data. Skipping.")
        except NotFittedError:
            print("Error: Min-Max scaler was not fitted. Artifacts might be missing or corrupted.")
        except Exception as e:
            print(f"Error applying Min-Max scaling: {e}")
    else:
        print("Min-Max scaler artifacts not loaded or missing. Skipping Min-Max scaling.")

    # Apply Standard Scaling (Z-score)
    if standard_scaler and standard_cols:
        print("Applying Standard scaling to its group...")
        try:
            # Ensure columns exist
            cols_to_transform_std = [col for col in standard_cols if col in df.columns]
            if cols_to_std_scale:
                df[cols_to_transform_std] = standard_scaler.transform(df[cols_to_transform_std])
                print(f"Applied Standard scaling to {len(cols_to_transform_std)} columns.")
            else:
                print("No columns found for Standard scaling in the input data. Skipping.")
        except NotFittedError:
            print("Error: Standard scaler was not fitted. Artifacts might be missing or corrupted.")
        except Exception as e:
            print(f"Error applying Standard scaling: {e}")
    else:
        print("Standard scaler artifacts not loaded or missing. Skipping Standard scaling.")

    # Log Transformation is applied directly in the API using static functions.
    # No .pkl needed for this.
    print("Applying Log transformation to flow/rate features...")
    flow_rate_cols_to_transform = [col for col in [
        "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"
    ] if col in df.columns]
    if flow_rate_cols_to_transform:
        df[flow_rate_cols_to_transform] = np.log1p(df[flow_rate_cols_to_transform])
        print(f"Applied Log transformation to {len(flow_rate_cols_to_transform)} columns.")
    else:
        print("No columns found for Log transformation. Skipping.")


    # Final check to ensure data is clean after all steps
    try:
        assert df.notna().all().all(), "Error: NaNs still exist after preprocessing!"
        assert np.isfinite(df.to_numpy()).all(), "Error: Infinite values still exist after preprocessing!"
        print("✅ Preprocessing checks passed: No NaNs or Infs found.")
    except AssertionError as e:
        print(f"Data integrity check failed: {e}")


    print("--- Preprocessing Complete ---")
    # Convert the final preprocessed DataFrame to a list of JSON objects to send back
    return df.to_dict(orient='records')

# --- 5. Define the Preprocessing Endpoint ---
@app.post("/preprocess/dos")
async def preprocess_dos_data_api(data: List[dict]): # Expecting list of dicts, as n8n sends it
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Convert the list of dicts into a DataFrame before passing to the function
    input_df = pd.DataFrame(data)

    # Call the preprocessing function
    try:
        processed_data = preprocess_dos_data(input_df)
        return processed_data # Return the preprocessed data
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during preprocessing: {e}")

# --- 6. (Optional) A simple root endpoint to check if the API is running ---
@app.get("/")
def read_root():
    return {"status": "DoS Preprocessing API is running"}