import streamlit as st
import pandas as pd
import os
import shutil
import datetime
import subprocess
import gdown

# ======================== DEFINE PATHS ===========================
# GitHub repository base URL
GITHUB_REPO = "https://github.com/GOUFANG2021/Calcium-tartrate-model/raw/main"

# File download URLs from GitHub
DATA_TEMPLATE_URL = f"{GITHUB_REPO}/Wine%20Data.xlsx"

# Local directory for storing session files
LOCAL_SESSION_DIR = os.path.expanduser("~/Downloads/CalciumTartrateSessions")  # User's downloads folder
os.makedirs(LOCAL_SESSION_DIR, exist_ok=True)

# ======================== FUNCTION TO DOWNLOAD FILE FROM GITHUB ===========================
def download_from_github(url, output_path):
    """Download a file from GitHub repository."""
    try:
        gdown.download(url, output_path, quiet=False)
        return f"✅ Downloaded {os.path.basename(output_path)} successfully!"
    except Exception as e:
        return f"❌ Failed to download {os.path.basename(output_path)}: {e}"

# ======================== FUNCTION TO RUN MODEL ===========================
def run_external_script(data_path):
    """Run the model stored in the Streamlit environment using the user's uploaded data."""
    original_dir = os.getcwd()
    model_path = "CaTarModel.py"  # Assumes model is already present in the environment
    try:
        os.chdir(os.path.dirname(data_path))  # Set working directory to user's Downloads folder
        process = subprocess.Popen(
            ["python", model_path, data_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate()
        os.chdir(original_dir)  # Change back to original directory
        
        if error:
            return f"❌ Model execution failed: {error}"
        return output  # Capture and return printed output
    except Exception as e:
        os.chdir(original_dir)  # Ensure we revert back to original directory
        return f"❌ Error running model: {e}"

# ======================== STREAMLIT UI ===========================
st.set_page_config(layout="wide")  
st.title("🍷 Calcium Tartrate Precipitation Predictor")

# Create two columns
col1, col2 = st.columns([1, 1])

# Ensure session state variables exist
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {}
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

with col1:
    # STEP 1: DOWNLOAD TEMPLATE
    st.subheader("Step 1: Download Required File")
    
    # Define paths for user’s local download folder
    template_path = os.path.join(LOCAL_SESSION_DIR, "Wine Data.xlsx")

    # Download template file from GitHub
    download_result_template = download_from_github(DATA_TEMPLATE_URL, template_path)

    # Provide download button
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            st.download_button("📥 Download Wine Data", f, file_name="Wine Data.xlsx")

    # Show download status message
    st.write(download_result_template)

    # STEP 2: UPLOAD MODIFIED WINE DATA
    st.subheader("Step 2: Upload Your Modified Wine Data (Excel)")
    uploaded_file = st.file_uploader("📤 Browse files to upload Your Wine Data (Excel)", type=["xlsx"])

    if uploaded_file:
        st.session_state.uploaded_data = uploaded_file  
        st.success(f"✅ Uploaded: {uploaded_file.name}")

    # STEP 3: RUN MODEL
    st.subheader("Step 3: Run Model")
    if st.button("🚀 Run Model"):
        if st.session_state.uploaded_data is None:
            st.error("⚠️ Please upload a wine data file before running the model.")
        else:
            # Create a new session folder inside user's Downloads folder
            session_number = len(st.session_state.simulation_results) + 1
            session_id = f"session_{session_number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_path = os.path.join(LOCAL_SESSION_DIR, session_id)
            os.makedirs(session_path, exist_ok=True)

            # Save uploaded file in the session folder
            uploaded_file_path = os.path.join(session_path, "Wine Data.xlsx")
            with open(uploaded_file_path, "wb") as f:
                f.write(st.session_state.uploaded_data.getbuffer())

            # Run model from user's local folder
            results = run_external_script(uploaded_file_path)

            # Store results
            st.session_state.simulation_results[f"Simulation {session_number}"] = results  
            st.success(f"✅ Simulation {session_number} completed! Check results on the right.")

with col2:
    # DISPLAY RESULTS FOR ALL SESSIONS
    st.subheader("📊 Simulation Results")
    for session_name, results in st.session_state.simulation_results.items():
        with st.expander(session_name):
            st.text(results)  

    # ALWAYS SHOW INTERPRETATION TEXT
    st.subheader("📌 Interpretation")
    st.write("If the supersaturation ratio > 1, there is a high risk of calcium tartrate precipitation.")

    # ADDITIONAL WARNING MESSAGE
    st.warning(
        "⚠️ If the model does not cover your data, please reformat and try again."
    )
