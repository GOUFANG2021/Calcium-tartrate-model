import streamlit as st
import pandas as pd
import os
import shutil
import datetime
import subprocess
import gdown
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ======================== GOOGLE DRIVE AUTHENTICATION ===========================
def authenticate_drive():
    """Authenticate and return Google Drive instance."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Open browser for authentication
    return GoogleDrive(gauth)

drive = authenticate_drive()

# Google Drive Folder ID for storing session files
GOOGLE_DRIVE_FOLDER_ID = "1YDVVq0Ac3k43Ikq02uV9N_F4FMj73aGM"

# ======================== DEFINE PATHS ===========================
# GitHub repository base URL
GITHUB_REPO = "https://github.com/GOUFANG2021/Calcium-tartrate-model/raw/main"

# File download URLs from GitHub
MODEL_PY_URL = f"{GITHUB_REPO}/CaTarModel.py"
DATA_TEMPLATE_URL = f"{GITHUB_REPO}/Wine%20Data.xlsx"

# Local directory to temporarily store files before uploading to Google Drive
LOCAL_SESSION_DIR = os.path.expanduser("~/sessions")
os.makedirs(LOCAL_SESSION_DIR, exist_ok=True)

# ======================== FUNCTION TO DOWNLOAD FILE FROM GITHUB ===========================
def download_from_github(url, output_path):
    """Download a file from GitHub repository."""
    try:
        gdown.download(url, output_path, quiet=False)
        return f"✅ Downloaded {os.path.basename(output_path)} successfully!"
    except Exception as e:
        return f"❌ Failed to download {os.path.basename(output_path)}: {e}"

# ======================== FUNCTION TO UPLOAD FILE TO GOOGLE DRIVE ===========================
def upload_to_google_drive(file_path, file_name):
    """Upload a file to Google Drive inside the specified folder."""
    try:
        file_drive = drive.CreateFile({
            "title": file_name,
            "parents": [{"id": GOOGLE_DRIVE_FOLDER_ID}]
        })
        file_drive.SetContentFile(file_path)
        file_drive.Upload()
        return f"✅ Uploaded {file_name} to Google Drive successfully!"
    except Exception as e:
        return f"❌ Google Drive upload failed: {e}"

# ======================== FUNCTION TO RUN MODEL FROM GOOGLE DRIVE ===========================
def run_external_script(model_drive_path, data_drive_path):
    """Run the model stored in Google Drive."""
    try:
        process = subprocess.Popen(
            ["python", model_drive_path, data_drive_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate()
        if error:
            return f"❌ Model execution failed: {error}"
        return output
    except Exception as e:
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
    st.subheader("Step 1: Download Wine Data")
    template_path = os.path.join(LOCAL_SESSION_DIR, "Wine Data.xlsx")
    download_result = download_from_github(DATA_TEMPLATE_URL, template_path)
    
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            st.download_button("📥 Download Wine Data", f, file_name="Wine Data.xlsx")

    st.write(download_result)

    # STEP 2: UPLOAD MODIFIED WINE DATA
    st.subheader("Step 2: Upload Your Wine Data (Excel)")
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
            # Create a new session folder
            session_number = len(st.session_state.simulation_results) + 1
            session_id = f"session_{session_number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_path = os.path.join(LOCAL_SESSION_DIR, session_id)
            os.makedirs(session_path, exist_ok=True)

            # Save uploaded file locally before uploading to Google Drive
            uploaded_file_path = os.path.join(session_path, "Wine Data.xlsx")
            with open(uploaded_file_path, "wb") as f:
                f.write(st.session_state.uploaded_data.getbuffer())

            # Download model script from GitHub
            model_path = os.path.join(session_path, f"CaTarModel_{session_number}.py")
            model_download_result = download_from_github(MODEL_PY_URL, model_path)

            # Upload session data to Google Drive
            upload_data_result = upload_to_google_drive(uploaded_file_path, f"{session_id}_Wine Data.xlsx")
            upload_model_result = upload_to_google_drive(model_path, f"{session_id}_CaTarModel.py")

            # Run model from Google Drive
            results = run_external_script(model_path, uploaded_file_path)

            # Store results
            st.session_state.simulation_results[f"Simulation {session_number}"] = results  
            st.success(f"✅ Simulation {session_number} completed! Check results on the right.")

            # Display messages
            st.write(model_download_result)
            st.write(upload_data_result)
            st.write(upload_model_result)

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
