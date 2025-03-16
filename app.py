import streamlit as st
import pandas as pd
import os
import shutil
import datetime
import subprocess

# ======================== DEFINE PATHS ===========================
BASE_DIR = r"C:\Users\Fang.Gou\caltarmodel"
MODEL_PY_PATH = os.path.join(BASE_DIR, "CaTarModel.py")
DATA_TEMPLATE_PATH = os.path.join(BASE_DIR, "Wine Data.xlsx")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

# Ensure session directory exists
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)

# ======================== FUNCTION TO RUN MODEL ===========================
def run_external_script(model_path, data_path, session_path):
    """Run the session-specific CaTarModel.py with the uploaded file."""
    original_dir = os.getcwd()
    try:
        os.chdir(session_path)  # Change working directory to session folder
        process = subprocess.Popen(
            ["python", model_path, data_path],  # Run the specific downloaded model file
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate()
        os.chdir(original_dir)  # Change back to the main directory
        
        if error:
            return f"❌ Model execution failed: {error}"
        return output  # Capture and return printed output
    except Exception as e:
        os.chdir(original_dir)  # Ensure we revert back to original directory
        return f"❌ Error changing directory or running model: {e}"

# ======================== STREAMLIT UI ===========================
st.set_page_config(layout="wide")  # Reduce page border for better layout
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
    st.download_button(
        "📥 Download Wine Data",
        open(DATA_TEMPLATE_PATH, "rb"),
        file_name="Wine Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.write("Please change the data with your wine data in the first 'Input Sheet' and keep the other data as it is. Keep the file name as 'Wine Data'.")

    # STEP 2: UPLOAD MODIFIED WINE DATA
    st.subheader("Step 2: Please upload Your Wine Data (Excel)")
    uploaded_file = st.file_uploader("📤 Click Browse files to upload Your Wine Data (Excel)", type=["xlsx"], key="file_uploader")
    
    if uploaded_file:
        st.session_state.uploaded_data = uploaded_file  # Ensure uploaded file is stored
        st.success(f"✅ File Uploaded Successfully: {uploaded_file.name}.    If you want to run the model for another wine sample, please clear data before uploading a new wine data file. ")

    # STEP 3: RUN MODEL
    st.subheader("Step 3: Run Model")
    if st.button("🚀 Run Model"):
        if st.session_state.uploaded_data is None:
            st.error("⚠️ Please upload a wine data file before running the model.")
        else:
            # Create a new session folder
            session_number = len(st.session_state.simulation_results) + 1
            session_name = f"Simulation {session_number}"
            session_id = f"session_{session_number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_path = os.path.join(SESSION_DIR, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Save uploaded file in session folder
            uploaded_file_path = os.path.join(session_path, "Wine Data.xlsx")
            with open(uploaded_file_path, "wb") as f:
                f.write(st.session_state.uploaded_data.getbuffer())
            
            # Copy CaTarModel.py to session folder with session-specific name
            model_script_path = os.path.join(session_path, f"CaTarModel_{session_number}.py")
            shutil.copy(MODEL_PY_PATH, model_script_path)
            
            # Run the downloaded model script with the uploaded file as input
            try:
                results = run_external_script(model_script_path, uploaded_file_path, session_path)
                st.session_state.simulation_results[session_name] = results  # Store results with a descriptive name
                st.success(f"✅ {session_name} completed! Please review your simulation results on the right ")
            except Exception as e:
                st.error(f"❌ Model execution failed: {e}")

with col2:
    # DISPLAY RESULTS FOR ALL SESSIONS
    st.subheader("📊 Simulation Results")
    for session_name, results in st.session_state.simulation_results.items():
        with st.expander(session_name):
            st.text(results)  # Print raw text output from the model
    
    # ALWAYS SHOW INTERPRETATION TEXT
    st.subheader("📌 Interpretation")
    st.write("The supersaturation ratio is a key indicator of calcium tartrate precipitation risk in wine. If the ratio is greater than 1, there is a high risk of precipitation. If the ratio is 1 or less, the risk is low.")

 # ADDITIONAL WARNING MESSAGE
    st.warning(
        "⚠️ The model may not achieve coverage if the input data falls outside the simulation range. "
        "If this occurs, please stop the simulation, delete the uploaded Excel file, and upload a new data file. "
        "Remember to maintain the same format and file name for the input file."
    )

