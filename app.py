import streamlit as st
import pandas as pd
import os
import subprocess
import gdown
import datetime
import streamlit as st
import subprocess

st.title("🔍 Environment Check")

# Run pip freeze to see installed packages
installed_packages = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

# Display results
st.text_area("Installed Packages", installed_packages.stdout)


# ======================== DEFINE PATHS ===========================
# GitHub repository base URL
GITHUB_REPO = "https://github.com/GOUFANG2021/Calcium-tartrate-model/raw/main"

# File download URLs from GitHub
MODEL_PY_URL = f"{GITHUB_REPO}/CaTarModel.py"
DATA_TEMPLATE_URL = f"{GITHUB_REPO}/Wine%20Data.xlsx"

# ======================== FUNCTION TO DOWNLOAD FILE FROM GITHUB ===========================
def download_from_github(url, output_path):
    """Download a file from GitHub repository."""
    try:
        gdown.download(url, output_path, quiet=False)
        return f"✅ Downloaded {os.path.basename(output_path)} successfully!"
    except Exception as e:
        return f"❌ Failed to download {os.path.basename(output_path)}: {e}"

# ======================== FUNCTION TO RUN MODEL DIRECTLY FROM GITHUB ===========================
def run_model_from_github(model_url, data_path, simulation_id):
    """Download the model from GitHub and execute it with the uploaded data file."""
    model_path = "CaTarModel.py"  # Temporary local path for execution

    # Download model file from GitHub
    download_result = download_from_github(model_url, model_path)

    # Run the model
    try:
        process = subprocess.Popen(
            ["python", model_path, data_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate()
        
        if error:
            return f"❌ Model execution failed: {error}"
        
        return f"✅ Simulation {simulation_id} completed successfully!\n\n{output}"
    except Exception as e:
        return f"❌ Error running model: {e}"

# ======================== STREAMLIT UI ===========================
st.set_page_config(layout="wide")  
st.title("🍷 Calcium Tartrate Precipitation Predictor")

# Create two columns
col1, col2 = st.columns([1, 1])

# Ensure session state variables exist
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {}  # Keep previous results
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

with col1:
    # STEP 1: DOWNLOAD TEMPLATE
    st.subheader("Step 1: Download Required File")
    
    template_path = "Wine Data.xlsx"  # Temporary path for execution
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
            # Generate a unique simulation ID using timestamp
            simulation_id = f"Simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save uploaded file temporarily
            uploaded_file_path = "Wine Data.xlsx"
            with open(uploaded_file_path, "wb") as f:
                f.write(st.session_state.uploaded_data.getbuffer())

            # Run model from GitHub
            results = run_model_from_github(MODEL_PY_URL, uploaded_file_path, simulation_id)

            # Store results with unique identifier
            st.session_state.simulation_results[simulation_id] = results  
            st.success(f"✅ {simulation_id} completed! Check results on the right.")

with col2:
    # DISPLAY RESULTS FOR ALL SIMULATIONS
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
