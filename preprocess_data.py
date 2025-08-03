import os
import time
import numpy as np
import scipy.ndimage

from final_utils import (
    load_CT_data, load_RT_data,
    extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, save_volumes,
    parse_patient_metadata
)

# ------------------------
# Define patient DICOM paths
# ------------------------
patient_1_files = "DICOM files/Healthy Brain/1575_First_Session/1575_SRS_LT occipital_02062021"
patient_2_files = "DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"
patients_files = [patient_1_files, patient_2_files]

# ------------------------
# Define resampling config
# ------------------------
resample_type = "shape"  # or "spacing"
new_size = np.array((512, 512, 512))
new_spacing = np.array([1.0, 1.0, 1.0])

# ------------------------
# Define global output folder
# ------------------------
global_output_dir = "Outputs"
os.makedirs(global_output_dir, exist_ok=True)

# ------------------------
# Process each patient
# ------------------------
for i, patient_files_path in enumerate(patients_files, start=1):
    patient_name = os.path.basename(patient_files_path.strip("/").split("/")[-1])
    meta = parse_patient_metadata(patient_name)
    print(
        f"\n📁 Processing Patient Record\n"
        f"   • Patient ID       : {meta['Patient ID']}\n"
        f"   • Treatment Type   : {meta['Treatment']}\n"
        f"   • Targeted Region  : {meta['Region']}\n"
        f"   • Session Date     : {meta['Date']}"
    )
    print(f"🗂️ DICOM path: {patient_files_path}")

    start_time = time.time()

    # ------------------------
    # Load DICOM datasets
    # ------------------------
    print("📥 Loading CT and RT data...")
    CT_data = load_CT_data(patient_files_path)
    RD_data, RS_data, RP_data = load_RT_data(patient_files_path)

    print("💉 Extracting Dose data...")
    Dose_data = extract_dose_data(RD_data)

    print("🧠 Extracting ROI structures...")
    ROIs_data = extract_ROIs_data(RS_data, CT_data)

    # ------------------------
    # Compute zoom factors
    # ------------------------
    if resample_type == "shape":
        zoom_factors = new_size / CT_data['Volume'].shape
        print(f"📏 Resampling by shape to {new_size.tolist()}")
    else:
        zoom_factors = CT_data['Spacing'] / new_spacing
        print(f"📏 Resampling by spacing to {new_spacing.tolist()}")

    # ------------------------
    # Resample CT
    # ------------------------
    print("   🔄 Resampling CT volume...")
    CT_data['Volume'] = scipy.ndimage.zoom(CT_data['Volume'], zoom=zoom_factors, order=1)
    CT_data['Spacing'] = CT_data['Spacing'] / zoom_factors

    # ------------------------
    # Resample ROIs
    # ------------------------
    print("   🔄 Aligning ROI structures to CT volume...")
    for ROI_Number in ROIs_data['ROIs']:
        ROI_entry = ROIs_data['ROIs'][ROI_Number]
        ROI_entry['Volume'] = scipy.ndimage.zoom(ROI_entry['Volume'], zoom=zoom_factors, order=0)
    ROIs_data['Spacing'] = CT_data['Spacing']
    ROIs_data['Position'] = CT_data['Position']

    # ------------------------
    # Resample Dose
    # ------------------------
    print("   🔄 Aligning Dose volume to CT volume...")
    Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

    # ------------------------
    # Time Summary
    # ------------------------
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"✅ Finished processing {patient_name}.")
    print(f"⏱ Processing time: {int(mins)}m {int(secs)}s\n{'=' * 60}")

    # ------------------------
    # Save Outputs
    # ------------------------
    output_path = os.path.join(global_output_dir, patient_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"💾 Saving volumes to: {output_path}")
    save_volumes(CT_data, Dose_data, ROIs_data, output_path)
