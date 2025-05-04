from new_utils import (
    load_RT_data,
    load_CT_data,
    extract_dose_data,
    extract_ROIs_data,
    preprocess_Dose_to_CT,
    create_ROIs_volume,
    extract_group1_metadata,
    extract_group2_metrics
)

# --- STEP 1: Load Files ---
RT_files_path = r"C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\64147\CT radiation maps\64147_radiation maps_24082022"
CT_files_path = r"C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\64147\CT anatomy\64147_SRS_3 METS_24082022"

RD_data, RS_data, RP_data = load_RT_data(RT_files_path)
CT_data = load_CT_data(CT_files_path)

# --- STEP 2: Extract and Align ---
Dose_data = extract_dose_data(RD_data)
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

ROIs_data = extract_ROIs_data(RS_data)
ROIs_data["Volume"] = create_ROIs_volume(ROIs_data, CT_data)

# --- STEP 3: Group 1 Metadata ---
print("\n=== Group 1: Plan Metadata ===")
group1 = extract_group1_metadata(RP_data, RD_data)
for k, v in group1.items():
    print(f"{k}: {v}")

print("Available ROIs in RTSTRUCT:")
for roi_id, roi_info in ROIs_data["ROIs"].items():
    print(f"  {roi_id}: {roi_info['Name']}")

# --- STEP 4: Group 2 Metrics (PTV/GTV) ---
print("\n=== Group 2: ROI-Based Metrics ===")
prescribed_dose = group1.get("Prescription Dose [Gy]", 20)  # Fallback if None
group2 = extract_group2_metrics(
    Dose_data,
    ROIs_data,

    prescribed_dose=group1["Prescription Dose [Gy]"]
)


for roi_name, metrics in group2.items():
    print(f"\n--- {roi_name} ---")
    for k, v in metrics.items():
        if k != "DVH":
            print(f"{k}: {v}")
import os

from new_utils import save_dose_histograms
output_folder = "outputs"

save_dose_histograms(group2, output_dir=os.path.join(output_folder, "dose_histograms"))


'''import os

from new_utils import (
    save_group1_to_excel, save_group2_to_excel, save_all_dvh_plots
)

# Save locations
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

save_group1_to_excel(group1, os.path.join(output_folder, "group1_metrics.xlsx"))
save_group2_to_excel(group2, os.path.join(output_folder, "group2_metrics.xlsx"))
save_all_dvh_plots(group2, os.path.join(output_folder, "dvh_figures"))

print("✅ Metrics and DVH plots saved to 'outputs' folder.")'''

''' 
from new_utils import extract_group3_metrics

# Replace with the actual ROI name in your DICOM structure
healthy_brain_roi_name = "Healthy Brain"

group3 = extract_group3_metrics(Dose_data, ROIs_data, healthy_brain_label_name=healthy_brain_roi_name)

print("\n=== Group 3: Healthy Brain Vx Metrics ===")
for k, v in group3.items():
    print(f"{k}: {v:.2f} cc")
    
'''