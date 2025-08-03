from new_utils_3 import *
import os

# Load Files
RT_files_path = r"DICOM files\Healthy Brain\1575"
CT_files_path = r"DICOM files\Healthy Brain\1575"

RD_data, RS_data, RP_data = load_RT_data(RT_files_path)
CT_data = load_CT_data(CT_files_path)

# Step 1: Extract data and Align
Dose_data = extract_dose_data(RD_data)
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

ROIs_data = extract_ROIs_data(RS_data)
ROIs_data["Volume"] = create_ROIs_volume(ROIs_data, CT_data)

# Step 2: Extract Group 1 Metadata
print("\nGroup 1: Plan Metadata")
group1 = extract_group1_metadata(RP_data, RD_data)
for k, v in group1.items():
    print(f"{k}: {v}")

print("Available ROIs in RTSTRUCT:")
for roi_id, roi_info in ROIs_data["ROIs"].items():
    print(f"{roi_id}: {roi_info['Name']}")

# Step 3: Calculate Group 2 Metrics
print("\nGroup 2: ROI-Based Metrics")
prescribed_dose = group1.get("Prescription Dose [Gy]", 20)  # Fallback if None
group2 = extract_group2_metrics(Dose_data, ROIs_data, prescribed_dose)

for roi_name, metrics in group2.items():
    print(f"\n{roi_name}")
    for k, v in metrics.items():
        if k != "DVH":
            print(f"{k}: {v}")

# Step 4: Calculate Group 3 Metrics
group3 = extract_group3_metrics(Dose_data, ROIs_data, healthy_brain_label_name="Healthy Brain")

print("\nGroup 3: Healthy Brain Vx Metrics")
for k, v in group3.items():
    print(f"{k}: {v} mm3")

# Save locations
output_folder = "outputs"
save_group1_to_excel(group1, os.path.join(output_folder, "group1_metrics.xlsx"))
save_group2_to_excel(group2, os.path.join(output_folder, "group2_metrics.xlsx"))
save_group3_to_excel(group2, os.path.join(output_folder, "group2_metrics.xlsx"))
save_dose_histograms(group2, output_dir=os.path.join(output_folder, "dose_histograms"))
