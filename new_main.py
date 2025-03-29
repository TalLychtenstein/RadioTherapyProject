from new_utils import *
from visualization import *

# Load Radiotherapy data.
RT_files_path = "DICOM files/64147/CT radiation maps/64147_radiation maps_24082022/64147_radiation maps_24082022"
RD_data, RS_data, RP_data = load_RT_data(RT_files_path)

# Extract Dose and ROIs data.
Dose_data = extract_dose_data(RD_data)
ROIs_data = extract_ROIs_data(RS_data)

# Load CT data.
CT_files_path = "DICOM files/64147/CT anatomy/64147_AnatomicCT_24082022/64147_SRS_3 METS_24082022"
CT_data = load_CT_data(CT_files_path)

# Define new spacing (resolution)
new_spacing = [1.0, 1.0]  # y-spacing, a-spacing

# Resample dose volume
dose_volume = Dose_data['Volume']
dose_spacing = Dose_data['Spacing']
Dose_data['Volume'] = resample_array(dose_volume, dose_spacing, new_spacing)
Dose_data['Spacing'] = new_spacing

# Resample CT slices
for slice_number in CT_data['Slices'].keys():
    slice_image = CT_data['Slices'][slice_number]["Image"]
    CT_data['Slices'][slice_number]["Image"] = resample_array(slice_image, CT_data['Spacing'], new_spacing)
CT_data['Spacing'] = new_spacing

# Create CT volume
CT_data['Volume'] = create_CT_volume(CT_data)

# Create ROIs volume
ROIs_data['Volume'] = create_ROIs_volume(ROIs_data, CT_data)

# Match Dose scale to CT scale
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

# Plot Dose on CT
plot_Dose_on_CT(CT_data, Dose_data, ROIs_data)