from final_utils import *
from visualization import *

# files_path = "DICOM files/Healthy Brain/1575_First_Session/1575_SRS_LT occipital_02062021"
files_path = "DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"

# Load data.
CT_data = load_CT_data(files_path)
RD_data, RS_data, RP_data = load_RT_data(files_path)

# Extract Dose data.
Dose_data = extract_dose_data(RD_data)

# Extract ROIs data.
ROIs_data = extract_ROIs_data(RS_data, CT_data)

# Define new resolution (shape/spacing)
# Option 1: shape
new_size = np.array((512, 512, 512))
zoom_factors = new_size / CT_data['Volume'].shape

# Option 2: spacing
# new_spacing = np.array([1.0, 1.0, 1.0])  # zyx-spacing
# zoom_factors = CT_data['Spacing'] / new_spacing

# Resample CT volume
CT_data['Volume'] = scipy.ndimage.zoom(CT_data['Volume'], zoom=zoom_factors, order=1)
CT_data['Spacing'] = CT_data['Spacing'] / zoom_factors

# Resample ROIs volume
for ROI_Number in ROIs_data['ROIs']:
    ROI_entry = ROIs_data['ROIs'][ROI_Number]
    ROI_entry['Volume'] = scipy.ndimage.zoom(ROI_entry['Volume'], zoom=zoom_factors, order=0)
ROIs_data['Spacing'] = CT_data['Spacing']

# Match Dose (scale and offset) to CT (scale and offset)
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

# Plot Dose on CT
plot_combined_plot_2(CT_data, Dose_data, ROIs_data)
