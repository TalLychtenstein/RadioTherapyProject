from utils import *
from visualization import *

# Load Radiotherapy data.
RT_files_path = "DICOM files/64147/CT radiation maps/64147_radiation maps_24082022/64147_radiation maps_24082022"
RD_data, RS_data, RP_data = load_RT_data(RT_files_path)

# Extract Dose and ROIs data.
Dose_data = extract_dose_data(RD_data)
ROIs_contours_data = extract_ROIs_contours_data(RS_data)

# Load CT data.
CT_files_path = "DICOM files/64147/CT anatomy/64147_AnatomicCT_24082022/64147_SRS_3 METS_24082022"
CT_data = load_CT_data(CT_files_path)

# Select CT slice number.
# CT_slice = "1.2.246.352.221.4756343653739451167.9950165951424806580"
CT_slice = '1.2.246.352.221.5368342218023505269.1309773173056034724'

# Extract relevant ROIs for given CT slice
ROIs_data = extract_ROI_data(ROIs_contours_data, CT_slice)

# Find relative slice index
slice_index = find_slice_index(CT_data, CT_slice)

# Preprocess ROIs for CT scale
ROIs_data = preprocess_ROIs_to_CT(ROIs_data, CT_data[CT_slice])

# Preprocess Dose map for CT scale
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data[CT_slice])

# Plot data
plot_Dose(Dose_data["Scaled Image"][slice_index])
plot_ROIs(ROIs_data)
plot_CT_slice(CT_data[CT_slice]["Image"])
plot_full_image(ROIs_data, CT_data[CT_slice]["Image"], Dose_data["Scaled Image"][slice_index])

