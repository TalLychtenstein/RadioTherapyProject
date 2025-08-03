from new_utils_2 import *
from visualization import *

# Load Radiotherapy data.
RT_files_path = "../DICOM files/64147/CT radiation maps/64147_radiation maps_24082022"
# RT_files_path = "DICOM files/7227/CT radiation maps/7227_Radiation plans_27102021/7227_Radiation plans_27102021"
# RT_files_path = "DICOM files/52724/CT radiation maps/52724_radiation plans_30062020/52724_radiation plans_30062020"
# RT_files_path = "DICOM files/Healthy Brain/1575_First_Session/1575_SRS_LT occipital_02062021"
# RT_files_path = "DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"
RD_data, RS_data, RP_data = load_RT_data(RT_files_path)

# Extract Dose and ROIs data.
Dose_data = extract_dose_data(RD_data)
ROIs_data = extract_ROIs_data(RS_data)

# Load CT data.
CT_files_path = "../DICOM files/64147/CT anatomy/64147_SRS_3 METS_24082022"
# CT_files_path = "DICOM files/7227/CT anatomy/7227_anatomyCT_27102021/7227_FSR_RT Hipocampus_27102021"
# CT_files_path = "DICOM files/52724/CT anatomy/52724_AnatomyCT_30062020/52724_SRS_2 METS_30062020"
# CT_files_path = "DICOM files/Healthy Brain/1575_First_Session/1575_SRS_LT occipital_02062021"
# CT_files_path = "DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"
CT_data = load_CT_data(CT_files_path)

# Create ROIs volume according to CT volume
ROIs_data['Volume'] = create_ROIs_volume(ROIs_data, CT_data)

# Define new resolution (spacing/shape)
CT_volume = CT_data['Volume']
CT_shape = CT_data['Volume'].shape
CT_spacing = CT_data['Spacing']

# Option 1
new_size = np.array((512, 512, 512))
zoom_factors = new_size / CT_shape

# Option 2
# new_spacing = np.array([1.0, 1.0, 1.0])  # zyx-spacing
# zoom_factors = CT_spacing / new_spacing

# Resample ROIs and CT volumes
ROIs_data['Volume'] = resample_array(ROIs_data['Volume'], zoom_factors, order=0)
CT_data['Volume'] = resample_array(CT_volume, zoom_factors, order=1)
CT_data['Spacing'] = CT_spacing / zoom_factors

# Match Dose (scale and offs    et) to CT (scale and offset)
Dose_data = preprocess_Dose_to_CT(Dose_data, CT_data)

# plot_CT_view(CT_data['Volume'], view='axial')
# plot_Dose_view(Dose_data['Volume'], view='axial')
# plot_ROI_contours(ROIs_data)

# Plot Dose on CT
plot_Dose_on_CT(CT_data, Dose_data, ROIs_data)

# Save CT volume
save_volume_as_nifti(CT_data["Volume"], CT_data["Spacing"], "CT_volume.nii.gz", affine_origin=CT_data["Position"])

# Save Dose volume
save_volume_as_nifti(Dose_data["Volume"], Dose_data["Spacing"], "Dose_volume.nii.gz", affine_origin=Dose_data["Position"])

# Save ROIs volume
save_volume_as_nifti(ROIs_data["Volume"], CT_data["Spacing"], "ROIs_volume.nii.gz", affine_origin=CT_data["Position"])



