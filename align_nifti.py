import numpy as np
import nibabel as nib

def load_nifti_file(file_path):
    """
    Load a NIfTI file and print its dimensions.
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    print(f"Loaded '{file_path}' with dimensions: {data.shape}")
    return data, nifti_img.affine

def align_dose_to_ct(dose_map, ct_map):
    """
    Align the dose map dimensions with the CT volume dimensions.
    """
    dose_z, dose_y, dose_x = dose_map.shape
    ct_z, ct_y, ct_x = ct_map.shape

    # Check if dimensions already match
    if dose_z == ct_z:
        print("Dose map already aligned with CT volume.")
        return dose_map

    # Padding or cropping along the Z-axis
    if dose_z < ct_z:
        padding = ct_z - dose_z
        pad_before = padding // 2
        pad_after = padding - pad_before
        print(f"Padding dose map: before={pad_before}, after={pad_after}")
        aligned_dose = np.pad(dose_map, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
    else:
        crop_before = (dose_z - ct_z) // 2
        crop_after = crop_before + ct_z
        print(f"Cropping dose map: from slice {crop_before} to {crop_after}")
        aligned_dose = dose_map[crop_before:crop_after, :, :]

    return aligned_dose

def save_nifti(data, affine, output_path):
    """
    Save a NIfTI file with the given data and affine matrix.
    """
    nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti_img, output_path)
    print(f"Saved aligned dose map to: {output_path}")

# File paths for the input NIfTI files
ct_file_path = 'ct_volume.nii.gz'               # Plain CT volume
ct_with_seg_file_path = '3d_ct_seg.nii.gz'  # CT with segmentation volume
dose_file_path = 'dose_map_3d.nii.gz'            # Dose map volume

# Load the NIfTI files
ct_data, ct_affine = load_nifti_file(ct_file_path)
ct_with_seg_data, ct_with_seg_affine = load_nifti_file(ct_with_seg_file_path)
dose_data, dose_affine = load_nifti_file(dose_file_path)

# Check if the CT with segmentation file is multi-channel
if len(ct_with_seg_data.shape) == 4:
    print("Detected multi-channel CT with segmentation file.")
    ct_with_seg_data = ct_with_seg_data[..., 0]  # Extract first channel as CT data

# Align the dose map with the CT with segmentation volume dimensions
aligned_dose_map = align_dose_to_ct(dose_data, ct_with_seg_data)

# Save the aligned dose map
output_dose_path = 'aligned_dose_map.nii.gz'
save_nifti(aligned_dose_map, ct_with_seg_affine, output_dose_path)

# Print final dimensions for verification
print(f"Final aligned dose map dimensions: {aligned_dose_map.shape}")
print(f"CT volume dimensions: {ct_data.shape}")
print(f"CT with segmentation volume dimensions: {ct_with_seg_data.shape}")
