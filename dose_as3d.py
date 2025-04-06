import os
import pydicom
import numpy as np
import scipy.ndimage
import nibabel as nib
from skimage.draw import polygon2mask
import cv2

def find_file_with_prefix(folder_path, prefix):
    """Finds a file in the given folder with the specified prefix."""
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            return os.path.join(folder_path, file_name)
    return None

def read_dicom_rd_file(file_path):
    """Reads and loads a DICOM Radiation Dose (RD) file."""
    rd = pydicom.dcmread(file_path)
    if rd.Modality == 'RTDOSE':
        return rd
    else:
        raise ValueError("The provided file is not a Radiation Dose (RD) DICOM file.")

def load_CT_data(ct_files_path):
    """
    Loads CT image data from DICOM files in a specified directory.

    Parameters:
    ct_files_path: str
        The directory path containing CT DICOM files.

    Returns:
    dict
        A dictionary containing the CT slices and relevant metadata.
    """
    CT_data = {'Slices': {}, 'Position': None, 'Spacing': None}
    for file_name in os.listdir(ct_files_path):
        if file_name.startswith('CT'):
            CT_slice_data = pydicom.dcmread(os.path.join(ct_files_path, file_name))
            slice_number = CT_slice_data.SOPInstanceUID
            slice_position = CT_slice_data.ImagePositionPatient
            pixel_spacing = CT_slice_data.PixelSpacing
            pixel_array = CT_slice_data.pixel_array
            CT_data['Slices'][slice_number] = {"Position": slice_position,
                                               "Spacing": pixel_spacing,
                                               "Image": pixel_array}

            # Set total CT position and spacing to be the same as the first slice
            if CT_data['Position'] is None:
                CT_data['Position'] = slice_position
                CT_data['Spacing'] = pixel_spacing

    # Create a 3D volume from the CT slices
    CT_data['Volume'] = create_CT_volume(CT_data)
    return CT_data

def create_CT_volume(CT_data):
    """Creates a 3D volume from the individual CT slices."""
    CT_slices_data = [(slice_number, CT_data['Slices'][slice_number]["Position"][2]) for slice_number in
                      CT_data['Slices']]
    CT_slices_data.sort(key=lambda s: s[1])

    # Extract CT volume shape
    z_len = len(CT_slices_data)
    y_len = list(CT_data['Slices'].values())[0]["Image"].shape[0]
    x_len = list(CT_data['Slices'].values())[0]["Image"].shape[1]

    # Create empty 3D CT volume
    CT_volume = np.zeros((z_len, y_len, x_len), dtype=np.float32)
    for i, (slice_number, _) in enumerate(CT_slices_data):
        CT_data['Slices'][slice_number]['Z Index'] = i
        CT_volume[i] = CT_data['Slices'][slice_number]['Image']

    return CT_volume

def extract_dose_data(RD_data):
    """
    Extracts relevant dose-related information from the RD (Radiotherapy Dose) dataset.

    Parameters:
    RD_data: pydicom Dataset
        The DICOM dataset containing radiotherapy dose information.

    Returns:
    dict
        A dictionary containing:
        - "Scaling Factor": The dose grid scaling factor.
        - "Position": The (x, y, z) position of the dose grid in patient coordinates.
        - "Spacing": The spacing between pixels in the dose grid.
        - "Image": The actual pixel array representing the dose distribution.
    """
    dose_data = {"Position": RD_data.ImagePositionPatient,
                 "Spacing": RD_data.PixelSpacing,
                 "Volume": RD_data.pixel_array * RD_data.DoseGridScaling}
    return dose_data

def resample_array(array, old_spacing, new_spacing, target_shape, order=1):
    """
    Resamples a 2D/3D NumPy array to a new spacing and shape.

    Parameters:
        array: 2D/3D NumPy array
        old_spacing: list or array of [y, x] spacing in mm
        new_spacing: desired spacing [y, x] in mm
        target_shape: tuple (z, y, x) representing the desired output shape
        order: interpolation order (1 = linear, 0 = nearest for masks)

    Returns:
        resampled_volume: volume resampled to new spacing and size
    """
    # Calculate the zoom factors based on spacing and target shape
    zoom_factors = [
        target_shape[0] / array.shape[0],  # Z-axis scaling
        target_shape[1] / array.shape[1],  # Y-axis scaling
        target_shape[2] / array.shape[2]   # X-axis scaling
    ]

    # Resample the entire 3D volume at once to match the target shape
    resampled = scipy.ndimage.zoom(array, zoom_factors, order=order)
    return resampled


def create_3d_dose_map(RD_data, CT_data):
    """
    Create a 3D dose map from the Radiation Dose (RD) data and align it with the CT slices.

    Parameters:
    RD_data: dict
        A dictionary containing the dose metadata.
    CT_data: dict
        A dictionary containing the CT metadata (volume, spacing, etc.).

    Returns:
    numpy.ndarray
        A 3D array of dose data aligned with the CT slices.
    """
    # Extract the target shape from the CT volume
    target_shape = CT_data["Volume"].shape

    # Rescale the dose data to match the CT resolution and shape
    rescaled_dose_data = resample_array(
        RD_data["Volume"],
        RD_data["Spacing"],
        CT_data["Spacing"],
        target_shape,
        order=1
    )

    # Check the shape after resampling
    if rescaled_dose_data.shape != target_shape:
        raise ValueError(f"Rescaled dose data shape {rescaled_dose_data.shape} does not match CT volume shape {target_shape}.")

    return rescaled_dose_data


def create_affine(ct_slices):
    """
    Create an affine matrix for NIfTI from CT DICOM slice metadata.

    Parameters:
    ct_slices: list of dictionaries containing 'Position' and 'Spacing'

    Returns:
    np.ndarray: A 4x4 affine transformation matrix
    """
    # Use the first slice's spacing and position
    first_slice = ct_slices[0]
    spacing = list(map(float, first_slice['Spacing']))  # [row_spacing, col_spacing]

    # Calculate Z spacing between consecutive slices
    if len(ct_slices) > 1:
        spacing_z = abs(ct_slices[1]['Position'][2] - first_slice['Position'][2])
    else:
        spacing_z = 1.0  # Fallback if only one slice

    spacing.append(spacing_z)

    origin = first_slice['Position']

    # Create the affine transformation matrix
    affine = np.eye(4)
    affine[0, 0] = spacing[1]  # X spacing (columns)
    affine[1, 1] = spacing[0]  # Y spacing (rows)
    affine[2, 2] = spacing[2]  # Z spacing (slices)
    affine[:3, 3] = origin  # Translation (origin)
    return affine


def save_dose_map_to_nifti(dose_map, ct_slices, output_path):
    """
    Save the 3D dose map to a NIfTI file.
    """
    # Create the affine matrix using the CT slices
    affine = create_affine(list(ct_slices))

    # Create the NIfTI image and save
    nifti_img = nib.Nifti1Image(dose_map.astype(np.float32), affine)
    nib.save(nifti_img, output_path)
    print(f"NIfTI dose map saved to: {output_path}")

# Example usage for loading CT and Radiation Dose Data

# Load CT Data
ct_files_path = r'C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\64147\CT anatomy\64147_SRS_3 METS_24082022'  # Directory containing CT DICOM files
CT_data = load_CT_data(ct_files_path)

# Load Radiation Dose Data
rd_file_path = find_file_with_prefix(r'C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\64147\CT radiation maps\64147_radiation maps_24082022', 'RD')  # Directory containing RD DICOM file

if rd_file_path is None:
    raise FileNotFoundError("RTDOSE file not found in the directory.")

# Read the Radiation Dose (RD) DICOM file
RD_data = read_dicom_rd_file(rd_file_path)

# Extract dose-related information from the RD file
Dose_data = extract_dose_data(RD_data)

# Create the 3D dose map by aligning the dose data with the CT slices
dose_map = create_3d_dose_map(Dose_data, CT_data)
# Save the 3D dose map as a NIfTI file
output_dose_path = 'dose_map_3d.nii.gz'
save_dose_map_to_nifti(dose_map, list(CT_data['Slices'].values()), output_dose_path)

