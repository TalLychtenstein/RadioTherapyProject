import pydicom
import scipy.ndimage
from scipy.spatial import ConvexHull, Delaunay


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


def read_dicom_rs_file(file_path):
    """Reads and loads a DICOM Radiation Therapy (RT) Structure Set (RS) file."""
    rs = pydicom.dcmread(file_path)
    if rs.Modality == 'RTSTRUCT':
        return rs
    else:
        raise ValueError("The provided file is not an RT Structure Set (RTSTRUCT) DICOM file.")


def read_dicom_rp_file(file_path):
    """Reads and loads a DICOM Radiation Therapy Plan (RP) file."""
    rp = pydicom.dcmread(file_path)
    if rp.Modality == 'RTPLAN':
        return rp
    else:
        raise ValueError("The provided file is not an RP (RTPLAN) DICOM file.")


def create_CT_volume(CT_data):
    # 1. Sort CT slices
    CT_slices_data = [(slice_number, CT_data['Slices'][slice_number]["Position"][0]) for slice_number in
                      CT_data['Slices']]
    CT_slices_data.sort(key=lambda s: s[1])

    # 2. Extract CT volume shape
    z_len = len(CT_slices_data)
    y_len = list(CT_data['Slices'].values())[0]["Image"].shape[0]
    x_len = list(CT_data['Slices'].values())[0]["Image"].shape[1]

    # Insert CT slices
    CT_volume = np.zeros((z_len, y_len, x_len), dtype=np.float32)
    for i, (slice_number, _) in enumerate(CT_slices_data):
        CT_data['Slices'][slice_number]['Z Index'] = i
        CT_volume[i] = CT_data['Slices'][slice_number]['Image']

    return CT_volume


def load_CT_data(files_path):
    """
    Loads CT image data from DICOM files in a specified directory.

    Parameters:
    files_path: str
        The directory path containing CT DICOM files.

    Returns:
    dict
        A dictionary where each key is a CT slice identifier (SOPInstanceUID), and values are dictionaries containing:
        - "Position": The (x, y, z) position of the slice.
        - "Spacing": The pixel spacing values.
        - "Image": The pixel array representing the CT image.
    """

    CT_data = {}
    CT_data['Slices'] = {}

    # Extract z positions
    z_positions = []

    min_z = np.inf
    for file_name in os.listdir(files_path):
        if file_name.startswith('CT'):
            CT_slice_data = pydicom.dcmread(os.path.join(files_path, file_name))

            slice_number = CT_slice_data.SOPInstanceUID
            pos_x, pos_y, pos_z = CT_slice_data.ImagePositionPatient
            slice_position = (pos_z, pos_y, pos_x)
            spacing_x, spacing_y = CT_slice_data.PixelSpacing
            pixel_spacing = (spacing_y, spacing_x)
            pixel_array = CT_slice_data.pixel_array

            CT_data['Slices'][slice_number] = {"Position": slice_position,
                                               "Spacing": pixel_spacing,
                                               "Image": pixel_array}
            z_positions.append(float(slice_position[0]))

            # Set total CT position and spacing to be the same as the slice with lowest Z index
            if min_z > slice_position[0]:
                min_z = slice_position[0]
                CT_data['Position'] = slice_position
                CT_data['Spacing'] = pixel_spacing

    # Calculate spacing as average distance between z-values
    z_diffs = np.diff(sorted(z_positions))
    z_spacing = np.abs(np.mean(z_diffs))
    CT_data['Spacing'] = [z_spacing] + list(CT_data['Spacing'])

    # Create volume
    CT_data['Volume'] = create_CT_volume(CT_data)
    return CT_data


def load_RT_data(files_path):
    """
    Load DICOM Radiation Therapy (RT) data from a given directory.

    This function loads three types of DICOM RT files:
    - RD (Radiation Dose) file: Contains radiation dose distribution data.
    - RS (Radiation Structure) file: Contains structure set data, defining target volumes and organs-at-risk.
    - RP (Radiation Plan) file: Contains treatment plan information.

    Parameters:
    ----------
    files_path : str
        The path to the directory containing the DICOM RT files.

    Returns:
    -------
    tuple
        A tuple containing three elements:
        - RD_data: Radiation dose data extracted from the RD file.
        - RS_data: Radiation structure data extracted from the RS file.
        - RP_data: Radiation plan data extracted from the RP file.

    Raises:
    ------
    FileNotFoundError:
        If any of the required files (RD, RS, RP) are not found in the directory.
    """

    # Load RD data
    RD_file_name = find_file_with_prefix(files_path, 'RD')
    if RD_file_name is None:
        raise FileNotFoundError("RD (Radiation Dose) file not found in the directory.")
    RD_data = read_dicom_rd_file(RD_file_name)  # load radiation dose information

    # Load RS data
    RS_file_name = find_file_with_prefix(files_path, 'RS')
    if RS_file_name is None:
        raise FileNotFoundError("RS (Radiation Structure) file not found in the directory.")
    RS_data = read_dicom_rs_file(RS_file_name)  # load radiation structure information

    # Load RP data
    RP_file_name = find_file_with_prefix(files_path, 'RP')
    if RP_file_name is None:
        raise FileNotFoundError("RP (Radiation Plan) file not found in the directory.")
    RP_data = read_dicom_rp_file(RP_file_name)  # load radiation plan information

    return RD_data, RS_data, RP_data


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
        - "Spacing": The spacing between voxels in the dose grid (z, y, x) in mm.
        - "Volume": The 3D dose array in Gy units.
    """
    pos_x, pos_y, pos_z = RD_data.ImagePositionPatient
    spacing_x, spacing_y = RD_data.PixelSpacing

    offset_vector = RD_data.GridFrameOffsetVector
    spacing_z = offset_vector[1] - offset_vector[0] if len(offset_vector) > 1 else 0.0

    dose_volume = RD_data.pixel_array * RD_data.DoseGridScaling

    return {
        "Position": (float(pos_z), float(pos_y), float(pos_x)),
        "Spacing": (float(spacing_z), float(spacing_y), float(spacing_x)),  # matches (z, y, x) shape
        "Volume": dose_volume
    }


def voxelize_convex_hull(points_voxel, volume_shape):
    hull = ConvexHull(points_voxel)
    delaunay = Delaunay(points_voxel[hull.vertices])
    zz, yy, xx = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing='ij'
    )
    test_points = np.vstack([zz.ravel(), yy.ravel(), xx.ravel()]).T
    mask = delaunay.find_simplex(test_points) >= 0
    filled = np.zeros(volume_shape, dtype=bool)
    filled[zz.ravel()[mask], yy.ravel()[mask], xx.ravel()[mask]] = True
    return filled


def extract_ROIs_data(RS_data, CT_data):
    """
    Extracts ROIs and fills their 3D volume using convex hull of contour points.

    Args:
        RS_data (pydicom Dataset): RT Structure Set (RS) DICOM dataset.
        CT_data (dict): Dictionary containing CT volume and geometry.

    Returns:
        dict: ROI data with names, colors, and filled 3D binary volumes.
    """

    # Extract sequences
    ROIs_raw_data = RS_data.StructureSetROISequence
    contours_data = RS_data.ROIContourSequence

    # CT grid reference
    z_origin, y_origin, x_origin = CT_data["Position"]
    z_spacing, y_spacing, x_spacing = CT_data["Spacing"]
    volume_shape = CT_data['Volume'].shape

    # Initialize output
    ROIs_data = {'ROIs': {}}

    for ROI in ROIs_raw_data:
        ROI_Number = str(ROI.ROINumber)
        ROIs_data['ROIs'][ROI_Number] = {
            "Name": ROI.ROIName,
            "Color": None,
            "Volume": np.zeros(volume_shape, dtype=bool)
        }

    for contour in contours_data:
        ROI_Number = str(contour.ReferencedROINumber)
        ROI_entry = ROIs_data['ROIs'][ROI_Number]

        if ROI_entry["Color"] is None:
            ROI_entry["Color"] = getattr(contour, "ROIDisplayColor", None)

        voxel_coords_list = []

        if hasattr(contour, "ContourSequence") and contour.ContourSequence:
            for contour_seq in contour.ContourSequence:
                slice_uid = contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID
                z_index = CT_data['Slices'][slice_uid]['Z Index']

                points = np.array(contour_seq.ContourData).reshape(-1, 3)
                x_indices = np.round((points[:, 0] - x_origin) / x_spacing).astype(int)
                y_indices = np.round((points[:, 1] - y_origin) / y_spacing).astype(int)
                z_indices = np.full_like(x_indices, z_index)

                coords = np.stack([z_indices, y_indices, x_indices], axis=1)
                voxel_coords_list.append(coords)

            all_voxel_coords = np.vstack(voxel_coords_list)
            filled_volume = voxelize_convex_hull(all_voxel_coords, volume_shape)
            ROI_entry['Volume'][filled_volume] = True  # Set matching cells to True

    return ROIs_data


def resample_array(array, zoom_factors, order=1):
    """
    Resamples a 2D/3D NumPy array to a new spacing.

    Parameters:
        array: 2D/3D NumPy array
        zoom_factors: desired zoom factors in [z, y, x]
        order: interpolation order (1 = linear, 0 = nearest for masks)

    Returns:
        resampled_volume: volume resampled to new spacing
    """
    resampled = scipy.ndimage.zoom(array, zoom=zoom_factors, order=order)
    return resampled


def match_Dose_to_CT(Dose_data, CT_data, scales, offsets):
    """
    Aligns Dose images to the CT slice dimensions.

    Parameters:
    Dose_data: numpy.ndarray
        A dictionary containing the dose metadata.
    CT_data: dict
        A dictionary containing the CT metadata.
    scale_x: float
        Scaling factor along the x-axis.
    scale_y: float
        Scaling factor along the y-axis.
    offset_x: int
        Offset in pixels along the x-axis.
    offset_y: int
        Offset in pixels along the y-axis.

    Returns:
    numpy.ndarray
        A transformed 3D array of dose images aligned with the CT slice.
    """
    scale_z, scale_y, scale_x = scales
    offset_z, offset_y, offset_x = offsets

    # Build the 3D affine transformation matrix (rotation + scale)
    affine_matrix = np.array([
        [1 / scale_z, 0, 0],
        [0, 1 / scale_y, 0],
        [0, 0, 1 / scale_x]
    ])

    # Note: affine_transform maps from output space to input space
    offset = [-offset_z / scale_z, -offset_y / scale_y, -offset_x / scale_x]

    # Apply the affine transformation
    Dose_data["Volume"] = scipy.ndimage.affine_transform(
        Dose_data["Volume"],
        matrix=affine_matrix,
        offset=offset,
        output_shape=CT_data["Volume"].shape,
        order=1  # Linear interpolation
    )


def preprocess_Dose_to_CT(Dose_data, CT_data):
    """
    Transforms dose images to align with CT slices dimensions.

    Parameters:
    Dose_data: dict
        A dictionary containing dose metadata.
    CT_data: dict
        A dictionary containing CT metadata.

    Returns:
    dict
        The updated Dose_data dictionary with transformed images aligned to CT.
    """
    # Obtain Dose and CT Spacings and Positions to match between them.
    Dose_z_spacing, Dose_y_spacing, Dose_x_spacing = Dose_data["Spacing"]
    Dose_z_position, Dose_y_position, Dose_x_position = Dose_data["Position"]

    CT_z_spacing, CT_y_spacing, CT_x_spacing = CT_data["Spacing"]
    CT_z_position, CT_y_position, CT_x_position = CT_data["Position"]

    # Define scales
    scale_z = Dose_z_spacing / CT_z_spacing
    scale_y = Dose_y_spacing / CT_y_spacing
    scale_x = Dose_x_spacing / CT_x_spacing
    scales = [scale_z, scale_y, scale_x]

    # Define offsets
    offset_z = int((Dose_z_position - CT_z_position) / CT_z_spacing)
    offset_y = int((Dose_y_position - CT_y_position) / CT_y_spacing)
    offset_x = int((Dose_x_position - CT_x_position) / CT_x_spacing)
    offsets = [offset_z, offset_y, offset_x]

    # Match Dose images to CT
    match_Dose_to_CT(Dose_data, CT_data, scales, offsets)
    Dose_data["Spacing"] = CT_data["Spacing"]
    Dose_data["Position"] = CT_data["Position"]
    return Dose_data


def save_volume_as_nifti(volume, spacing, output_path, affine_origin=(0, 0, 0)):
    """
    Save a 3D NumPy volume as a NIfTI file.

    Parameters:
    - volume (np.ndarray): The 3D volume (Z, Y, X).
    - spacing (tuple): Voxel spacing (Z, Y, X) in mm.
    - output_path (str): Where to save the .nii.gz file.
    - affine_origin (tuple): Origin of the image (Z, Y, X), defaults to (0,0,0).
    """
    affine = np.diag(list(spacing)[::-1] + [1])
    affine[:3, 3] = affine_origin[::-1]  # Put origin in correct position (X, Y, Z)

    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nifti_img, output_path)


import json


def save_volumes(CT_data, Dose_data, ROIs_data, output_path):
    # Save CT
    CT_output_path = os.path.join(output_path, "CT_volume.nii.gz")
    save_volume_as_nifti(volume=CT_data["Volume"], spacing=CT_data["Spacing"],
                         output_path=CT_output_path, affine_origin=CT_data["Position"])

    # Save Dose
    Dose_output_path = os.path.join(output_path, "Dose_volume.nii.gz")
    save_volume_as_nifti(volume=Dose_data["Volume"], spacing=Dose_data["Spacing"],
                         output_path=Dose_output_path, affine_origin=Dose_data["Position"])

    # Save ROIs
    ROIs_output_path = os.path.join(output_path, "ROIs/")
    os.makedirs(ROIs_output_path, exist_ok=True)

    roi_metadata = {}  # Dictionary to store ROI metadata

    for ROI_Number, ROI_entry in ROIs_data['ROIs'].items():
        # Save the volume
        ROI_volume_path = os.path.join(ROIs_output_path, f"ROI_{ROI_Number}_volume.nii.gz")
        save_volume_as_nifti(volume=ROI_entry["Volume"], spacing=ROIs_data["Spacing"],
                             output_path=ROI_volume_path, affine_origin=ROIs_data["Position"])

        # Store metadata
        roi_name = ROI_entry.get("Name")
        roi_color = ROI_entry.get("Color")

        # Convert color to standard Python list of ints
        if roi_color is not None:
            try:
                color_list = [int(c) for c in roi_color]
            except:
                color_list = None
        else:
            color_list = None

        roi_metadata[ROI_Number] = {
            "Name": str(roi_name) if roi_name is not None else f"ROI_{ROI_Number}",
            "Color": color_list
        }
    # Save ROI metadata as JSON
    metadata_path = os.path.join(ROIs_output_path, "roi_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(roi_metadata, f, indent=4)


import re


def parse_patient_metadata(name):
    """
    Parse metadata from patient folder name.
    Example format: 1575_SRS_LT occipital_02062021
    """
    pattern = r"(\d+)[ _]+([A-Z]+)(?:[ _]+([A-Z0-9]+))?(?:[ _]+([A-Za-z0-9 ]+))?[ _]+(\d{2})(\d{2})(\d{4})"
    match = re.match(pattern, name)

    if not match:
        return {"Patient ID": "Unknown", "Treatment": "Unknown", "Region": "Unknown", "Date": "Unknown"}

    pid, treatment, side, region, day, month, year = match.groups()
    date = f"{day}/{month}/{year}"
    return {
        "Patient ID": pid,
        "Treatment": treatment,
        "Region": f"{side} {region}",
        "Date": date
    }


import os
import glob
import numpy as np
import nibabel as nib


def load_nifti_volume(path):
    """
    Load a NIfTI file and return a dictionary with volume, spacing, and origin.
    """
    nii = nib.load(path)
    volume = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()[::-1]  # Z, Y, X
    origin = nii.affine[:3, 3][::-1]  # Z, Y, X
    return {
        "Volume": volume,
        "Spacing": np.array(spacing),
        "Position": np.array(origin)
    }


def load_preprocessed_volumes(files_path):
    """
    Load CT, Dose, and ROI volumes from .nii.gz files in the specified directory.

    Returns:
        CT_data, Dose_data, ROIs_data
    """

    # --- Load CT ---
    ct_path = os.path.join(files_path, "CT_volume.nii.gz")
    CT_data = load_nifti_volume(ct_path)

    # --- Load Dose ---
    dose_path = os.path.join(files_path, "Dose_volume.nii.gz")
    Dose_data = load_nifti_volume(dose_path)

    # --- Load ROI metadata ---
    roi_metadata_path = os.path.join(files_path, "ROIs", "roi_metadata.json")
    roi_metadata = {}
    if os.path.exists(roi_metadata_path):
        with open(roi_metadata_path, "r") as f:
            roi_metadata = json.load(f)

    # --- Load ROI volumes ---
    ROIs_data = {
        "ROIs": {},
        "Spacing": CT_data["Spacing"],
        "Position": CT_data["Position"]
    }

    roi_paths = sorted(glob.glob(os.path.join(files_path, "ROIs", "ROI_*_volume.nii.gz")))
    for path in roi_paths:
        filename = os.path.basename(path)
        parts = filename.split("_")
        if len(parts) < 3:
            continue  # Skip malformed files

        roi_number = parts[1]
        volume = load_nifti_volume(path)["Volume"]

        ROIs_data["ROIs"][roi_number] = {
            "Name": roi_metadata.get(roi_number, {}).get("Name", f"ROI_{roi_number}"),
            "Color": roi_metadata.get(roi_number, {}).get("Color", None),
            "Volume": volume.astype(bool)
        }

    return CT_data, Dose_data, ROIs_data
