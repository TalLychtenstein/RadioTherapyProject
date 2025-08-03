import os
import pydicom
import numpy as np
import scipy.ndimage
import cv2
import nibabel as nib
import SimpleITK as sitk
from skimage.draw import polygon

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
        "Scaling Factor": RD_data.DoseGridScaling,
        "Position": (float(pos_z), float(pos_y), float(pos_x)),
        "Spacing": (float(spacing_z), float(spacing_y), float(spacing_x)),  # matches (z, y, x) shape
        "Volume": dose_volume
    }


def extract_ROIs_data(RS_data):
    """
    Extracts ROIs and their associated contour data from an RT Structure Set (RS) DICOM file.

    Args:
        RS_data (pydicom Dataset): RT Structure Set (RS) DICOM dataset.

    Returns:
        ROIs_contours_data (dict): Dictionary containing ROI numbers as keys with their name, color, and contour data.
    """

    # Extract sequences
    ROIs_data = RS_data.StructureSetROISequence
    contours_data = RS_data.ROIContourSequence

    # Initialize storage dictionary
    ROIs_contours_data = {}
    ROIs_contours_data['ROIs'] = {}

    Outer_ROI_number = 0

    # Process ROIs
    for ROI in ROIs_data:
        ROI_Number = str(ROI.ROINumber)
        if ROI.ROIName == 'Outer Contour':
            Outer_ROI_number = ROI_Number
        ROIs_contours_data['ROIs'][ROI_Number] = {
            "Name": ROI.ROIName,  # ROI's name
            "Color": None,  # Placeholder for ROI's color
            "CT Contours": {},  # Placeholder for ROI's CT contours
        }

    # Process Contours
    for contour in contours_data:
        # Find corresponding ROI number
        ROI_Number = str(contour.ReferencedROINumber)

        # if ROI_Number == Outer_ROI_number:
        #     continue

        # Define the ROI color to be the color of its contour (if not already set)
        if "Color" in ROIs_contours_data['ROIs'][ROI_Number] and ROIs_contours_data['ROIs'][ROI_Number][
            "Color"] is None:
            ROIs_contours_data['ROIs'][ROI_Number]["Color"] = getattr(contour, "ROIDisplayColor", None)

        # Extract Contour Data (dict of CT slice number and corresponding list of (x, y, z) contour points)
        for contour_seq in contour.ContourSequence:
            # Find corresponding CT number
            CT_slice = str(contour_seq.ContourImageSequence[0].ReferencedSOPInstanceUID)

            # Extract Contour Data
            contour_points = contour_seq.ContourData  # Flat list of points
            x_points = contour_points[0::3]
            y_points = contour_points[1::3]
            z_points = contour_points[2::3]
            ROIs_contours_data['ROIs'][ROI_Number]["CT Contours"][CT_slice] = list(zip(x_points, y_points, z_points))
    return ROIs_contours_data


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


def rasterize_rtstruct_to_separate_images(rtstruct_path, reference_image):
    """
    Rasterizes RTSTRUCT contours into separate binary masks per ROI.

    Returns:
        dict: {roi_name: SimpleITK Image (binary mask)}
        list: metadata (list of dicts)
    """
    # Load RTSTRUCT
    ds = pydicom.dcmread(rtstruct_path)

    # Build ROI number -> name mapping
    roi_number_to_name = {
        roi.ROINumber: roi.ROIName for roi in ds.StructureSetROISequence
    }

    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()
    size = reference_image.GetSize()

    # Initialize empty arrays per ROI
    mask_images = {}

    metadata = []

    # For each ROIContourSequence
    for idx, roi_contour in enumerate(ds.ROIContourSequence, start=1):
        roi_number = roi_contour.ReferencedROINumber
        roi_name = roi_number_to_name[roi_number]
        mask_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        for contour_item in roi_contour.ContourSequence:
            data = contour_item.ContourData
            coords = np.array(data).reshape((-1, 3))

            indices = [
                reference_image.TransformPhysicalPointToIndex(tuple(pt))
                for pt in coords
            ]

            xs = [i[0] for i in indices]
            ys = [i[1] for i in indices]
            zs = [i[2] for i in indices]

            unique_z = set(zs)
            for zz in unique_z:
                slice_pts = [(x, y) for x, y, z in zip(xs, ys, zs) if z == zz]
                if len(slice_pts) < 3:
                    continue
                rr, cc = polygon(
                    [p[1] for p in slice_pts],
                    [p[0] for p in slice_pts],
                    shape=mask_array.shape[1:]
                )
                mask_array[zz, rr, cc] = 1

        # Convert to SimpleITK Image
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.SetSpacing(spacing)
        mask_img.SetOrigin(origin)
        mask_img.SetDirection(direction)

        mask_images[roi_name] = mask_img

        metadata.append({
            "index": idx,
            "name": roi_name,
            "color": (
                list(map(int, roi_contour.ROIDisplayColor))
                if hasattr(roi_contour, "ROIDisplayColor")
                else [255, 255, 255]
            )
        })

    return mask_images, metadata


def create_ROIs_volume_2(RT_files_path, CT_data):
    """
    Creates a ROIs volume (label map) using rasterization of RTSTRUCT contours
    with reference to the given CT volume.

    Parameters:
    - RT_files_path (str): Path to the DICOM folder containing RS file.
    - CT_data (dict): Dictionary containing CT volume, spacing, and position.

    Returns:
    - ROIs_data (dict): Dictionary with 'Volume' (3D label map) and 'ROIs' metadata.
    """
    # Create reference SimpleITK image from CT
    ct_sitk_image = sitk.GetImageFromArray(CT_data["Volume"])
    ct_sitk_image.SetSpacing(CT_data["Spacing"])
    ct_sitk_image.SetOrigin(CT_data["Position"])
    ct_sitk_image.SetDirection([1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0])

    # Get RS file path
    rtstruct_path = find_file_with_prefix(RT_files_path, 'RS')

    # Rasterize RTSTRUCT into binary masks
    roi_masks_dict, roi_metadata = rasterize_rtstruct_to_separate_images(
        rtstruct_path=rtstruct_path,
        reference_image=ct_sitk_image
    )

    # Initialize label volume and metadata
    volume_shape = CT_data["Volume"].shape
    label_volume = np.zeros(volume_shape, dtype=np.uint16)
    rois_dict = {}

    for idx, metadata in enumerate(roi_metadata, start=1):
        name = metadata["name"]
        color = metadata["color"]
        mask_image = roi_masks_dict[name]
        mask_array = sitk.GetArrayFromImage(mask_image)

        label_volume[mask_array > 0] = idx
        rois_dict[str(idx)] = {
            "Name": name,
            "Color": color,
            "CT Contours": {}  # Empty, kept for compatibility
        }

    ROIs_data = {
        "Volume": label_volume,
        "ROIs": rois_dict
    }

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


def get_CT_slice_by_slice_number(CT_data, slice_number):
    return CT_data['Slices'][slice_number]['Image']


def get_CT_slice_by_z_index(CT_data, z_index):
    return CT_data['Volume'][z_index]


# def create_ROIs_volume(ROIs_data, CT_data):
#     ROIs_volume = np.zeros_like(CT_data['Volume'])
#
#     # Extract metadata for proper contour alignment
#     z_position, y_position, x_position = CT_data["Position"]
#     z_spacing, y_spacing, x_spacing = CT_data["Spacing"]\
#
#     for ROI_number in ROIs_data['ROIs']:
#         for CT_slice_number in ROIs_data['ROIs'][ROI_number]['CT Contours']:
#
#             # Obtain the Z index of this CT slice in the CT volume
#             z_index = CT_data['Slices'][CT_slice_number]['Z Index']
#
#             # Preprocess the ROI contour
#             ROI_contour = ROIs_data['ROIs'][ROI_number]['CT Contours'][CT_slice_number]
#             preprocessed_x = [int((x - x_position) / x_spacing) for (x, y, z) in ROI_contour]
#             preprocessed_y = [int((y - y_position) / y_spacing) for (x, y, z) in ROI_contour]
#
#             # Fill ROI volume
#             for (y, x) in list(zip(preprocessed_y, preprocessed_x)):
#                 ROIs_volume[z_index, y, x] = ROI_number
#     return ROIs_volume

# import numpy as np
# from skimage.draw import polygon  # For filling ROI contours
#
# def create_ROIs_volume(ROIs_data, CT_data):
#     ROIs_volume = np.zeros_like(CT_data['Volume'], dtype=np.uint16)
#
#     # Extract metadata for proper contour alignment
#     z_position, y_position, x_position = CT_data["Position"]
#     z_spacing, y_spacing, x_spacing = CT_data["Spacing"]
#
#
#     for ROI_number in ROIs_data['ROIs']:
#         for CT_slice_number in ROIs_data['ROIs'][ROI_number]['CT Contours']:
#
#             # Obtain the Z index of this CT slice in the CT volume
#             z_index = CT_data['Slices'][CT_slice_number]['Z Index']
#
#             # Preprocess the ROI contour points
#             ROI_contour = ROIs_data['ROIs'][ROI_number]['CT Contours'][CT_slice_number]
#             x_coords = [(x - x_position) / x_spacing for (x, y, z) in ROI_contour]
#             y_coords = [(y - y_position) / y_spacing for (x, y, z) in ROI_contour]
#
#             # Convert to integer pixel indices
#             rr, cc = polygon(y_coords, x_coords, shape=ROIs_volume[z_index].shape)
#
#             ROIs_volume[z_index, rr, cc] = ROI_number
#
#     return ROIs_volume

from skimage.draw import polygon
import numpy as np


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def create_ROIs_volume(ROIs_data, CT_data):
    ROIs_volume = np.zeros_like(CT_data['Volume'], dtype=np.uint16)

    z_position, y_position, x_position = CT_data["Position"]
    z_spacing, y_spacing, x_spacing = CT_data["Spacing"]

    # Group all contours by slice
    contours_by_slice = {}

    for ROI_number in ROIs_data['ROIs']:
        for CT_slice_number, contour in ROIs_data['ROIs'][ROI_number]['CT Contours'].items():
            z_index = CT_data['Slices'][CT_slice_number]['Z Index']
            x_coords = [(x - x_position) / x_spacing for (x, y, z) in contour]
            y_coords = [(y - y_position) / y_spacing for (x, y, z) in contour]

            area = polygon_area(x_coords, y_coords)

            if z_index not in contours_by_slice:
                contours_by_slice[z_index] = []
            contours_by_slice[z_index].append((area, ROI_number, x_coords, y_coords))

    # Now insert per slice sorted by area
    for z_index in contours_by_slice:
        # Sort by increasing area
        for area, ROI_number, x_coords, y_coords in sorted(contours_by_slice[z_index], key=lambda x: x[0], reverse=True):
            rr, cc = polygon(y_coords, x_coords, shape=ROIs_volume[z_index].shape)

            insert_mask = np.zeros_like(ROIs_volume[z_index], dtype=bool)
            insert_mask[rr, cc] = True

            # insert_mask = np.logical_and(temp_mask, ROIs_volume[z_index] == 0)
            ROIs_volume[z_index][insert_mask] = ROI_number

    return ROIs_volume


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

def extract_group1_metadata(RP_data, RD_data):
    """
    Extract Group 1 metadata parameters from RTPLAN and RTDOSE files.

    Parameters:
    - RP_data: DICOM dataset (RTPLAN)
    - RD_data: DICOM dataset (RTDOSE)

    Returns:
    dict with keys:
        - Prescription Dose
        - Number of Fractions
        - Dose per Fraction
        - Planning Software
        - Planning Technique
        - Dose Calculation Algorithm (if available)
        - Dose Grid Size (mm)
    """
    result = {}

    # Prescription dose (from DoseReferenceSequence)
    try:
        dose_ref = RP_data.DoseReferenceSequence[0]
        result["Prescription Dose [Gy]"] = float(dose_ref.TargetPrescriptionDose)
    except:
        result["Prescription Dose [Gy]"] = None

    # Fractions
    try:
        result["Number of Fractions"] = int(RP_data.FractionGroupSequence[0].NumberOfFractionsPlanned)
    except:
        result["Number of Fractions"] = None

    # Dose per Fraction
    try:
        if result["Prescription Dose [Gy]"] and result["Number of Fractions"]:
            result["Dose per Fraction [Gy]"] = (
                result["Prescription Dose [Gy]"] / result["Number of Fractions"]
            )
        else:
            result["Dose per Fraction [Gy]"] = None
    except:
        result["Dose per Fraction [Gy]"] = None

    # Software
    result["Planning Software"] = RP_data.get("SoftwareVersions", None)

    # Planning Technique (first beam)
    try:
        technique = RP_data.BeamSequence[0].BeamTechnique
        result["Planning Technique"] = technique
    except:
        result["Planning Technique"] = None

    # Dose Calculation Algorithm (not always standard)
    try:
        algo = RP_data.BeamSequence[0].DoseCalculationAlgorithm
        result["Dose Calculation Algorithm"] = algo
    except:
        result["Dose Calculation Algorithm"] = "Unknown or vendor-specific"

    # Dose Grid Size
    try:
        spacing_x, spacing_y = RD_data.PixelSpacing
        offset_vector = RD_data.GridFrameOffsetVector
        spacing_z = offset_vector[1] - offset_vector[0]
        result["Dose Grid Size [mm]"] = (spacing_z, spacing_y, spacing_x)
    except:
        result["Dose Grid Size [mm]"] = None

    return result


import numpy as np

def compute_volume_cc(roi_mask, spacing):
    voxel_volume_mm3 = np.prod(spacing)
    voxel_volume_cc = voxel_volume_mm3 / 1000
    return np.sum(roi_mask > 0) * voxel_volume_cc

def compute_volume(roi_mask, spacing):
    voxel_volume_mm3 = np.prod(spacing)
    return np.sum(roi_mask > 0) * voxel_volume_mm3


def compute_dvh(dose_volume, roi_mask, bin_width=1.0, max_dose=None):
    masked_dose = dose_volume[roi_mask > 0]
    if max_dose is None:
        max_dose = np.max(masked_dose)
    bins = np.arange(0, max_dose + bin_width, bin_width)
    hist, _ = np.histogram(masked_dose, bins=bins)
    return {"dose_bins": bins[:-1], "voxel_counts": hist}


def compute_dose_percentiles(dose_volume, roi_mask):
    masked_dose = dose_volume[roi_mask > 0]
    return {
        "D2% [Gy]": np.percentile(masked_dose, 98),
        "D50% [Gy]": np.percentile(masked_dose, 50),
        "D98% [Gy]": np.percentile(masked_dose, 2)
    }


def compute_homogeneity_index(d2, d98):
    return (d2 - d98) / d2 if d2 > 0 else None


def compute_conformity_index(dose_volume, roi_mask, prescribed_dose):
    ptv_covered = np.sum((dose_volume >= prescribed_dose) & (roi_mask > 0))
    total_high_dose = np.sum(dose_volume >= prescribed_dose)
    if total_high_dose == 0:
        return None
    return ptv_covered / total_high_dose


def extract_group2_metrics(Dose_data, ROIs_data, target_labels=None, prescribed_dose=None):
    dose_volume = Dose_data["Volume"]
    spacing = Dose_data["Spacing"]
    label_map = ROIs_data["Volume"]
    results = {}

    for roi_number, roi_info in ROIs_data["ROIs"].items():
        name = roi_info["Name"]
        if target_labels:
            name_upper = name.upper()
            if not any(target.upper() in name_upper for target in target_labels):
                continue

        roi_mask = (label_map == int(roi_number))
        if not np.any(roi_mask):
            continue

        volume = compute_volume(roi_mask, spacing)
        percentiles = compute_dose_percentiles(dose_volume, roi_mask)
        hi = compute_homogeneity_index(percentiles["D2% [Gy]"], percentiles["D98% [Gy]"])
        ci = compute_conformity_index(dose_volume, roi_mask, prescribed_dose) if prescribed_dose else None
        dvh = compute_dvh(dose_volume, roi_mask)

        results[name] = {
            "ROI Number": roi_number,
            "Volume [mm3]": volume,
            **percentiles,
            "Homogeneity Index": hi,
            "Conformity Index": ci,
            "DVH": dvh
        }

    return results


import matplotlib.pyplot as plt


def plot_dvh_curves(group2_metrics):
    """
    Plot DVH curves for each ROI using the extracted group2 metrics.

    Parameters:
    - group2_metrics: dict output from extract_group2_metrics()
    """
    plt.figure(figsize=(8, 6))
    for roi_name, metrics in group2_metrics.items():
        dvh = metrics["DVH"]
        doses = dvh["dose_bins"]
        counts = dvh["voxel_counts"]
        volume_percent = 100 * counts.cumsum()[::-1] / counts.sum()  # Reverse cumulative %

        plt.plot(doses, volume_percent, label=roi_name)

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Volume [%]")
    plt.title("Dose Volume Histogram (DVH)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import statistics

def plot_histogram_summary(group2_metrics):
    for roi_name, metrics in group2_metrics.items():
        doses = metrics["DVH"]["dose_bins"]
        counts = metrics["DVH"]["voxel_counts"]
        expanded = []
        for d, c in zip(doses, counts):
            expanded.extend([d] * c)

        if not expanded:
            continue

        fig, ax = plt.subplots()
        plt.hist(expanded, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Dose [Gy]')
        plt.ylabel('Number of Voxels')
        plt.title(roi_name)

        plt.text(.51, .99, f"Min = {min(expanded):.2f}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .95, f"Max = {max(expanded):.2f}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .91, f"Mean = {statistics.mean(expanded):.2f}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .87, f"Median = {statistics.median(expanded):.2f}", ha='left', va='top', transform=ax.transAxes)
        plt.tight_layout()
        plt.show()

def get_group2_roi_names(ROIs_data, target_labels=None):
    names = []
    for roi in ROIs_data["ROIs"].values():
        name = roi["Name"]
        if not target_labels or any(t.lower() in name.lower() for t in target_labels):
            names.append(name)
    return names


def get_group2_metrics_for_roi(metrics_dict, roi_name):
    if roi_name not in metrics_dict:
        return "ROI not found.", None

    m = metrics_dict[roi_name]
    info = {
        "Volume [cc]": m["Volume [cc]"],
        "D2% [Gy]": m["D2% [Gy]"],
        "D50% [Gy]": m["D50% [Gy]"],
        "D98% [Gy]": m["D98% [Gy]"],
        "Homogeneity Index": m["Homogeneity Index"],
        "Conformity Index": m["Conformity Index"]
    }

    # Plot
    fig, ax = plt.subplots()
    doses = m["DVH"]["dose_bins"]
    counts = m["DVH"]["voxel_counts"]
    volume_percent = 100 * counts.cumsum()[::-1] / counts.sum()
    ax.plot(doses, volume_percent, label=roi_name)
    ax.set_xlabel("Dose [Gy]")
    ax.set_ylabel("Volume [%]")
    ax.set_title(f"DVH: {roi_name}")
    ax.grid(True)
    ax.legend()
    return info, fig

import pandas as pd
import os
import matplotlib.pyplot as plt

def save_group1_to_excel(group1_dict, output_path):
    """
    Save Group 1 metrics (plan metadata) to Excel.
    """
    df = pd.DataFrame.from_dict(group1_dict, orient='index', columns=["Value"])
    df.to_excel(output_path, sheet_name="Group 1")

def save_group2_to_excel(group2_dict, output_path):
    """
    Save Group 2 metrics (one row per ROI) to Excel (excluding DVH).
    """
    rows = []
    for roi_name, data in group2_dict.items():
        row = {
            "ROI Name": roi_name,
            "ROI Number": data["ROI Number"],
            "Volume [cc]": data["Volume [cc]"],
            "D2% [Gy]": data["D2% [Gy]"],
            "D50% [Gy]": data["D50% [Gy]"],
            "D98% [Gy]": data["D98% [Gy]"],
            "Homogeneity Index": data["Homogeneity Index"],
            "Conformity Index": data["Conformity Index"]
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, sheet_name="Group 2")

def save_all_dvh_plots(group2_dict, output_dir):
    """
    Save DVH plots for all ROIs as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for roi_name, data in group2_dict.items():
        dvh = data["DVH"]
        doses = dvh["dose_bins"]
        counts = dvh["voxel_counts"]
        volume_percent = 100 * counts.cumsum()[::-1] / counts.sum()

        plt.figure()
        plt.plot(doses, volume_percent, label=roi_name)
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Volume [%]")
        plt.title(f"DVH Curve: {roi_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{roi_name.replace(' ', '_')}_DVH.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

import matplotlib.pyplot as plt
import statistics
import os

def save_dose_histograms(group2_metrics, output_dir, bins=30):
    """
    Save dose histograms (not DVH) for all ROIs in group2_metrics.

    Parameters:
    - group2_metrics: output from extract_group2_metrics()
    - output_dir: where to save PNGs
    - bins: number of bins for histogram
    """
    os.makedirs(output_dir, exist_ok=True)

    for roi_name, metrics in group2_metrics.items():
        dvh = metrics["DVH"]
        doses = dvh["dose_bins"]
        counts = dvh["voxel_counts"]

        # Expand doses into a flat array (each voxel dose)
        expanded = []
        for d, c in zip(doses, counts):
            expanded.extend([d] * c)

        if not expanded:
            continue

        # Compute stats
        mean = statistics.mean(expanded)
        median = statistics.median(expanded)

        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(expanded, bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(roi_name)
        ax.set_xlabel("Gray")
        ax.set_ylabel("Number of Voxels")

        # Stats annotations
        ax.text(0.95, 0.95, f"Mean = {mean:.4f}", ha='right', va='top', transform=ax.transAxes)
        ax.text(0.95, 0.90, f"Median = {median:.4f}", ha='right', va='top', transform=ax.transAxes)

        fig.tight_layout()
        filename = f"{roi_name.replace(' ', '_')}_DoseHistogram.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

def extract_group3_metrics(Dose_data, ROIs_data, healthy_brain_label_name, dose_thresholds=[5, 10, 12, 18, 20, 23, 24, 25, 27, 30]):
    """
    Extract Group 3 Vx metrics from the healthy brain segmentation.

    Parameters:
    - Dose_data: dict with 'Volume' and 'Spacing'
    - ROIs_data: dict with 'Volume' (label map) and ROI names
    - healthy_brain_label_name: name of the ROI representing healthy brain
    - dose_thresholds: list of Gy thresholds to evaluate (default: [5, 10, 12, 18, 20, 23, 24, 25, 27, 30])

    Returns:
    - dict of {"Vx [Gy]": volume in cc}
    """
    label_map = ROIs_data["Volume"]
    spacing = Dose_data["Spacing"]
    dose = Dose_data["Volume"]

    # Find ROI number by name
    roi_number = None
    for num, info in ROIs_data["ROIs"].items():
        if info["Name"].lower() == healthy_brain_label_name.lower():
            roi_number = int(num)
            break

    if roi_number is None:
        raise ValueError(f"Healthy Brain ROI '{healthy_brain_label_name}' not found in ROIs.")

    brain_mask = (label_map == roi_number)
    voxel_volume_cc = np.prod(spacing) / 1000

    results = {}
    for threshold in dose_thresholds:
        mask = (dose >= threshold) & brain_mask
        results[f"V{threshold} [cc]"] = np.sum(mask) * voxel_volume_cc

    return results
