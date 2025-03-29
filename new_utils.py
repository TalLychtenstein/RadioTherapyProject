import os
import pydicom
import numpy as np
import scipy.ndimage
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
        - "Spacing": The spacing between pixels in the dose grid.
        - "Image": The actual pixel array representing the dose distribution.
    """

    Dose_data = {"Position": RD_data.ImagePositionPatient,
                 "Spacing": RD_data.PixelSpacing,
                 "Volume": RD_data.pixel_array * RD_data.DoseGridScaling}
    return Dose_data


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
    # Process ROIs
    for ROI in ROIs_data:
        ROI_Number = str(ROI.ROINumber)
        ROIs_contours_data['ROIs'][ROI_Number] = {
            "Name": ROI.ROIName,  # ROI's name
            "Color": None,  # Placeholder for ROI's color
            "CT Contours": {},  # Placeholder for ROI's CT contours
        }

    # Process Contours
    for contour in contours_data:
        # Find corresponding ROI number
        ROI_Number = str(contour.ReferencedROINumber)

        # Define the ROI color to be the color of its contour (if not already set)
        if "Color" in ROIs_contours_data['ROIs'][ROI_Number] and ROIs_contours_data['ROIs'][ROI_Number]["Color"] is None:
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
    min_z = np.inf
    for file_name in os.listdir(files_path):
        if file_name.startswith('CT'):
            CT_slice_data = pydicom.dcmread(os.path.join(files_path, file_name))
            slice_number = CT_slice_data.SOPInstanceUID
            slice_position = CT_slice_data.ImagePositionPatient
            pixel_spacing = CT_slice_data.PixelSpacing
            pixel_array = CT_slice_data.pixel_array
            CT_data['Slices'][slice_number] = {"Position": slice_position,
                                               "Spacing": pixel_spacing,
                                               "Image": pixel_array}

            # Set total CT position and spacing to be the same as the slice with lowest Z index
            if min_z > slice_position[2]:
                min_z = slice_position[2]
                CT_data['Position'] = slice_position
                CT_data['Spacing'] = pixel_spacing

    return CT_data


def resample_array(array, old_spacing, new_spacing, order=1):
    """
    Resamples a 2D/3D NumPy array to a new spacing.

    Parameters:
        array: 2D/3D NumPy array
        old_spacing: list or array of [y, x] spacing in mm
        new_spacing: desired spacing [y, x] in mm
        order: interpolation order (1 = linear, 0 = nearest for masks)

    Returns:
        resampled_volume: volume resampled to new spacing
    """

    # Add fake z spacing to maintain same z-axis resolution
    if array.ndim == 3:
        old_spacing = [1] + list(old_spacing)
        new_spacing = [1] + list(new_spacing)

    zoom_factors = np.array(old_spacing) / np.array(new_spacing)
    resampled = scipy.ndimage.zoom(array, zoom=zoom_factors, order=order)
    return resampled


def create_CT_volume(CT_data):
    # 1. Sort CT slices
    CT_slices_data = [(slice_number, CT_data['Slices'][slice_number]["Position"][2]) for slice_number in
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


def match_Dose_to_CT(Dose_data, CT_data, scale_x, scale_y, offset_x, offset_y):
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
    num_slices = Dose_data["Volume"].shape[0]

    # Compute the combined transformation matrix (scaling + translation)
    transformation_matrix = np.float32([
        [scale_x, 0, offset_x],  # Scale X and shift X
        [0, scale_y, offset_y]  # Scale Y and shift Y
    ])

    # Determine new image shape
    new_shape = (CT_data["Volume"].shape[1], CT_data["Volume"].shape[2])

    # Apply transformation to all slices
    transformed_images = np.zeros((num_slices, new_shape[0], new_shape[1]), dtype=np.float32)

    for i in range(num_slices):
        transformed_images[i] = cv2.warpAffine(Dose_data["Volume"][i].astype(np.float32),
                                               transformation_matrix,
                                               new_shape,
                                               flags=cv2.INTER_LINEAR)

    return transformed_images


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
    Dose_x_spacing, Dose_y_spacing = Dose_data["Spacing"]
    Dose_x_position, Dose_y_position, _ = Dose_data["Position"]

    CT_x_spacing, CT_y_spacing = CT_data["Spacing"]
    CT_x_position, CT_y_position, _ = CT_data["Position"]

    # Define scales
    scale_x = Dose_x_spacing / CT_x_spacing
    scale_y = Dose_y_spacing / CT_y_spacing

    # Define offsets
    offset_x = int((Dose_x_position - CT_x_position) / CT_x_spacing)
    offset_y = int((Dose_y_position - CT_y_position) / CT_y_spacing)

    # Match Dose images to CT
    Dose_data["Volume"] = match_Dose_to_CT(Dose_data, CT_data, scale_x, scale_y, offset_x, offset_y)
    return Dose_data


def get_CT_slice_by_slice_number(CT_data, slice_number):
    return CT_data['Slices'][slice_number]['Image']


def get_CT_slice_by_z_index(CT_data, z_index):
    return CT_data['Volume'][z_index]

def create_ROIs_volume(ROIs_data, CT_data):
    ROIs_volume = np.zeros_like(CT_data['Volume'])

    # Extract metadata for proper contour alignment
    x_position, y_position, _ = CT_data["Position"]
    x_spacing, y_spacing = CT_data["Spacing"]

    for ROI_number in ROIs_data['ROIs']:
        for CT_slice_number in ROIs_data['ROIs'][ROI_number]['CT Contours']:

            # Obtain the Z index of this CT slice in the CT volume
            z_index = CT_data['Slices'][CT_slice_number]['Z Index']

            # Preprocess the ROI contour
            ROI_contour = ROIs_data['ROIs'][ROI_number]['CT Contours'][CT_slice_number]
            preprocessed_x = [int((x - x_position) / x_spacing) for (x, y, z) in ROI_contour]
            preprocessed_y = [int((y - y_position) / y_spacing) for (x, y, z) in ROI_contour]
            preprocessed_z = [int(z_index) for (x, y, z) in ROI_contour]

            # Fill ROI volume
            for (z, y, x) in list(zip(preprocessed_z, preprocessed_y, preprocessed_x)):
                ROIs_volume[z, y, x] = ROI_number
    return ROIs_volume
