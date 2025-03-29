import numpy as np
import pydicom
import os


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


def get_contour_data(rtstruct):
    """Extracts contour data and colors from the RT Structure Set."""
    contour_data = {}
    contour_color = {}
    structure_set_ROI_sequence = rtstruct.StructureSetROISequence
    roi_contour = rtstruct.ROIContourSequence

    for i in range(len(structure_set_ROI_sequence)):
        roi_name = structure_set_ROI_sequence[i].ROIName
        contour_data[roi_name] = []
        contour_color[roi_name] = roi_contour[i].ROIDisplayColor
        roi_contour1 = roi_contour[i]
        for contour_sequence in roi_contour1.ContourSequence:
            contour_points = contour_sequence.ContourData
            if isinstance(contour_points, pydicom.multival.MultiValue):
                points = np.array(contour_points).reshape(-1, 3)  # Reshape into Nx3 array (x, y, z)
            else:
                points = np.array(list(map(float, contour_points.split()))).reshape(-1, 3)
            contour_data[roi_name].append(points)
    return contour_data, contour_color


def load_contour_slices(rs_dataset):
    """Loads contour slices from CT images in the given RS dataset."""
    contour_slices = {}

    # For each ROI, extract corresponding sequence of contours
    roi_contour_sequence = rs_dataset.ROIContourSequence
    structure_set_ROI_sequence = rs_dataset.StructureSetROISequence

    for i in range(len(structure_set_ROI_sequence)):
        roi_contour = roi_contour_sequence[i].ContourSequence
        roi_name = structure_set_ROI_sequence[i].ROIName
        slice_data = []
        contour_slices[roi_name] = []

        for i in range(len(roi_contour)):
            slice_number = roi_contour[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            contour_points = roi_contour[i].ContourData
            slice_data.append((slice_number, contour_points))

        contour_slices[roi_name] = slice_data
    return contour_slices


def load_ct_slices(ct_folder_path):
    """Loads CT slices and extracts pixel spacing from CT images in the given RS dataset."""
    ct_slices = {}
    for file_name in os.listdir(ct_folder_path):
        if file_name.startswith('CT'):
            ct_slice = pydicom.dcmread(os.path.join(ct_folder_path, file_name))
            slice_number = ct_slice.SOPInstanceUID
            slice_position = ct_slice.ImagePositionPatient
            pixel_spacing = ct_slice.PixelSpacing

            ct_slices[slice_number] = (ct_slice.pixel_array, slice_position)
            x_spacing = pixel_spacing[0]
            y_spacing = pixel_spacing[1]

    return (ct_slices, x_spacing, y_spacing)


def sort_ct(ct_slices):
    """Sorts CT slices based on slice position (the most bottom image is first)."""
    slices_position = [ct_slices[i][1] for i in sorted(ct_slices.keys())]
    slices_order = [position[2] for position in slices_position]
    return sorted(slices_order)


def match_contour_to_ct(contours_across_slices, ct_slices):
    """Matches contour slices to corresponding CT slices."""
    matched_slices = {}
    # looping over specific rois and their contour through all contours across all slices.
    for roi_name, contour_across_slices in contours_across_slices.items():
        matched_slices[roi_name] = []
        # looping over a specific slice and contour locations through all slices.
        for slice_number, contour in contour_across_slices:
            if slice_number in ct_slices:
                matched_slices[roi_name].append(
                    (slice_number, contour, ct_slices[slice_number][0], ct_slices[slice_number][1]))

    return matched_slices
