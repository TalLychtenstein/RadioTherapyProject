import os
import pydicom
import numpy as np
import nibabel as nib


def load_ct_series(ct_dir):
    """
    Load all CT slices from a folder and sort them by their z-position
    """
    slices = []
    for file in os.listdir(ct_dir):
        path = os.path.join(ct_dir, file)
        ds = pydicom.dcmread(path)
        if ds.Modality == "CT":
            slices.append(ds)
    # Sort slices by z-position (ImagePositionPatient[2])
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def create_ct_nifti(ct_slices, output_path):
    """
    Create a NIfTI file from the CT slices and save it
    """
    # Get the shape of the volume (z, y, x)
    shape = (len(ct_slices), ct_slices[0].Rows, ct_slices[0].Columns)
    ct_volume = np.zeros(shape, dtype=np.int16)

    # Fill the volume with the pixel data from each slice
    for i, slice_ds in enumerate(ct_slices):
        ct_volume[i] = slice_ds.pixel_array

    # Get spacing and origin from the DICOM metadata
    spacing = [float(ct_slices[0].PixelSpacing[1]), float(ct_slices[0].PixelSpacing[0])]  # [row_spacing, col_spacing]
    spacing.append(
        float(ct_slices[1].ImagePositionPatient[2]) - float(ct_slices[0].ImagePositionPatient[2]))  # z spacing

    origin = ct_slices[0].ImagePositionPatient  # The origin of the image

    # Create affine transformation matrix
    affine = np.eye(4)
    affine[0, 0] = spacing[1]  # x spacing (columns)
    affine[1, 1] = spacing[0]  # y spacing (rows)
    affine[2, 2] = spacing[2]  # z spacing (slices)
    affine[:3, 3] = origin  # set the origin translation

    # Create NIfTI image and save it
    nifti_img = nib.Nifti1Image(ct_volume, affine)
    nib.save(nifti_img, output_path)
    print(f"NIfTI file saved to: {output_path}")


# Example usage
ct_dir = r'C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\64147\CT anatomy\64147_SRS_3 METS_24082022'  # Directory containing your DICOM CT slices
output_path = 'ct_volume.nii.gz'  # Output path for the NIfTI file

# Load the CT slices and create the NIfTI file
ct_slices = load_ct_series(ct_dir)
create_ct_nifti(ct_slices, output_path)
