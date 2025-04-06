import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from matplotlib.patches import Patch, Polygon
from matplotlib.widgets import Slider
from collections import defaultdict


def load_ct_series(ct_dir):
    slices = []
    for file in os.listdir(ct_dir):
        path = os.path.join(ct_dir, file)
        ds = pydicom.dcmread(path)
        if ds.Modality == "CT":
            slices.append(ds)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def load_rs_file(rs_path):
    return pydicom.dcmread(rs_path)


def get_roi_mappings(rs):
    roi_dict = {}
    for roi in rs.StructureSetROISequence:
        roi_dict[roi.ROINumber] = roi.ROIName
    return roi_dict


def create_label_map(ct_slices, rs):
    z_map = {round(float(s.ImagePositionPatient[2]), 2): i for i, s in enumerate(ct_slices)}
    shape = (len(ct_slices), ct_slices[0].Rows, ct_slices[0].Columns)
    label_map = np.zeros(shape, dtype=np.uint8)  # Ensure you define label_map here
    roi_map = {}
    contours = defaultdict(list)
    for idx, roi_contour in enumerate(rs.ROIContourSequence, start=1):
        roi_number = roi_contour.ReferencedROINumber
        roi_map[roi_number] = idx
        if not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            z = round(float(contour.ContourData[2]), 2)
            if z not in z_map:
                continue
            slice_idx = z_map[z]
            coords = np.array(contour.ContourData).reshape(-1, 3)
            ds = ct_slices[slice_idx]
            origin = np.array(ds.ImagePositionPatient)
            spacing = np.array([float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])])
            pixel_coords = ((coords[:, :2] - origin[:2]) / spacing).astype(np.int32)
            pixel_coords = np.clip(pixel_coords, 0, [ds.Columns - 1, ds.Rows - 1])
            contours[slice_idx].append((roi_number, pixel_coords))
            mask = polygon2mask((ds.Rows, ds.Columns), np.flip(pixel_coords, axis=1))
            label_map[slice_idx][mask] = roi_number  # Add this line to fill label_map

    return label_map, roi_map, contours  # Return label_map here


def update(val):
    slice_index = int(slider.val)
    img = ct_slices[slice_index].pixel_array
    ax.clear()
    ax.imshow(img, cmap='gray')
    handles = []
    for roi_number, contour in contours[slice_index]:
        color = plt.cm.nipy_spectral(roi_number / (len(roi_names) + 1))
        poly = Polygon(contour, closed=True, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(poly)
    for roi_number, roi_name in roi_names.items():
        color = plt.cm.nipy_spectral(roi_number / (len(roi_names) + 1))
        handles.append(Patch(color=color, label=f"{roi_number}: {roi_name}"))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title(f"Contour Map (Slice {slice_index})")
    fig.canvas.draw_idle()



ct_dir = r"DICOM files/52724/CT anatomy/52724_SRS_2 METS_30062020"
rs_path = r"DICOM files/52724/CT radiation maps/52724_radiation plans_30062020/RS.1.2.246.352.221.5606847014205161347.14613639006987443369.dcm"
ct_slices = load_ct_series(ct_dir)
rs = load_rs_file(rs_path)
roi_names = get_roi_mappings(rs)
label_map, roi_map, contours = create_label_map(ct_slices, rs)  # Get the label_map here

output_path = "3d_ct_seg.nii.gz"



fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.imshow(ct_slices[0].pixel_array, cmap='gray')

ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Slice', 0, len(ct_slices) - 1, valinit=0, valstep=1)
slider.on_changed(update)
plt.show()

import nibabel as nib
import numpy as np

def export_label_map_to_nifti(label_map, ct_slices, output_path):
    # Get image spacing and origin from the first slice
    spacing = list(map(float, ct_slices[0].PixelSpacing))  # [row_spacing, col_spacing]
    spacing_z = float(ct_slices[1].ImagePositionPatient[2]) - float(ct_slices[0].ImagePositionPatient[2])
    spacing.append(spacing_z)

    origin = ct_slices[0].ImagePositionPatient

    # Create affine transform matrix
    affine = np.eye(4)
    affine[0, 0] = spacing[1]  # x spacing (columns)
    affine[1, 1] = spacing[0]  # y spacing (rows)
    affine[2, 2] = spacing[2]  # z spacing (slices)
    affine[:3, 3] = origin     # set translation (origin)

    # Create and save the NIfTI image
    nifti_img = nib.Nifti1Image(label_map.astype(np.uint8), affine)
    nib.save(nifti_img, output_path)
    print(f"NIfTI file saved to: {output_path}")

# Example usage
output_path = "3d_ct_seg.nii.gz"
export_label_map_to_nifti(label_map, ct_slices, output_path)  # Pass label_map to the export function


