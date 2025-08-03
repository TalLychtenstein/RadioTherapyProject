import os, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.draw import polygon
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import itk


def classify_dicom_files(dicom_folder):
    dicom_files = list(Path(dicom_folder).glob("*.dcm"))
    modality_map = defaultdict(list)
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            modality_map[ds.Modality].append(str(f))
        except:
            continue
    return modality_map


def get_dicom_components(dicom_folder):
    """
    Classify and extract DICOM components:
    Returns:
        ct_dir: Path to CT series folder
        rd_file: Path to RTDOSE file
        rs_file: Path to RTSTRUCT file
    """
    modality_map = classify_dicom_files(dicom_folder)

    ct_files = modality_map["CT"]
    rd_files = modality_map["RTDOSE"]
    rs_files = modality_map["RTSTRUCT"]

    assert len(ct_files) > 0, "❌ No CT files found"
    assert len(rd_files) == 1, "❌ Expected exactly one RD file"
    assert len(rs_files) == 1, "❌ Expected exactly one RS file"

    ct_dir = Path(ct_files[0]).parent
    rd_file = rd_files[0]
    rs_file = rs_files[0]

    return ct_dir, ct_files, rd_file, rs_file


def load_sitk_volume_from_series(dicom_files):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(Path(dicom_files[0]).parent))
    series_files = reader.GetGDCMSeriesFileNames(str(Path(dicom_files[0]).parent), series_IDs[0])
    reader.SetFileNames(series_files)
    return reader.Execute()


def get_rescale_params(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    return intercept, slope


def get_display_params(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    window_center = float(ds.WindowCenter)
    window_width = float(ds.WindowWidth)
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    return vmin, vmax


def convert_ct_to_hu_sitk(sitk_img, dicom_path):
    """
    Returns a SimpleITK.Image with voxel values in HU (Hounsfield Units).
    note: no need to convert ct images from BrainLAB iPlan RT Dose - the images are derived and already saved in HU
    """
    intercept, slope = get_rescale_params(dicom_path)
    return sitk.Cast(sitk_img, sitk.sitkFloat32)*slope + intercept


def convert_ct_to_hu_sitk_safe(sitk_img, dicom_path):
    """
    Returns a SimpleITK.Image with voxel values in HU (Hounsfield Units).
    Handles derived images that are already in HU.
    """
    ds = pydicom.dcmread(dicom_path)

    intercept = float(ds.RescaleIntercept)
    slope = float(ds.RescaleSlope)

    image_type = ds.get("ImageType", [])
    if isinstance(image_type, str):
        image_type = image_type.split("\\")

    # Check raw min
    raw_arr = sitk.GetArrayFromImage(sitk_img)
    raw_min = raw_arr.min()

    # Heuristic: Derived + signed + + raw_min ≈ intercept = already HU
    if (
        "DERIVED" in image_type and
        abs(raw_min - intercept) < 2
    ):
        print("✅ Detected DERIVED signed image with negative stored values; assuming already HU.")
        return sitk.Cast(sitk_img, sitk.sitkFloat32)

    # Otherwise apply intercept
    print(f"✅ Applying rescale: slope={slope}, intercept={intercept}")
    return sitk.Cast(sitk_img, sitk.sitkFloat32)*slope + intercept


def resample_image(image, full_target_spacing, axes_to_resample=(0, 1, 2), interpolator=sitk.sitkLinear):

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    new_spacing = original_spacing.copy()
    for axis in axes_to_resample:
        new_spacing[axis] = full_target_spacing[axis]
    new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize([int(sz) for sz in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def crop_center(image, target_size, axes_to_crop=(0,1,2)):
    """
    Crops a SimpleITK image to target_size in the specified axes.
    In other axes, the size is left unchanged, regardless of target_size.

    Args:
        image (sitk.Image): The input image.
        target_size (tuple): (X, Y, Z) target size.
        axes_to_crop (tuple): Which axes to crop. Other axes remain unchanged.

    Returns:
        sitk.Image: Cropped image.
    """
    original_size = image.GetSize()
    target_size_final = list(original_size)
    start_index = [0]*3

    for axis in axes_to_crop:
        tsize = target_size[axis]
        if tsize > original_size[axis]:
            raise ValueError(f"Target size {tsize} exceeds original size {original_size[axis]} on axis {axis}")
        target_size_final[axis] = tsize
        start_index[axis] = (original_size[axis] - tsize) // 2

    return sitk.RegionOfInterest(image, target_size_final, start_index)


def multilabel_rasterize_rtstruct_to_reference(rtstruct_path, reference_image):
    """
    Rasterizes RTSTRUCT contours into reference_image grid.
    Returns mask image and metadata.
    """
    # Load RTSTRUCT DICOM
    ds = pydicom.dcmread(rtstruct_path)

    # Build ROINumber -> ROIName mapping
    roi_number_to_name = {}
    for roi in ds.StructureSetROISequence:
        roi_number_to_name[roi.ROINumber] = roi.ROIName

    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()
    size = reference_image.GetSize()

    # Prepare empty label array
    mask_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

    metadata = []

    # For each ROIContourSequence
    for idx, roi_contour in enumerate(ds.ROIContourSequence, start=1):
        roi_number = roi_contour.ReferencedROINumber
        roi_name = roi_number_to_name[roi_number]
        print(f"Rasterizing ROI '{roi_name}'...")

        # Each ROI can have multiple contours
        for contour_item in tqdm(roi_contour.ContourSequence):
            data = contour_item.ContourData  # flat list of floats
            num_points = contour_item.NumberOfContourPoints

            # Reshape to Nx3
            coords = np.array(data).reshape((-1, 3))  # (N,3)

            # Transform to indices
            indices = [reference_image.TransformPhysicalPointToIndex(tuple(pt)) for pt in coords]

            # Split indices
            zs = [i[2] for i in indices]
            ys = [i[1] for i in indices]
            xs = [i[0] for i in indices]

            # For each unique Z slice, rasterize
            unique_z = set(zs)
            print("Unique Z indices in contour:", sorted(unique_z))
            for z in unique_z:
                slice_pts = [(x, y) for x, y, zz in zip(xs, ys, zs) if zz == z]
                print(f"Z={z}, number of points: {len(slice_pts)}")
                if len(slice_pts) < 3:
                    continue  # Not enough points

                rr, cc = polygon(
                    [p[1] for p in slice_pts],
                    [p[0] for p in slice_pts],
                    shape=mask_array.shape[1:]
                )
                mask_array[z, rr, cc] = idx

        metadata.append({
            "index": idx,
            "name": roi_name,
            "color": (roi_contour.ROIDisplayColor if hasattr(roi_contour, 'ROIDisplayColor') else [255,255,255])
        })

    # Convert to SimpleITK Image
    mask_img = sitk.GetImageFromArray(mask_array)
    mask_img.SetSpacing(spacing)
    mask_img.SetOrigin(origin)
    mask_img.SetDirection(direction)

    return mask_img, metadata


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
        # print(f"Rasterizing ROI '{roi_name}'...")

        # Initialize empty binary array for this ROI
        mask_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        for contour_item in roi_contour.ContourSequence: # to see progress: tqdm(roi_contour.ContourSequence)
            data = contour_item.ContourData
            coords = np.array(data).reshape((-1, 3))

            indices = [
                reference_image.TransformPhysicalPointToIndex(tuple(pt))
                for pt in coords
            ]

            zs = [i[2] for i in indices]
            ys = [i[1] for i in indices]
            xs = [i[0] for i in indices]

            unique_z = set(zs)
            # print("Unique Z indices in contour:", sorted(unique_z))
            for z in unique_z:
                slice_pts = [(x, y) for x, y, zz in zip(xs, ys, zs) if zz == z]
                # print(f"Z={z}, number of points: {len(slice_pts)}")
                if len(slice_pts) < 3:
                    # print(f"Skipping contour on Z={z} with {len(slice_pts)} points (likely invalid).")
                    continue

                rr, cc = polygon(
                    [p[1] for p in slice_pts],
                    [p[0] for p in slice_pts],
                    shape=mask_array.shape[1:]
                )
                mask_array[z, rr, cc] = 1

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


def extract_roi_color(rs_path, roi_name):
    ds = pydicom.dcmread(rs_path)
    structure_set_roi_sequence = {roi.ROINumber: roi.ROIName for roi in ds.StructureSetROISequence}
    roi_contour_sequence = {rc.ReferencedROINumber: rc for rc in ds.ROIContourSequence}
    for num, name in structure_set_roi_sequence.items():
        if name == roi_name:
            rc = roi_contour_sequence.get(num)
            if rc and hasattr(rc, "ROIDisplayColor"):
                return [v / 255.0 for v in rc.ROIDisplayColor]
    return [1.0, 1.0, 1.0]


def sitk_to_nifti_ras(sitk_image, output_path, description=None):
    """
    Convert SimpleITK image to NIfTI in RAS orientation using nibabel.
    """
    array = sitk.GetArrayFromImage(sitk_image)  # [Z, Y, X]
    array = array[:, ::-1, ::-1]  # flip LPS->RAS
    array = np.transpose(array, (2, 1, 0))  # [X,Y,Z]

    # Build affine
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()

    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = origin

    nii = nib.Nifti1Image(array, affine)
    nii.header['descrip'] = description
    nib.save(nii, output_path)


def save_4d_seg_nifti_ras(sitk_images, output_path, metadata):
    """
    Convert list of SimpleITK images to a 4D NIfTI in RAS orientation.
    """
    arrays = []
    for im in sitk_images.values():
        a = sitk.GetArrayFromImage(im)[:, ::-1, ::-1]  # Flip LPS->RAS
        arrays.append(a)

    stacked = np.stack(arrays, axis=0)          # (N, Z, Y, X)
    stacked = np.transpose(stacked, (3,2,1,0))  # (X,Y,Z,N)

    first_im = next(iter(sitk_images.values()))
    spacing = first_im.GetSpacing()
    origin = first_im.GetOrigin()


    affine = np.eye(4)
    affine[0,0] = spacing[0]
    affine[1,1] = spacing[1]
    affine[2,2] = spacing[2]
    affine[:3,3] = origin

    json_filename = output_path.parent / "seg_map_labels.json"
    with open(json_filename, "w") as f:
        json.dump(metadata, f, indent=2)

    nii = nib.Nifti1Image(stacked, affine)
    nii.header['descrip'] = f"Segmentation labels saved in {json_filename.name}"
    nib.save(nii, str(output_path))


def plot_orthogonal_slices_with_overlay(
    base_image,
    overlay_image=None,
    alpha=0.5,
    title="",
    overlay_cmap="jet",
    base_cmap="gray",
    vmin_base=None,
    vmax_base=None,
    vmin_overlay=None,
    vmax_overlay=None,
    slice_point=None,
):
    """
    Plots axial, coronal, sagittal slices with optional overlay and physical axes.

    Args:
        base_image (SimpleITK.Image): Base image (e.g. CT).
        overlay_image (SimpleITK.Image or None): Optional overlay (e.g. RD or seg).
        alpha (float): Transparency of overlay.
        title (str): Plot title.
        overlay_cmap (str): Colormap for overlay.
        base_cmap (str): Colormap for base image.
        vmin_base (float or None): Min intensity for base image display.
        vmax_base (float or None): Max intensity for base image display.
        vmin_overlay (float or None): Min intensity for overlay display.
        vmax_overlay (float or None): Max intensity for overlay display.
    """
    base_array = sitk.GetArrayFromImage(base_image)  # [Z, Y, X]
    spacing = base_image.GetSpacing()  # (X, Y, Z)
    origin = base_image.GetOrigin()
    direction = base_image.GetDirection()

    # Determine orientation labels
    orientation_labels = determine_orientation_labels(direction)
    # print(f"Orientation per axis (i,j,k): {orientation_labels}")

    if overlay_image is not None:
        overlay_array = sitk.GetArrayFromImage(overlay_image)
        assert overlay_array.shape == base_array.shape, "Overlay must match base image shape"
        is_binary = np.all(np.isin(overlay_array, [0, 1]))
    else:
        overlay_array = None

    if slice_point is not None:
        x, y, z = slice_point
    else:
        z, y, x = [s // 2 for s in base_array.shape]

    # print("Direction matrix:", base_image.GetDirection())

    slices = {
        "Axial": (base_array[z, :, :], overlay_array[z, :, :] if overlay_array is not None else None,
                  spacing[0], spacing[1], origin[0], origin[1],
                  orientation_labels[0], orientation_labels[1]),  # X, Y
        "Coronal": (base_array[:, y, :], overlay_array[:, y, :] if overlay_array is not None else None,
                    spacing[0], spacing[2], origin[0], origin[2],
                    orientation_labels[0], orientation_labels[2]),  # X, Z
        "Sagittal": (base_array[:, :, x], overlay_array[:, :, x] if overlay_array is not None else None,
                     spacing[1], spacing[2], origin[1], origin[2],
                     orientation_labels[1], orientation_labels[2])  # Y, Z
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (label, (base_slice, overlay_slice, dx, dy, x0, y0, xlabel, ylabel)) in zip(axes, slices.items()):
        extent = [x0, x0 + dx * base_slice.shape[1],
                  y0 + dy * base_slice.shape[0], y0]  # [xmin, xmax, ymin, ymax]
        ax.imshow(
            base_slice,
            cmap=base_cmap,
            extent=extent,
            origin="lower",
            vmin=vmin_base,
            vmax=vmax_base
        )
        if overlay_slice is not None:
            if is_binary:
                # Mask zeros so they are transparent
                overlay_masked = np.ma.masked_where(overlay_slice == 0, overlay_slice)
                # vmin/vmax set to 1 so only 1s are colored
                vmin_ = 1
                vmax_ = 1
            else:
                overlay_masked = overlay_slice
                vmin_ = vmin_overlay
                vmax_ = vmax_overlay

            ax.imshow(
                overlay_masked,
                cmap=overlay_cmap,
                alpha=alpha,
                extent=extent,
                origin="lower",
                vmin=vmin_,
                vmax=vmax_
            )

        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def determine_orientation_labels(direction):
    """
    Given a 3x3 direction matrix (flattened), return the anatomical labels
    along each axis.
    """
    axes = []
    dir_matrix = np.array(direction).reshape(3,3)
    for col in dir_matrix.T:
        # Determine major axis
        max_idx = np.argmax(np.abs(col))
        sign = np.sign(col[max_idx])
        if max_idx == 0:
            axes.append("R→L" if sign > 0 else "L→R")
        elif max_idx == 1:
            axes.append("A→P" if sign > 0 else "P→A")
        else:
            axes.append("I→S" if sign > 0 else "S→I")
    return axes


def plot_orthogonal_slices_with_orientation(
    base_image,
    overlay_image=None,
    alpha=0.5,
    title="",
    overlay_cmap="jet",
    base_cmap="gray",
    vmin_base=None,
    vmax_base=None,
    vmin_overlay=None,
    vmax_overlay=None
):
    """
    Plots orthogonal slices using physical coordinates and correct anatomical labels.
    """
    base_array = sitk.GetArrayFromImage(base_image)
    spacing = base_image.GetSpacing()
    origin = base_image.GetOrigin()
    direction = base_image.GetDirection()

    # Determine orientation labels
    orientation_labels = determine_orientation_labels(direction)
    # print(f"Orientation per axis (i,j,k): {orientation_labels}")

    if overlay_image is not None:
        overlay_array = sitk.GetArrayFromImage(overlay_image)
        assert overlay_array.shape == base_array.shape, "Overlay must match base image shape"
    else:
        overlay_array = None

    z, y, x = [s // 2 for s in base_array.shape]

    # Each plane needs proper extent in physical coordinates
    slices = {
        "Axial": {
            "base": base_array[z,:,:],
            "overlay": overlay_array[z,:,:] if overlay_array is not None else None,
            "extent": [
                origin[0],
                origin[0] + spacing[0]*base_array.shape[2],
                origin[1],
                origin[1] + spacing[1]*base_array.shape[1]
            ],
            "xlabel": orientation_labels[0],
            "ylabel": orientation_labels[1]
        },
        "Coronal": {
            "base": base_array[:,y,:],
            "overlay": overlay_array[:,y,:] if overlay_array is not None else None,
            "extent": [
                origin[0],
                origin[0] + spacing[0]*base_array.shape[2],
                origin[2],
                origin[2] + spacing[2]*base_array.shape[0]
            ],
            "xlabel": orientation_labels[0],
            "ylabel": orientation_labels[2]
        },
        "Sagittal": {
            "base": base_array[:,:,x],
            "overlay": overlay_array[:,:,x] if overlay_array is not None else None,
            "extent": [
                origin[1],
                origin[1] + spacing[1]*base_array.shape[1],
                origin[2],
                origin[2] + spacing[2]*base_array.shape[0]
            ],
            "xlabel": orientation_labels[1],
            "ylabel": orientation_labels[2]
        }
    }

    fig, axes = plt.subplots(1,3,figsize=(18,6))

    for ax, (plane, data) in zip(axes, slices.items()):
        ax.imshow(
            data["base"],
            cmap=base_cmap,
            extent=data["extent"],
            origin="lower",
            vmin=vmin_base,
            vmax=vmax_base
        )
        if data["overlay"] is not None:
            ax.imshow(
                data["overlay"],
                cmap=overlay_cmap,
                alpha=alpha,
                extent=data["extent"],
                origin="lower",
                vmin=vmin_overlay,
                vmax=vmax_overlay
            )
        ax.set_title(plane)
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel"])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def check_multilabel_seg_map(seg_image):
    seg_array = sitk.GetArrayFromImage(seg_image)
    unique_values = np.unique(seg_array)
    print("Unique labels in segmentation:", unique_values)

    if np.all(np.equal(np.mod(unique_values, 1), 0)):
        print("✅ All labels are integers.")
    else:
        print("❌ Found non-integer labels!")

    if np.all(np.floor(seg_array) == seg_array):
        print("✅ All voxels are integers.")
    else:
        print("❌ Some voxels are not integers.")

    print("Segmentation dtype:", seg_array.dtype)
    if np.issubdtype(seg_array.dtype, np.integer):
        print("✅ Array data type is integer.")
    else:
        print("⚠️ Array data type is not integer—check contents.")


def assert_seg_map_is_binary(seg_image):
    """
    Assert that the segmentation image is binary (only 0 and 1).
    """
    seg_array = sitk.GetArrayFromImage(seg_image)
    unique_values = np.unique(seg_array)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise AssertionError(f"❌ Segmentation contains non-binary labels: {unique_values}")


def label_axis_from_physical_lps(axis_min, axis_max, axis_name):
    """
    Return label string for LPS axes.
    """
    if axis_name == "X":
        neg, pos = "R", "L"
    elif axis_name == "Y":
        neg, pos = "A", "P"
    elif axis_name == "Z":
        neg, pos = "I", "S"
    else:
        raise ValueError("Unknown axis name")

    if axis_max > axis_min:
        return f"{neg} ←→ {pos}"
    else:
        return f"{pos} ←→ {neg}"


os.environ.pop("QT_PLUGIN_PATH", None)
# import napari
#
# def napari_viewer(
#         base_image,
#         seg_image=None,
#         dose_image=None,
#         dose_window=(0, 70)
# ):
#     """
#     Launch an interactive napari viewer with:
#     - base_image: SimpleITK Image
#     - seg_image: optional segmentation SimpleITK Image
#     - dose_image: optional dose map SimpleITK Image
#     - dose_window: tuple (min,max) for dose display in Gy
#     """
#     # Convert base image to array
#     base_array = sitk.GetArrayFromImage(base_image)
#
#     # Create viewer
#     viewer = napari.Viewer()
#
#     # Add base image
#     viewer.add_image(
#         base_array,
#         name="Base Image",
#         colormap="gray",
#         blending="opaque",
#         contrast_limits=[np.min(base_array), np.max(base_array)],
#     )
#
#     # Add segmentation overlay if provided
#     if seg_image is not None:
#         seg_array = sitk.GetArrayFromImage(seg_image).astype(np.int32)
#         viewer.add_labels(
#             seg_array,
#             name="Segmentation",
#             opacity=0.5,
#         )
#
#     # Add dose map overlay if provided
#     if dose_image is not None:
#         dose_array = sitk.GetArrayFromImage(dose_image)
#         viewer.add_image(
#             dose_array,
#             name="Dose (Gy)",
#             colormap="hot",
#             blending="additive",
#             opacity=0.4,
#             contrast_limits=list(dose_window),
#         )
#
#     napari.run()


def find_central_voxel(segmentation):
    """
    Finds the central voxel index in a binary segmentation.

    Parameters:
        segmentation (sitk.Image or np.ndarray): Binary segmentation.

    Returns:
        tuple: Index coordinates of the central voxel.
            - For SimpleITK.Image: (x,y,z)
            - For NumPy array: (i,j,k) in array coordinates
    """
    if isinstance(segmentation, sitk.Image):
        # Sitk case
        array = sitk.GetArrayFromImage(segmentation)
        coords = np.argwhere(array)
        if coords.size == 0:
            raise ValueError("Segmentation map is empty (no foreground voxels).")
        centroid = coords.mean(axis=0)
        rounded = np.round(centroid).astype(int)
        sitk_index = tuple(reversed(rounded))  # Convert z,y,x -> x,y,z
        return sitk_index

    elif isinstance(segmentation, np.ndarray):
        # Numpy case
        coords = np.argwhere(segmentation)
        if coords.size == 0:
            raise ValueError("Segmentation map is empty (no foreground voxels).")
        centroid = coords.mean(axis=0)
        central_voxel = tuple(np.round(centroid).astype(int))
        return central_voxel

    else:
        raise TypeError("Input must be a SimpleITK.Image or a NumPy array.")


def get_rgb_colormap_from_name(structures, target_name):
    """
    Given a list of structures and a target name, returns a ListedColormap with the corresponding color.

    Args:
        structures (list): List of dicts with 'name' and 'color' keys.
        target_name (str): Name of the structure to find.

    Returns:
        ListedColormap: A colormap with the specified RGB color.
    """
    for s in structures:
        if s["name"] == target_name:
            rgb = [c / 255 for c in s["color"]]
            from matplotlib.colors import ListedColormap
            return ListedColormap([rgb])

    raise ValueError(f"Structure name '{target_name}' not found.")


def parse_folder_metadata(folder_path):
    # Get the last part of the path
    folder_name = os.path.basename(folder_path)
    # Split by underscore
    parts = folder_name.split('_')
    if len(parts) < 3:
        raise ValueError(f"Unexpected folder format: '{folder_name}'")

    patient_id = parts[0]
    treatment_type = parts[1]

    # Tumor location may include underscores if more than 1 word
    # e.g., 'Lt_Frontal' or 'Lt_Temporal_Lobe'
    tumor_location = '_'.join(parts[2:-1]).replace('_', ' ')

    date_str = parts[-1]
    # Convert to datetime
    date_obj = datetime.strptime(date_str, "%d%m%Y")
    date_formatted = date_obj.strftime("%Y-%m-%d")

    return {
        "patient_id": patient_id,
        "treatment_type": treatment_type,
        "tumor_location": tumor_location,
        "session_date": date_formatted
    }


def process_patient_session(dicom_dir, output_dir,
                            final_spacing=(1.0, 1.0, 1.0),
                            crop_size=(240, 240, 120)):

    treatment_data = parse_folder_metadata(dicom_dir)
    patient_id = treatment_data["patient_id"]
    print(f"\n📂 Processing patient: {patient_id}")

    ct_dir, ct_files, rd_file, rs_file = get_dicom_components(dicom_dir)

    # CT
    ct = convert_ct_to_hu_sitk_safe(load_sitk_volume_from_series(ct_files), ct_files[0])
    ct_vmin, ct_vmax = get_display_params(ct_files[0])
    ct_xy = resample_image(ct, full_target_spacing=final_spacing, axes_to_resample=(0, 1)) # interpolate x,y dims
    ct_crop = crop_center(ct_xy, crop_size, axes_to_crop=(0, 1)) # crop in x,y dims
    ct_final = resample_image(ct_crop, final_spacing, axes_to_resample=(2,)) # resample z dim
    # plot_orthogonal_slices_with_orientation(base_image=ct_final, title=f"CT, size: {ct_final.GetSize()}", vmin_base=ct_vmin,
    #                                         vmax_base=ct_vmax)

    # RD: Dose map
    rd = sitk.ReadImage(rd_file)
    rd_final = sitk.Resample(rd, referenceImage=ct_final, interpolator=sitk.sitkLinear, defaultPixelValue=0.0)  # resample and crop according to ct
    # plot_orthogonal_slices_with_overlay(base_image=rd_final, title=f"Dose, size: {rd_final.GetSize()}", base_cmap="jet")
    # plot_orthogonal_slices_with_overlay(ct_final, rd_final, alpha=0.4,
    #                                     title=f"Final CT + Dose, size: {ct_final.GetSize()}", vmin_base=ct_vmin,
    #                                     vmax_base=ct_vmax)

    # RS: rasterize and interpolate
    seg_masks, seg_metadata = rasterize_rtstruct_to_separate_images(rs_file, ct_crop)
    print(seg_metadata)

    resampled_masks = {}
    for ind, (roi_name, roi_mask) in enumerate(seg_masks.items(), start=1):
        # print(f"({ind}). {roi_name}")
        assert_seg_map_is_binary(roi_mask)
        cmap = get_rgb_colormap_from_name(seg_metadata, roi_name)
        red_cmap = ListedColormap(["red"])
        resampled = resample_image(roi_mask, final_spacing, axes_to_resample=(2,),
                                   interpolator=sitk.sitkNearestNeighbor)
        assert_seg_map_is_binary(resampled)
        plot_orthogonal_slices_with_overlay(ct_final, resampled, alpha=0.4, title=f"({ind}). Segmentation of {roi_name} after resample",
                                            vmin_base=ct_vmin, vmax_base=ct_vmax, overlay_cmap=red_cmap,
                                            slice_point=find_central_voxel(resampled))
        resampled_masks[roi_name] = resampled

    # Save outputs
    out_path = Path(output_dir) / patient_id / treatment_data["session_date"]
    description = f"{treatment_data['treatment_type']} | {treatment_data['tumor_location']}"
    out_path.mkdir(parents=True, exist_ok=True)
    sitk_to_nifti_ras(ct_final, out_path / "CT.nii.gz")
    sitk_to_nifti_ras(rd_final, out_path / "dose_map.nii.gz", description)
    save_4d_seg_nifti_ras(resampled_masks, out_path / "seg_map.nii.gz", seg_metadata)

    print(f"✅ Saved NIfTI and metadata to {out_path}")

    return ct_final, rd_final, resampled_masks, seg_metadata, ct_vmin, ct_vmax, out_path


# Run when script is executed
if __name__ == "__main__":
    # main()
    dicom_dir = "../DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"
    # dicom_dir = "/data/datasets/Ichilov_radiation_dataset/Radiation_plans_TABM/10295_FSR_Lt Frontal_04072022"
    output_dir = "../outputs"
    show = False

    # Run the pipeline
    ct, rd, seg, metadata, ct_vmin, ct_vmax, out_path = process_patient_session(
        dicom_dir=dicom_dir,
        output_dir=output_dir
    )

    # Optional visualization
    if show:
        nifti_files = sorted(out_path.glob("**/*.nii*"))
        for nifti_path in nifti_files:
            itk_image = itk.imread(nifti_path)
            viewer = itk.view(itk_image)

