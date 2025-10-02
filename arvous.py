import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import re
import SimpleITK as sitk
import numpy as np
import pydicom
import math
import time
import scipy.ndimage
import nibabel as nib
import json
import traceback
import glob
import subprocess
import shutil

matplotlib.use('TkAgg')

from datetime import datetime
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Dict, Optional, Tuple, OrderedDict, Any
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QIcon
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, CheckButtons
from skimage.measure import find_contours

SMOOTH_SIGMA_MM: float = 0.75

def plot_combined_plot(CT_data, Dose_data, ROIs_data):
    """
        Interactive viewer for CT, Dose, and ROI data in three orthogonal views
        (axial, coronal, sagittal).

        Features:
        ----------
        • Displays CT grayscale slices in three views.
        • Overlays dose distribution with adjustable min/max thresholds.
        • Provides checkboxes to select which ROIs to display.
        • Allows toggling between contour mode and filled mask mode for ROIs.
        • Interactive sliders to navigate through slices for each view.

        Parameters:
        -----------
        CT_data : dict
            Dictionary with keys:
              - "Volume": 3D numpy array [Z,Y,X] with CT values.
              - "Spacing": voxel spacing in mm (z, y, x).
              - "Position": origin coordinates (z, y, x).
        Dose_data : dict
            Dictionary with keys:
              - "Volume": 3D numpy array [Z,Y,X] with dose values (Gy).
              - "Spacing": voxel spacing in mm (z, y, x).
              - "Position": origin coordinates (z, y, x).
        ROIs_data : dict
            Dictionary with keys:
              - "ROIs": mapping {roi_id: {"Name": str, "Color": list|None, "Volume": 3D mask}}.
              - "Spacing": voxel spacing in mm (z, y, x).
              - "Position": origin coordinates (z, y, x).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.38, left=0.22)  # leave space for checkboxes

    views = ['axial', 'coronal', 'sagittal']
    get_ct = {
        'axial': lambda i: CT_data['Volume'][i, :, :],
        'coronal': lambda i: CT_data['Volume'][:, i, :],
        'sagittal': lambda i: CT_data['Volume'][:, :, i]
    }
    get_dose = {
        'axial': lambda i: Dose_data['Volume'][i, :, :],
        'coronal': lambda i: Dose_data['Volume'][:, i, :],
        'sagittal': lambda i: Dose_data['Volume'][:, :, i]
    }

    shapes = {
        'axial': CT_data['Volume'].shape[0],
        'coronal': CT_data['Volume'].shape[1],
        'sagittal': CT_data['Volume'].shape[2]
    }
    origins = {'axial': 'upper', 'coronal': 'lower', 'sagittal': 'lower'}

    imgs_ct, imgs_dose, sliders = [], [], []
    roi_patches_by_axis = [[] for _ in range(3)]
    roi_masks = [None, None, None]

    dose_global_max = float(np.max(Dose_data['Volume']))
    dose_min, dose_max = 0.0, dose_global_max

    cmap_ct = plt.cm.gray.copy()
    cmap_ct.set_bad(color='black')

    show_mode = {'mode': 'contour'}
    view_extents = {}
    selected_rois = set()

    # ROI mask colormap (still needed for mask display)
    unique_rois = list(ROIs_data.get('ROIs', {}).keys())
    color_list = [plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)] for i in range(len(unique_rois))]
    cmap_roi = mcolors.ListedColormap([(0, 0, 0, 0)] + list(color_list))
    norm_roi = mcolors.BoundaryNorm(
        boundaries=np.arange(len(unique_rois) + 2) - 0.5,
        ncolors=len(unique_rois) + 1
    )

    # Precompute pixel -> mm extents
    for view in views:
        if view == 'axial':
            extent = [0, CT_data["Spacing"][2] * shapes['sagittal'],
                      0, CT_data["Spacing"][1] * shapes['coronal']]
        elif view == 'coronal':
            extent = [0, CT_data["Spacing"][2] * shapes['sagittal'],
                      0, CT_data["Spacing"][0] * shapes['axial']]
        else:
            extent = [0, CT_data["Spacing"][1] * shapes['coronal'],
                      0, CT_data["Spacing"][0] * shapes['axial']]
        view_extents[view] = extent

    def rgba_from_dose(dose_slice, min_val, max_val):
        norm_color = np.clip(dose_slice / max(dose_global_max, 1e-9), 0, 1)
        rgba = plt.cm.jet(norm_color)
        alpha = np.zeros_like(dose_slice, dtype=float)
        if max_val > min_val:
            in_range = (dose_slice >= min_val) & (dose_slice <= max_val)
            alpha[in_range] = (dose_slice[in_range] - min_val) / (max_val - min_val)
        rgba[..., 3] = alpha
        return rgba

    # Initial draw
    for i, view in enumerate(views):
        ax = axes[i]
        ct_data0 = get_ct[view](0)
        dose0 = get_dose[view](0)

        ct_img = ax.imshow(ct_data0, cmap=cmap_ct, vmin=900, vmax=1200,
                           origin=origins[view], extent=view_extents[view])
        dose_img = ax.imshow(rgba_from_dose(dose0, dose_min, dose_max),
                             origin=origins[view], extent=view_extents[view])

        ax.set_title(f"{view.capitalize()} Slice 0")
        ax.axis('off')
        imgs_ct.append(ct_img)
        imgs_dose.append(dose_img)

    # Dose colorbar
    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_clim(0, dose_global_max)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Dose (Gy)")

    # Sliders
    slider_axes = [
        plt.axes([0.30, 0.25, 0.5, 0.02]),
        plt.axes([0.30, 0.20, 0.5, 0.02]),
        plt.axes([0.30, 0.15, 0.5, 0.02])
    ]
    dose_min_ax = plt.axes([0.30, 0.08, 0.5, 0.02])
    dose_max_ax = plt.axes([0.30, 0.03, 0.5, 0.02])
    button_ax = plt.axes([0.50, 0.30, 0.12, 0.04])

    dose_min_slider = Slider(dose_min_ax, 'Min Dose', 0.0, dose_global_max, valinit=dose_min, valstep=0.001)
    dose_max_slider = Slider(dose_max_ax, 'Max Dose', 0.0, dose_global_max, valinit=dose_max, valstep=0.001)
    toggle_button = Button(button_ax, 'Show Mask')

    # Checkbox for ROI selection
    checkbox_ax = plt.axes([0.02, 0.15, 0.15, 0.7])
    checkbox_labels = ["All"] + [ROIs_data['ROIs'][roi]['Name'] for roi in unique_rois]
    checkbox_states = [False] * len(checkbox_labels)
    checkbox = CheckButtons(checkbox_ax, labels=checkbox_labels, actives=checkbox_states)

    # Set label colors: black for "All", ROI colors for others
    checkbox.labels[0].set_color("black")
    for i, color in enumerate(color_list, start=1):
        checkbox.labels[i].set_color(color)

    def refresh_all(*_):
        min_val = dose_min_slider.val
        max_val = dose_max_slider.val

        for i, view in enumerate(views):
            idx = int(sliders[i].val)

            ct_slice = get_ct[view](idx)
            dose_slice = get_dose[view](idx)
            ct_slice = np.ma.masked_where(dose_slice <= 0, ct_slice)


            imgs_ct[i].set_data(ct_slice)
            imgs_dose[i].set_data(rgba_from_dose(dose_slice, min_val, max_val))

            for line in roi_patches_by_axis[i]:
                line.remove()
            roi_patches_by_axis[i].clear()
            if roi_masks[i] is not None:
                roi_masks[i].remove()
                roi_masks[i] = None

            if unique_rois:
                if show_mode['mode'] == 'mask':
                    combined = np.zeros_like(dose_slice, dtype=int)
                    roi_sizes = []
                    for j, roi_id in enumerate(unique_rois):
                        if j not in selected_rois:
                            continue
                        vol = ROIs_data['ROIs'][roi_id]['Volume']
                        if view == 'axial':
                            slice_mask = vol[idx, :, :]
                        elif view == 'coronal':
                            slice_mask = vol[:, idx, :]
                        else:
                            slice_mask = vol[:, :, idx]
                        roi_sizes.append((j, roi_id, np.count_nonzero(slice_mask)))

                    roi_sizes.sort(key=lambda x: x[2], reverse=True)

                    for j, roi_id, _size in roi_sizes:
                        vol = ROIs_data['ROIs'][roi_id]['Volume']
                        if view == 'axial':
                            slice_mask = vol[idx, :, :]
                        elif view == 'coronal':
                            slice_mask = vol[:, idx, :]
                        else:
                            slice_mask = vol[:, :, idx]
                        combined[slice_mask.astype(bool)] = j + 1

                    combined = np.ma.masked_where(dose_slice <= 0, combined)
                    roi_masks[i] = axes[i].imshow(
                        combined, cmap=cmap_roi, norm=norm_roi, alpha=0.5,
                        origin=origins[view], extent=view_extents[view]
                    )

                else:  # contour mode
                    if view == 'axial':
                        spacing_x = CT_data["Spacing"][2]
                        spacing_y = CT_data["Spacing"][1]
                    elif view == 'coronal':
                        spacing_x = CT_data["Spacing"][2]
                        spacing_y = CT_data["Spacing"][0]
                    else:
                        spacing_x = CT_data["Spacing"][1]
                        spacing_y = CT_data["Spacing"][0]

                    ext = view_extents[view]
                    for j, roi_id in enumerate(unique_rois):
                        if j not in selected_rois:
                            continue
                        vol = ROIs_data['ROIs'][roi_id]['Volume']
                        roi_slice = vol[idx, :, :] if view == 'axial' else \
                                    (vol[:, idx, :] if view == 'coronal' else vol[:, :, idx])

                        if not np.any(roi_slice):
                            continue

                        for contour in find_contours(roi_slice.astype(float), level=0.5):
                            y_mm = contour[:, 0] * spacing_y
                            x_mm = contour[:, 1] * spacing_x
                            if origins[view] == 'upper':
                                y_mm = ext[3] - y_mm
                            line, = axes[i].plot(x_mm, y_mm, color=color_list[j], linewidth=1.5)
                            roi_patches_by_axis[i].append(line)

            axes[i].set_title(f"{view.capitalize()} Slice {idx}")

        fig.canvas.draw_idle()

    def toggle_display(_event):
        if show_mode['mode'] == 'contour':
            show_mode['mode'] = 'mask'
            toggle_button.label.set_text('Show Contour')
        else:
            show_mode['mode'] = 'contour'
            toggle_button.label.set_text('Show Mask')
        refresh_all()

    updating_checkboxes = False
    def on_checkbox_clicked(label):
        nonlocal selected_rois, updating_checkboxes
        if updating_checkboxes:
            return

        if label == "All":
            updating_checkboxes = True
            if len(selected_rois) == len(unique_rois):
                selected_rois.clear()
                for i in range(1, len(checkbox_labels)):
                    if checkbox.get_status()[i]:
                        checkbox.set_active(i)
            else:
                selected_rois = set(range(len(unique_rois)))
                for i in range(1, len(checkbox_labels)):
                    if not checkbox.get_status()[i]:
                        checkbox.set_active(i)
            updating_checkboxes = False
        else:
            idx = checkbox_labels.index(label) - 1
            if idx in selected_rois:
                selected_rois.remove(idx)
            else:
                selected_rois.add(idx)

            updating_checkboxes = True
            all_checked = (len(selected_rois) == len(unique_rois))
            if checkbox.get_status()[0] != all_checked:
                checkbox.set_active(0)
            updating_checkboxes = False

        refresh_all()

    toggle_button.on_clicked(toggle_display)
    dose_min_slider.on_changed(refresh_all)
    dose_max_slider.on_changed(refresh_all)
    checkbox.on_clicked(on_checkbox_clicked)

    for i, view in enumerate(views):
        sld = Slider(slider_axes[i], f'{view.capitalize()} Slice', 0, shapes[view] - 1, valinit=0, valstep=1)
        sld.on_changed(lambda _val, i=i: refresh_all())
        sliders.append(sld)

    plt.show()

def load_nifti_volume(path):
    """
    Load a NIfTI medical image (.nii or .nii.gz) and convert it into a
    consistent Python dictionary format.

    Parameters
    ----------
    path : str or pathlib.Path
        File path to the NIfTI file on disk.

    Returns
    -------
    dict
        A dictionary containing:
          • "Volume" : numpy.ndarray
              3D array [Z, Y, X] with voxel values (float32).
          • "Spacing" : numpy.ndarray
              Physical size of each voxel in millimeters [z, y, x].
          • "Position" : numpy.ndarray
              World-coordinate origin of the first voxel (in mm), ordered [z, y, x].

    Notes
    -----
    • This function standardizes all outputs to the internal convention
      used in this project: arrays are indexed as [Z, Y, X].
    • NIfTI spacing is originally stored as (x, y, z) in the header;
      it is reversed here to (z, y, x) for consistency.
    • The affine matrix encodes the spatial position of the volume;
      we extract its translation part (x, y, z) and reverse it to (z, y, x).
    • All voxel values are cast to float32 to save memory and ensure
      compatibility with downstream computations.
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
       Load preprocessed CT, Dose, and ROI volumes from NIfTI files in a directory.

       This function expects that the given folder contains:
         • CT_volume.nii.gz
         • Dose_volume.nii.gz
         • ROIs/roi_metadata.json         (optional metadata)
         • ROIs/ROI_*_volume.nii.gz       (one file per ROI mask)

       Parameters
       ----------
       files_path : str or pathlib.Path
           Path to the directory that holds the preprocessed volumes.

       Returns
       -------
       CT_data : dict
           Dictionary from `load_nifti_volume` with CT volume, spacing, and position.
       Dose_data : dict
           Dictionary from `load_nifti_volume` with Dose volume, spacing, and position.
       ROIs_data : dict
           Dictionary with ROI information:
             • "ROIs": mapping {roi_number: {"Name", "Color", "Volume"}}
                 - "Name": ROI name (from metadata, or fallback ROI_x)
                 - "Color": Optional RGB color (from metadata, else None)
                 - "Volume": 3D binary mask [Z,Y,X] (bool)
             • "Spacing": voxel spacing (z,y,x) inherited from CT
             • "Position": origin coordinates (z,y,x) inherited from CT

       Notes
       -----
       • The function uses `load_nifti_volume` to ensure consistent [Z,Y,X] arrays.
       • ROI volumes are cast to boolean masks (True for ROI voxels).
       • If `roi_metadata.json` is missing, ROI names default to "ROI_<number>".
       • Malformed ROI filenames (not matching `ROI_<number>_...`) are skipped.
       • Spacing/position for all ROIs are assumed identical to CT.
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

def save_volume_as_nifti(volume, spacing, output_path, affine_origin=(0, 0, 0)):
    """
    Save a 3D NumPy volume to disk as a NIfTI (.nii.gz) file.

    Parameters
    ----------
    volume : np.ndarray
        3D array [Z, Y, X] containing the image data to be saved.
    spacing : tuple or list of float
        Physical voxel spacing in millimeters, ordered (z, y, x).
    output_path : str or pathlib.Path
        File path where the NIfTI file will be saved (typically ending with .nii.gz).
    affine_origin : tuple of float, optional
        World-coordinate origin of the volume (in mm), ordered (z, y, x).
        Defaults to (0, 0, 0).

    Notes
    -----
    • Internally, the affine transformation matrix is built as a diagonal
      matrix with voxel spacings on the diagonal, reordered into (x, y, z).
    • The origin is inserted into the affine matrix as translation values.
    • The saved file can be loaded later with :func:`load_nifti_volume`
      to recover the same volume, spacing, and position (within rounding).
    • Data is always cast to float32 to ensure compatibility and reduce file size.
    """
    affine = np.diag(list(spacing)[::-1] + [1])
    affine[:3, 3] = affine_origin[::-1]  # Put origin in correct position (X, Y, Z)

    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nifti_img, output_path)

def save_volumes(CT_data, Dose_data, ROIs_data, output_path):
    """
        Save CT, Dose, and ROI volumes to NIfTI (.nii.gz) files, along with ROI metadata.

        This function creates a standardized folder structure inside `output_path`:
          • CT_volume.nii.gz
          • Dose_volume.nii.gz
          • ROIs/
              ├── ROI_<number>_volume.nii.gz   (one file per ROI mask)
              └── roi_metadata.json            (ROI names and colors)

        Parameters
        ----------
        CT_data : dict
            Dictionary containing CT volume, spacing, and position.
            Typically produced by `load_nifti_volume`.
        Dose_data : dict
            Dictionary containing Dose volume, spacing, and position.
        ROIs_data : dict
            Dictionary containing ROI data with keys:
              • "ROIs": mapping {roi_number: {"Name", "Color", "Volume"}}
                  - "Name": ROI name (str, defaults to "ROI_<number>")
                  - "Color": Optional list of 3 ints [R, G, B], or None
                  - "Volume": 3D binary mask [Z,Y,X] (bool or int)
              • "Spacing": voxel spacing (z,y,x)
              • "Position": origin coordinates (z,y,x)
        output_path : str or pathlib.Path
            Directory where all volumes and metadata will be saved.

        Notes
        -----
        • Volumes are saved with :func:`save_volume_as_nifti`, ensuring consistent
          affine construction from spacing and position.
        • ROI masks are saved as individual files under `ROIs/`.
        • ROI metadata (name, color) is collected and written to JSON
          (`roi_metadata.json`) in the ROIs folder.
        • Colors are validated and converted to standard Python int lists before saving.
        • The function automatically creates the `ROIs/` folder if it does not exist.
    """
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

def match_Dose_to_CT(Dose_data, CT_data, scales, offsets):
    """
    Resample and align the Dose volume so that it matches the CT volume grid.

    Parameters
    ----------
    Dose_data : dict
        Dictionary containing the dose volume and metadata, typically from
        :func:`load_nifti_volume`. Must contain key:
          • "Volume": 3D numpy array [Z,Y,X] (float32 Gy).
    CT_data : dict
        Dictionary containing the CT volume and metadata, typically from
        :func:`load_nifti_volume`. Must contain key:
          • "Volume": 3D numpy array [Z,Y,X] (float32 HU).
    scales : tuple of float (scale_z, scale_y, scale_x)
        Scaling factors between Dose and CT voxel dimensions.
        Values > 1 shrink the dose grid; values < 1 expand it.
    offsets : tuple of float (offset_z, offset_y, offset_x)
        Translation offsets (in voxels) applied after scaling,
        mapping Dose coordinates into CT coordinates.

    Returns
    -------
    numpy.ndarray
        Resampled 3D numpy array [Z,Y,X] of the dose volume aligned
        to the CT grid (linear interpolation).

    Notes
    -----
    • The transformation is performed using `scipy.ndimage.affine_transform`,
      which maps output coordinates (CT space) back into input coordinates (Dose space).
    • The `affine_matrix` rescales the Dose axes to match CT voxel dimensions.
    • The `offset` shifts the Dose volume so that anatomical landmarks align.
    • Interpolation order is set to 1 (linear) to balance accuracy and speed.
    • The aligned dose replaces `Dose_data["Volume"]` in place.
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
    Resample and align the Dose volume so it matches the CT grid in both
    dimensions and spatial origin.

    Parameters
    ----------
    Dose_data : dict
        Dictionary containing the dose volume and metadata, typically from
        :func:`load_nifti_volume`. Expected keys:
          • "Volume": 3D numpy array [Z,Y,X] (float32 Gy)
          • "Spacing": voxel spacing (z,y,x) in mm
          • "Position": origin coordinates (z,y,x) in mm
    CT_data : dict
        Dictionary containing the CT volume and metadata, typically from
        :func:`load_nifti_volume`. Expected keys:
          • "Volume": 3D numpy array [Z,Y,X] (float32 HU)
          • "Spacing": voxel spacing (z,y,x) in mm
          • "Position": origin coordinates (z,y,x) in mm

    Returns
    -------
    dict
        Updated `Dose_data` dictionary where:
          • "Volume" has been resampled and aligned to CT grid
          • "Spacing" is set to CT spacing
          • "Position" is set to CT origin

    Notes
    -----
    • Computes scaling factors by dividing Dose voxel spacing by CT voxel spacing.
    • Computes voxel offsets by comparing the physical positions of Dose and CT origins.
    • Calls :func:`match_Dose_to_CT` to apply an affine transformation
      (scaling + translation) so the Dose matches CT geometry.
    • The function **modifies Dose_data in place** and also returns it
      for convenience.
    • After this step, CT and Dose arrays are guaranteed to have the
      same shape, spacing, and origin, which is required for DVH computation.
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

def voxelize_convex_hull(points_voxel, volume_shape):
    """
        Create a 3D binary mask by voxelizing the convex hull of a set of points.

        Parameters
        ----------
        points_voxel : (N, 3) array_like
            Array of 3D points in voxel coordinates [z, y, x].
            These points define the region of interest to be enclosed.
        volume_shape : tuple of int (Z, Y, X)
            Shape of the target 3D volume into which the convex hull will be voxelized.

        Returns
        -------
        filled : numpy.ndarray (bool)
            3D binary mask [Z, Y, X] where voxels inside the convex hull are True,
            and voxels outside are False.

        Notes
        -----
        • The convex hull is computed using `scipy.spatial.ConvexHull`.
        • A Delaunay triangulation of the hull vertices is built to efficiently
          test whether each voxel lies inside the convex hull.
        • All voxel coordinates within the bounding box of `volume_shape`
          are tested against the Delaunay simplex.
        • The output mask can be used for rasterizing structures (e.g., ROIs)
          into voxel space from point-based definitions.
        • Complexity grows with the number of voxels in `volume_shape`;
          for large grids this step may be computationally expensive.
    """
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
    Extract 3D ROI masks from an RT Structure Set (RTSTRUCT) DICOM file and
    voxelize them onto the CT grid.

    Parameters
    ----------
    RS_data : pydicom.Dataset
        RT Structure Set (RS) DICOM dataset containing ROI definitions
        (StructureSetROISequence and ROIContourSequence).
    CT_data : dict
        Dictionary containing the CT volume and geometry, typically from
        :func:`load_nifti_volume`. Must include:
          • "Volume": 3D CT array [Z,Y,X]
          • "Spacing": voxel spacing (z,y,x) in mm
          • "Position": origin coordinates (z,y,x) in mm
          • "Slices": mapping from SOPInstanceUID → {"Z Index": int}

    Returns
    -------
    ROIs_data : dict
        Dictionary of ROI information with structure:
          • "ROIs": mapping {roi_number: {"Name", "Color", "Volume"}}
              - "Name": ROI name from RS (string)
              - "Color": ROI display color (list of 3 ints [R,G,B] or None)
              - "Volume": 3D binary mask [Z,Y,X] (bool), True = inside ROI
        Each ROI volume is voxelized from its DICOM contour data.

    Notes
    -----
    • Each ROI contour is defined in patient coordinates (x,y,z). These
      are converted to voxel indices using CT origin and spacing.
    • All contour points for an ROI are collected, and a convex hull
      is computed via :func:`voxelize_convex_hull` to fill the volume.
    • ROI volumes are stored as **binary masks**:
        - True (1) → voxel inside ROI
        - False (0) → voxel outside ROI
    • Colors are taken from the DICOM `ROIDisplayColor` field if present.
    • Multiple slices are combined by stacking voxel coordinates before hull fill.
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

def extract_dose_data(RD_data):
    """
    Extract dose grid information from a Radiotherapy Dose (RD) DICOM dataset.

    Parameters
    ----------
    RD_data : pydicom.Dataset
        The RT Dose DICOM object, typically read with `pydicom.dcmread().

    Returns
    -------
    dict
        Dictionary with the following keys:
          • "Position": tuple of float (z,y,x)
              Physical coordinates (mm) of the first voxel in patient space,
              reordered to match internal [Z,Y,X] convention.
          • "Spacing": tuple of float (z,y,x)
              Voxel spacing in mm, derived from PixelSpacing (x,y) and
              GridFrameOffsetVector (z).
          • "Volume": numpy.ndarray [Z,Y,X]
              3D dose array in units of Gray (Gy), scaled by `DoseGridScaling`.

    Notes
    -----
    • DICOM stores coordinates as (x,y,z); this function reorders them to (z,y,x)
      for consistency with the rest of the pipeline.
    • `PixelSpacing` provides in-plane resolution (y,x) in mm.
    • `GridFrameOffsetVector` gives slice positions along z; spacing is inferred
      from consecutive offsets.
    • The raw pixel array is multiplied by `DoseGridScaling` to convert stored
      integer values into absolute dose in Gray.
    • The dataset’s `DoseUnits` attribute is assumed to be GY.
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

def read_dicom_rd_file(file_path):
    """
    Read a DICOM Radiotherapy Dose (RD) file and validate its modality.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the DICOM file to be read.

    Returns
    -------
    pydicom.Dataset
        The loaded DICOM dataset if the file is a valid RT Dose object
        (Modality == "RTDOSE").

    Raises
    ------
    ValueError
        If the file exists but its `Modality` is not "RTDOSE".

    Notes
    -----
    • Uses `pydicom.dcmread()` to parse the file.
    • The `Modality` attribute is checked to confirm that the file is indeed
      a Radiotherapy Dose (RD) object.
    • This function only validates file type; dose grid extraction should be
      done separately (e.g., with :func:`extract_dose_data`).
    """
    rd = pydicom.dcmread(file_path)
    if rd.Modality == 'RTDOSE':
        return rd
    else:
        raise ValueError("The provided file is not a Radiation Dose (RD) DICOM file.")

def read_dicom_rs_file(file_path):
    """
    Read a DICOM Radiotherapy Structure Set (RS) file and validate its modality.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the DICOM file to be read.

    Returns
    -------
    pydicom.Dataset
        The loaded DICOM dataset if the file is a valid RT Structure Set
        (Modality == "RTSTRUCT").

    Raises
    ------
    ValueError
        If the file exists but its `Modality` is not "RTSTRUCT".

    Notes
    -----
    • Uses `pydicom.dcmread()` to parse the file.
    • The `Modality` attribute is checked to confirm that the file is indeed
      a Radiotherapy Structure Set (RS).
    • This function only validates and loads the dataset. To extract ROI volumes
      and metadata from the RS file, use :func:`extract_ROIs_data`.
    """
    rs = pydicom.dcmread(file_path)
    if rs.Modality == 'RTSTRUCT':
        return rs
    else:
        raise ValueError("The provided file is not an RT Structure Set (RTSTRUCT) DICOM file.")


def read_dicom_rp_file(file_path):
    """
        Read a DICOM Radiotherapy Plan (RP) file and validate its modality.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the DICOM file to be read.

        Returns
        -------
        pydicom.Dataset
            The loaded DICOM dataset if the file is a valid RT Plan
            (Modality == "RTPLAN").

        Raises
        ------
        ValueError
            If the file exists but its `Modality` is not "RTPLAN".

        Notes
        -----
        • Uses `pydicom.dcmread()` to parse the file.
        • The `Modality` attribute is checked to confirm that the file is indeed
          a Radiotherapy Plan (RP).
        • This function only validates and loads the dataset. Downstream processing
          of beams, fractions, or prescriptions should be implemented separately.
    """
    rp = pydicom.dcmread(file_path)
    if rp.Modality == 'RTPLAN':
        return rp
    else:
        raise ValueError("The provided file is not an RP (RTPLAN) DICOM file.")

def load_RT_data(files_path):
    """
    Load core DICOM Radiotherapy (RT) datasets from a directory.

    This function searches for and loads three essential RT DICOM files:
      • RD (Radiotherapy Dose)   – 3D dose distribution grid.
      • RS (RT Structure Set)    – Contours defining target volumes (PTVs, CTVs)
                                    and organs-at-risk (OARs).
      • RP (RT Plan)             – Treatment plan including beams, fractions,
                                    and prescriptions.

    Parameters
    ----------
    files_path : str or pathlib.Path
        Path to the directory containing the DICOM RT files.

    Returns
    -------
    tuple
        (RD_data, RS_data, RP_data), where each element is a pydicom.Dataset:
          • RD_data : DICOM dataset of type "RTDOSE"
          • RS_data : DICOM dataset of type "RTSTRUCT"
          • RP_data : DICOM dataset of type "RTPLAN"

    Raises
    ------
    FileNotFoundError
        If any of the required files (RD, RS, RP) are not found in the directory.

    Notes
    -----
    • Internally uses `find_file_with_prefix` to locate files by DICOM prefix.
    • Each dataset is validated for its `Modality` using the respective reader:
        - :func:`read_dicom_rd_file`
        - :func:`read_dicom_rs_file`
        - :func:`read_dicom_rp_file`
    • Only one file of each type is expected in the directory. If multiple are
      present, the first match will be used.
    • After loading, the datasets should be passed to higher-level extractors:
        - :func:`extract_dose_data` for RD
        - :func:`extract_ROIs_data` for RS
        - custom RP parsing for plan details
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

def create_CT_volume(CT_data):
    """
        Construct a 3D CT volume from individual CT slices.

        Parameters
        ----------
        CT_data : dict
            Dictionary containing CT slice information. Must include:
              • "Slices": dict mapping slice identifiers → {
                    "Image": 2D numpy array (pixels),
                    "Position": tuple (z,y,x) position of slice origin in mm
                }

        Returns
        -------
        CT_volume : numpy.ndarray
            3D numpy array [Z,Y,X] representing the reconstructed CT volume,
            where Z corresponds to slice index.

        Notes
        -----
        • CT slices are sorted along the z-axis using their spatial position
          (from `Position[0]` in mm).
        • The function assigns a `"Z Index"` field to each slice in `CT_data["Slices"]`,
          indicating its slice index in the final 3D volume.
        • The output volume is assembled by stacking 2D images in slice order,
          producing a float32 array.
        • All slices are assumed to have the same in-plane dimensions (Y,X).
        """
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
    Load CT image series from DICOM files in a directory and assemble them into a 3D volume.

    Parameters
    ----------
    files_path : str or pathlib.Path
        Path to the directory containing CT DICOM slices (files with "CT" prefix).

    Returns
    -------
    CT_data : dict
        Dictionary containing CT geometry and voxel data:
          • "Slices": dict mapping {SOPInstanceUID → {
                "Position": tuple (z,y,x) in mm, slice origin in patient space,
                "Spacing": tuple (y,x) pixel spacing in mm,
                "Image": 2D numpy array of the CT slice
            }}
          • "Position": tuple (z,y,x) position of the lowest-z slice (mm).
          • "Spacing": tuple (z,y,x) voxel spacing in mm.
              - z-spacing is computed as the mean difference between consecutive slice positions.
              - (y,x) spacing is taken from DICOM `PixelSpacing`.
          • "Volume": 3D numpy array [Z,Y,X], float32 CT voxel values (HU).

    Notes
    -----
    • Slice identifiers are taken from DICOM `SOPInstanceUID`.
    • Slice positions are reordered to (z,y,x) to match internal convention.
    • Pixel spacing is reordered to (y,x). Combined with z-spacing, this yields full (z,y,x) spacing.
    • The final 3D CT volume is assembled using :func:`create_CT_volume`, which sorts slices
      along the z-axis and stacks them into [Z,Y,X].
    • CT values are stored as raw DICOM pixel values; HU scaling (RescaleSlope/Intercept)
      should be applied beforehand if needed.
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

class PreprocessWorker(QThread):
    """
        Background worker for preprocessing radiotherapy patient data.

        This worker is intended to be run inside a PyQt application. It loads DICOM
        CT, dose, and structure data for a single patient, optionally resamples them
        to a new voxel grid, aligns dose and ROIs to the CT, and saves standardized
        NIfTI outputs.

        Signals
        -------
        finished : pyqtSignal(dict)
            Emitted when processing ends. Dictionary includes:
              • "ok": bool – True if successful, False if error
              • "msg": str – status or error message
              • "out_dir": str – output directory path ("" if failed)
        log : pyqtSignal(str)
            Emitted with progress messages for logging to the GUI.

        Parameters
        ----------
        patient_data : str or pathlib.Path
            Path to directory containing the patient’s DICOM files.
        patient_output_dir : str or pathlib.Path
            Directory where preprocessed NIfTI outputs will be saved.
        resample_type : str
            Resampling mode:
              • "shape" – resample CT to match `new_size`
              • otherwise – resample CT to match `new_spacing`
        new_size : tuple of int, optional
            Desired (Z,Y,X) shape for CT resampling (used if resample_type="shape").
        new_spacing : tuple of float, optional
            Desired voxel spacing (z,y,x) in mm (used if resample_type="spacing").

        Workflow
        --------
        1. Check if outputs already exist → skip if present.
        2. Load DICOM CT, RT Dose, and RT Structure Set.
        3. Extract Dose and ROI masks.
        4. Compute resampling zoom factors (from `new_size` or `new_spacing`).
        5. Resample CT, ROIs (nearest-neighbor), and align Dose to CT.
        6. Save CT, Dose, and ROI volumes as NIfTI (.nii.gz) with metadata.
        7. Emit `finished` signal with results or error message.

        Notes
        -----
        • CT resampling uses linear interpolation (order=1).
        • ROI masks use nearest-neighbor interpolation (order=0) to preserve labels.
        • Dose is aligned to CT grid using :func:`preprocess_Dose_to_CT`.
        • Outputs are saved in NIfTI format via :func:`save_volumes`.
        • Errors are caught and emitted in the `finished` signal with traceback.
        """
    finished = pyqtSignal(dict)  # {ok: bool, msg: str, out_dir: str}
    log = pyqtSignal(str)

    def __init__(self, patient_data, patient_output_dir, resample_type, new_size, new_spacing):
        super().__init__()
        self.patient_data = patient_data
        self.patient_output_dir = patient_output_dir
        self.resample_type = resample_type
        self.new_size = np.array(new_size) if new_size is not None else None
        self.new_spacing = np.array(new_spacing) if new_spacing is not None else None

    def run(self):
        try:
            t0 = time.time()

            self.log.emit("🔍 Checking if outputs already exist...")
            ready = all(os.path.exists(os.path.join(self.patient_output_dir, f))
                        for f in ("CT_volume.nii.gz", "Dose_volume.nii.gz"))
            if ready:
                self.log.emit(f"✅ NIfTI already present in {self.patient_output_dir}")
                self.finished.emit({
                    "ok": True,
                    "msg": f"NIfTI already present in {self.patient_output_dir}",
                    "out_dir": self.patient_output_dir
                })
                return

            # ------------------------
            # Load DICOM data
            # ------------------------
            self.log.emit("📥 Loading CT and RT data...")
            CT = load_CT_data(self.patient_data)
            RD, RS, _ = load_RT_data(self.patient_data)

            self.log.emit("💉 Extracting Dose data...")
            Dose = extract_dose_data(RD)

            self.log.emit("🧠 Extracting ROI structures...")
            ROIs = extract_ROIs_data(RS, CT)

            # ------------------------
            # Compute zoom factors
            # ------------------------
            if self.resample_type == "shape":
                zoom = self.new_size / CT["Volume"].shape
                self.log.emit(f"📏 Resampling by shape to {self.new_size.tolist()}")
            else:
                zoom = CT["Spacing"] / self.new_spacing
                self.log.emit(f"📏 Resampling by spacing to {self.new_spacing.tolist()}")

            # ------------------------
            # Resample CT
            # ------------------------
            self.log.emit("   🔄 Resampling CT volume...")
            CT["Volume"] = scipy.ndimage.zoom(CT["Volume"], zoom, order=1)
            CT["Spacing"] = CT["Spacing"] / zoom

            # ------------------------
            # Resample ROIs
            # ------------------------
            self.log.emit("   🔄 Aligning ROI structures to CT volume...")
            for r in ROIs["ROIs"].values():
                r["Volume"] = scipy.ndimage.zoom(r["Volume"], zoom, order=0)
            ROIs["Spacing"], ROIs["Position"] = CT["Spacing"], CT["Position"]

            # ------------------------
            # Resample Dose
            # ------------------------
            self.log.emit("   🔄 Aligning Dose volume to CT volume...")
            Dose = preprocess_Dose_to_CT(Dose, CT)

            # ------------------------
            # Save Outputs
            # ------------------------
            os.makedirs(self.patient_output_dir, exist_ok=True)
            self.log.emit(f"💾 Saving volumes to: {self.patient_output_dir}")
            save_volumes(CT, Dose, ROIs, self.patient_output_dir)

            dt = time.time() - t0
            self.finished.emit({
                "ok": True,
                "msg": f"Preprocess finished in {dt / 60:.1f} min. Saved to {self.patient_output_dir}",
                "out_dir": self.patient_output_dir
            })
        except Exception:
            self.finished.emit({
                "ok": False,
                "msg": traceback.format_exc(),
                "out_dir": ""
            })

class PandasModel(QAbstractTableModel):
    """
        Qt table model wrapper for a pandas DataFrame.

        This model allows a pandas DataFrame to be displayed in Qt views
        such as QTableView, with support for formatted display values
        and custom headers.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be displayed. A copy is stored internally.
        parent : QObject, optional
            Optional Qt parent object.

        Methods
        -------
        rowCount(parent=None)
            Returns the number of rows in the DataFrame.
        columnCount(parent=None)
            Returns the number of columns in the DataFrame.
        data(idx, role=Qt.DisplayRole)
            Returns the display value for a given cell, formatted as string.
            Floats are formatted with two decimals; NaN values are shown as empty.
        headerData(sec, ori, role=Qt.DisplayRole)
            Returns column headers (Horizontal) or index labels (Vertical).

        Notes
        -----
        • Implements a minimal QAbstractTableModel for read-only display.
        • Only DisplayRole and ToolTipRole are supported for cell data.
        • Floats are rendered with 2 decimal places, other types use str().
        • NaN values are displayed as empty strings.
        • Column headers come from `df.columns`; row headers from `df.index`.
        """
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df.copy()

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, idx, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.ToolTipRole) and idx and idx.isValid():
            v = self._df.iat[idx.row(), idx.column()]
            if pd.isna(v):
                return ""
            return f"{v:.2f}" if isinstance(v, float) else str(v)

    def headerData(self, sec, ori, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if ori == Qt.Horizontal else str(self._df.index[sec])

def _dvh_cumsum_weighted(dose_values: np.ndarray, weights: np.ndarray, voxel_vol_cc: float, step_gy: float = 0.1, max_dose: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a weighted cumulative Dose-Volume Histogram (DVH).

    Parameters
    ----------
    dose_values : np.ndarray
        1D array of dose values (Gy) for all voxels of interest.
    weights : np.ndarray
        1D array of per-voxel weights (e.g., mask values or ROI fractions),
        same shape as `dose_values`.
    voxel_vol_cc : float
        Volume of a single voxel in cubic centimeters (cc).
    step_gy : float, optional
        Bin width in Gy for histogramming (default = 0.1 Gy).
    max_dose : float, optional
        Maximum dose value to consider (Gy). If None, the maximum from
        `dose_values` is used.

    Returns
    -------
    bins : np.ndarray
        Left edges of dose bins (Gy), shape [N].
    cumvol : np.ndarray
        Cumulative volume (cc) corresponding to each bin, shape [N].
        Decreases monotonically with dose.

    Notes
    -----
    • Computes a weighted histogram of dose values, scaling by voxel volume (cc).
    • The cumulative sum is taken from high dose → low dose so that `cumvol[i]`
      represents the total volume receiving at least `bins[i]` Gy.
    • If `weights` is binary (0/1), the result is a standard DVH for the ROI.
    • Non-binary weights allow for partial-volume or probabilistic ROIs.
    • Returned arrays are float32 for memory efficiency.
    """
    if dose_values.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    end = max_dose if max_dose is not None else float(dose_values.max())
    last_edge = math.ceil(end / step_gy) * step_gy
    bins = np.arange(0.0, last_edge + step_gy, step_gy, dtype=np.float32)

    vol_hist, _ = np.histogram(dose_values, bins=bins,
                               weights=weights * float(voxel_vol_cc))
    cumvol = np.cumsum(vol_hist[::-1])[::-1]
    return bins[:-1], cumvol.astype(np.float32)

def dvh_table_abs(dose_values: np.ndarray, weights: np.ndarray, voxel_vol_cc: float, step_gy: float = 0.1, prescription: Optional[float] = None, max_dose: Optional[float] = None) -> pd.DataFrame:
    """
    Build an absolute Dose-Volume Histogram (DVH) as a pandas DataFrame.

    Parameters
    ----------
    dose_values : np.ndarray
        1D array of dose values (Gy) for all voxels of interest.
    weights : np.ndarray
        1D array of per-voxel weights (e.g., ROI mask values).
        Must have the same shape as `dose_values`.
    voxel_vol_cc : float
        Volume of a single voxel in cubic centimeters (cc).
    step_gy : float, optional
        Dose bin width in Gy (default = 0.1 Gy).
    prescription : float, optional
        Prescription dose in Gy. If provided, relative dose is expressed
        as a percentage of this value. If None, the maximum dose from
        the curve is used as the reference.
    max_dose : float, optional
        Maximum dose value (Gy) to include in the DVH. If None, uses the
        maximum in `dose_values`.

    Returns
    -------
    dvh_df : pandas.DataFrame
        DataFrame with the following columns:
          • "Dose [Gy]"      – Dose bin edges (float32)
          • "Rel. Dose [%]"  – Dose as % of prescription (or max dose)
          • "Volume [cm³]"   – Absolute cumulative volume in cc

    Notes
    -----
    • Internally calls :func:`_dvh_cumsum_weighted` to compute cumulative
      volume curves.
    • Relative dose normalization:
        - If `prescription` is provided → relative dose = 100 × dose / prescription.
        - Else → relative dose = 100 × dose / max_curve_dose.
    • Volume is cumulative: "Volume [cm³]" at dose d represents the volume
      receiving at least d Gy.
    • If `dose_values` is empty, an empty DataFrame with headers is returned.
    """
    d, v = _dvh_cumsum_weighted(dose_values, weights, voxel_vol_cc, step_gy, max_dose=max_dose)
    if d.size == 0:
        return pd.DataFrame(columns=["Dose [Gy]", "Rel. Dose [%]", "Volume [cm³]"])

    denom = float(prescription) if prescription else float(max(d[-1], 1e-12))
    rel = (100.0 * d / denom).astype(np.float32)

    return pd.DataFrame({
        "Dose [Gy]": d.astype(np.float32),
        "Rel. Dose [%]": rel,
        "Volume [cm³]": v.astype(np.float32),
    })

def compute_abs_dvhs(masks, dose_arr, voxel_vol_cc, prescription, spacing_mm, log):
    """
    Compute absolute DVH tables (0.1 Gy bins) for multiple ROIs.

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        Mapping from ROI name → 3D binary mask [Z,Y,X].
        Each mask should be boolean or integer (1 inside ROI, 0 outside).
    dose_arr : np.ndarray
        3D dose distribution array [Z,Y,X] in Gy, aligned to the masks.
    voxel_vol_cc : float
        Volume of a single voxel in cubic centimeters (cc).
    prescription : float
        Prescription dose in Gy. Used to normalize relative dose in DVH tables.
    spacing_mm : tuple[float, float, float]
        Voxel spacing (z,y,x) in millimeters. Passed to dose sampling.
    log : pyqtSignal or callable
        Logging function/signal used to emit progress updates.

    Returns
    -------
    dvh_abs : dict[str, pandas.DataFrame]
        Mapping from ROI name → DVH DataFrame with columns:
          • "Dose [Gy]"      – Dose bin edges (float32, 0.1 Gy bins)
          • "Rel. Dose [%]"  – Dose as % of prescription
          • "Volume [cm³]"   – Absolute cumulative volume in cc

    Notes
    -----
    • Internally calls :func:`_sample_native_dose` to extract per-voxel dose
      values within each ROI mask.
    • Uses :func:`dvh_table_abs` to build per-ROI DVH tables.
    • Bins are fixed at 0.1 Gy increments from 0 to global maximum dose.
    • The cumulative DVH volume decreases monotonically with dose.
    • A log message is emitted for each ROI upon completion.
    """
    dvh_abs: Dict[str, pd.DataFrame] = {}
    global_max = float(dose_arr.max()) if dose_arr.size else 0.0

    for roi, mask in masks.items():
        dose_vals, weights = _sample_native_dose(mask, spacing_mm, dose_arr)
        df = dvh_table_abs(dose_vals, weights, voxel_vol_cc, 0.1, prescription, max_dose=global_max)
        dvh_abs[roi] = df
        log.emit("📈     Finished calculating DVH curve for ROI '{}'".format(roi))

    return dvh_abs

def extract_group1_metadata(RP_data, RD_data, patient_folder_path) -> Dict[str, Any]:
    """
    Extract “Group-1” plan-level metadata from RT Plan (RP), RT Dose (RD),
    and the patient’s folder name.

    The function is designed to be robust against missing DICOM tags and
    non-standard planners: unavailable values are returned as None or
    'Unknown…'.

    Parameters
    ----------
    RP_data : pydicom.Dataset
        DICOM Radiotherapy Plan (RTPLAN) dataset.
    RD_data : pydicom.Dataset
        DICOM Radiotherapy Dose (RTDOSE) dataset.
    patient_folder_path : str or pathlib.Path
        Path to the patient’s folder; folder name is parsed for patient ID
        and session date.

    Returns
    -------
    meta : dict[str, Any]
        Dictionary with nine standardized plan-level metadata fields:

          • "Patient ID"                – ID parsed from folder name prefix
          • "Session Date"              – Parsed from folder name (DD/MM/YYYY if possible)
          • "Prescription Dose [Gy]"    – Total prescribed dose
          • "Number of Fractions"       – Planned number of fractions
          • "Dose per Fraction [Gy]"    – Prescription ÷ number of fractions
          • "Planning Software"         – Planner software version string
          • "Planning Technique"        – Beam technique (e.g. IMRT, VMAT)
          • "Dose Calculation Algorithm"– Algorithm used for dose calc.
          • "Dose Grid Size [mm]"       – Grid spacing (z,y,x) in mm

    Notes
    -----
    • Patient ID is parsed as the substring before the first “_” in the folder name.
    • Session date is inferred from any 8-digit block in the folder name;
      common formats (%d%m%Y, %Y%m%d) are recognized, otherwise raw digits are returned.
    • Prescription dose is extracted via `_get_prescription(RP_data)`.
    • Number of fractions is taken from `FractionGroupSequence[0].NumberOfFractionsPlanned`.
    • Dose per fraction is derived only if both total dose and fraction count are present.
    • Planning technique and algorithm are taken from the first entry in `BeamSequence`.
    • Dose grid size is extracted from `RD_data.PixelSpacing` and
      `GridFrameOffsetVector` (z spacing); falls back to `SliceThickness` if needed.
    • All missing values are returned as None (or 'Unknown…' for IDs/dates).
    """
    meta: Dict[str, Any] = {
        "Patient ID":               "UnknownID",
        "Session Date":             "UnknownDate",
        "Prescription Dose [Gy]":   None,
        "Number of Fractions":      None,
        "Dose per Fraction [Gy]":   None,
        "Planning Software":        None,
        "Planning Technique":       None,
        "Dose Calculation Algorithm": None,
        "Dose Grid Size [mm]":      None,
    }

    # ── 1) Patient‑ID & session date from the folder name ─────────────────
    folder = Path(patient_folder_path).name
    if folder:
        #   • Patient‑ID = everything before first “_”
        meta["Patient ID"] = folder.split("_")[0] or meta["Patient ID"]

        #   • Any 8‑digit block interpreted as date
        m = re.search(r"(\d{8})", folder)
        if m:
            raw = m.group(1)
            for fmt in ("%d%m%Y", "%Y%m%d"):
                try:
                    meta["Session Date"] = datetime.strptime(
                        raw, fmt).strftime("%d/%m/%Y")
                    break
                except ValueError:
                    continue
            else:
                meta["Session Date"] = raw          # unknown order

    # ── 2) Prescription, n‑fractions, dose/fraction from the RTPLAN ───────
    rx = _get_prescription(RP_data)                    # helper already in file :contentReference[oaicite:2]{index=2}
    if rx is not None:
        meta["Prescription Dose [Gy]"] = float(rx)

    for fg in getattr(RP_data, "FractionGroupSequence", []):
        n_frac = getattr(fg, "NumberOfFractionsPlanned", None)
        if n_frac:
            meta["Number of Fractions"] = int(n_frac)
            break

    if meta["Prescription Dose [Gy]"] and meta["Number of Fractions"]:
        meta["Dose per Fraction [Gy]"] = (
            meta["Prescription Dose [Gy]"] / meta["Number of Fractions"]
        )

    # ── 3) Planning software / technique / algorithm ──────────────────────
    sw = getattr(RP_data, "SoftwareVersions", None)
    meta["Planning Software"] = str(sw) if sw is not None else None

    beam = next(iter(getattr(RP_data, "BeamSequence", [])), None)
    if beam:
        meta["Planning Technique"] = getattr(beam, "BeamTechnique", None)
        meta["Dose Calculation Algorithm"] = getattr(
            beam, "DoseCalculationAlgorithm", None)

    # ── 4) Dose‑grid spacing from RTDOSE ───────────────────────────────────
    try:
        spacing_x, spacing_y = map(float, RD_data.PixelSpacing)
        gfv = RD_data.GridFrameOffsetVector
        spacing_z = float(gfv[1] - gfv[0]) if len(gfv) > 1 else float(
            getattr(RD_data, "SliceThickness", np.nan))
        meta["Dose Grid Size [mm]"] = (spacing_z, spacing_y, spacing_x)
    except Exception:
        pass

    return meta

def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Compute the weighted percentile of a set of values.

    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values (e.g., dose values).
    weights : np.ndarray
        1D array of non-negative weights, same shape as `values`.
        Defines the contribution of each value to the percentile.
    q : float
        Percentile to compute, in the range [0, 100].

    Returns
    -------
    float
        Weighted q-th percentile of the input values.

    Notes
    -----
    • The method sorts `values`, builds the cumulative distribution function (CDF)
      weighted by `weights`, and performs linear interpolation.
    • If all weights are equal, result is identical to the standard percentile.
    • If `weights` contain zeros, those entries are effectively ignored.
    • Implementation uses `numpy.interp` for interpolation on the weighted CDF.
    """
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q / 100.0, cdf, v))

def _weighted_mode(values: np.ndarray, weights: np.ndarray, bin_width: float = 0.1) -> float:
    """
    Compute the weighted mode of a distribution using a histogram.

    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values.
    weights : np.ndarray
        1D array of non-negative weights, same shape as `values`.
        Defines the contribution of each value to the mode estimate.
    bin_width : float, optional
        Width of histogram bins (default = 0.1). Smaller bin widths yield
        more precise but potentially noisier results.

    Returns
    -------
    float
        Weighted mode estimate. If the input is empty, returns NaN.
        If all values are nearly identical, returns that value.

    Notes
    -----
    • The mode is estimated as the center of the histogram bin with
      the maximum weighted count.
    • Special cases are handled:
        - Empty input → returns NaN.
        - Degenerate distribution (all values equal or nearly so) →
          returns that constant value.
        - Very narrow ranges (< `bin_width`) → single bin used.
    • If no bin has weight > 0, falls back to the weighted average.
    • Result depends on the choice of `bin_width`; using a smaller
      bin may better capture fine-grained peaks.
    """
    if values.size == 0:
        return float("nan")
    if np.allclose(values.ptp(), 0):
        return float(values[0])
    v_min, v_max = float(values.min()), float(values.max())
    if v_max - v_min < bin_width:
        bins = np.array([v_min, v_max + bin_width], dtype=np.float32)
    else:
        bins = np.arange(v_min, v_max + bin_width, bin_width, dtype=np.float32)
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    if hist.size == 0 or hist.max() == 0:
        return float(np.average(values, weights=weights if weights.sum() > 0 else None))
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) / 2.0)

def _sample_native_dose(mask: np.ndarray, spacing_mm: Tuple[float, float, float], dose: np.ndarray, smooth_sigma: float = SMOOTH_SIGMA_MM) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-voxel dose values and fractional occupancy weights
    inside a binary ROI mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary array [Z,Y,X], True (1) inside the ROI, False (0) outside.
    spacing_mm : tuple[float, float, float]
        Physical voxel spacing (z,y,x) in millimeters. Used to convert
        the smoothing kernel from mm to voxel units.
    dose : np.ndarray
        3D dose distribution [Z,Y,X] in Gy. Must have the same shape as `mask`.
    smooth_sigma : float, optional
        Gaussian smoothing kernel in millimeters (default = SMOOTH_SIGMA_MM).
        Smoothing softens mask edges and assigns partial (fractional) weights
        to voxels on ROI boundaries. If set to 0, no smoothing is applied.

    Returns
    -------
    dose_values : np.ndarray
        1D array of dose values (Gy) for voxels inside the mask.
    weights : np.ndarray
        1D array of occupancy weights corresponding to `dose_values`.
        Values are 1.0 for fully inside voxels, fractional [0,1] for
        edge voxels (if smoothing applied).

    Notes
    -----
    • The mask and dose arrays must have identical shapes.
    • Smoothing is applied in voxel units:
        sigma_vox = (σ/Δx, σ/Δy, σ/Δz), where σ = smooth_sigma (mm).
    • This fractional-weight sampling approximates sub-voxel interpolation
      and reduces aliasing in DVH calculations.
    • When `smooth_sigma=0`, weights are exactly binary (1 inside, 0 outside).
    """
    if mask.shape != dose.shape:
        raise ValueError(f"Mask shape {mask.shape} must match dose shape {dose.shape}")
    if smooth_sigma > 0:
        sigma_vox = (smooth_sigma/spacing_mm[2], smooth_sigma/spacing_mm[1], smooth_sigma/spacing_mm[0])
        weight_mask = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox)
    else:
        weight_mask = mask.astype(np.float32)

    inside = mask.astype(bool)
    return dose[inside], weight_mask[inside]

def compute_roi_metrics(masks: Dict[str, np.ndarray], dose_arr: np.ndarray, voxel_vol_cc: float, prescription: Optional[float], spacing_mm: Tuple[float, float, float], smooth_sigma: float, log) -> pd.DataFrame:
    """
    Compute per-ROI dose–volume statistics and plan quality metrics.

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        Mapping from ROI name → 3D binary mask [Z,Y,X].
        Each mask should be boolean or integer (1 inside ROI, 0 outside).
    dose_arr : np.ndarray
        3D dose distribution [Z,Y,X] in Gy, aligned with the masks.
    voxel_vol_cc : float
        Volume of a single voxel in cubic centimeters (cc).
    prescription : float, optional
        Prescription dose in Gy. Used for conformity index (CI) and
        homogeneity index (HI) calculations. If None, CI/HI = NaN.
    spacing_mm : tuple[float, float, float]
        Voxel spacing (z,y,x) in millimeters. Used for smoothing conversion.
    smooth_sigma : float
        Gaussian smoothing kernel (in mm) applied to ROI masks before
        sampling dose. Produces fractional weights at edges.
    log : pyqtSignal or callable
        Logging function/signal used to emit progress updates.

    Returns
    -------
    pd.DataFrame
        Table of per-ROI metrics. Each row corresponds to one ROI and
        includes the following columns:

          • ROI            – ROI name
          • Volume_cc      – ROI volume (cc)
          • Min_Gy, Max_Gy – Minimum / maximum dose within ROI
          • Mean_Gy        – Weighted mean dose (Gy)
          • Median_Gy      – Weighted 50th percentile (Gy)
          • Mode_Gy        – Weighted mode (Gy, histogram-based)
          • Std_Gy         – Weighted dose standard deviation (Gy)
          • D2_Gy          – Dose received by 2% of ROI (≈ near-max dose)
          • D50_Gy         – Median dose (Gy)
          • D98_Gy         – Dose received by 98% of ROI (≈ near-min dose)
          • HI             – Homogeneity index = (D98 - D2) / prescription
          • CI             – Conformity index (for PTVs only):
                               (V_PTV,100%)² / (V_PTV,80% × ROI volume)

        For healthy brain ROIs (name contains “brain” but not “brainstem”),
        additional columns are included:
          • V5_cc, V10_cc, V12_cc, V18_cc, V20_cc, V23_cc, V24_cc,
            V25_cc, V27_cc, V30_cc
          representing absolute ROI volume (cc) receiving ≥ threshold Gy.

    Notes
    -----
    • Uses :func:`_sample_native_dose` for sub-voxel weighted sampling of dose.
    • Percentiles are computed via :func:`_weighted_percentile`.
    • Mode is estimated via :func:`_weighted_mode`.
    • CI is only computed for ROIs with "ptv" in their name (case-insensitive).
    • Healthy brain Vx metrics are hard-coded at clinically relevant thresholds.
    • Logging emits a message for each ROI when metrics are complete.
    """
    rows = []
    hb_thrs = [5, 10, 12, 18, 20, 23, 24, 25, 27, 30]

    for roi, mask in masks.items():
        dose_vals, w = _sample_native_dose(mask, spacing_mm, dose_arr, smooth_sigma)
        if dose_vals.size == 0:
            continue

        vol_total = float(np.sum(w) * voxel_vol_cc)
        mean = float(np.average(dose_vals, weights=w))
        std = float(np.sqrt(np.average((dose_vals - mean) ** 2, weights=w)))
        median = _weighted_percentile(dose_vals, w, 50.0)

        d2 = _weighted_percentile(dose_vals, w, 98.0)   # high-dose tail
        d50 = median
        d98 = _weighted_percentile(dose_vals, w, 2.0)   # low-dose tail
        mode = _weighted_mode(dose_vals, w)

        hi = float(abs((d98 - d2) / prescription)) if prescription else float("nan")

        # Simple conformity index for PTVs (if name contains 'ptv')
        roi_is_ptv = ("ptv" in roi.lower())
        if prescription and roi_is_ptv:
            v_ptv_100 = float(np.sum(w[dose_vals >= prescription]) * voxel_vol_cc)
            v_ptv_80  = float(np.sum(w[dose_vals >= 0.8 * prescription]) * voxel_vol_cc)
            ci = (v_ptv_100 ** 2) / (v_ptv_80 * vol_total) if v_ptv_80 > 0 else float("nan")
        else:
            ci = float("nan")

        row = OrderedDict([
            ("ROI", roi),
            ("Volume_cc", vol_total),
            ("Min_Gy", float(dose_vals.min())),
            ("Max_Gy", float(dose_vals.max())),
            ("Mean_Gy", mean),
            ("Median_Gy", float(median)),
            ("Mode_Gy", float(mode)),
            ("Std_Gy", std),
            ("D2_Gy", float(d2)),
            ("D50_Gy", float(d50)),
            ("D98_Gy", float(d98)),
            ("HI", hi),
            ("CI", float(ci)),
        ])

        roi_clean = roi.strip().lower()
        if "brain" in roi_clean and "brainstem" not in roi_clean:
            for thr in hb_thrs:
                vx = float(np.sum(w[dose_vals >= thr]) * voxel_vol_cc)
                row[f"V{thr}_cc"] = vx

        rows.append(row)
        log.emit("📊     Finished calculating metrics for ROI '{}'".format(roi))

    return pd.DataFrame(rows)

def _get_prescription(rtplan: pydicom.dataset.FileDataset) -> Optional[float]:
    """
    Heuristically extract the prescription dose (in Gy) from an RT Plan DICOM.

    The DICOM RTPLAN standard does not enforce a single tag for prescription
    dose. Different TPS vendors store it under different attributes or derive
    it indirectly. This helper searches through the most common locations.

    Parameters
    ----------
    rtplan : pydicom.dataset.FileDataset
        The RTPLAN DICOM dataset.

    Returns
    -------
    float or None
        The prescription dose in Gray (Gy) if found, otherwise None.

    Heuristic Search Order
    ----------------------
    1. **DoseReferenceSequence (TARGET entries)**
       Looks for fields such as:
         - `TargetPrescriptionDose`
         - `DeliveryMaximumDose`
         - `DeliveryWarningDose`
         - `DeliveryUnit`
       The first valid positive value is returned.

    2. **Top-level attributes**
       Attempts to read:
         - `DoseReferenceTreatmentMaxDose`
         - `PrescriptionDescription` (if numeric)

    3. **Beam-sequence derived estimate**
       Computes prescription ≈ (#fractions × dose_per_fraction), using:
         - `NumberOfFractionsPlanned`
         - `FinalCumulativeMetersetWeight`
       Returns the median across beams if available.

    Notes
    -----
    • Values are rounded to the nearest whole number when obtained from
      `DoseReferenceSequence`.
    • Some vendors (e.g., Varian, Elekta, RayStation) use different fields;
      this heuristic covers the most common.
    • If no valid positive value is found, returns None.
    """
    # 1) DoseReferenceSequence (TARGET)
    try:
        for dr in getattr(rtplan, "DoseReferenceSequence", []):
            if str(getattr(dr, "DoseReferenceType", "")).upper() == "TARGET":
                for tag in ("TargetPrescriptionDose", "DeliveryMaximumDose",
                            "DeliveryWarningDose", "DeliveryUnit"):
                    if hasattr(dr, tag):
                        val = float(getattr(dr, tag))
                        if val > 0:
                            return round(val)
    except Exception:
        pass

    # 2) Common top-level fields
    for tag in ("DoseReferenceTreatmentMaxDose", "PrescriptionDescription"):
        try:
            val = float(getattr(rtplan, tag))
            if val > 0:
                return val
        except Exception:
            pass

    # 3) Fraction × number of fractions
    try:
        beams = getattr(rtplan, "BeamSequence", [])
        if beams:
            doses = []
            for b in beams:
                fracs = int(getattr(b, "NumberOfFractionsPlanned", 0) or 0)
                dose_per_frac = float(getattr(b, "FinalCumulativeMetersetWeight", 0.0) or 0.0)
                if fracs > 0 and dose_per_frac > 0:
                    doses.append(fracs * dose_per_frac)
            if doses:
                est = float(np.median(doses))
                if est > 0:
                    return est
    except Exception:
        pass

    return None

def build_roi_masks(rs: pydicom.dataset.FileDataset, ct_img: sitk.Image,  log, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Convert RTSTRUCT polygon contours into binary ROI masks aligned to a CT grid.

    Parameters
    ----------
    rs : pydicom.dataset.FileDataset
        RT Structure Set (RS) DICOM dataset containing ROI definitions and
        contour sequences.
    ct_img : sitk.Image
        Reference CT image (SimpleITK) providing the target voxel grid and
        patient-to-voxel coordinate transform.
    log : pyqtSignal or callable
        Logger for progress updates.
    verbose : bool, optional
        If True (default), warnings are printed for missing/invalid ROIs.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping ROI name → binary mask with shape [Z,Y,X].
        Each mask is uint8 with values:
          - 1 inside the ROI
          - 0 outside the ROI

    Notes
    -----
    • Each RTSTRUCT polygon is rasterized slice-by-slice.
    • ROI contours are projected onto the nearest CT slice using the **median Z index**
      of the contour points. This is the common convention for planar RTSTRUCT data.
    • No cross-slice interpolation is performed — polygons are drawn only on their
      corresponding CT slices.
    • Contours that lie completely outside the CT grid are skipped.
    • Empty or missing contours produce no mask for that ROI.

    Logging
    -------
    • A progress message is emitted for each ROI successfully processed.
    • If `verbose=True`, warnings are printed for ROIs with no contours or with
      contours outside the CT volume.
    """
    size_x, size_y, size_z = ct_img.GetSize()     # (X, Y, Z) voxel counts
    masks: Dict[str, np.ndarray] = {}

    roi_contour_by_number = {}
    if hasattr(rs, "ROIContourSequence"):
        for rc in rs.ROIContourSequence:
            roi_contour_by_number[int(rc.ReferencedROINumber)] = rc

    for roi in getattr(rs, "StructureSetROISequence", []):
        num = int(roi.ROINumber)
        name = str(getattr(roi, "ROIName", f"ROI_{num}")).strip()

        rc = roi_contour_by_number.get(num, None)
        if rc is None or not getattr(rc, "ContourSequence", []):
            if verbose:
                print(f"⚠️  ROI '{name}' has no contours – skipped")
            continue

        m = np.zeros((size_z, size_y, size_x), np.uint8)
        for cs in rc.ContourSequence:
            pts = np.asarray(cs.ContourData, dtype=np.float64).reshape(-1, 3)
            if pts.size == 0:
                continue

            # Convert patient (x,y,z) to CT index (i,j,k) then split
            ijk = [ct_img.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
            xs, ys, zs = map(np.array, zip(*ijk))

            # Round/clip XY; use median Z slice for the polygon
            xs = np.clip(np.round(xs).astype(int), 0, size_x - 1)
            ys = np.clip(np.round(ys).astype(int), 0, size_y - 1)
            z  = int(np.clip(np.round(np.median(zs)), 0, size_z - 1))

            rr, cc = polygon(ys, xs, (size_y, size_x))
            m[z, rr, cc] = 1

        if m.any():
            masks[name] = m
        elif verbose:
            print(f"⚠️  ROI '{name}' contours lay outside the CT – skipped")
        log.emit("🔬     Finished building mask for ROI '{}'".format(name))

    return masks

def find_file_with_prefix(folder_path, prefix):
    """
    Search for the first file in a folder whose name starts with a given prefix.

    Parameters
    ----------
    folder_path : str
        Path to the directory where the search should be performed.
    prefix : str
        Filename prefix to match (case-sensitive).

    Returns
    -------
    str or None
        The absolute path to the first matching file if found,
        otherwise None.

    Notes
    -----
    • Only the first match is returned; if multiple files share the prefix,
      later ones are ignored.
    • The match is **prefix-only** (e.g., prefix="CT" will match
      "CT123.dcm" and "CT_volume.nii.gz").
    • Matching is case-sensitive; "CT" will not match "ct001.dcm".
    • Useful for locating RT DICOM files (e.g., "RD", "RS", "RP") inside
      patient study directories.
    """
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            return os.path.join(folder_path, file_name)
    return None

def resample_dose_to_ct(dose_ds: pydicom.dataset.FileDataset, dose_arr: np.ndarray, ct_img: sitk.Image) -> np.ndarray:
    """
    Resample a DICOM RTDOSE array onto the CT voxel grid.

    The dose array is first wrapped into a SimpleITK image with correct
    spacing and origin based on RTDOSE metadata, then resampled onto the
    CT grid using linear interpolation.

    Parameters
    ----------
    dose_ds : pydicom.dataset.FileDataset
        RTDOSE DICOM dataset providing dose geometry metadata:
          • PixelSpacing (dy, dx) in mm
          • GridFrameOffsetVector (for dz spacing) or SliceThickness
          • ImagePositionPatient (dose grid origin in patient coords)
    dose_arr : np.ndarray
        Raw 3D dose array [Z,Y,X] in Gy (scaled by DoseGridScaling).
    ct_img : sitk.Image
        Reference CT SimpleITK image defining the target voxel grid
        (spacing, size, orientation, origin).

    Returns
    -------
    np.ndarray
        Resampled dose array [Z,Y,X] in Gy, aligned voxel-for-voxel with
        the CT grid.

    Notes
    -----
    • Input spacing: DICOM stores PixelSpacing as (row=dy, col=dx).
    • Z-spacing is derived from GridFrameOffsetVector if present,
      otherwise from SliceThickness.
    • SimpleITK requires spacing order = (x, y, z).
    • Resampling uses linear interpolation (`sitk.sitkLinear`) with
      default value = 0 outside the original grid.
    • The output is cast to float32 for memory efficiency.
    """
    img = sitk.GetImageFromArray(dose_arr)  # [Z,Y,X]

    # DICOM PixelSpacing is [row_spacing (dy), col_spacing (dx)]
    dy, dx = map(float, dose_ds.PixelSpacing)
    # z spacing from GridFrameOffsetVector or fallback to SliceThickness
    if hasattr(dose_ds, "GridFrameOffsetVector") and len(dose_ds.GridFrameOffsetVector) > 1:
        dz = float(np.diff(dose_ds.GridFrameOffsetVector).mean())
    else:
        dz = float(getattr(dose_ds, "SliceThickness", 1.0))

    # IMPORTANT: SimpleITK expects spacing order (x, y, z)
    img.SetSpacing((dx, dy, dz))
    # Origin in patient space
    img.SetOrigin(tuple(map(float, dose_ds.ImagePositionPatient)))

    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ct_img)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    out = rf.Execute(img)
    return sitk.GetArrayFromImage(out).astype(np.float32)  # [Z,Y,X]

def _find_dcm(folder: str, modality: str) -> str:
    """
    Find the first DICOM file in a folder matching a given Modality.

    Parameters
    ----------
    folder : str
        Path to the directory to search. Only immediate files are scanned
        (no recursive subdirectory search).
    modality : str
        Expected DICOM Modality value (e.g., "CT", "RTDOSE", "RTSTRUCT", "RTPLAN").

    Returns
    -------
    str
        Absolute path to the first matching DICOM file.

    Raises
    ------
    FileNotFoundError
        If no readable DICOM in the folder matches the requested modality.

    Notes
    -----
    • Reads DICOM headers with `stop_before_pixels=True` for speed and
      robustness (pixel data is skipped).
    • Files that are not valid DICOMs or that fail to parse are ignored.
    • Returns only the **first match** encountered. If multiple files of the
      same modality exist (e.g., multiple CT slices), only one is returned.
    • For modalities with multiple files (e.g., CT series), downstream
      functions should use series-based loaders instead of `_find_dcm`.
    """
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        try:
            if pydicom.dcmread(p, stop_before_pixels=True).Modality == modality:
                return p
        except Exception:
            # Skip unreadable files
            pass
    raise FileNotFoundError(f"Could not find a DICOM with modality={modality!r} in {folder!r}")

def load_dose(folder: str) -> Tuple[pydicom.dataset.FileDataset, np.ndarray]:
    """
    Load an RTDOSE DICOM file and return both the dataset and dose grid.

    Parameters
    ----------
    folder : str
        Path to a directory containing at least one RTDOSE DICOM file.

    Returns
    -------
    tuple
        (ds, dose)
        • ds : pydicom.dataset.FileDataset
          Parsed RTDOSE DICOM dataset with metadata.
        • dose : np.ndarray
          3D NumPy array [Z,Y,X] in Gray (Gy), scaled by DoseGridScaling.

    Notes
    -----
    • Uses `_find_dcm(folder, "RTDOSE")` to locate the first RTDOSE file.
    • Dose values are converted to Gy by multiplying `pixel_array`
      with `DoseGridScaling`.
    • Some RTDOSE datasets are 4D (time × Z × Y × X). In that case,
      only the first frame (time=0) is returned.
    • The returned dataset (`ds`) still contains all header fields and
      should be used for geometry (PixelSpacing, GridFrameOffsetVector, etc.).
    """
    ds = pydicom.dcmread(_find_dcm(folder, "RTDOSE"))
    dose = ds.pixel_array * ds.DoseGridScaling
    # Some RTDOSE are 4D (time). Use first frame if so.
    if dose.ndim == 4:
        dose = dose[0]
    return ds, dose

def load_ct(folder: str) -> Tuple[sitk.Image, np.ndarray]:
    """
    Load a CT series under `folder` using SimpleITK.
    Returns (itk_image, array[Z,Y,X] in HU).
    """
    rdr = sitk.ImageSeriesReader()
    ids = rdr.GetGDCMSeriesIDs(folder) or []
    for sid in ids:
        fns = rdr.GetGDCMSeriesFileNames(folder, sid)
        if not fns:
            continue
        try:
            if pydicom.dcmread(fns[0], stop_before_pixels=True).Modality != "CT":
                continue
        except Exception:
            continue
        rdr.SetFileNames(fns)
        img = rdr.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X]
        # Convert to HU if unsigned storage (common: 0..4095)
        if arr.min() >= 0:
            arr -= 1024.0
        return img, arr
    raise RuntimeError(f"No CT series found under {folder!r}")

class MetricsWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    result = pyqtSignal(object)

    def __init__(self, patient_data, patient_output_dir):
        super().__init__()
        self.patient_data = patient_data
        self.patient_output_dir = patient_output_dir

    def run(self):
        try:
            self.log.emit("📥 Starting metric extraction...")

            # Define expected output file paths
            roi_path = os.path.join(self.patient_output_dir, "roi_metrics.csv")
            meta_path = os.path.join(self.patient_output_dir, "patient_metadata.csv")
            rx_path = os.path.join(self.patient_output_dir, "prescription.txt")

            # Check for existing DVH files
            dvh_files = [
                f for f in os.listdir(self.patient_output_dir)
                if f.startswith("dvh_") and f.endswith(".csv")
            ]

            # If all expected files exist, load instead of recompute
            if os.path.exists(roi_path) and os.path.exists(meta_path) and os.path.exists(rx_path) and dvh_files:
                self.log.emit("📂 Found existing extraction files. Loading instead of recomputing...")

                roi_df = pd.read_csv(roi_path)
                meta_df = pd.read_csv(meta_path)

                with open(rx_path, "r") as f:
                    rx = f.read().strip()

                try:
                    rx = float(rx)
                except ValueError:
                    pass  # leave as string if it can't be cast

                dvh_abs = {}
                for f in dvh_files:
                    roi_name = f[len("dvh_"):-len(".csv")].replace("_", " ")
                    df = pd.read_csv(os.path.join(self.patient_output_dir, f))
                    dvh_abs[roi_name] = df
                    self.log.emit(f"📄 Loaded DVH: {f}")

                result = {
                    "ROI_DF": roi_df,
                    "Meta_DF": meta_df,
                    "DVH": dvh_abs,
                    "Rx": rx
                }

                self.log.emit("✅ Loaded result object from disk.")
                self.result.emit(result)
                self.finished.emit()
                return  # Skip extraction

            # If not all files are present, run full extraction
            self.log.emit("🚧 Running full extraction...")

            self.log.emit("🔄 Loading CT and Dose data...")
            ct_img, _ = load_ct(self.patient_data)
            self.log.emit("🔄   Finished Loading CT")
            ds_dose, dose_raw = load_dose(self.patient_data)
            self.log.emit("🔄   Finished Loading Dose data")

            self.log.emit("   ↳ Resampling Dose to CT grid...")
            dose_arr = resample_dose_to_ct(ds_dose, dose_raw, ct_img)

            self.log.emit("🔄 Reading RT Structure and Plan...")
            rs = pydicom.dcmread(find_file_with_prefix(self.patient_data, "RS"))
            rp = pydicom.dcmread(find_file_with_prefix(self.patient_data, "RP"))

            self.log.emit("🔬 Building ROI masks...")
            masks = build_roi_masks(rs, ct_img, self.log)

            self.log.emit("💊 Extracting prescription...")
            rx = _get_prescription(rp)

            sx, sy, sz = ct_img.GetSpacing()
            spacing = (sx, sy, sz)
            vv_cc = (sx * sy * sz) / 1000.0

            self.log.emit("📊 Computing ROI metrics...")
            roi_df = compute_roi_metrics(masks, dose_arr, vv_cc, rx, spacing, SMOOTH_SIGMA_MM, self.log).round(2)

            self.log.emit("📄 Extracting patient metadata...")
            meta_df = pd.DataFrame(
                extract_group1_metadata(rp, None, self.patient_data).items(),
                columns=["Field", "Value"]
            )

            self.log.emit("📈 Calculating DVH curves...")
            dvh_abs = compute_abs_dvhs(masks, dose_arr, vv_cc, rx, spacing_mm=spacing, log=self.log)

            result = {
                "ROI_DF": roi_df,
                "Meta_DF": meta_df,
                "DVH": dvh_abs,
                "Rx": rx
            }

            # 🔽 Save outputs
            os.makedirs(self.patient_output_dir, exist_ok=True)

            roi_df.to_csv(roi_path, index=False)
            self.log.emit(f"💾 Saved ROI metrics to {roi_path}")

            meta_df.to_csv(meta_path, index=False)
            self.log.emit(f"💾 Saved patient metadata to {meta_path}")

            with open(rx_path, "w") as f:
                f.write(str(rx))
            self.log.emit(f"💾 Saved prescription to {rx_path}")

            for roi_name, dvh_df in dvh_abs.items():
                safe_name = roi_name.replace(" ", "_").replace("/", "_")
                dvh_csv_path = os.path.join(self.patient_output_dir, f"dvh_{safe_name}.csv")
                dvh_df.to_csv(dvh_csv_path, index=False)
                self.log.emit(f"💾 Saved DVH for '{roi_name}' to {dvh_csv_path}")

            self.result.emit(result)
            self.finished.emit()

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())
            self.finished.emit()


class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing in Progress")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        self.log_area = QPlainTextEdit(self)
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        self.close_button = QPushButton("Close", self)
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

    def write(self, message):
        self.log_area.appendPlainText(message.strip())

    def mark_done(self):
        self.progress_bar.setRange(0, 1)  # Set to determinate
        self.progress_bar.setValue(1)
        self.close_button.setEnabled(True)
        self.write("✅ Preprocessing complete.")

    def show_error(self, message):
        self.write(f"❌ Error: {message}")
        self.mark_done()

class SettingsDialog(QDialog):
    def __init__(self, default_output_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resample and Output Settings")
        self.setMinimumWidth(400)

        # ─── Widgets ──────────────────────────────────────────────
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Select resample type...", "shape", "spacing"])
        self.value_edit = QLineEdit()
        self.value_edit.setEnabled(False)  # Disabled initially

        self.output_edit = QLineEdit(default_output_dir)
        self.browse_btn = QPushButton("Browse...")

        # ─── Layout ────────────────────────────────────────────────
        form = QVBoxLayout()

        form.addWidget(QLabel("Resample Type:"))
        form.addWidget(self.type_combo)

        form.addWidget(QLabel("Resample Values:"))
        form.addWidget(self.value_edit)

        form.addWidget(QLabel("Output Folder:"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.browse_btn)
        form.addLayout(output_row)

        # ─── Buttons ──────────────────────────────────────────────
        button_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        button_row.addStretch()
        button_row.addWidget(self.ok_btn)
        button_row.addWidget(self.cancel_btn)
        form.addLayout(button_row)

        self.setLayout(form)

        # ─── Events ───────────────────────────────────────────────
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_type_change(self, text):
        if text == "shape":
            self.value_edit.setEnabled(True)
            self.value_edit.setText("512,512,512")
        elif text == "spacing":
            self.value_edit.setEnabled(True)
            self.value_edit.setText("1.0,1.0,1.0")
        else:
            self.value_edit.setEnabled(False)
            self.value_edit.setText("")

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def get_values(self):
        return {
            "resample_type": self.type_combo.currentText(),
            "resample_value": self.value_edit.text(),
            "output_dir": self.output_edit.text()
        }

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
        "Region": f"{side} {region}" if side else region,
        "Date": date
    }

def plot_all_roi_dvhs(dvh_abs: Dict[str, pd.DataFrame],
                      prescription: Optional[float],
                      *,
                      x_mode: str = "dose",
                      y_mode: str = "volume",
                      ax: Optional[Axes] = None,
                      linewidth: float = 1.5) -> Axes:
    """
    Plot all DVHs with flexible axes.
      x_mode: {'dose','relative'}
      y_mode: {'volume','relative'}
    """
    if x_mode not in {"dose", "relative"}:
        raise ValueError("x_mode must be 'dose' or 'relative'")
    if y_mode not in {"volume", "relative"}:
        raise ValueError("y_mode must be 'volume' or 'relative'")
    if x_mode == "relative" and not prescription:
        raise ValueError("Relative x‑axis requires a prescription dose")

    if ax is None:
        _, ax = plt.subplots()

    for roi, df in dvh_abs.items():
        if df.empty:
            continue

        if x_mode == "dose":
            x = df["Dose [Gy]"].to_numpy()
            xlabel = "Dose [Gy]"
        else:
            x = df["Dose [Gy]"].to_numpy() / float(prescription) * 100.0
            xlabel = "Relative dose [% Rx]"

        if y_mode == "volume":
            y = df["Volume [cm³]"].to_numpy()
            ylabel = "Volume [cm³]"
        else:
            y0 = float(df["Volume [cm³]"].iloc[0]) if not df.empty else 1.0
            y = df["Volume [cm³]"].to_numpy() / max(y0, 1e-12) * 100.0
            ylabel = "Relative volume [%]"

        ax.plot(x, y, label=roi, linewidth=linewidth)

    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize="small")
    ax.figure.tight_layout()
    return ax

class DVHPlotWidget(QWidget):
    """
    Composite widget that shows cumulative DVHs and lets the user
    choose which ROIs to overlay.  Each checklist entry now carries
    the same colour as its curve in the plot.
    """
    def __init__(self, dvh_abs: dict[str, "pd.DataFrame"] | None = None,
                 prescription: float | None = None,
                 parent=None):
        super().__init__(parent)
        self._dvh_abs = dvh_abs or {}
        self._prescription = prescription

        # colour bookkeeping ----------------------------------------------
        self._roi_colors: dict[str, str] = {}     # ROI name → "#rrggbb"
        self._mpl_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ROI checklist ----------------------------------------------------
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.NoSelection)
        self.roi_list.itemChanged.connect(self._redraw)

        # axis selectors ---------------------------------------------------
        self.x_selector = QComboBox();  self.x_selector.addItems(["dose", "relative"])
        self.y_selector = QComboBox();  self.y_selector.addItems(["volume", "relative"])
        self.x_selector.currentIndexChanged.connect(self._redraw)
        self.y_selector.currentIndexChanged.connect(self._redraw)

        # Matplotlib canvas ------------------------------------------------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # layout -----------------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X‑axis:")); ctrl.addWidget(self.x_selector)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Y‑axis:")); ctrl.addWidget(self.y_selector)
        ctrl.addStretch()

        left = QVBoxLayout()
        left.addWidget(QLabel("ROIs to display:"))
        left.addWidget(self.roi_list)
        left.addLayout(ctrl)

        main = QHBoxLayout(self)
        main.addLayout(left, 0)
        main.addWidget(self.canvas, 1)

        # first fill (if data were supplied) ------------------------------
        self._populate_list()
        self._redraw()

    # =====================================================================
    def set_data(self, dvh_abs: dict[str, "pd.DataFrame"],
                 prescription: float | None):
        """Replace DVH dictionary and Rx dose, then refresh widget."""
        self._dvh_abs = dvh_abs
        self._prescription = prescription
        self._populate_list()
        self._redraw()

    def clear(self):
        self.figure.clear()
        self.canvas.draw()
        self.roi_list.clear()

    # =====================================================================
    # internals
    # =====================================================================
    def _make_color_icon(self, hex_color: str) -> QIcon:
        """Return a 12×12 pixmap filled with *hex_color*."""
        pix = QPixmap(12, 12)
        pix.fill(QColor(hex_color))
        return QIcon(pix)

    def _populate_list(self):
        """Populate ROI checklist with coloured bullets (all ticked)."""
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        self._roi_colors.clear()

        for idx, roi in enumerate(sorted(self._dvh_abs.keys())):
            # deterministic colour from the Matplotlib cycle
            color = self._mpl_cycle[idx % len(self._mpl_cycle)]
            self._roi_colors[roi] = color

            item = QListWidgetItem(self._make_color_icon(color), roi)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.roi_list.addItem(item)

        self.roi_list.blockSignals(False)

    def _selected_rois(self):
        return [
            self.roi_list.item(i).text()
            for i in range(self.roi_list.count())
            if self.roi_list.item(i).checkState() == Qt.Checked
        ]

    def _redraw(self):
        self.figure.clear()
        rois = self._selected_rois()
        if not self._dvh_abs or not rois:
            self.canvas.draw_idle()
            return

        ax = self.figure.add_subplot(111)

        # set colour cycle to match the order in *rois*
        ax.set_prop_cycle('color', [self._roi_colors[r] for r in rois])

        subset = {roi: self._dvh_abs[roi] for roi in rois}
        plot_all_roi_dvhs(
            subset,
            self._prescription,
            ax=ax,
            x_mode=self.x_selector.currentText(),
            y_mode=self.y_selector.currentText(),
        )
        self.canvas.draw_idle()


class ArnousUnifiedGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARVOUS – Automatic Radiotherapy Visualization and Utility Output System")
        self.resize(1420, 860)

        # ── top bar
        self.btn_settings = QPushButton("Settings")
        self.btn_batch = QPushButton("Batch Processing")
        self.btn_load = QPushButton("Load Patient Data")
        self.btn_metrics = QPushButton("Extract Metrics")
        self.btn_nifti = QPushButton("Create NIfTI Volumes")
        self.btn_vis = QPushButton("Show Visualization")
        for b in (self.btn_batch, self.btn_load , self.btn_metrics, self.btn_nifti, self.btn_vis):
            b.setEnabled(False)

        self.btn_settings.clicked.connect(self.settings)
        self.btn_batch.clicked.connect(self.batch_process)
        self.btn_load.clicked.connect(self.load_patient_data)
        self.btn_metrics.clicked.connect(self.extract_metrics)
        self.btn_nifti.clicked.connect(self.create_nifti_volumes)
        self.btn_vis.clicked.connect(self.show_visualization)

        bar = QHBoxLayout()
        [bar.addWidget(w) for w in (self.btn_settings, self.btn_batch , self.btn_load, self.btn_metrics, self.btn_nifti, self.btn_vis)]
        bar.addStretch()

        # ── patient info display
        self.patient_info_label = QLabel()
        self.patient_info_label.setStyleSheet("font: 14px 'Courier New'; padding: 5px;")
        self.patient_info_label.setAlignment(Qt.AlignLeft)
        self.patient_info_label.setText("🧾 No patient selected yet.")

        # ── settings info display
        self.settings_info_label = QLabel()
        self.settings_info_label.setStyleSheet("font: 14px 'Courier New'; padding: 5px;")
        self.settings_info_label.setAlignment(Qt.AlignLeft)
        self.settings_info_label.setText("🧾 No settings defined yet.")

        # ── tables + DVH plot
        self.tbl_meta = QTableView()
        self.tbl_roi = QTableView()
        self._make_table_interactive(self.tbl_meta)
        self._make_table_interactive(self.tbl_roi)

        self.dvh_widget = DVHPlotWidget()

        split_tables = QSplitter(Qt.Horizontal)
        split_tables.addWidget(self.tbl_meta)
        split_tables.addWidget(self.tbl_roi)
        split_tables.setStretchFactor(0, 1)
        split_tables.setStretchFactor(1, 4)

        split_vert = QSplitter(Qt.Vertical)
        split_vert.addWidget(split_tables)
        split_vert.addWidget(self.dvh_widget)
        split_vert.setStretchFactor(0, 3)
        split_vert.setStretchFactor(1, 2)

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(self.settings_info_label)
        root.addWidget(self.patient_info_label)
        root.addWidget(split_vert)

    # ────────────────────────── UI helpers ────────────────────────────────
    def clear_all_data(self):
        self.tbl_meta.setModel(None)
        self.tbl_roi.setModel(None)
        self.dvh_widget.clear()
        self.metric_data = None

    def _make_table_interactive(self, tbl: QTableView):
        hh = tbl.horizontalHeader()
        vh = tbl.verticalHeader()
        hh.setSectionsMovable(True)
        hh.setSectionsClickable(True)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(False)
        vh.setSectionsMovable(True)
        vh.setSectionsClickable(True)
        vh.setSectionResizeMode(QHeaderView.Interactive)
        tbl.setSortingEnabled(True)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tbl.setAlternatingRowColors(True)

    # ══════════════════════════ Main actions ═════════════════════════════
    def settings(self):
        # Show combined dialog
        dialog = SettingsDialog(default_output_dir="", parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        # Extract values
        settings = dialog.get_values()
        self.resample_type = settings["resample_type"]
        self.resample_value = settings["resample_value"]
        self.outputs_dir = settings["output_dir"]
        info_text = (
            f"⚙️ Settings:\n"
            f"   • Resampling Type: {self.resample_type}\n"
            f"   • Resampling Value: {self.resample_value}\n"
            f"   • Output Path: {self.outputs_dir}"
        )
        self.settings_info_label.setText(info_text)

        # Parse resample values
        try:
            if self.resample_type == "shape":
                self.new_size = tuple(map(int, self.resample_value.split(",")))
                self.new_spacing = None
            else:
                self.new_spacing = tuple(map(float, self.resample_value.split(",")))
                self.new_size = None
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid resample values.")
            return
        self.btn_load.setEnabled(True)
        self.btn_batch.setEnabled(True)

    def load_patient_data(self):
        # Let user select patient data
        self.patient_data = QFileDialog.getExistingDirectory(self, "Select patient DICOM folder")
        if not self.patient_data:
            return
        self.patient_name = Path(self.patient_data).name

        self.patient_output_dir = os.path.join(self.outputs_dir, Path(self.patient_data).name, f"Resample by {self.resample_type}", self.resample_value)
        os.makedirs(self.patient_output_dir, exist_ok=True)

        # Parse and display patient metadata
        meta = parse_patient_metadata(self.patient_name)
        info_text = (
            f"📁 Processing Patient Record:\n"
            f"   • Patient ID: {meta['Patient ID']}\n"
            f"   • Treatment Type: {meta['Treatment']}\n"
            f"   • Targeted Region: {meta['Region']}\n"
            f"   • Session Date: {meta['Date']}\n"
            f"   • Input Path: {self.patient_data}"
        )
        self.patient_info_label.setText(info_text)
        self.clear_all_data()
        self.btn_metrics.setEnabled(True)
        self.btn_nifti.setEnabled(True)

    def extract_metrics(self):
        # Open dialog to show extraction progress
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.setWindowTitle("Extracting Dosimetric and Volumetric Metrics…")
        self.processing_dialog.show()

        # Set up thread and worker
        self.metrics_thread = QThread()
        self.metrics_worker = MetricsWorker(self.patient_data, self.patient_output_dir)
        self.metrics_worker.moveToThread(self.metrics_thread)

        # Connect signals
        self.metrics_thread.started.connect(self.metrics_worker.run)
        self.metrics_worker.log.connect(self.processing_dialog.write)
        self.metrics_worker.error.connect(self.processing_dialog.show_error)

        def handle_result(result):
            self.metric_data = result
            self.tbl_meta.setModel(PandasModel(result["Meta_DF"]))
            self.tbl_meta.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

            self.tbl_roi.setModel(PandasModel(result["ROI_DF"]))
            self.tbl_roi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            self.dvh_widget.set_data(result["DVH"], result["Rx"])

        self.metrics_worker.result.connect(handle_result)

        def cleanup():
            self.processing_dialog.mark_done()
            self.processing_dialog.close()
            self.metrics_thread.quit()
            self.metrics_thread.wait()
            self.metrics_worker.deleteLater()
            self.metrics_thread.deleteLater()

        # Start thread
        self.metrics_worker.finished.connect(cleanup)
        self.metrics_thread.start()

    def create_nifti_volumes(self):
        # Show progress/log dialog
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.show()

        self.preprocess_thread = PreprocessWorker(
            patient_data=self.patient_data,
            patient_output_dir=self.patient_output_dir,
            resample_type=self.resample_type,
            new_size=self.new_size,
            new_spacing=self.new_spacing
        )
        self.preprocess_thread.log.connect(self.processing_dialog.write)

        def on_finish(res):
            self.processing_dialog.mark_done()
            self.processing_dialog.close()
            ok, msg, out_dir = res.get("ok", False), res.get("msg", ""), res.get("out_dir", "")
            if not ok:
                QMessageBox.critical(self, "Preprocess error", msg)
            else:
                self.btn_vis.setEnabled(True)

        self.preprocess_thread.finished.connect(on_finish)
        self.preprocess_thread.start()

    # ------------------------------------------------------------------
    def show_visualization(self):
        CT, Dose, ROIs = load_preprocessed_volumes(self.patient_output_dir)
        plot_combined_plot(CT, Dose, ROIs)

    # ------------------------------------------------------------------
    def batch_process(self):
        # Let user select patient data
        batch_data = QFileDialog.getExistingDirectory(self, "Select batch folder")
        if not batch_data:
            return

        patients_files = [
            patient_files_folder
            for patient_folder in glob.glob(os.path.join(batch_data, "*"))
            if os.path.isdir(patient_folder)
            for patient_files_folder in glob.glob(os.path.join(patient_folder, "CT", "*"))
            if os.path.isdir(patient_files_folder)
        ]

        if not patients_files:
            QMessageBox.warning(self, "No Patients Found", "No valid patient folders found in the selected batch.")
            return

        # Store patient list and index for sequential processing
        self.batch_patients = patients_files
        self.current_patient_index = 0

        # Set up dialog
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.setWindowTitle("Processing Batch…")
        self.processing_dialog.show()

        # Start first patient
        self.process_next_patient()

    def process_next_patient(self):
        if self.current_patient_index >= len(self.batch_patients):
            self.processing_dialog.write("\n✅ Batch processing completed.")
            self.processing_dialog.close()
            return

        patient_data = self.batch_patients[self.current_patient_index]
        meta = parse_patient_metadata(Path(patient_data).name)

        self.processing_dialog.write(
            f"📁 Processing Patient Record ({self.current_patient_index + 1}/{len(self.batch_patients)})\n"
            f"   • Patient ID: {meta['Patient ID']}\n"
            f"   • Treatment Type: {meta['Treatment']}\n"
            f"   • Targeted Region: {meta['Region']}\n"
            f"   • Session Date: {meta['Date']}\n"
            f"   • DICOM path: {patient_data}\n\n"
        )

        patient_output_dir = os.path.join(
            self.outputs_dir,
            Path(patient_data).name,
            f"Resample by {self.resample_type}",
            self.resample_value
        )
        os.makedirs(patient_output_dir, exist_ok=True)

        # Create and run worker
        self.metrics_thread = QThread()
        self.metrics_worker = MetricsWorker(patient_data, patient_output_dir)
        self.metrics_worker.moveToThread(self.metrics_thread)

        # Connect signals
        self.metrics_thread.started.connect(self.metrics_worker.run)
        self.metrics_worker.log.connect(self.processing_dialog.write)
        self.metrics_worker.error.connect(self.processing_dialog.show_error)
        self.metrics_worker.finished.connect(self.handle_worker_finished)

        # Start thread
        self.metrics_thread.start()

    def handle_worker_finished(self):
        # Clean up metrics worker
        self.metrics_thread.quit()
        self.metrics_thread.wait()
        self.metrics_worker.deleteLater()
        self.metrics_thread.deleteLater()

        # Start preprocessing
        self.start_preprocessing_for_current_patient()

    def start_preprocessing_for_current_patient(self):
        patient_data = self.batch_patients[self.current_patient_index]
        patient_output_dir = os.path.join(
            self.outputs_dir,
            Path(patient_data).name,
            f"Resample by {self.resample_type}",
            self.resample_value
        )

        # Start PreprocessWorker
        self.preprocess_thread = PreprocessWorker(
            patient_data=patient_data,
            patient_output_dir=patient_output_dir,
            resample_type=self.resample_type,
            new_size=self.new_size,
            new_spacing=self.new_spacing
        )

        def on_finish(res):
            self.processing_dialog.write(f"✅ Finished preprocessing for: {Path(patient_data).name}\n")

            # Move to next patient
            self.current_patient_index += 1
            self.process_next_patient()

        self.preprocess_thread.log.connect(self.processing_dialog.write)
        self.preprocess_thread.finished.connect(on_finish)
        self.preprocess_thread.start()

    # ── helpers copied from your metrics UI ─────────────────────────────
    def _choose_multiple_dirs_under_parent(self, parent: Path):
        dlg = QFileDialog(self, "Select patients/sessions (multi-select)")
        dlg.setDirectory(str(parent))
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        for view in dlg.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.MultiSelection)
        if dlg.exec_() != QFileDialog.Accepted:
            return []
        picks = [Path(p) for p in dlg.selectedFiles() if Path(p).is_dir()]
        picks = [p for p in picks if p.parent == parent]
        return picks

    def _likely_has_ct(self, folder: Path, max_files: int = 200) -> bool:
        try:
            dcm_files = [p for p in folder.rglob("*.dcm")]
            if not dcm_files:
                return False
            uid_ct = "1.2.840.10008.5.1.4.1.1.2"
            for f in dcm_files[:max_files]:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    if getattr(ds, "Modality", "").upper() == "CT":
                        return True
                    if str(getattr(ds, "SOPClassUID", "")) == uid_ct:
                        return True
                    if "ct" in str(getattr(ds, "SeriesDescription", "")).lower():
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _discover_session_folders(self, root_path: Path):
        if self._contains_dicoms_nonrecursive(root_path):
            return [root_path]
        level1_dirs = [p for p in root_path.iterdir() if p.is_dir()]
        sessions = []
        for d in level1_dirs:
            if self._contains_dicoms_nonrecursive(d) or self._looks_like_session(d):
                sessions.append(d)
        if sessions:
            return sorted(set(sessions), key=lambda x: x.name)
        return []

    def _looks_like_session(self, folder: Path, max_checks: int = 200) -> bool:
        try:
            modalities = set()
            checked = 0
            for f in folder.rglob("*"):
                if not f.is_file():
                    continue
                checked += 1
                if f.suffix.lower() in {".dcm", ".dicom"}:
                    modalities.add("ANY")
                else:
                    try:
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                        mod = str(getattr(ds, "Modality", "")).upper()
                        if mod:
                            modalities.add(mod)
                        elif getattr(ds, "SOPClassUID", None):
                            modalities.add("ANY")
                    except Exception:
                        pass
                if {"CT", "RTDOSE", "RTPLAN", "RTSTRUCT"} & modalities or ("ANY" in modalities and checked >= 20):
                    return True
                if checked >= max_checks:
                    break
        except Exception:
            return False
        return False

    def _contains_dicoms_nonrecursive(self, folder: Path, max_checks: int = 200) -> bool:
        checked = 0
        for f in folder.iterdir():
            if not f.is_file():
                continue
            checked += 1
            if f.suffix.lower() in {".dcm", ".dicom"}:
                return True
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if getattr(ds, "SOPClassUID", None) or getattr(ds, "Modality", None):
                    return True
            except Exception:
                pass
            if checked >= max_checks:
                break
        return False



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousUnifiedGUI()
    gui.show()
    sys.exit(app.exec_())