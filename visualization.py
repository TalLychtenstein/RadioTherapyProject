import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

matplotlib.use('TkAgg')


def plot_Dose(Dose_map):
    """
    Plots a dose distribution map.

    Parameters:
    Dose_map: numpy.ndarray
        A 2D array representing the dose map.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(Dose_map, cmap="jet")

    # Add colorbar for dose intensity
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dose Intensity")

    plt.show()


def plot_ROIs(ROIs_data):
    """
    Plots the contours of Regions of Interest (ROIs) on a specified CT slice.

    Parameters:
    ROIs_data (dict): A list of tuples, each containing:
        - name (str): The name of the ROI.
        - color (tuple): The color of the ROI in normalized RGB format.
        - contour (list): A list of (x, y) coordinate pairs defining the contour.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    for name, color, contour in ROIs_data:
        x, y = zip(*contour)
        ax.plot(x, y, color=color, label=name, linewidth=1)

    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.legend(loc="upper left", fontsize=9)
    plt.show()


def plot_CT_slice(CT_slice):
    """
    Displays a CT scan slice as a grayscale image.

    Parameters:
    CT_data (dict): A dictionary containing CT scan slice images.
    CT_slice (int): The index of the CT slice to display.
    """
    plt.imshow(CT_slice, cmap="gray")
    plt.show()


def plot_full_image(ROIs_data, CT_slice, Dose_map, alpha=0.5):
    """
    Overlays ROIs and dose map onto a CT slice.

    Parameters:
    ROIs_data: list
        A list of tuples with ROI information.
    CT_slice: numpy.ndarray
        A 2D array representing the CT scan slice.
    Dose_map: numpy.ndarray
        A 2D array representing the dose distribution.
    alpha: float, optional
        Transparency level for the dose overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display CT slice in grayscale
    ax.imshow(CT_slice, cmap="gray", origin="lower")

    # Overlay dose map with transparency
    im = ax.imshow(Dose_map, cmap="jet", alpha=alpha)

    # Add colorbar for dose intensity
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dose Intensity")

    # Plot ROI contours
    for name, color, contour in ROIs_data:
        x = [x for (x, y) in contour]
        y = [y for (x, y) in contour]
        ax.plot(x, y, color=color, label=name, linewidth=1)

    plt.legend(loc="upper right", fontsize="9")
    plt.show()


def plot_CT_view(volume, view='axial'):
    """
    Plot CT slices in a specified anatomical view with a slider.

    Parameters:
        volume: 3D NumPy array of shape (Z, Y, X)
        view: 'axial', 'sagittal', or 'coronal'
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Determine slice axis and initial image
    if view == 'axial':
        axis = 0
        get_slice = lambda i: volume[i, :, :]
        max_index = volume.shape[0] - 1
    elif view == 'coronal':
        axis = 1
        get_slice = lambda i: volume[:, i, :]
        max_index = volume.shape[1] - 1
    elif view == 'sagittal':
        axis = 2
        get_slice = lambda i: np.flipud(volume[:, :, i])  # Flip vertically
        max_index = volume.shape[2] - 1
    else:
        raise ValueError("Invalid view. Choose from 'axial', 'sagittal', or 'coronal'.")

    index = 0
    img = ax.imshow(get_slice(index), cmap='gray')
    ax.set_title(f"{view.capitalize()} Slice {index}")
    ax.axis('off')

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, f'{view.capitalize()} Slice', 0, max_index, valinit=index, valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(get_slice(idx))
        ax.set_title(f"{view.capitalize()} Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_Dose_view(volume, view='axial'):
    """
    Plot Dose slices in a specified anatomical view with a slider.

    Parameters:
        volume: 3D NumPy array of shape (Z, Y, X)
        view: 'axial', 'sagittal', or 'coronal'
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Define which axis to slice through
    if view == 'axial':
        get_slice = lambda i: volume[i, :, :]
        max_index = volume.shape[0] - 1
    elif view == 'coronal':
        get_slice = lambda i: volume[:, i, :]
        max_index = volume.shape[1] - 1
    elif view == 'sagittal':
        get_slice = lambda i: np.flipud(volume[:, :, i])  # Flip vertically
        max_index = volume.shape[2] - 1
    else:
        raise ValueError("Invalid view. Choose from 'axial', 'coronal', or 'sagittal'.")

    index = 0
    vmin = 0
    vmax = np.max(volume)
    img = ax.imshow(get_slice(index), cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"{view.capitalize()} Slice {index}")
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Dose [Gy]')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, f'{view.capitalize()} Slice', 0, max_index, valinit=index, valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(get_slice(idx))
        ax.set_title(f"{view.capitalize()} Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_Dose_on_CT_view(ct_volume, dose_volume, view='axial', alpha=0.5):
    """
    Plot CT with dose overlay in a specified anatomical view using a slider.

    Parameters:
        ct_volume: 3D NumPy array of CT (Z, Y, X)
        dose_volume: 3D NumPy array of dose (same shape as ct_volume)
        view: 'axial', 'sagittal', or 'coronal'
        alpha: transparency for dose overlay (0 to 1)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    # Slice extractors
    if view == 'axial':
        get_ct_slice = lambda i: ct_volume[i, :, :]
        get_dose_slice = lambda i: dose_volume[i, :, :]
        max_index = ct_volume.shape[0] - 1
        origin = 'upper'
    elif view == 'coronal':
        get_ct_slice = lambda i: ct_volume[:, i, :]
        get_dose_slice = lambda i: dose_volume[:, i, :]
        max_index = ct_volume.shape[1] - 1
        origin = 'lower'
    elif view == 'sagittal':
        get_ct_slice = lambda i: ct_volume[:, :, i]
        get_dose_slice = lambda i: dose_volume[:, :, i]
        max_index = ct_volume.shape[2] - 1
        origin = 'lower'
    else:
        raise ValueError("Invalid view. Choose from 'axial', 'coronal', or 'sagittal'.")

    ct_slice = get_ct_slice(0)
    dose_slice = get_dose_slice(0)
    dose_slice = np.ma.masked_where(dose_slice <= 0, dose_slice)

    # CT grayscale base
    ax.imshow(ct_slice, cmap='gray', origin=origin)

    # Dose overlay
    im = ax.imshow(dose_slice, cmap='jet', alpha=alpha, origin=origin, vmin=0, vmax=np.max(dose_volume))
    ax.set_title(f"{view.capitalize()} Slice 0")
    ax.axis('off')

    # Colorbar for dose
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Dose [Gy]')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, f'{view.capitalize()} Slice', 0, max_index, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        ct = get_ct_slice(idx)
        dose = get_dose_slice(idx)
        dose = np.ma.masked_where(dose <= 0, dose)
        ax.images[0].set_data(ct)  # CT layer
        im.set_data(dose)  # Dose overlay
        ax.set_title(f"{view.capitalize()} Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_Dose_on_CT(ct_volume, dose_volume, alpha=0.5):
    """
    Plot CT with dose overlay in axial, sagittal, and coronal views using 3 sliders.

    Parameters:
        ct_volume: 3D NumPy array of CT (Z, Y, X)
        dose_volume: 3D NumPy array of dose (same shape)
        alpha: transparency for dose overlay (0 to 1)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.3)

    views = ['axial', 'coronal', 'sagittal']
    get_ct = {
        'axial': lambda i: ct_volume[i, :, :],
        'coronal': lambda i: ct_volume[:, i, :],
        'sagittal': lambda i: ct_volume[:, :, i]
    }
    get_dose = {
        'axial': lambda i: dose_volume[i, :, :],
        'coronal': lambda i: dose_volume[:, i, :],
        'sagittal': lambda i: dose_volume[:, :, i]
    }
    shapes = {
        'axial': ct_volume.shape[0],
        'coronal': ct_volume.shape[1],
        'sagittal': ct_volume.shape[2]
    }
    origins = {
        'axial': 'upper',
        'coronal': 'lower',
        'sagittal': 'lower'
    }

    imgs_ct = []
    imgs_dose = []
    sliders = []

    for i, view in enumerate(views):
        ax = axes[i]
        ct_slice = get_ct[view](0)
        dose_slice = np.ma.masked_where(get_dose[view](0) <= 0, get_dose[view](0))
        ct_img = ax.imshow(ct_slice, cmap='gray', origin=origins[view])
        dose_img = ax.imshow(dose_slice, cmap='jet', alpha=alpha, origin=origins[view],
                             vmin=0, vmax=np.max(dose_volume))
        ax.set_title(f"{view.capitalize()} Slice 0")
        ax.axis('off')
        imgs_ct.append(ct_img)
        imgs_dose.append(dose_img)

    # Shared colorbar
    cbar = fig.colorbar(imgs_dose[0], ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Dose [Gy]")

    # Add 3 vertically stacked sliders
    slider_axes = [
        plt.axes([0.25, 0.18, 0.5, 0.025]),  # Axial
        plt.axes([0.25, 0.12, 0.5, 0.025]),  # Coronal
        plt.axes([0.25, 0.06, 0.5, 0.025])   # Sagittal
    ]
    for i, view in enumerate(views):
        slider = Slider(slider_axes[i], f'{view.capitalize()} Slice', 0, shapes[view] - 1, valinit=0, valstep=1)

        def make_update_func(view=view, i=i):
            def update(val):
                idx = int(val)
                new_ct = get_ct[view](idx)
                new_dose = np.ma.masked_where(get_dose[view](idx) <= 0, get_dose[view](idx))
                imgs_ct[i].set_data(new_ct)
                imgs_dose[i].set_data(new_dose)
                axes[i].set_title(f"{view.capitalize()} Slice {idx}")
                fig.canvas.draw_idle()

            return update

        slider.on_changed(make_update_func())
        sliders.append(slider)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation

def dilate_labeled_rois(roi_slice, iterations=1):
    """Dilate each ROI label individually."""
    dilated = np.zeros_like(roi_slice)
    unique_labels = np.unique(roi_slice)
    for label in unique_labels:
        if label == 0:
            continue  # skip background
        mask = roi_slice == label
        dilated_mask = binary_dilation(mask, iterations=iterations)
        dilated[dilated_mask] = label
    return dilated

def plot_Dose_on_CT(CT_data, Dose_data, ROIs_data, alpha=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.3)

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
    get_roi = {
        'axial': lambda i: ROIs_data['Volume'][i, :, :],
        'coronal': lambda i: ROIs_data['Volume'][:, i, :],
        'sagittal': lambda i: ROIs_data['Volume'][:, :, i]
    }

    unique_rois = np.unique(ROIs_data['Volume'])
    unique_rois = unique_rois[unique_rois != 0]

    base_colors = plt.cm.tab20.colors
    color_list = [base_colors[i % len(base_colors)] for i in range(len(unique_rois))]

    # color_list = []
    roi_patches = []
    for i, label in enumerate(unique_rois):
        name = ROIs_data['ROIs'][str(int(label))]['Name']
        # color = ROIs_data['ROIs'][str(int(label))]['Color']
        # color = [x / 255 for x in color]
        roi_patches.append(mpatches.Patch(color=color_list[i], label=name))
        # color_list. append(color)

    roi_cmap = mcolors.ListedColormap(['none'] + color_list)
    roi_norm = mcolors.BoundaryNorm(boundaries=np.arange(0, len(unique_rois) + 2) - 0.5,
                                    ncolors=len(unique_rois) + 1)

    shapes = {
        'axial': CT_data['Volume'].shape[0],
        'coronal': CT_data['Volume'].shape[1],
        'sagittal': CT_data['Volume'].shape[2]
    }
    origins = {
        'axial': 'upper',
        'coronal': 'lower',
        'sagittal': 'lower'
    }

    imgs_ct = []
    imgs_dose = []
    imgs_roi = []
    sliders = []

    for i, view in enumerate(views):
        ax = axes[i]
        ct_slice = get_ct[view](0)
        dose_slice = np.ma.masked_where(get_dose[view](0) <= 0, get_dose[view](0))

        ct_img = ax.imshow(ct_slice, cmap='gray', origin=origins[view])
        dose_img = ax.imshow(dose_slice, cmap='jet', alpha=alpha, origin=origins[view],
                             vmin=0, vmax=np.max(Dose_data['Volume']))

        # ROI overlay
        roi_slice = get_roi[view](0)
        roi_slice = dilate_labeled_rois(roi_slice)
        roi_slice = np.ma.masked_where(roi_slice == 0, roi_slice)
        roi_img = ax.imshow(roi_slice, cmap=roi_cmap, norm=roi_norm, alpha=0.6, origin=origins[view])

        ax.set_title(f"{view.capitalize()} Slice 0")
        ax.axis('off')

        imgs_ct.append(ct_img)
        imgs_dose.append(dose_img)
        imgs_roi.append(roi_img)

    # Shared colorbar for dose
    cbar = fig.colorbar(imgs_dose[0], ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Dose [Gy]")

    fig.legend(handles=roi_patches, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

    # 3 vertically stacked sliders
    slider_axes = [
        plt.axes([0.25, 0.18, 0.5, 0.025]),  # Axial
        plt.axes([0.25, 0.12, 0.5, 0.025]),  # Coronal
        plt.axes([0.25, 0.06, 0.5, 0.025])   # Sagittal
    ]

    for i, view in enumerate(views):
        slider = Slider(slider_axes[i], f'{view.capitalize()} Slice', 0, shapes[view] - 1, valinit=0, valstep=1)

        def make_update_func(view=view, i=i):
            def update(val):
                idx = int(val)
                new_ct = get_ct[view](idx)
                new_dose = np.ma.masked_where(get_dose[view](idx) <= 0, get_dose[view](idx))

                imgs_ct[i].set_data(new_ct)
                imgs_dose[i].set_data(new_dose)
                roi_data = get_roi[view](idx)
                roi_data = dilate_labeled_rois(roi_data)
                roi_data = np.ma.masked_where(roi_data == 0, roi_data)
                imgs_roi[i].set_data(roi_data)

                axes[i].set_title(f"{view.capitalize()} Slice {idx}")
                fig.canvas.draw_idle()

            return update

        slider.on_changed(make_update_func())
        sliders.append(slider)

    plt.show()
