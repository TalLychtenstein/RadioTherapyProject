import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation

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


from pydicom.pixel_data_handlers.util import apply_color_lut, apply_modality_lut, apply_voi_lut


def plot_CT_view(volume, view='axial'):
    """
    Plot CT slices in a specified anatomical view with a slider.

    Parameters:
        volume: 3D NumPy array of shape (Z, Y, X)
        view: 'axial', 'sagittal', or 'coronal'
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    cmap = plt.cm.gray.copy()
    cmap.set_bad(color='black')

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
    img = ax.imshow(get_slice(index), cmap=cmap, vmin=900, vmax=1200)
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

    dose_data = get_slice(index)

    normed = dose_data / np.max(volume)
    normed = np.clip(normed, 0, 1)
    colored = plt.cm.jet(normed)
    alpha = np.where((dose_data < vmin) | (dose_data > vmax), 0.0, dose_data)
    colored[..., 3] = alpha

    img = ax.imshow(colored)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_clim(0, np.max(volume))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Dose (Gy)")

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, f'{view.capitalize()} Slice', 0, max_index, valinit=index, valstep=1)

    def update(val):
        idx = int(slider.val)

        dose_data = get_slice(idx)

        normed = dose_data / np.max(volume)
        normed = np.clip(normed, 0, 1)
        colored = plt.cm.jet(normed)
        alpha = np.where((dose_data < vmin) | (dose_data > vmax), 0.0, dose_data)
        colored[..., 3] = alpha

        img.set_data(colored)
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

def dilate_labeled_rois(roi_slice, iterations=3):
    """Dilate each ROI label individually."""
    dilated = np.zeros_like(roi_slice)
    unique_labels = np.unique(roi_slice)
    for label in unique_labels:
        mask = roi_slice == label
        dilated_mask = binary_dilation(mask, iterations=iterations)
        dilated[dilated_mask] = label
    return dilated


def black_to_jet_colormap():
    jet = plt.cm.get_cmap('jet', 256)
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[0] = [0, 0, 0, 1]  # Replace the first color with black
    return mcolors.ListedColormap(newcolors, name='black_jet')

def plot_Dose_on_CT(CT_data, Dose_data, ROIs_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.38)

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

    unique_rois = np.unique(ROIs_data['Volume'].astype(int))
    unique_rois = unique_rois[unique_rois != 0]
    color_list = [plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)] for i in range(len(unique_rois))]

    roi_patches = []
    for i, label in enumerate(unique_rois):
        name = ROIs_data['ROIs'][str(label)]['Name']
        roi_patches.append(mpatches.Patch(color=color_list[i], label=name))

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

    imgs_ct, imgs_dose, imgs_roi, sliders = [], [], [], []
    dose_min = 0
    dose_max = np.max(Dose_data['Volume'])

    cmap = plt.cm.gray.copy()
    cmap.set_bad(color='black')

    def update_dose_image(dose_data, view_index):
        min_val = dose_min_slider.val
        max_val = dose_max_slider.val

        normed = dose_data / np.max(Dose_data['Volume'])
        normed = np.clip(normed, 0, 1)
        colored = plt.cm.jet(normed)
        alpha = np.where((dose_data < min_val) | (dose_data > max_val), 0.0, dose_data)
        colored[..., 3] = alpha

        imgs_dose[view_index].set_data(colored)

    for i, view in enumerate(views):
        ax = axes[i]
        ct_data = get_ct[view](0)
        dose_data = get_dose[view](0)

        ct_img = ax.imshow(ct_data, cmap=cmap, vmin=900, vmax=1200, origin=origins[view])

        normed = dose_data / np.max(Dose_data['Volume'])
        normed = np.clip(normed, 0, 1)
        colored = plt.cm.jet(normed)
        alpha = np.where((dose_data < dose_min) | (dose_data > dose_max), 0.0, dose_data)
        colored[..., 3] = alpha

        dose_img = ax.imshow(colored, origin=origins[view], alpha=0.4)

        roi_data = get_roi[view](0)
        # roi_data = dilate_labeled_rois(roi_data)
        # roi_data = np.where(roi_data, roi_data, 0)
        roi_img = ax.imshow(roi_data, cmap=roi_cmap, norm=roi_norm, alpha=0.6, origin=origins[view])

        ax.set_title(f"{view.capitalize()} Slice 0")
        ax.axis('off')

        imgs_ct.append(ct_img)
        imgs_dose.append(dose_img)
        imgs_roi.append(roi_img)

    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_clim(0, np.max(Dose_data['Volume']))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Dose (Gy)")

    fig.legend(handles=roi_patches, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

    slider_axes = [
        plt.axes([0.25, 0.25, 0.5, 0.02]),
        plt.axes([0.25, 0.20, 0.5, 0.02]),
        plt.axes([0.25, 0.15, 0.5, 0.02])
    ]

    dose_min_ax = plt.axes([0.25, 0.08, 0.5, 0.02])
    dose_max_ax = plt.axes([0.25, 0.03, 0.5, 0.02])

    dose_min_slider = Slider(dose_min_ax, 'Min Dose', 0.0, dose_max, valinit=dose_min, valstep=0.001)
    dose_max_slider = Slider(dose_max_ax, 'Max Dose', 0.0, dose_max, valinit=dose_max, valstep=0.001)
    roi_patches_by_axis = [[] for _ in range(3)]  # One list per axis

    def refresh_all():
        for i, view in enumerate(views):

            idx = int(sliders[i].val)
            ct_data = get_ct[view](idx)
            dose_data = get_dose[view](idx)
            ct_data = np.ma.masked_where(dose_data <= 0, ct_data)

            imgs_ct[i].set_data(ct_data)
            update_dose_image(dose_data, i)

            for patch in roi_patches_by_axis[i]:
                patch.remove()
            roi_patches_by_axis[i].clear()

            roi_data = get_roi[view](idx)
            # imgs_roi[i].set_data(roi_data)
            for j, label in enumerate(unique_rois):
                mask = roi_data == label

                if mask.max():
                    points = np.argwhere(mask)
                    if len(points) < 3:
                        continue
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    # Convert contour pixel coordinates to physical coordinates based on spacing
                    if view == 'axial':
                        x_coords = hull_points[:, 1] * CT_data["Spacing"][2]
                        y_coords = hull_points[:, 0] * CT_data["Spacing"][1]
                    elif view == 'coronal':
                        x_coords = hull_points[:, 1] * CT_data["Spacing"][2]
                        y_coords = hull_points[:, 0] * CT_data["Spacing"][0]
                    elif view == 'sagittal':
                        x_coords = hull_points[:, 1] * CT_data["Spacing"][1]
                        y_coords = hull_points[:, 0] * CT_data["Spacing"][0]
                    else:
                        x_coords = hull_points[:, 1]
                        y_coords = hull_points[:, 0]

                    if i == 0:
                        y_coords = view_extents[view][3] - (y_coords - view_extents[view][2])

                    line, = axes[i].plot(x_coords, y_coords, color=color_list[j], linewidth=1.5)
                    roi_patches_by_axis[i].append(line)

            axes[i].set_title(f"{view.capitalize()} Slice {idx}")

        fig.canvas.draw_idle()

    dose_min_slider.on_changed(lambda val: refresh_all())
    dose_max_slider.on_changed(lambda val: refresh_all())

    view_extents = {
        'axial': None,
        'coronal': None,
        'sagittal': None
    }

    for i, view in enumerate(views):
        view_slider = Slider(slider_axes[i], f'{view.capitalize()} Slice', 0, shapes[view] - 1, valinit=0, valstep=1)

        ct_data = get_ct[view](int(len(CT_data['Volume']) / 2))
        dose_data = get_dose[view](int(len(Dose_data['Volume']) / 2))
        # Compute physical extent for current view
        if view == 'axial':
            extent = [0, CT_data["Spacing"][2] * ct_data.shape[1], 0, CT_data["Spacing"][1] * ct_data.shape[0]]
        elif view == 'coronal':
            extent = [0, CT_data["Spacing"][2] * ct_data.shape[1], 0, CT_data["Spacing"][0] * ct_data.shape[0]]
        elif view == 'sagittal':
            extent = [0, CT_data["Spacing"][1] * ct_data.shape[1], 0, CT_data["Spacing"][0] * ct_data.shape[0]]
        else:
            extent = None

        view_extents[view] = extent

        if extent is not None:
            imgs_ct[i].set_extent(extent)
            imgs_dose[i].set_extent(extent)
            imgs_roi[i].set_extent(extent)
            axes[i].set_aspect('auto')

        # Zoom to dose mask bounding box
        mask = dose_data > 0
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            margin = 25  # pixels

            y_min = max(0, y_min - margin)
            y_max = min(mask.shape[0] - 1, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(mask.shape[1] - 1, x_max + margin)

            # Convert pixel bounds to physical units using spacing
            if view == 'axial':
                axes[i].set_xlim(x_min * CT_data["Spacing"][2], x_max * CT_data["Spacing"][2])
                axes[i].set_ylim(y_min * CT_data["Spacing"][1], y_max * CT_data["Spacing"][1])
            elif view == 'coronal':
                axes[i].set_xlim(x_min * CT_data["Spacing"][2], x_max * CT_data["Spacing"][2])
                axes[i].set_ylim(y_min * CT_data["Spacing"][0], y_max * CT_data["Spacing"][0])
            elif view == 'sagittal':
                axes[i].set_xlim(x_min * CT_data["Spacing"][1], x_max * CT_data["Spacing"][1])
                axes[i].set_ylim(y_min * CT_data["Spacing"][0], y_max * CT_data["Spacing"][0])
        else:
            axes[i].autoscale()

        def make_update_func(view=view, i=i):
            def update(val):
                refresh_all()

            return update

        view_slider.on_changed(make_update_func())
        sliders.append(view_slider)

    plt.show()


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import patches
from skimage import measure
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib.patches import Polygon


def connect_contour_points(contour):
    contour = contour.copy()
    used = np.zeros(len(contour), dtype=bool)
    path = [0]
    used[0] = True

    for _ in range(1, len(contour)):
        last = contour[path[-1]]
        dists = np.linalg.norm(contour - last, axis=1)
        next_idx = np.argmin(dists)
        path.append(next_idx)
        used[next_idx] = True

    return contour[path]


def plot_ROI_contours(ROIs_data):
    views = ['axial', 'coronal', 'sagittal']
    get_roi = {
        'axial': lambda i: ROIs_data['Volume'][i, :, :],
        'coronal': lambda i: ROIs_data['Volume'][:, i, :],
        'sagittal': lambda i: ROIs_data['Volume'][:, :, i]
    }

    shapes = {
        'axial': ROIs_data['Volume'].shape[0],
        'coronal': ROIs_data['Volume'].shape[1],
        'sagittal': ROIs_data['Volume'].shape[2]
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    unique_rois = np.unique(ROIs_data['Volume'].astype(int))
    unique_rois = unique_rois[unique_rois != 0]
    color_list = [plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)] for i in range(len(unique_rois))]

    contour_lines_by_view = [[] for _ in range(3)]  # to clear and redraw

    def draw_contours(ax, roi_slice, view_index, view):
        for line in contour_lines_by_view[view_index]:
            line.remove()
        contour_lines_by_view[view_index].clear()

        for j, label in enumerate(unique_rois):
            mask = roi_slice == label
            if mask.max():
                contours = measure.find_contours(mask)
                for contour in contours:
                    x_coords = [xy[1] for xy in contour]
                    y_coords = [xy[0] for xy in contour]

                line, = ax.plot(x_coords, y_coords, color=color_list[j], linewidth=1.5)
                contour_lines_by_view[view_index].append(line)

    sliders = []
    slider_axes = [
        plt.axes([0.25, 0.18, 0.5, 0.02]),
        plt.axes([0.25, 0.13, 0.5, 0.02]),
        plt.axes([0.25, 0.08, 0.5, 0.02])
    ]

    for i, view in enumerate(views):
        idx = shapes[view] // 2
        ax = axes[i]
        roi_slice = get_roi[view](idx)

        draw_contours(ax, roi_slice, i, view)

        ax.set_title(f"{view.capitalize()} Slice {idx}")
        ax.set_aspect('equal')
        ax.axis('off')

        slider = Slider(slider_axes[i], f'{view.capitalize()} Slice', 0, shapes[view] - 1, valinit=idx, valstep=1)

        def make_update_func(i=i, view=view):
            def update(val):
                idx = int(val)
                roi_slice = get_roi[view](idx)
                draw_contours(axes[i], roi_slice, i, view)
                axes[i].set_title(f"{view.capitalize()} Slice {idx}")
                fig.canvas.draw_idle()

            return update

        slider.on_changed(make_update_func())
        sliders.append(slider)

    roi_patches = [patches.Patch(color=color_list[i], label=ROIs_data['ROIs'][str(label)]['Name']) for i, label in
                   enumerate(unique_rois)]
    fig.legend(handles=roi_patches, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

    plt.show()

def interactive_3axis_viewer(volume, title="3-Axis Viewer"):
    """
    Interactive viewer for 3D volume with sliders along Z, Y, and X axes.

    Parameters:
        volume (3D np.ndarray): (Z, Y, X)
        title (str): Window title
    """
    z_dim, y_dim, x_dim = volume.shape

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)

    # Initial slices
    z_init, y_init, x_init = 0, 0, 0

    # Display images
    im_z = axs[0].imshow(volume[z_init], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title(f'Axial (Z={z_init})')

    im_y = axs[1].imshow(volume[:, y_init, :], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title(f'Coronal (Y={y_init})')

    im_x = axs[2].imshow(volume[:, :, x_init], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title(f'Sagittal (X={x_init})')

    for ax in axs:
        ax.axis('off')

    # Sliders
    ax_z = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_y = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_x = plt.axes([0.15, 0.05, 0.7, 0.03])

    slider_z = Slider(ax_z, 'Z', 0, z_dim - 1, valinit=z_init, valstep=1)
    slider_y = Slider(ax_y, 'Y', 0, y_dim - 1, valinit=y_init, valstep=1)
    slider_x = Slider(ax_x, 'X', 0, x_dim - 1, valinit=x_init, valstep=1)

    def update(val):
        z = int(slider_z.val)
        y = int(slider_y.val)
        x = int(slider_x.val)

        im_z.set_data(volume[z])
        axs[0].set_title(f'Axial (Z={z})')

        im_y.set_data(volume[:, y, :])
        axs[1].set_title(f'Coronal (Y={y})')

        im_x.set_data(volume[:, :, x])
        axs[2].set_title(f'Sagittal (X={x})')

        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)

    plt.suptitle(title, fontsize=16)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, CheckButtons
from skimage.measure import find_contours

def plot_combined_plot(CT_data, Dose_data, ROIs_data):
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



