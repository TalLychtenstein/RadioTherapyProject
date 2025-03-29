import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from shapely import Point, within, Polygon
import os


def show_dose_map(rd_data):
    """Displays dose maps as heatmaps."""
    grid_scaling = rd_data.DoseGridScaling
    Vmax = rd_data.pixel_array.max()
    Vmax1 = Vmax * grid_scaling
    for n in range(147, len(rd_data.pixel_array)):
        dose_map = np.flipud(rd_data.pixel_array[n])
        plt.imshow(dose_map * grid_scaling, cmap='jet', origin='lower', vmin=0, vmax=Vmax1)
        plt.colorbar(label='Dose (Gy)')
        plt.title(f'Dose map at slice {n}')
        plt.xlabel('X Axis (pixels)')
        plt.ylabel('Y Axis (pixels)')
        plt.show()


def draw_contour_on_dose_map(rd_data, matched_slices, x_spacing, y_spacing, slices_order, contour_color, output_dir):
    """Overlays contours on dose maps and saves the results as images."""
    grid_scaling = rd_data.DoseGridScaling
    Vmax = rd_data.pixel_array.max()
    Vmax1 = Vmax * grid_scaling
    slice_position1 = rd_data.ImagePositionPatient[0]
    slice_position2 = rd_data.ImagePositionPatient[1]

    for n in range(len(rd_data.pixel_array)):
        labels = []
        fig, ax = plt.subplots()
        dose_map = rd_data.pixel_array[n] * grid_scaling
        im = ax.imshow(dose_map, cmap='cool', origin='lower', vmin=0, vmax=Vmax1)
        plt.colorbar(im, ax=ax, label='Dose (Gy)')
        ax.set_xlabel('X Axis (pixels)')
        ax.set_ylabel('Y Axis (pixels)')
        ax.set_title(f'Dose map at slice {n}')

        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    # Extract x, y coordinates from contour points
                    x = [float(contour_points[i]) for i in range(0, len(contour_points), 3)]
                    y = [float(contour_points[i + 1]) for i in range(0, len(contour_points), 3)]

                    # Convert world coordinates to pixel coordinates
                    x_pixel = (np.array(x) - slice_position1) / x_spacing
                    y_pixel = (np.array(y) - slice_position2) / y_spacing
                    normalized_color = [c / 255.0 for c in contour_color[roi_name]]

                    if roi_name in labels:
                        ax.plot(x_pixel, y_pixel, color=normalized_color, linewidth=1)
                    else:
                        ax.plot(x_pixel, y_pixel, color=normalized_color, label=roi_name, linewidth=1)
                        labels.append(roi_name)

        if labels:
            ax.legend(loc="upper left", fontsize="9")

        # Save the plot as an image file
        filename = f"contour_on_dose_map/slice_{n}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory


def draw_dose_map_on_CT(rd_data, matched_slices, slices_order, x_spacing, y_spacing, contour_color, output_dir):
    """Overlays dose maps on CT images and saves the results as images."""
    grid_scaling = rd_data.DoseGridScaling
    dose_position = rd_data.ImagePositionPatient

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for n in range(len(rd_data.pixel_array)):
        labels = []
        dose_grid = rd_data.pixel_array[n]

        fig, ax = plt.subplots()
        dose_im = None  # Initialize dose_im to ensure it's defined

        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    scale_factors = [rd_data.PixelSpacing[0] / x_spacing, rd_data.PixelSpacing[1] / y_spacing]
                    resampled_dose = zoom(dose_grid, scale_factors, order=1)
                    Vmax = resampled_dose.max()
                    Vmax_scaled = Vmax * grid_scaling

                    y_offset = round((dose_position[0] - slice_position[0]) / x_spacing)
                    x_offset = round((dose_position[1] - slice_position[1]) / y_spacing)

                    new_ct = np.zeros_like(resampled_dose)

                    for x in range(new_ct.shape[0]):
                        for y in range(new_ct.shape[1]):
                            x_index = x + x_offset
                            y_index = y + y_offset
                            if 0 <= x_index < ct_image.shape[0] and 0 <= y_index < ct_image.shape[1]:
                                new_ct[x, y] = ct_image[int(x_index), int(y_index)]

                    ax.imshow(new_ct, cmap='gray', origin='lower')
                    dose_im = ax.imshow(resampled_dose * grid_scaling, cmap='cool', origin='lower', vmin=0,
                                        vmax=Vmax_scaled, alpha=0.5)

                    x = [float(contour_points[i]) for i in range(0, len(contour_points), 3)]
                    y = [float(contour_points[i + 1]) for i in range(0, len(contour_points), 3)]

                    x_pixel = ((np.array(x) - slice_position[0]) / x_spacing) - y_offset
                    y_pixel = (np.array(y) - slice_position[1]) / y_spacing - x_offset
                    normalized_color = [c / 255.0 for c in contour_color[roi_name]]

                    if roi_name in labels:
                        ax.plot(x_pixel, y_pixel, color=normalized_color, linewidth=1)
                    else:
                        ax.plot(x_pixel, y_pixel, color=normalized_color, label=roi_name, linewidth=1)
                        labels.append(roi_name)

        if labels:
            ax.legend(loc="upper left", fontsize="9")

        ax.set_title(f'Dose map at slice {n}')
        ax.set_xlabel('X Axis (pixels)')
        ax.set_ylabel('Y Axis (pixels)')

        if dose_im is not None:
            cbar = fig.colorbar(dose_im, ax=ax, label='Dose (Gy)')
        else:
            print(f"No dose image for slice {n}; skipping colorbar.")

        filename = f"dose_map_on_CT/slice_{n}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory


def create_histogram(rd_data, matched_slices, slices_order, x_spacing, y_spacing, rs_contour_data, contour_color):
    """Generates histograms by summing radiation doses within each ROI polygon.

    Args:
        rd_data (pydicom.Dataset): DICOM RD file data.
        matched_slices (dict): Matched slices containing contour and CT data.
        slices_order (list): Sorted CT slice positions.
        x_spacing (float): X-axis pixel spacing.
        y_spacing (float): Y-axis pixel spacing.
        rs_contour_data (dict): Extracted contour data from RTSTRUCT file.
        contour_color (dict): Dictionary of ROI colors.

    Returns:
        dict: Dictionary containing radiation values and ROI areas.
    """
    dose_position = rd_data.ImagePositionPatient
    grid_scaling = rd_data.DoseGridScaling
    histograms = {}
    for n in range(len(rd_data.pixel_array)):
        dose_grid = rd_data.pixel_array[n]
        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    contour_data_list = rs_contour_data[roi_name]
                    for contour_data in contour_data_list:
                        if contour_data[0][2] == np.float64("{:.2f}".format(slices_order[n])):
                            # Outer Contour has been excluded for faster results
                            if roi_name != 'Outer Contour':

                                scale_factors = [rd_data.PixelSpacing[0] / x_spacing,
                                                 rd_data.PixelSpacing[1] / y_spacing]
                                resampled_dose1 = zoom(dose_grid, scale_factors, order=1)
                                resampled_dose2 = resampled_dose1
                                Vmax = resampled_dose1.max()
                                Vmax1 = Vmax * grid_scaling

                                points = []
                                for j in range(contour_data.shape[0]):
                                    points.append((contour_data[j][0] - dose_position[0]) / x_spacing)
                                    points.append((contour_data[j][1] - dose_position[1]) / y_spacing)

                                points = np.array(points).reshape(-1, 2)
                                polygon = Polygon(points)

                                radiation = []
                                for x in range(resampled_dose1.shape[1]):
                                    for y in range(resampled_dose1.shape[0]):

                                        if within(Point(x, y), polygon):
                                            radiation.append(resampled_dose1[y, x] * grid_scaling)
                                            resampled_dose2[y, x] = Vmax

                                plt.imshow(resampled_dose2 * grid_scaling, cmap='cool', origin='lower', vmin=0,
                                           vmax=Vmax1, alpha=0.5)

                                if roi_name in histograms.keys():
                                    radiation1 = histograms[roi_name] + radiation
                                    histograms[roi_name] = radiation1
                                    roi_area = histograms[roi_name + ' Area'] + polygon.area
                                    histograms[roi_name + ' Area'] = roi_area
                                else:
                                    histograms[roi_name] = radiation
                                    histograms[roi_name + ' Area'] = polygon.area
    return histograms
