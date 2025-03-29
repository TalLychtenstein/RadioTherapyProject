import time
import statistics

from utils import *
from visualization import *

# Define output folder for plots
output_dir = '../output_plots'
os.makedirs(output_dir, exist_ok=True)

# load data.
RD_file_name = find_file_with_prefix(
    "../DICOM files/64147/CT radiation maps/64147_radiation maps_24082022/64147_radiation maps_24082022", 'RD')
RD_data = read_dicom_rd_file(RD_file_name)  # load radiation dose information

RS_file_name = find_file_with_prefix(
    "../DICOM files/64147/CT radiation maps/64147_radiation maps_24082022/64147_radiation maps_24082022", 'RS')
RS_data = read_dicom_rs_file(RS_file_name)  # load radiation structure information
RS_contours_across_slices = load_contour_slices(RS_data)  # load radiation contours
(RS_contour_data, Contour_color) = get_contour_data(RS_data)

(CT_slices, X_spacing, Y_spacing) = load_ct_slices(
    "../DICOM files/64147/CT anatomy/64147_AnatomicCT_24082022/64147_SRS_3 METS_24082022")  # load CT slices

# temporary add
RP_file_name = find_file_with_prefix(
    "../DICOM files/64147/CT radiation maps/64147_radiation maps_24082022/64147_radiation maps_24082022", 'RP')
RP_data = read_dicom_rp_file(RP_file_name)

# Match radiation contours to CT slices.
CT_slices_order = sort_ct(CT_slices)
matched_slices = match_contour_to_ct(RS_contours_across_slices, CT_slices)

# Plot radiation contours on dose map and CT map
slice_thickness = abs(CT_slices_order[0]) - abs(CT_slices_order[1])
draw_dose_map_on_CT(RD_data, matched_slices, CT_slices_order, X_spacing, Y_spacing, Contour_color, output_dir)

# Create histogram of radiation dose in different ROIs.
start_time = time.time()
histograms = create_histogram(RD_data, matched_slices, CT_slices_order, X_spacing, Y_spacing, RS_contour_data,
                              Contour_color)
print("--- %s seconds ---" % (time.time() - start_time))

# Display the histogram of each ROI including key parameters
for roi in RS_contour_data.keys():
    # Outer Contour has been excluded for faster results
    if roi != 'Outer Contour':
        f, ax = plt.subplots()
        plt.hist(histograms[roi], bins=30, color='skyblue', edgecolor='black')
        # Adding labels and title
        plt.xlabel('Gray')
        plt.ylabel('Number of Voxels')
        plt.title(roi)

        plt.text(.51, .99, f"Min = {min(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .95, f"Max = {max(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .91, f"Mean = {statistics.mean(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .87, f"Median = {statistics.median(histograms[roi])}", ha='left', va='top',
                 transform=ax.transAxes)

        # Display the plot
        filename = f"{roi.replace(' ', '_')}.png"  # Replace spaces with underscores for the filename
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(f)  # Close the figure to free up memory
