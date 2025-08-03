from final_utils import  load_preprocessed_volumes
from visualization import plot_combined_plot

# files_path = "Outputs/1575_SRS_LT occipital_02062021"
# files_path = "Outputs/1807_SRS_4 METS_22012020"
files_path = "Outputs/7227_WBRT_02102022"

# Load everything from current directory
CT_data, Dose_data, ROIs_data = load_preprocessed_volumes(files_path=files_path)

# Visualize
plot_combined_plot(CT_data, Dose_data, ROIs_data)
