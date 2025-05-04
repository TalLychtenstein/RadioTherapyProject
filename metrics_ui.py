import gradio as gr
from new_utils_2 import (
    load_RT_data, load_CT_data, extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, create_ROIs_volume,
    extract_group1_metadata, extract_group2_metrics,
    get_group2_roi_names
)
import matplotlib.pyplot as plt
import statistics

# Load once
RT_files_path = r"C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\52724\CT radiation maps\52724_radiation plans_30062020"
CT_files_path = r"C:\Users\dansa\PycharmProjects\pythonProject2\DICOM files\52724\CT anatomy\52724_SRS_2 METS_30062020"
RD_data, RS_data, RP_data = load_RT_data(RT_files_path)
CT_data = load_CT_data(CT_files_path)
Dose_data = preprocess_Dose_to_CT(extract_dose_data(RD_data), CT_data)
ROIs_data = extract_ROIs_data(RS_data)
ROIs_data["Volume"] = create_ROIs_volume(ROIs_data, CT_data)
group1 = extract_group1_metadata(RP_data, RD_data)

# Extract for all ROIs
group2 = extract_group2_metrics(Dose_data, ROIs_data, prescribed_dose=group1["Prescription Dose [Gy]"])
roi_names = get_group2_roi_names(ROIs_data)  # all names

# --------- Plot histogram instead of DVH ----------
def show_group1():
    return group1, None

def show_group2_histogram(roi_name):
    if roi_name not in group2:
        return "ROI not found", None

    data = group2[roi_name]
    dvh = data["DVH"]
    doses = dvh["dose_bins"]
    counts = dvh["voxel_counts"]

    # Expand dose array
    expanded = []
    for d, c in zip(doses, counts):
        expanded.extend([d] * c)

    if not expanded:
        return "Empty ROI", None

    mean = statistics.mean(expanded)
    median = statistics.median(expanded)

    fig, ax = plt.subplots()
    ax.hist(expanded, bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel("Gray")
    ax.set_ylabel("Number of Voxels")
    ax.set_title(f"Dose Histogram: {roi_name}")
    ax.text(0.95, 0.95, f"Mean = {mean:.4f}", ha='right', va='top', transform=ax.transAxes)
    ax.text(0.95, 0.90, f"Median = {median:.4f}", ha='right', va='top', transform=ax.transAxes)
    plt.tight_layout()

    # Metrics for JSON display
    metrics = {
        "Volume [cc]": data["Volume [cc]"],
        "D2% [Gy]": data["D2% [Gy]"],
        "D50% [Gy]": data["D50% [Gy]"],
        "D98% [Gy]": data["D98% [Gy]"],
        "Homogeneity Index": data["Homogeneity Index"],
        "Conformity Index": data["Conformity Index"]
    }

    return metrics, fig

# --------- Gradio App ----------
with gr.Blocks() as demo:
    gr.Markdown("# Radiotherapy Metrics Viewer")

    group = gr.Radio(["Group 1: Plan Metadata", "Group 2: ROI-Based Metrics"], label="Select Group")
    roi_dropdown = gr.Dropdown(choices=roi_names, visible=False, label="Select ROI")
    out_metrics = gr.JSON(label="Metrics")
    out_plot = gr.Plot(label="Dose Histogram")

    def toggle_dropdown(selected_group):
        return gr.update(visible=(selected_group == "Group 2: ROI-Based Metrics"))

    group.change(toggle_dropdown, inputs=group, outputs=roi_dropdown)

    group.select(
        lambda g: show_group1() if g == "Group 1: Plan Metadata" else (None, None),
        inputs=group,
        outputs=[out_metrics, out_plot]
    )

    roi_dropdown.change(
        show_group2_histogram,
        inputs=roi_dropdown,
        outputs=[out_metrics, out_plot]
    )

demo.launch()
