from final_utils import process_patient_session
import SimpleITK as sitk
from visualization import plot_Dose_on_CT
import numpy as np

if __name__ == "__main__":
    dicom_dir = "../DICOM files/Healthy Brain/1807_SRS_4 METS_22012020"
    output_dir = "../outputs"

    # Run the final processing pipeline
    ct, rd, seg, metadata = process_patient_session(
        dicom_dir=dicom_dir,
        output_dir=output_dir,
        final_spacing=(1.0, 1.0, 1.0),
        crop_size=(240, 240, 120)
    )

    # Convert SITK images to NumPy arrays
    ct_array = sitk.GetArrayFromImage(ct)
    rd_array = sitk.GetArrayFromImage(rd)

    # Convert ROI dict to label map
    seg_array = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint16)
    for idx, (roi_name, mask_image) in enumerate(seg.items(), start=1):
        mask_array = sitk.GetArrayFromImage(mask_image)
        seg_array[mask_array > 0] = idx

    # Reverse spacing and origin for visualization compatibility
    ct_spacing = ct.GetSpacing()[::-1]
    ct_origin = ct.GetOrigin()[::-1]
    rd_spacing = rd.GetSpacing()[::-1]
    rd_origin = rd.GetOrigin()[::-1]

    # Construct visualization-compatible input format
    CT_data = {
        "Volume": ct_array,
        "Spacing": ct_spacing,
        "Position": ct_origin,
        "ROIs": {str(i+1): {"Name": m["name"], "Color": m["color"]} for i, m in enumerate(metadata)}
    }

    Dose_data = {
        "Volume": rd_array,
        "Spacing": rd_spacing,
        "Position": rd_origin,
    }

    ROIs_data = {
        "Volume": seg_array,
        "ROIs": CT_data["ROIs"]
    }

    # Launch interactive dose/CT/ROI viewer
    plot_Dose_on_CT(CT_data, Dose_data, ROIs_data)
