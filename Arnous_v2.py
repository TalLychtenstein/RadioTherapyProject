# Arnous_v3.py  ───────────────────────────────────────────────────────────────
import os, sys, shutil, subprocess
from pathlib import Path
import pydicom
import pandas as pd


from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableView, QHeaderView,
    QPushButton, QFileDialog, QMessageBox,
    QSplitter, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex

# ── utilities ---------------------------------------------------------------
from final_utils import (
    load_preprocessed_volumes,                # existing reader
    load_CT_data, load_RT_data,               # ↓ new – for automatic build
    extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, save_volumes
)
import final_utils as partner_utils
from aligned_metrics_full import (
    load_ct, load_dose, resample_dose_to_ct, build_roi_masks, _get_prescription,
    compute_abs_dvhs, compute_roi_metrics, extract_group1_metadata,
    SMOOTH_SIGMA_MM
)
from dvh_widget import DVHPlotWidget


# ─────────────────────────────────  Qt‑table model  ──────────────────────────
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent); self._df = df.copy()
    def rowCount(self, parent=QModelIndex()):    return self._df.shape[0]
    def columnCount(self, parent=QModelIndex()): return self._df.shape[1]
    def data(self, idx, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.ToolTipRole) and idx.isValid():
            v = self._df.iat[idx.row(), idx.column()]
            if pd.isna(v): return ""
            return f"{v:.2f}" if isinstance(v, float) else str(v)
    def headerData(self, sec, ori, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if ori == Qt.Horizontal else str(self._df.index[sec])


# ─────────────────────────────────  Main window  ─────────────────────────────
class ArnousViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARNOUS Viewer")
        self.resize(1400, 820)

        # runtime state -----------------------------------------------------
        self.metric_data = None
        self.vis_data    = None
        self.patient_dir = None
        self.output_folder = None

        # top bar -----------------------------------------------------------
        self.btn_load  = QPushButton("Load Patient")
        self.btn_save  = QPushButton("Save NIfTI")
        self.btn_batch = QPushButton("Batch Process")
        self.btn_viz   = QPushButton("Visualize 3‑D")
        for b in (self.btn_save, self.btn_viz): b.setEnabled(False)

        self.btn_load.clicked.connect(self.load_patient)
        self.btn_save.clicked.connect(self.save_nifti)
        self.btn_batch.clicked.connect(self.batch_process)
        self.btn_viz.clicked.connect(self.show_visualization)

        bar = QHBoxLayout()
        [bar.addWidget(w) for w in (self.btn_load, self.btn_save,
                                    self.btn_batch, self.btn_viz)];  bar.addStretch()

        # tables -----------------------------------------------------------
        self.tbl_meta = QTableView(); self.tbl_meta.setMinimumWidth(220)
        self.tbl_roi  = QTableView()

        split_tables = QSplitter(Qt.Horizontal)
        split_tables.addWidget(self.tbl_meta); split_tables.addWidget(self.tbl_roi)
        split_tables.setStretchFactor(0,1); split_tables.setStretchFactor(1,4)

        # DVH plot widget ---------------------------------------------------
        self.dvh_widget = DVHPlotWidget()

        split_vert = QSplitter(Qt.Vertical)
        split_vert.addWidget(split_tables); split_vert.addWidget(self.dvh_widget)
        split_vert.setStretchFactor(0,3); split_vert.setStretchFactor(1,2)

        root = QVBoxLayout(self); root.addLayout(bar); root.addWidget(split_vert)

    # ══════════════════════════  NEW helper  ════════════════════════════════
    def _autogenerate_nifti(self):
        """
        Build CT, Dose and ROI volumes from the current DICOM folder
        and save them under self.output_folder/CT_volume.nii.gz  etc.
        Uses the same pipeline you had in preprocess_data.py
        """
        if not self.patient_dir or not self.output_folder:
            return None

        # --- load DICOM ----------------------------------------------------
        CT = load_CT_data(self.patient_dir)
        RD, RS, _ = load_RT_data(self.patient_dir)
        Dose = extract_dose_data(RD)
        Dose = preprocess_Dose_to_CT(Dose, CT)
        ROIs = extract_ROIs_data(RS, CT)

        # --- make sure target dir exists ----------------------------------
        os.makedirs(self.output_folder, exist_ok=True)
        save_volumes(CT, Dose, ROIs, self.output_folder)

        # --- return dicts so caller can reuse in‑memory objects -----------
        return {"CT": CT, "Dose": Dose, "ROIs": ROIs}

    # ══════════════════════════  main actions  ══════════════════════════════
    def load_patient(self):
        folder = QFileDialog.getExistingDirectory(self, "Select patient folder")
        if not folder: return
        self.setDisabled(True)

        self.patient_dir = folder
        patient_name = os.path.basename(folder)
        self.output_folder = os.path.join("outputs", patient_name)

        messages = []

        # 1) try to load NIfTI volumes -------------------------------------
        try:
            CT, Dose, ROIs = load_preprocessed_volumes(self.output_folder)
            self.vis_data = {"CT": CT, "Dose": Dose, "ROIs": ROIs}
            messages.append("✔ Vis data loaded")
        except FileNotFoundError:
            # auto‑generate, then load
            gen = self._autogenerate_nifti()
            if gen:
                self.vis_data = gen
                messages.append("✔ Vis data auto‑generated and loaded")
            else:
                self.vis_data = None
                messages.append("• Vis data not available")
        except Exception as e:            # any other unexpected error
            self.vis_data = None
            messages.append(f"❌ Vis data failed: {e}")

        # 2) DICOM metrics + DVHs ------------------------------------------
        try:
            ct_img,_ = load_ct(folder)
            ds_dose, dose_raw = load_dose(folder)
            dose_arr = resample_dose_to_ct(ds_dose, dose_raw, ct_img)

            rs = pydicom.dcmread(partner_utils.find_file_with_prefix(folder,"RS"))
            rp = pydicom.dcmread(partner_utils.find_file_with_prefix(folder,"RP"))

            masks = build_roi_masks(rs, ct_img)
            rx    = _get_prescription(rp)

            sx,sy,sz = ct_img.GetSpacing()
            spacing = (sx,sy,sz); vv_cc = (sx*sy*sz)/1000.0

            roi_df = compute_roi_metrics(masks,dose_arr,vv_cc,rx,spacing,SMOOTH_SIGMA_MM).round(2)
            meta_df = pd.DataFrame(
                extract_group1_metadata(rp, None, folder).items(),
                columns=["Field","Value"]
            )
            dvh_abs = compute_abs_dvhs(masks,dose_arr,vv_cc,rx,spacing_mm=spacing)

            self.metric_data = {"ROI_DF":roi_df,"Meta_DF":meta_df,
                                "DVH":dvh_abs,"Rx":rx}

            # populate UI
            self.tbl_meta.setModel(PandasModel(meta_df))
            self.tbl_meta.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tbl_roi.setModel(PandasModel(roi_df))
            self.tbl_roi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.dvh_widget.set_data(dvh_abs, rx)

            messages.append("✔ DICOM metrics loaded")
        except Exception as e:
            self.metric_data=None
            self.dvh_widget.set_data({},None)
            messages.append(f"❌ DICOM processing: {e}")

        # enable/disable buttons ------------------------------------------
        self.btn_save.setEnabled(bool(self.vis_data))
        self.btn_viz.setEnabled(bool(self.vis_data))

        self.setDisabled(False)
        QMessageBox.information(self,"Load result","\n".join(messages))

    # ----------------------------------------------------------------------
    def save_nifti(self):
        if not self.vis_data:
            QMessageBox.warning(self,"Nothing to save","No NIfTI data.")
            return
        dest = QFileDialog.getExistingDirectory(self,"Choose save folder")
        if not dest: return
        try:
            save_volumes(self.vis_data["CT"],self.vis_data["Dose"],self.vis_data["ROIs"],dest)
            QMessageBox.information(self,"Saved",f"Volumes saved to:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self,"Save error",str(e))

    # ----------------------------------------------------------------------
    def batch_process(self):
        root = QFileDialog.getExistingDirectory(self,"Folder with many patients")
        if not root: return
        root = Path(root); out_dir = root/"DVH_files"; out_dir.mkdir(exist_ok=True)
        script = Path(__file__).parent/"aligned_metrics_full.py"
        failed=[]
        for p in sorted(root.iterdir()):
            if not p.is_dir() or p.name=="DVH_files": continue
            try:
                subprocess.run([sys.executable,str(script),str(p)],check=True)
                xlsx = p/"abs_DVH_CTgrid.xlsx"
                if xlsx.exists(): shutil.copy2(xlsx,out_dir/f"{p.name}_DVH.xlsx")
            except Exception as e: failed.append(f"{p.name}: {e}")
        msg = ("Batch finished!\n\nFailures:\n"+"\n".join(failed)) if failed else "Batch finished!"
        QMessageBox.information(self,"Batch result",msg)

    # ----------------------------------------------------------------------
    def show_visualization(self):
        if not self.vis_data:
            QMessageBox.warning(self, "No data", "NIfTI volumes not loaded.")
            return

        import matplotlib

        # Block any attempt to swap to Tk while Qt is active
        orig_use = matplotlib.use

        def safe_use(backend, *a, **k):
            if backend.lower().startswith("tk") and \
                    matplotlib.get_backend().lower().startswith("qt"):
                return  # silently ignore
            return orig_use(backend, *a, **k)

        matplotlib.use = safe_use

        try:
            # ✅import happens *after* the guard is in place
            from visualization import plot_combined_plot
            plot_combined_plot(self.vis_data["CT"],
                               self.vis_data["Dose"],
                               self.vis_data["ROIs"])
        except Exception as e:
            QMessageBox.critical(self, "Visualization error", str(e))
        finally:
            matplotlib.use = orig_use  # restore


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousViewer(); gui.show()
    sys.exit(app.exec_())
