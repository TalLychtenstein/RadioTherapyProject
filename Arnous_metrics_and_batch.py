# Arnous_v3.py  ───────────────────────────────────────────────────────────────
import os, sys, shutil, subprocess
from pathlib import Path
import pydicom
import pandas as pd

import final_utils as partner_utils
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

        self.btn_batch = QPushButton("Batch Process")


        self.btn_load.clicked.connect(self.load_patient)

        self.btn_batch.clicked.connect(self.batch_process)


        bar = QHBoxLayout()
        [bar.addWidget(w) for w in (self.btn_load,
                                    self.btn_batch)];  bar.addStretch()

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


    # ══════════════════════════  main actions  ══════════════════════════════
    def load_patient(self):
        folder = QFileDialog.getExistingDirectory(self, "Select patient folder")
        if not folder: return
        self.setDisabled(True)

        self.patient_dir = folder


        messages = []

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


        self.setDisabled(False)
        QMessageBox.information(self,"Load result","\n".join(messages))


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


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousViewer(); gui.show()
    sys.exit(app.exec_())
