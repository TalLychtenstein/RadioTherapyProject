#!/usr/bin/env python
# arnous_preprocess_viewer.py  – v1.1  (2025‑08‑04)
#
#  • Pre‑process DICOM → NIfTI (same pipeline as preprocess_data.py)
#  • Visualise the volumes
#  • NEW: Save NIfTI button to copy the ready files anywhere you like
#
#  Requirements: PyQt5, numpy, scipy, matplotlib, your own final_utils + visualization modules
# -------------------------------------------------------------------------

import os, sys, time, shutil, traceback
from pathlib import Path

import numpy as np
import scipy.ndimage

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLabel,
    QHBoxLayout, QVBoxLayout, QTextEdit, QMessageBox, QProgressDialog
)

# ── your utility functions ───────────────────────────────────────────────
from final_utils import (
    load_CT_data, load_RT_data,
    extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, save_volumes,
    load_preprocessed_volumes                   # needed for Save button
)
from visualization import plot_combined_plot     # 3‑D viewer

# ........................................................................
#                           Worker thread
# ........................................................................
class PreprocessThread(QThread):
    finished = pyqtSignal(dict)          # {"ok": bool, "msg": str}

    def __init__(self, dicom_dir, output_root="Outputs",
                 resample_type="shape",
                 new_size=(512, 512, 512),
                 new_spacing=(1.0, 1.0, 1.0)):
        super().__init__()
        self.dicom_dir, self.output_root = dicom_dir, output_root
        self.resample_type  = resample_type
        self.new_size       = np.array(new_size)
        self.new_spacing    = np.array(new_spacing)

    # ............................................................
    def run(self):
        try:
            t0 = time.time()
            patient = Path(self.dicom_dir).name
            out_dir = Path(self.output_root) / patient
            out_dir.mkdir(parents=True, exist_ok=True)

            already_ready = all((out_dir / f).exists() for f in
                                ("CT_volume.nii.gz", "Dose_volume.nii.gz", "ROIs"))
            if already_ready:
                self.finished.emit({"ok": True, "msg": f"NIfTI already exist in {out_dir}"})
                return

            # --- pipeline copied from preprocess_data.py ----------------
            CT = load_CT_data(self.dicom_dir)
            RD, RS, _ = load_RT_data(self.dicom_dir)
            Dose = extract_dose_data(RD)
            ROIs = extract_ROIs_data(RS, CT)

            zoom = (self.new_size / CT["Volume"].shape) if self.resample_type == "shape" \
                    else (CT["Spacing"] / self.new_spacing)

            CT["Volume"]  = scipy.ndimage.zoom(CT["Volume"], zoom, order=1)
            CT["Spacing"] = CT["Spacing"] / zoom
            for r in ROIs["ROIs"].values():
                r["Volume"] = scipy.ndimage.zoom(r["Volume"], zoom, order=0)
            ROIs["Spacing"], ROIs["Position"] = CT["Spacing"], CT["Position"]

            Dose = preprocess_Dose_to_CT(Dose, CT)      # align dose grid

            save_volumes(CT, Dose, ROIs, str(out_dir))  # write .nii.gz
            elapsed = time.time() - t0
            self.finished.emit({"ok": True,
                                "msg": f"Finished in {elapsed/60:.1f} min; files saved to {out_dir}"})
        except Exception:
            self.finished.emit({"ok": False, "msg": traceback.format_exc()})


# ........................................................................
#                               Main GUI
# ........................................................................
class ArnousPreprocessViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARNOUS – Preprocess & Visualize")
        self.resize(640, 320)

        # runtime state
        self.dicom_dir = None       # original DICOM path
        self.output_dir = None      # Outputs/<patient>
        self.thread = None

        # ── buttons
        self.btn_browse = QPushButton("Choose DICOM Folder")
        self.btn_run    = QPushButton("Run Pre‑process")
        self.btn_view   = QPushButton("Visualize")
        self.btn_save   = QPushButton("Save NIfTI")     # ← NEW
        for b in (self.btn_view, self.btn_save):
            b.setEnabled(False)

        # ── layouts
        bar = QHBoxLayout()
        bar.addWidget(self.btn_browse)
        bar.addWidget(self.btn_run)
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_view)
        bar.addStretch()

        self.lbl_dir = QLabel("No patient loaded")
        self.log     = QTextEdit(); self.log.setReadOnly(True)

        root = QVBoxLayout(self)
        root.addLayout(bar); root.addWidget(self.lbl_dir); root.addWidget(self.log, 1)

        # ── signals
        self.btn_browse.clicked.connect(self.pick_folder)
        self.btn_run.clicked.connect(self.start_preprocess)
        self.btn_view.clicked.connect(self.launch_visualization)
        self.btn_save.clicked.connect(self.save_nifti)            # ← NEW

    # ............................................................
    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select patient DICOM folder")
        if not folder:
            return
        self.dicom_dir   = folder
        self.output_dir  = str(Path("Outputs") / Path(folder).name)
        self.lbl_dir.setText(f"📁 {folder}")

        # try loading existing volumes to enable View / Save
        try:
            load_preprocessed_volumes(self.output_dir)
            self.btn_view.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.log.append(f"✔ NIfTI detected in {self.output_dir}")
        except FileNotFoundError:
            self.btn_view.setEnabled(False)
            self.btn_save.setEnabled(False)

    # ............................................................
    def start_preprocess(self):
        if not self.dicom_dir:
            QMessageBox.warning(self, "No folder", "Please choose a DICOM folder first.")
            return

        self.btn_run.setEnabled(False)
        dlg = QProgressDialog("Pre‑processing…", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(0)

        self.thread = PreprocessThread(self.dicom_dir)
        self.thread.finished.connect(lambda res: self._on_done(res, dlg))
        self.thread.start()

    def _on_done(self, res, dlg):
        dlg.close()
        self.btn_run.setEnabled(True)
        ok, msg = res["ok"], res["msg"]
        self.log.append(msg)
        if ok:
            self.btn_view.setEnabled(True)
            self.btn_save.setEnabled(True)
        else:
            QMessageBox.critical(self, "Pre‑process failed", msg)

    # ............................................................
    def launch_visualization(self):
        if not self.output_dir: return
        try:
            CT, Dose, ROIs = load_preprocessed_volumes(self.output_dir)
            plot_combined_plot(CT, Dose, ROIs)          # interactive 3‑D view :contentReference[oaicite:2]{index=2}
        except Exception as e:
            QMessageBox.critical(self, "Visualization error", str(e))

    # ............................................................
    def save_nifti(self):
        """Copy the .nii.gz files somewhere else."""
        if not self.output_dir:
            QMessageBox.warning(self, "Nothing to save", "Run the pre‑process first.")
            return
        dest = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not dest:
            return
        try:
            Path(dest).mkdir(parents=True, exist_ok=True)
            for f in Path(self.output_dir).glob("*.nii.gz"):
                shutil.copy2(f, Path(dest) / f.name)
            QMessageBox.information(self, "Saved",
                                     f"Copied NIfTI files to:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

# ........................................................................
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousPreprocessViewer(); gui.show()
    sys.exit(app.exec_())

