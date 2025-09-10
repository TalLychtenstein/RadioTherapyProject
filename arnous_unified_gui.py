import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel
from PyQt5.QtWidgets import (
    QProgressDialog, QInputDialog, QLineEdit, QDialog, QPlainTextEdit, QApplication, QWidget, QTableView, QHeaderView,
    QPushButton, QFileDialog, QMessageBox, QProgressBar, QLabel, QComboBox,
    QSplitter, QHBoxLayout, QVBoxLayout, QListView, QTreeView, QAbstractItemView
)

# ── metrics‑pipeline utils (from your metrics UI) ────────────────────────
from aligned_metrics_full_sigmafix_calls import (
    load_ct, load_dose, resample_dose_to_ct, build_roi_masks, _get_prescription,
    compute_abs_dvhs, compute_roi_metrics, extract_group1_metadata,
    SMOOTH_SIGMA_MM
)
from dvh_widget import DVHPlotWidget
# ── vis‑pipeline utils (from your preprocess viewer) ─────────────────────
from final_utils import (
    find_file_with_prefix,
    load_CT_data, load_RT_data,
    extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, save_volumes,
    load_preprocessed_volumes,
    parse_patient_metadata
)
from visualization import plot_combined_plot


class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing in Progress")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        self.log_area = QPlainTextEdit(self)
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        self.close_button = QPushButton("Close", self)
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

    def write(self, message):
        self.log_area.appendPlainText(message.strip())

    def mark_done(self):
        self.progress_bar.setRange(0, 1)  # Set to determinate
        self.progress_bar.setValue(1)
        self.close_button.setEnabled(True)
        self.write("✅ Preprocessing complete.")

    def show_error(self, message):
        self.write(f"❌ Error: {message}")
        self.mark_done()

class NIfTISettingsDialog(QDialog):
    def __init__(self, default_output_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resample and Output Settings")
        self.setMinimumWidth(400)

        # ─── Widgets ──────────────────────────────────────────────
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Select resample type...", "shape", "spacing"])
        self.value_edit = QLineEdit()
        self.value_edit.setEnabled(False)  # Disabled initially

        self.output_edit = QLineEdit(default_output_dir)
        self.browse_btn = QPushButton("Browse...")

        # ─── Layout ────────────────────────────────────────────────
        form = QVBoxLayout()

        form.addWidget(QLabel("Resample Type:"))
        form.addWidget(self.type_combo)

        form.addWidget(QLabel("Resample Values:"))
        form.addWidget(self.value_edit)

        form.addWidget(QLabel("Output Folder:"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.browse_btn)
        form.addLayout(output_row)

        # ─── Buttons ──────────────────────────────────────────────
        button_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        button_row.addStretch()
        button_row.addWidget(self.ok_btn)
        button_row.addWidget(self.cancel_btn)
        form.addLayout(button_row)

        self.setLayout(form)

        # ─── Events ───────────────────────────────────────────────
        self.type_combo.currentTextChanged.connect(self._on_type_change)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _on_type_change(self, text):
        if text == "shape":
            self.value_edit.setEnabled(True)
            self.value_edit.setText("512,512,512")
        elif text == "spacing":
            self.value_edit.setEnabled(True)
            self.value_edit.setText("1.0,1.0,1.0")
        else:
            self.value_edit.setEnabled(False)
            self.value_edit.setText("")

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def get_values(self):
        return {
            "resample_type": self.type_combo.currentText(),
            "resample_value": self.value_edit.text(),
            "output_dir": self.output_edit.text()
        }

# ╔═════════════════════════ Qt table model ═══════════════════════════════╗
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df.copy()

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, idx, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.ToolTipRole) and idx and idx.isValid():
            v = self._df.iat[idx.row(), idx.column()]
            if pd.isna(v):
                return ""
            return f"{v:.2f}" if isinstance(v, float) else str(v)

    def headerData(self, sec, ori, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if ori == Qt.Horizontal else str(self._df.index[sec])


# ╔═════════════════════════ Worker to create NIfTI volumes ══════════════════╗
class PreprocessWorker(QThread):
    finished = pyqtSignal(dict)  # {ok: bool, msg: str, out_dir: str}
    log = pyqtSignal(str)

    def __init__(self, patient_data, outputs_dir, resample_type, new_size, new_spacing):
        super().__init__()
        self.patient_data = patient_data
        self.outputs_dir = outputs_dir
        self.resample_type = resample_type
        self.new_size = np.array(new_size) if new_size is not None else None
        self.new_spacing = np.array(new_spacing) if new_spacing is not None else None

    def run(self):
        try:
            t0 = time.time()

            self.log.emit("🔍 Checking if outputs already exist...")
            ready = all(os.path.exists(os.path.join(self.outputs_dir, f))
                        for f in ("CT_volume.nii.gz", "Dose_volume.nii.gz", "ROIs"))
            if ready:
                self.log.emit(f"✅ NIfTI already present in {self.outputs_dir}")
                self.finished.emit({
                    "ok": True,
                    "msg": f"NIfTI already present in {self.outputs_dir}",
                    "out_dir": self.outputs_dir
                })
                return

            # ------------------------
            # Load DICOM data
            # ------------------------
            self.log.emit("📥 Loading CT and RT data...")
            CT = load_CT_data(self.patient_data)
            RD, RS, _ = load_RT_data(self.patient_data)

            self.log.emit("💉 Extracting Dose data...")
            Dose = extract_dose_data(RD)

            self.log.emit("🧠 Extracting ROI structures...")
            ROIs = extract_ROIs_data(RS, CT)

            # ------------------------
            # Compute zoom factors
            # ------------------------
            if self.resample_type == "shape":
                zoom = self.new_size / CT["Volume"].shape
                self.log.emit(f"📏 Resampling by shape to {self.new_size.tolist()}")
            else:
                zoom = CT["Spacing"] / self.new_spacing
                self.log.emit(f"📏 Resampling by spacing to {self.new_spacing.tolist()}")

            # ------------------------
            # Resample CT
            # ------------------------
            self.log.emit("   🔄 Resampling CT volume...")
            CT["Volume"] = scipy.ndimage.zoom(CT["Volume"], zoom, order=1)
            CT["Spacing"] = CT["Spacing"] / zoom

            # ------------------------
            # Resample ROIs
            # ------------------------
            self.log.emit("   🔄 Aligning ROI structures to CT volume...")
            for r in ROIs["ROIs"].values():
                r["Volume"] = scipy.ndimage.zoom(r["Volume"], zoom, order=0)
            ROIs["Spacing"], ROIs["Position"] = CT["Spacing"], CT["Position"]

            # ------------------------
            # Resample Dose
            # ------------------------
            self.log.emit("   🔄 Aligning Dose volume to CT volume...")
            Dose = preprocess_Dose_to_CT(Dose, CT)

            # ------------------------
            # Save Outputs
            # ------------------------
            os.makedirs(self.outputs_dir, exist_ok=True)
            self.log.emit(f"💾 Saving volumes to: {self.outputs_dir}")
            save_volumes(CT, Dose, ROIs, self.outputs_dir)

            dt = time.time() - t0
            self.finished.emit({
                "ok": True,
                "msg": f"Preprocess finished in {dt / 60:.1f} min. Saved to {self.outputs_dir}",
                "out_dir": self.outputs_dir
            })
        except Exception:
            self.finished.emit({
                "ok": False,
                "msg": traceback.format_exc(),
                "out_dir": ""
            })

# ╔═════════════════════════ Worker to extract metrics ══════════════════╗
class MetricsWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    result = pyqtSignal(object)

    def __init__(self, patient_data):
        super().__init__()
        self.patient_data = patient_data

    def run(self):
        try:
            self.log.emit("📥 Starting metric extraction...")

            self.log.emit("🔄 Loading CT and Dose data...")
            ct_img, _ = load_ct(self.patient_data)
            self.log.emit("🔄   Finished Loading CT")
            ds_dose, dose_raw = load_dose(self.patient_data)
            self.log.emit("🔄   Finished Loading Dose data")

            self.log.emit("   ↳ Resampling Dose to CT grid...")
            dose_arr = resample_dose_to_ct(ds_dose, dose_raw, ct_img)

            self.log.emit("🔄 Reading RT Structure and Plan...")
            rs = pydicom.dcmread(find_file_with_prefix(self.patient_data, "RS"))
            rp = pydicom.dcmread(find_file_with_prefix(self.patient_data, "RP"))

            self.log.emit("🔬 Building ROI masks...")
            masks = build_roi_masks(rs, ct_img, self.log)

            self.log.emit("💊 Extracting prescription...")
            rx = _get_prescription(rp)

            sx, sy, sz = ct_img.GetSpacing()
            spacing = (sx, sy, sz)
            vv_cc = (sx * sy * sz) / 1000.0

            self.log.emit("📊 Computing ROI metrics...")
            roi_df = compute_roi_metrics(masks, dose_arr, vv_cc, rx, spacing, SMOOTH_SIGMA_MM, self.log).round(2)

            self.log.emit("📄 Extracting patient metadata...")
            meta_df = pd.DataFrame(
                extract_group1_metadata(rp, None, self.patient_data).items(),
                columns=["Field", "Value"]
            )

            self.log.emit("📈 Calculating DVH curves...")
            dvh_abs = compute_abs_dvhs(masks, dose_arr, vv_cc, rx, spacing_mm=spacing, log=self.log)

            result = {
                "ROI_DF": roi_df,
                "Meta_DF": meta_df,
                "DVH": dvh_abs,
                "Rx": rx
            }

            self.result.emit(result)
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())
            self.finished.emit()

# ╔═════════════════════════ Main Window ══════════════════════════════════╗
class ArnousUnifiedGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARVOUS – Automatic Radiotherapy Visualization and Utility Output System")
        self.resize(1420, 860)

        # define output folders
        self.outputs_dir = os.path.join(os.getcwd(), "Outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

        # ── top bar
        self.btn_load = QPushButton("Load Patient Data")
        self.btn_metrics = QPushButton("Extract Metrics")
        self.btn_nifti = QPushButton("Create NIfTI Volumes")
        self.btn_vis = QPushButton("Show Visualization")
        for b in (self.btn_metrics, self.btn_nifti, self.btn_vis):
            b.setEnabled(False)

        self.btn_load.clicked.connect(self.load_patient_data)
        self.btn_metrics.clicked.connect(self.extract_metrics)
        self.btn_nifti.clicked.connect(self.create_nifti_volumes)
        self.btn_vis.clicked.connect(self.show_visualization)

        bar = QHBoxLayout()
        [bar.addWidget(w) for w in (self.btn_load, self.btn_metrics, self.btn_nifti, self.btn_vis)]
        bar.addStretch()

        # ── patient info display
        self.patient_info_label = QLabel()
        self.patient_info_label.setStyleSheet("font: 14px 'Courier New'; padding: 5px;")
        self.patient_info_label.setAlignment(Qt.AlignLeft)
        self.patient_info_label.setText("🧾 No patient selected yet.")

        # ── tables + DVH plot
        self.tbl_meta = QTableView()
        self.tbl_roi = QTableView()
        self._make_table_interactive(self.tbl_meta)
        self._make_table_interactive(self.tbl_roi)

        self.dvh_widget = DVHPlotWidget()

        split_tables = QSplitter(Qt.Horizontal)
        split_tables.addWidget(self.tbl_meta)
        split_tables.addWidget(self.tbl_roi)
        split_tables.setStretchFactor(0, 1)
        split_tables.setStretchFactor(1, 4)

        split_vert = QSplitter(Qt.Vertical)
        split_vert.addWidget(split_tables)
        split_vert.addWidget(self.dvh_widget)
        split_vert.setStretchFactor(0, 3)
        split_vert.setStretchFactor(1, 2)

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(self.patient_info_label)
        root.addWidget(split_vert)

    # ────────────────────────── UI helpers ────────────────────────────────
    def clear_all_data(self):
        self.tbl_meta.setModel(None)
        self.tbl_roi.setModel(None)
        self.dvh_widget.clear()
        self.metric_data = None

    def _make_table_interactive(self, tbl: QTableView):
        hh = tbl.horizontalHeader()
        vh = tbl.verticalHeader()
        hh.setSectionsMovable(True)
        hh.setSectionsClickable(True)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(False)
        vh.setSectionsMovable(True)
        vh.setSectionsClickable(True)
        vh.setSectionResizeMode(QHeaderView.Interactive)
        tbl.setSortingEnabled(True)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tbl.setAlternatingRowColors(True)

    # ══════════════════════════ Main actions ═════════════════════════════
    def load_patient_data(self):
        # Let user select patient data
        self.patient_data = QFileDialog.getExistingDirectory(self, "Select patient DICOM folder")
        if not self.patient_data:
            return
        self.patient_name = Path(self.patient_data).name

        # Parse and display patient metadata
        meta = parse_patient_metadata(self.patient_name)
        info_text = (
            f"📁 <b>Processing Patient Record</b><br>"
            f"   • <b>Patient ID:</b> {meta['Patient ID']}<br>"
            f"   • <b>Treatment Type:</b> {meta['Treatment']}<br>"
            f"   • <b>Targeted Region:</b> {meta['Region']}<br>"
            f"   • <b>Session Date:</b> {meta['Date']}<br>"
            f"🗂️  DICOM path: <i>{self.patient_data}</i>"
        )
        self.patient_info_label.setText(info_text)
        self.clear_all_data()
        self.btn_metrics.setEnabled(True)
        self.btn_nifti.setEnabled(True)

    def extract_metrics(self):
        # Open dialog to show extraction progress
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.setWindowTitle("Extracting Dosimetric and Volumetric Metrics…")
        self.processing_dialog.show()

        # Set up thread and worker
        self.metrics_thread = QThread()
        self.metrics_worker = MetricsWorker(self.patient_data)
        self.metrics_worker.moveToThread(self.metrics_thread)

        # Connect signals
        self.metrics_thread.started.connect(self.metrics_worker.run)
        self.metrics_worker.log.connect(self.processing_dialog.write)
        self.metrics_worker.error.connect(self.processing_dialog.show_error)

        def handle_result(result):
            self.metric_data = result
            self.tbl_meta.setModel(PandasModel(result["Meta_DF"]))
            self.tbl_meta.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

            self.tbl_roi.setModel(PandasModel(result["ROI_DF"]))
            self.tbl_roi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            self.dvh_widget.set_data(result["DVH"], result["Rx"])
            self.btn_metrics.setEnabled(False)

        self.metrics_worker.result.connect(handle_result)

        def cleanup():
            self.processing_dialog.mark_done()
            self.processing_dialog.close()
            self.metrics_thread.quit()
            self.metrics_thread.wait()
            self.metrics_worker.deleteLater()
            self.metrics_thread.deleteLater()

        # Start thread
        self.metrics_worker.finished.connect(cleanup)
        self.metrics_thread.start()

    def create_nifti_volumes(self):
        # Show combined dialog
        dialog = NIfTISettingsDialog(default_output_dir=self.patient_data, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        # Extract values
        settings = dialog.get_values()
        resample_type = settings["resample_type"]
        self.outputs_dir = os.path.join(settings["output_dir"], Path(self.patient_data).name, f"Resample by {settings['resample_type']}", settings["resample_value"])

        # Parse resample values
        try:
            if resample_type == "shape":
                new_size = tuple(map(int, settings["resample_value"].split(",")))
                new_spacing = None
            else:
                new_spacing = tuple(map(float, settings["resample_value"].split(",")))
                new_size = None
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid resample values.")
            return

        # Show progress/log dialog
        self.processing_dialog = ProcessingDialog(self)
        self.processing_dialog.show()

        self.preprocess_thread = PreprocessWorker(
            patient_data=self.patient_data,
            outputs_dir=self.outputs_dir,
            resample_type=resample_type,
            new_size=new_size,
            new_spacing=new_spacing
        )
        self.preprocess_thread.log.connect(self.processing_dialog.write)

        def on_finish(res):
            self.processing_dialog.mark_done()
            self.processing_dialog.close()
            ok, msg, out_dir = res.get("ok", False), res.get("msg", ""), res.get("out_dir", "")
            if not ok:
                QMessageBox.critical(self, "Preprocess error", msg)
            else:
                self.btn_vis.setEnabled(True)

        self.preprocess_thread.finished.connect(on_finish)
        self.preprocess_thread.start()

    # ------------------------------------------------------------------
    def show_visualization(self):
        CT, Dose, ROIs = load_preprocessed_volumes(self.outputs_dir)
        plot_combined_plot(CT, Dose, ROIs)

    # ------------------------------------------------------------------
    def batch_process(self):
        # Adapted from your metrics UI unchanged logic, wrapped here
        parent_str = QFileDialog.getExistingDirectory(self, "Select a parent folder (contains patient/session folders)")
        if not parent_str:
            return
        parent = Path(parent_str)

        selected_dirs = self._choose_multiple_dirs_under_parent(parent)
        if not selected_dirs:
            return

        # Discover concrete session folders
        session_folders = []
        for p in selected_dirs:
            session_folders.extend(self._discover_session_folders(p))
        session_folders = sorted(set(session_folders), key=lambda x: (x.parent.name, x.name))
        if not session_folders:
            QMessageBox.warning(self, "Nothing to process", "No valid patient/session folders found.")
            return

        n_max = len(session_folders)
        n, ok = QInputDialog.getInt(self, "How many?",
                                    f"Found {n_max} session folders.\nHow many do you want to process (1–{n_max})?",
                                    value=n_max, min=1, max=n_max)
        if not ok:
            return
        to_process = session_folders[:n]

        # Output root: common parent → DVH_files
        try:
            common_root = Path(os.path.commonpath([str(p.parent) for p in to_process]))
        except Exception:
            common_root = parent
        dvh_out = common_root / "DVH_files"
        dvh_out.mkdir(parents=True, exist_ok=True)

        # Locate engine script
        aligned_script = (Path(__file__).parent / "aligned_metrics_full_sigmafix_calls.py").resolve()
        if not aligned_script.exists():
            QMessageBox.critical(self, "Missing script", f"Cannot find {aligned_script}")
            return

        prog = QProgressDialog("Preparing…", "Cancel", 0, len(to_process), self)
        prog.setWindowTitle("Batch Process")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.show()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        errors = []
        processed = 0
        for i, sess in enumerate(to_process, start=1):
            if prog.wasCanceled(): break
            prog.setLabelText(f"Processing {i}/{len(to_process)}:\n{sess}")
            prog.setValue(i - 1)
            QApplication.processEvents()

            # quick skip if no CT
            if not self._likely_has_ct(sess):
                errors.append((sess, "Skipped: no CT series detected"))
                prog.setValue(i)
                QApplication.processEvents()
                continue
            try:
                cmd = [sys.executable, str(aligned_script), str(sess)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env)
                stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
                if proc.returncode != 0:
                    errors.append((sess, f"Return code {proc.returncode}\n{stderr[-2000:]}"))
                    continue
                produced = sess / "abs_DVH_CTgrid.xlsx"
                if not produced.exists():
                    cands = list(sess.glob("*.xlsx"))
                    if cands:
                        produced = cands[0]
                    else:
                        errors.append((sess, "No Excel output found after processing"))
                        continue
                if sess.parent not in (common_root, parent):
                    out_name = f"{sess.parent.name}__{sess.name}_DVH.xlsx"
                else:
                    out_name = f"{sess.name}_DVH.xlsx"
                shutil.copyfile(str(produced), str(dvh_out / out_name))
                processed += 1
            except Exception as e:
                errors.append((sess, repr(e)))
            prog.setValue(i)
            QApplication.processEvents()
        prog.close()

        if errors:
            lines = [f"Processed: {processed}", f"Errors: {len(errors)}", "", "Details:"]
            for s, err in errors[:12]:
                lines.append(f"- {s}: {err}")
            if len(errors) > 12:
                lines.append(f"...and {len(errors) - 12} more.")
            QMessageBox.warning(self, "Batch completed with errors", "\n".join(lines))
        else:
            QMessageBox.information(self, "Batch completed", f"Processed: {processed}\nSaved in: {dvh_out}")

    # ── helpers copied from your metrics UI ─────────────────────────────
    def _choose_multiple_dirs_under_parent(self, parent: Path):
        dlg = QFileDialog(self, "Select patients/sessions (multi-select)")
        dlg.setDirectory(str(parent))
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        for view in dlg.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.MultiSelection)
        if dlg.exec_() != QFileDialog.Accepted:
            return []
        picks = [Path(p) for p in dlg.selectedFiles() if Path(p).is_dir()]
        picks = [p for p in picks if p.parent == parent]
        return picks

    def _likely_has_ct(self, folder: Path, max_files: int = 200) -> bool:
        try:
            dcm_files = [p for p in folder.rglob("*.dcm")]
            if not dcm_files:
                return False
            uid_ct = "1.2.840.10008.5.1.4.1.1.2"
            for f in dcm_files[:max_files]:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    if getattr(ds, "Modality", "").upper() == "CT":
                        return True
                    if str(getattr(ds, "SOPClassUID", "")) == uid_ct:
                        return True
                    if "ct" in str(getattr(ds, "SeriesDescription", "")).lower():
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _discover_session_folders(self, root_path: Path):
        if self._contains_dicoms_nonrecursive(root_path):
            return [root_path]
        level1_dirs = [p for p in root_path.iterdir() if p.is_dir()]
        sessions = []
        for d in level1_dirs:
            if self._contains_dicoms_nonrecursive(d) or self._looks_like_session(d):
                sessions.append(d)
        if sessions:
            return sorted(set(sessions), key=lambda x: x.name)
        return []

    def _looks_like_session(self, folder: Path, max_checks: int = 200) -> bool:
        try:
            modalities = set()
            checked = 0
            for f in folder.rglob("*"):
                if not f.is_file():
                    continue
                checked += 1
                if f.suffix.lower() in {".dcm", ".dicom"}:
                    modalities.add("ANY")
                else:
                    try:
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                        mod = str(getattr(ds, "Modality", "")).upper()
                        if mod:
                            modalities.add(mod)
                        elif getattr(ds, "SOPClassUID", None):
                            modalities.add("ANY")
                    except Exception:
                        pass
                if {"CT", "RTDOSE", "RTPLAN", "RTSTRUCT"} & modalities or ("ANY" in modalities and checked >= 20):
                    return True
                if checked >= max_checks:
                    break
        except Exception:
            return False
        return False

    def _contains_dicoms_nonrecursive(self, folder: Path, max_checks: int = 200) -> bool:
        checked = 0
        for f in folder.iterdir():
            if not f.is_file():
                continue
            checked += 1
            if f.suffix.lower() in {".dcm", ".dicom"}:
                return True
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if getattr(ds, "SOPClassUID", None) or getattr(ds, "Modality", None):
                    return True
            except Exception:
                pass
            if checked >= max_checks:
                break
        return False


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousUnifiedGUI()
    gui.show()
    sys.exit(app.exec_())
