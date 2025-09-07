#!/usr/bin/env python
# ArnousUnifiedGUI.py — combines preprocessing+visualization with metrics+batch
#
# What you get in one window:
#   1) Load Patient  → runs preprocessing (if needed), computes metrics & DVHs, populates tables, readies DVH plot
#   2) Show Visualization  → launches your existing 3‑D viewer (plot_combined_plot)
#   3) Save NIfTIs  → copies the generated .nii.gz wherever you choose
#   4) Batch Process  → same robust batch engine from metrics UI (multi‑select, progress, UTF‑8 safe)
#
# Dependencies assumed to exist in your env (as in your two source UIs):
#   - final_utils (load_CT_data/load_RT_data/extract_dose_data/extract_ROIs_data/preprocess_Dose_to_CT/save_volumes/load_preprocessed_volumes)
#   - visualization.plot_combined_plot
#   - aligned_metrics_full_sigmafix_calls (load_ct, load_dose, resample_dose_to_ct, build_roi_masks, _get_prescription,
#       compute_abs_dvhs, compute_roi_metrics, extract_group1_metadata, SMOOTH_SIGMA_MM)
#   - dvh_widget.DVHPlotWidget
#   - pydicom, numpy, scipy, pandas, PyQt5
#
#

import os, sys, shutil, traceback, time
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage

import os, sys, subprocess, shutil

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableView, QHeaderView, QAbstractItemView,
    QPushButton, QFileDialog, QMessageBox,
    QSplitter, QHBoxLayout, QVBoxLayout, QListView, QTreeView, QProgressDialog, QInputDialog
)

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableView, QHeaderView,
    QPushButton, QFileDialog, QMessageBox,
    QSplitter, QHBoxLayout, QVBoxLayout,QListView, QTreeView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex

# ── vis‑pipeline utils (from your preprocess viewer) ─────────────────────
from final_utils import (
    load_CT_data, load_RT_data,
    extract_dose_data, extract_ROIs_data,
    preprocess_Dose_to_CT, save_volumes,
    load_preprocessed_volumes
)
from visualization import plot_combined_plot

# ── metrics‑pipeline utils (from your metrics UI) ────────────────────────
from aligned_metrics_full_sigmafix_calls import (
    load_ct, load_dose, resample_dose_to_ct, build_roi_masks, _get_prescription,
    compute_abs_dvhs, compute_roi_metrics, extract_group1_metadata,
    SMOOTH_SIGMA_MM
)
from dvh_widget import DVHPlotWidget
import final_utils as partner_utils  # for find_file_with_prefix


# ╔═════════════════════════ Qt table model ═══════════════════════════════╗
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent); self._df = df.copy()
    def rowCount(self, parent=None):    return self._df.shape[0]
    def columnCount(self, parent=None): return self._df.shape[1]
    def data(self, idx, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.ToolTipRole) and idx and idx.isValid():
            v = self._df.iat[idx.row(), idx.column()]
            if pd.isna(v):
                return ""
            return f"{v:.2f}" if isinstance(v, float) else str(v)
    def headerData(self, sec, ori, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if ori == Qt.Horizontal else str(self._df.index[sec])


# ╔═════════════════════════ Worker to preprocess NIfTIs ══════════════════╗
class PreprocessWorker(QThread):
    finished = pyqtSignal(dict)  # {ok: bool, msg: str, out_dir: str}

    def __init__(self, dicom_dir: str, output_root: str = "Outputs",
                 resample_type: str = "shape",
                 new_size=(512,512,512), new_spacing=(1.0,1.0,1.0)):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.output_root = output_root
        self.resample_type = resample_type
        self.new_size = np.array(new_size)
        self.new_spacing = np.array(new_spacing)

    def run(self):
        try:
            t0 = time.time()
            patient = Path(self.dicom_dir).name
            out_dir = Path(self.output_root) / patient
            out_dir.mkdir(parents=True, exist_ok=True)

            ready = all((out_dir / f).exists() for f in ("CT_volume.nii.gz","Dose_volume.nii.gz","ROIs_volume.nii.gz"))
            if ready:
                self.finished.emit({"ok": True, "msg": f"NIfTI already present in {out_dir}", "out_dir": str(out_dir)})
                return

            # Reuse your preprocess pipeline
            CT = load_CT_data(self.dicom_dir)
            RD, RS, _ = load_RT_data(self.dicom_dir)
            Dose = extract_dose_data(RD)
            ROIs = extract_ROIs_data(RS, CT)

            # shape‑based resample to 512³ (default) or spacing‑based if requested
            zoom = (self.new_size / CT["Volume"].shape) if self.resample_type == "shape" \
                    else (CT["Spacing"] / self.new_spacing)
            CT["Volume"]  = scipy.ndimage.zoom(CT["Volume"], zoom, order=1)
            CT["Spacing"] = CT["Spacing"] / zoom
            for r in ROIs["ROIs"].values():
                r["Volume"] = scipy.ndimage.zoom(r["Volume"], zoom, order=0)
            ROIs["Spacing"], ROIs["Position"] = CT["Spacing"], CT["Position"]

            Dose = preprocess_Dose_to_CT(Dose, CT)
            save_volumes(CT, Dose, ROIs, str(out_dir))
            dt = time.time() - t0
            self.finished.emit({"ok": True, "msg": f"Preprocess finished in {dt/60:.1f} min. Saved to {out_dir}", "out_dir": str(out_dir)})
        except Exception:
            self.finished.emit({"ok": False, "msg": traceback.format_exc(), "out_dir": ""})


# ╔═════════════════════════ Main Window ══════════════════════════════════╗
class ArnousUnifiedGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARNOUS – Unified Metrics + Preprocess/Visualize")
        self.resize(1420, 860)

        # state
        self.patient_dir = None
        self.outputs_dir = None  # Outputs/<patient>
        self.metric_data = None  # dict with ROI_DF, Meta_DF, DVH, Rx
        self.vis_ready = False   # have CT/Dose/ROIs loaded/saved
        self._prep_thread = None

        # ── top bar
        self.btn_load  = QPushButton("Load Patient")
        self.btn_save  = QPushButton("Save NIfTIs")
        self.btn_view  = QPushButton("Show Visualization")
        self.btn_batch = QPushButton("Batch Process")
        for b in (self.btn_save, self.btn_view):
            b.setEnabled(False)

        self.btn_load.clicked.connect(self.load_patient)
        self.btn_save.clicked.connect(self.save_niftis)
        self.btn_view.clicked.connect(self.show_visualization)
        self.btn_batch.clicked.connect(self.batch_process)

        bar = QHBoxLayout(); [bar.addWidget(w) for w in (self.btn_load, self.btn_save, self.btn_view, self.btn_batch)]; bar.addStretch()

        # ── tables + DVH plot
        self.tbl_meta = QTableView(); self.tbl_roi = QTableView()
        self._make_table_interactive(self.tbl_meta)
        self._make_table_interactive(self.tbl_roi)

        self.dvh_widget = DVHPlotWidget()

        split_tables = QSplitter(Qt.Horizontal)
        split_tables.addWidget(self.tbl_meta); split_tables.addWidget(self.tbl_roi)
        split_tables.setStretchFactor(0,1); split_tables.setStretchFactor(1,4)

        split_vert = QSplitter(Qt.Vertical)
        split_vert.addWidget(split_tables); split_vert.addWidget(self.dvh_widget)
        split_vert.setStretchFactor(0,3); split_vert.setStretchFactor(1,2)

        root = QVBoxLayout(self); root.addLayout(bar); root.addWidget(split_vert)

    # ────────────────────────── UI helpers ────────────────────────────────
    def _make_table_interactive(self, tbl: QTableView):
        hh = tbl.horizontalHeader(); vh = tbl.verticalHeader()
        hh.setSectionsMovable(True); hh.setSectionsClickable(True); hh.setSectionResizeMode(QHeaderView.Interactive); hh.setStretchLastSection(False)
        vh.setSectionsMovable(True); vh.setSectionsClickable(True); vh.setSectionResizeMode(QHeaderView.Interactive)
        tbl.setSortingEnabled(True); tbl.setSelectionBehavior(QAbstractItemView.SelectRows); tbl.setSelectionMode(QAbstractItemView.ExtendedSelection); tbl.setAlternatingRowColors(True)

    # ══════════════════════════ Main actions ═════════════════════════════
    def load_patient(self):
        folder = QFileDialog.getExistingDirectory(self, "Select patient DICOM folder")
        if not folder:
            return
        self.patient_dir = folder
        patient_name = Path(folder).name
        self.outputs_dir = str(Path("Outputs") / patient_name)

        # 1) Start preprocessing (if needed) in the background
        prog = QProgressDialog("Preparing volumes (if needed)…", None, 0, 0, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0); prog.show()

        self._prep_thread = PreprocessWorker(folder)
        self._prep_thread.finished.connect(lambda res: self._after_preprocess(res, prog))
        self._prep_thread.start()

    def _after_preprocess(self, res: dict, prog: QProgressDialog):
        prog.close()
        ok, msg, out_dir = res.get("ok", False), res.get("msg", ""), res.get("out_dir", "")
        if not ok:
            QMessageBox.critical(self, "Preprocess error", msg)
            return
        self.vis_ready = True
        self.btn_save.setEnabled(True)
        self.btn_view.setEnabled(True)

        # 2) Run metrics pipeline from DICOM (independent of NIfTI)
        try:
            ct_img,_ = load_ct(self.patient_dir)
            ds_dose, dose_raw = load_dose(self.patient_dir)
            dose_arr = resample_dose_to_ct(ds_dose, dose_raw, ct_img)

            rs = pydicom.dcmread(partner_utils.find_file_with_prefix(self.patient_dir, "RS"))
            rp = pydicom.dcmread(partner_utils.find_file_with_prefix(self.patient_dir, "RP"))

            masks = build_roi_masks(rs, ct_img)
            rx    = _get_prescription(rp)

            sx,sy,sz = ct_img.GetSpacing(); spacing = (sx,sy,sz); vv_cc = (sx*sy*sz)/1000.0

            roi_df = compute_roi_metrics(masks, dose_arr, vv_cc, rx, spacing, SMOOTH_SIGMA_MM).round(2)
            meta_df = pd.DataFrame(extract_group1_metadata(rp, None, self.patient_dir).items(), columns=["Field","Value"])
            dvh_abs = compute_abs_dvhs(masks, dose_arr, vv_cc, rx, spacing_mm=spacing)

            self.metric_data = {"ROI_DF": roi_df, "Meta_DF": meta_df, "DVH": dvh_abs, "Rx": rx}

            # Populate UI
            self.tbl_meta.setModel(PandasModel(meta_df)); self.tbl_meta.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tbl_roi.setModel(PandasModel(roi_df));   self.tbl_roi.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.dvh_widget.set_data(dvh_abs, rx)

            QMessageBox.information(self, "Loaded", f"✔ Preprocess ready.\n✔ Metrics computed and UI populated.\n{msg}")
        except Exception as e:
            self.metric_data = None
            self.dvh_widget.set_data({}, None)
            QMessageBox.critical(self, "Metrics error", str(e))

    # ------------------------------------------------------------------
    def show_visualization(self):
        if not self.vis_ready or not self.outputs_dir:
            QMessageBox.warning(self, "Not ready", "Run Load Patient first.")
            return
        try:
            CT, Dose, ROIs = load_preprocessed_volumes(self.outputs_dir)
            plot_combined_plot(CT, Dose, ROIs)
        except Exception as e:
            QMessageBox.critical(self, "Visualization error", str(e))

    # ------------------------------------------------------------------
    def save_niftis(self):
        if not self.outputs_dir:
            QMessageBox.warning(self, "Nothing to save", "Load a patient first.")
            return
        dest = QFileDialog.getExistingDirectory(self, "Select destination for NIfTIs")
        if not dest:
            return
        try:
            Path(dest).mkdir(parents=True, exist_ok=True)
            for f in Path(self.outputs_dir).glob("*.nii.gz"):
                shutil.copy2(f, Path(dest) / f.name)
            QMessageBox.information(self, "Saved", f"Copied NIfTI files to:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    # ------------------------------------------------------------------
    def batch_process(self):
        # Adapted from your metrics UI; unchanged logic, wrapped here
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
        n, ok = QInputDialog.getInt(self, "How many?", f"Found {n_max} session folders.\nHow many do you want to process (1–{n_max})?", value=n_max, min=1, max=n_max)
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
        prog.setWindowTitle("Batch Process"); prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0); prog.setAutoClose(False); prog.setAutoReset(False); prog.show()

        env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"

        errors = []; processed = 0
        for i, sess in enumerate(to_process, start=1):
            if prog.wasCanceled(): break
            prog.setLabelText(f"Processing {i}/{len(to_process)}:\n{sess}"); prog.setValue(i-1); QApplication.processEvents()

            # quick skip if no CT
            if not self._likely_has_ct(sess):
                errors.append((sess, "Skipped: no CT series detected"))
                prog.setValue(i); QApplication.processEvents(); continue
            try:
                cmd = [sys.executable, str(aligned_script), str(sess)]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env)
                stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
                if proc.returncode != 0:
                    errors.append((sess, f"Return code {proc.returncode}\n{stderr[-2000:]}")); continue
                produced = sess / "abs_DVH_CTgrid.xlsx"
                if not produced.exists():
                    cands = list(sess.glob("*.xlsx"))
                    if cands: produced = cands[0]
                    else:
                        errors.append((sess, "No Excel output found after processing")); continue
                if sess.parent not in (common_root, parent):
                    out_name = f"{sess.parent.name}__{sess.name}_DVH.xlsx"
                else:
                    out_name = f"{sess.name}_DVH.xlsx"
                shutil.copyfile(str(produced), str(dvh_out / out_name))
                processed += 1
            except Exception as e:
                errors.append((sess, repr(e)))
            prog.setValue(i); QApplication.processEvents()
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
            modalities = set(); checked = 0
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
                if {"CT","RTDOSE","RTPLAN","RTSTRUCT"} & modalities or ("ANY" in modalities and checked >= 20):
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
    gui = ArnousUnifiedGUI(); gui.show()
    sys.exit(app.exec_())
