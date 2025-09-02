# Arnous_v3.py  ───────────────────────────────────────────────────────────────
import os, sys, shutil, subprocess

from pathlib import Path
import pydicom
import pandas as pd

import final_utils as partner_utils

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableView, QHeaderView,
    QPushButton, QFileDialog, QMessageBox,
    QSplitter, QHBoxLayout, QVBoxLayout,QListView, QTreeView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex

from aligned_metrics_full import (
    load_ct, load_dose, resample_dose_to_ct, build_roi_masks, _get_prescription,
    compute_abs_dvhs, compute_roi_metrics, extract_group1_metadata,
    SMOOTH_SIGMA_MM
)
from dvh_widget import DVHPlotWidget
from pathlib import Path
import sys, subprocess, shutil
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QProgressDialog

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
        self.setWindowTitle("ARNOUS Metrics and Batch")
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
        from pathlib import Path
        import os, sys, subprocess, shutil
        from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QProgressDialog
        from PyQt5.QtCore import Qt

        # 0) Pick a parent folder that CONTAINS the patient/session folders
        parent_str = QFileDialog.getExistingDirectory(
            self, "Select a parent folder (contains patient/session folders)"
        )
        if not parent_str:
            return
        parent = Path(parent_str)

        # 1) Multi-select specific patients/sessions under that parent
        selected_dirs = self._choose_multiple_dirs_under_parent(parent)
        if not selected_dirs:
            return

        # 2) Expand each selected dir into concrete session folders
        session_folders = []
        for p in selected_dirs:
            session_folders.extend(self._discover_session_folders(p))
        # de-dup + sort
        session_folders = sorted(set(session_folders), key=lambda x: (x.parent.name, x.name))
        if not session_folders:
            QMessageBox.warning(self, "Nothing to process", "No valid patient/session folders found.")
            return

        # 3) Ask how many to process
        n_max = len(session_folders)
        n, ok = QInputDialog.getInt(
            self, "How many?",
            f"Found {n_max} session folders.\nHow many do you want to process (1–{n_max})?",
            value=n_max, min=1, max=n_max
        )
        if not ok:
            return
        to_process = session_folders[:n]

        # 4) Output root: common parent → DVH_files
        try:
            common_root = Path(os.path.commonpath([str(p.parent) for p in to_process]))
        except Exception:
            common_root = parent
        dvh_out = common_root / "DVH_files"
        dvh_out.mkdir(parents=True, exist_ok=True)

        # 5) Locate core script
        aligned_script = (Path(__file__).parent / "aligned_metrics_full.py").resolve()
        if not aligned_script.exists():
            QMessageBox.critical(self, "Missing script", f"Cannot find {aligned_script}")
            return

        # 6) Progress UI
        prog = QProgressDialog("Preparing…", "Cancel", 0, len(to_process), self)
        prog.setWindowTitle("Batch Process")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.show()

        # Prepare env to force UTF-8 from child
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        errors = []
        processed = 0

        for i, sess in enumerate(to_process, start=1):
            if prog.wasCanceled():
                break

            # Update progress
            prog.setLabelText(f"Processing {i}/{len(to_process)}:\n{sess}")
            prog.setValue(i - 1)
            QApplication.processEvents()

            # Optional quick skip if no CT found
            if not self._likely_has_ct(sess):
                errors.append((sess, "Skipped: no CT series detected (quick scan)"))
                prog.setValue(i)
                QApplication.processEvents()
                continue

            try:
                # Run the engine (capture BYTES; we’ll decode safely)
                cmd = [sys.executable, str(aligned_script), str(sess)]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,  # get raw bytes to avoid Windows cp1252 decoding errors
                    env=env
                )

                stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

                if proc.returncode != 0:
                    errors.append((sess, f"Return code {proc.returncode}\n{stderr[-2000:]}"))
                    continue

                # Find output Excel
                produced = sess / "abs_DVH_CTgrid.xlsx"
                if not produced.exists():
                    cands = list(sess.glob("*.xlsx"))
                    if cands:
                        produced = cands[0]
                    else:
                        errors.append((sess, "No Excel output found after processing"))
                        continue

                # Compose readable unique name
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

        # 7) Summary
        if errors:
            lines = [f"Processed: {processed}", f"Errors: {len(errors)}", "", "Details:"]
            for s, err in errors[:12]:
                lines.append(f"- {s}: {err}")
            if len(errors) > 12:
                lines.append(f"...and {len(errors) - 12} more.")
            QMessageBox.warning(self, "Batch completed with errors", "\n".join(lines))
        else:
            QMessageBox.information(self, "Batch completed", f"Processed: {processed}\nSaved in: {dvh_out}")

    def _choose_multiple_dirs_under_parent(self, parent: Path):
        """
        Show a non-native QFileDialog that allows MULTI-SELECT of directories under 'parent'.
        Returns a list[Path] of the selected subfolders (not recursive).
        """
        from PyQt5.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView

        dlg = QFileDialog(self, "Select patients/sessions (multi-select)")
        dlg.setDirectory(str(parent))
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        # Enable multi-select on internal views
        for view in dlg.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.MultiSelection)

        if dlg.exec_() != QFileDialog.Accepted:
            return []

        picks = [Path(p) for p in dlg.selectedFiles() if Path(p).is_dir()]
        # Keep only immediate children of 'parent'
        picks = [p for p in picks if p.parent == parent]
        return picks

    def _likely_has_ct(self, folder: Path, max_files: int = 200) -> bool:
        """Fast check: does this session contain any CT images?"""
        try:
            import pydicom
            dcm_files = [p for p in folder.rglob("*.dcm")]
            if not dcm_files:
                return False
            uid_ct = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
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
        """
        Return a list of *session* folders to process.

        Cases handled:
          • root_path itself is a session (DICOMs directly inside)  -> [root_path]
          • root_path contains many session subfolders               -> [each subfolder]
          • If neither, return [] (we don't guess deeper to avoid false positives).
        """
        # Case 1: the selected folder is itself a session (has DICOMs directly inside)
        if self._contains_dicoms_nonrecursive(root_path):
            return [root_path]

        # Case 2: immediate children that are sessions
        level1_dirs = [p for p in root_path.iterdir() if p.is_dir()]
        sessions = []
        for d in level1_dirs:
            # consider a dir a session if it has dicoms directly inside,
            # or if it "looks like session" (read a few files anywhere inside)
            if self._contains_dicoms_nonrecursive(d) or self._looks_like_session(d):
                sessions.append(d)

        if sessions:
            return sorted(set(sessions), key=lambda x: x.name)

        # Nothing found -> return empty; caller can warn the user.
        return []

    def _looks_like_session(self, folder: Path, max_checks: int = 200) -> bool:
        """
        Heuristic: treat 'folder' as a session if we can detect DICOM files
        inside it (recursively) and find at least one relevant Modality.
        This is used ONLY for *child* directories (not the root selection),
        so it's OK to scan recursively here.
        """
        try:
            modalities = set()
            checked = 0
            # Scan recursively but stop early when we’re confident
            for f in folder.rglob("*"):
                if not f.is_file():
                    continue
                checked += 1
                # Quick extension hint (helps performance, but we don't rely on it)
                if f.suffix.lower() in {".dcm", ".dicom"}:
                    modalities.add("ANY")
                else:
                    # Peek header safely; ignore non-dicom files
                    try:
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                        mod = str(getattr(ds, "Modality", "")).upper()
                        if mod:
                            modalities.add(mod)
                        else:
                            # If no Modality, still count as DICOM if a SOPClassUID exists
                            if getattr(ds, "SOPClassUID", None):
                                modalities.add("ANY")
                    except Exception:
                        pass

                # Confidence threshold:
                # if we see any of CT / RTDOSE / RTPLAN / RTSTRUCT (or many dicoms), call it a session
                if {"CT", "RTDOSE", "RTPLAN", "RTSTRUCT"} & modalities or "ANY" in modalities and checked >= 20:
                    return True

                if checked >= max_checks:
                    break
        except Exception:
            return False
        return False

    def _contains_dicoms_nonrecursive(self, folder: Path, max_checks: int = 200) -> bool:
        """
        Fast check: does 'folder' contain DICOM files *directly* (not in subfolders)?
        This prevents the top-level 'all patients' folder from being misclassified
        as a session just because deeper descendants have DICOMs.
        """
        checked = 0
        for f in folder.iterdir():
            if not f.is_file():
                continue
            checked += 1
            # Quick extension hint
            if f.suffix.lower() in {".dcm", ".dicom"}:
                return True
            # Peek header to catch DICOMs with no extension
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if getattr(ds, "SOPClassUID", None) or getattr(ds, "Modality", None):
                    return True
            except Exception:
                pass
            if checked >= max_checks:
                break
        return False


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArnousViewer(); gui.show()
    sys.exit(app.exec_())
