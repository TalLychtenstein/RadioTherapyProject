#!/usr/bin/env python
"""
batch_dvh_gui.py — bulk DVH export with a folder‑chooser dialog
────────────────────────────────────────────────────────────────
• Requires Python ≥3.8 on the same machine where you already run
  aligned_metrics_full.py.
• Put this script in the SAME directory as aligned_metrics_full.py
  (or adjust ALIGNED_SCRIPT below).

How to use
──────────
1.  Double‑click batch_dvh_gui.py        (or run `python batch_dvh_gui.py`)
2.  Pick the folder that holds all patient sub‑directories.
3.  Wait; a DVH_files folder will appear beside your patients.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ‑‑‑ edit here if aligned_metrics_full.py lives elsewhere ‑‑‑
ALIGNED_SCRIPT = Path(__file__).with_name("aligned_metrics_full.py")

REQUIRED_MODS = {"CT", "RTSTRUCT", "RTDOSE", "RTPLAN"}


def has_all_modalities(folder: Path) -> bool:
    """Return True once *folder* has at least one file for every modality."""
    found = set()
    for dcm in folder.rglob("*.dcm"):
        try:
            import pydicom
            m = pydicom.dcmread(str(dcm), stop_before_pixels=True).Modality
            if m in REQUIRED_MODS:
                found.add(m)
                if found == REQUIRED_MODS:
                    return True
        except Exception:
            continue
    return False


def ask_patients_root() -> Path | None:
    """Open a folder chooser and return the selected path or None."""
    tk_root = tk.Tk()
    tk_root.withdraw()                     # hide the main window
    tk_root.attributes("-topmost", True)   # bring dialog to front
    path = filedialog.askdirectory(title="Select folder that contains the PATIENT sub‑folders")
    tk_root.destroy()
    return Path(path) if path else None


def main() -> None:
    # 1–pick the root folder interactively -------------------------------
    patients_root = ask_patients_root()
    if patients_root is None:
        print("No folder selected – aborting.")
        return
    if not patients_root.is_dir():
        messagebox.showerror("Error", f"Folder not found:\n{patients_root}")
        return

    if not ALIGNED_SCRIPT.is_file():
        messagebox.showerror("Error",
                             f"aligned_metrics_full.py not found next to this script:\n{ALIGNED_SCRIPT}")
        return

    # 2–prepare output directory -----------------------------------------
    out_dir = patients_root / "DVH_files"
    out_dir.mkdir(exist_ok=True)

    patient_dirs = [p for p in patients_root.iterdir()
                    if p.is_dir() and p.name != "DVH_files"]

    if not patient_dirs:
        messagebox.showinfo("Batch DVH", "No patient sub‑folders found.")
        return

    # 3–loop over patients ----------------------------------------------
    processed = 0
    skipped   = 0
    t_batch0  = time.perf_counter()

    log_lines = []
    log_lines.append(f"Patients root  : {patients_root}")
    log_lines.append(f"Output folder  : {out_dir}")
    log_lines.append("")

    for p_dir in sorted(patient_dirs, key=lambda p: p.name):
        line = f"{p_dir.name.ljust(30)} "
        if not has_all_modalities(p_dir):
            line += "SKIPPED (missing CT/RS/RD/RP)"
            skipped += 1
            print(line)
            log_lines.append(line)
            continue

        t0 = time.perf_counter()
        try:
            subprocess.run(
                [sys.executable, str(ALIGNED_SCRIPT), str(p_dir)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError:
            line += "FAILED (script error)"
            print(line)
            log_lines.append(line)
            continue

        src = p_dir / "abs_DVH_CTgrid.xlsx"
        if not src.is_file():
            line += "FAILED (Excel not created)"
            print(line)
            log_lines.append(line)
            continue

        dst = out_dir / f"{p_dir.name}_DVH.xlsx"
        shutil.copy2(src, dst)
        dt = time.perf_counter() - t0
        line += f"DONE  ({dt:5.1f} s)"
        processed += 1
        print(line)
        log_lines.append(line)

    # 4–summary ----------------------------------------------------------
    t_total = time.perf_counter() - t_batch0
    summary = (f"\nProcessed : {processed}\n"
               f"Skipped   : {skipped}\n"
               f"Elapsed   : {t_total/60:4.1f} min  "
               f"(avg {t_total/processed:4.1f}s per patient)"
               if processed else "\nNo patient processed.")

    print(summary)
    log_lines.append(summary)

    # optional: save log to text file
    (out_dir / "batch_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    messagebox.showinfo("Batch DVH finished",
                        f"{processed} patient(s) processed.\n"
                        f"Excel files ➜ {out_dir}")


if __name__ == "__main__":
    main()
