#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────────
#  roi_dvh_backend_patched.py
#
#  • Builds binary ROI masks from RTSTRUCT on the CT grid
#  • Resamples RTDOSE → CT grid and exports **absolute‑dose DVHs**
#    (one Excel sheet per ROI, padded to the global max dose)
#
#  Version  : 2025‑08‑01  (σ hard‑wired to 0.85mm)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sys
from pathlib import Path
import matplotlib as plt


import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter

def _folder_with_all_modalities(path: Path) -> bool:
    """Returns True when *path* (any depth) has ≥1 CT, RTSTRUCT, RTDOSE, RTPLAN"""
    mods = {"CT", "RTSTRUCT", "RTDOSE", "RTPLAN"}
    found = set()
    for p in path.glob("**/*.dcm"):
        try:
            m = pydicom.dcmread(str(p), stop_before_pixels=True).Modality
            if m in mods:
                found.add(m)
                if found == mods:
                    return True
        except Exception:
            continue
    return False


# ╔════════════════════════════════  DICOM I/O  ════════════════════════════════╗
def _find_dcm(folder: str, modality: str) -> str:
    """Return the first DICOM file in *folder* whose Modality matches."""
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        try:
            if pydicom.dcmread(p, stop_before_pixels=True).Modality == modality:
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"{modality} not found in “{folder}”")


def load_ct(folder: str):
    """Load the CT image (SimpleITK Image) and its numpy array (Z,Y,X)."""
    rdr = sitk.ImageSeriesReader()
    for sid in rdr.GetGDCMSeriesIDs(folder) or []:
        fns = rdr.GetGDCMSeriesFileNames(folder, sid)
        if pydicom.dcmread(fns[0], stop_before_pixels=True).Modality != "CT":
            continue
        rdr.SetFileNames(fns)
        img = rdr.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.min() >= 0:           # unsigned → convert to HU
            arr -= 1024.0
        return img, arr
    raise RuntimeError("CT not found")


def load_dose(folder: str):
    ds  = pydicom.dcmread(_find_dcm(folder, "RTDOSE"))
    arr = ds.pixel_array.astype(np.float32) * ds.DoseGridScaling
    if arr.ndim == 4:               # (1,Z,Y,X)
        arr = arr[0]
    return ds, arr


def resample_dose_to_ct(dose_ds, dose_arr, ct_img):
    """Linear‑resample RTDOSE grid onto the CT grid and return a numpy array."""
    img = sitk.GetImageFromArray(dose_arr)
    dy, dx = map(float, dose_ds.PixelSpacing)
    dz     = (np.diff(dose_ds.GridFrameOffsetVector).mean()
              if len(dose_ds.GridFrameOffsetVector) > 1
              else float(dose_ds.SliceThickness))
    img.SetSpacing((dy, dx, dz))
    img.SetOrigin(tuple(map(float, dose_ds.ImagePositionPatient)))

    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ct_img)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    return sitk.GetArrayFromImage(rf.Execute(img)).astype(np.float32)
# ╚═════════════════════════════════════════════════════════════════════════════╝


# ╔═══════════════════════  Build ROI masks on CT grid  ════════════════════════╗
def build_roi_masks(rs, ct_img, verbose=True):
    """
    Build masks  {ROI‑name: uint8 (Z,Y,X)}  on the CT grid.
    ROIs with **no contours** are skipped with a warning.
    """
    sx, sy, sz = ct_img.GetSize()
    masks = {}

    for roi in rs.StructureSetROISequence:
        num = roi.ROINumber
        roi_name = roi.ROIName.strip()           # trim whitespace

        # Find the matching contour block, if any
        rc = next((r for r in rs.ROIContourSequence
                   if r.ReferencedROINumber == num), None)
        if rc is None or not getattr(rc, "ContourSequence", []):
            if verbose:
                print(f"⚠️   ROI '{roi_name}' has no contours – skipped")
            continue

        m = np.zeros((sz, sy, sx), np.uint8)
        for cs in rc.ContourSequence:
            pts = np.array(cs.ContourData).reshape(-1, 3)
            if pts.size == 0:
                continue
            ijk = [ct_img.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
            xs, ys, zs = map(np.array, zip(*ijk))

            xs = np.clip(np.round(xs).astype(int), 0, sx - 1)
            ys = np.clip(np.round(ys).astype(int), 0, sy - 1)
            z  = int(np.clip(np.round(np.median(zs)), 0, sz - 1))

            rr, cc = polygon(ys, xs, (sy, sx))
            m[z, rr, cc] = 1

        if m.any():
            masks[roi_name] = m
        elif verbose:
            print(f"⚠️   ROI '{roi_name}' contours lay outside the CT – skipped")

    return masks

# ╚═════════════════════════════════════════════════════════════════════════════╝


# ╔══════════════════════════  DVH calculation helpers  ════════════════════════╗
SMOOTH_SIGMA_MM = 0.75      # ← fixed σ for all patients

def _sample_native_dose(mask: np.ndarray,
                        dose: np.ndarray,
                        smooth_sigma: float = SMOOTH_SIGMA_MM):
    """
    Return (dose_values, occupancy_weights) restricted to the **inside** voxels.
    * The mask and dose arrays must share the same shape.
    * Edge voxels get fractional weights via Gaussian blurring,
      but voxels outside the binary ROI are never included.
    """
    if mask.shape != dose.shape:
        raise ValueError(
            f"Mask shape {mask.shape} differs from dose shape {dose.shape}")

    if smooth_sigma > 0:
        weight_mask = gaussian_filter(mask.astype(np.float32),
                                      sigma=smooth_sigma)
    else:
        weight_mask = mask.astype(np.float32)

    idx = mask.astype(bool)               # keep strictly inside the ROI
    return dose[idx], weight_mask[idx]


def _dvh_cumsum_weighted(dose_values,
                         weights,
                         voxel_vol_cc,
                         step_gy: float = 0.1,
                         max_dose: float | None = None):
    """Weighted cumulative‑volume histogram ⇒ (dose_axis, cum_volume_cm³)."""
    if dose_values.size == 0:
        return np.empty(0), np.empty(0)

    end = max_dose if max_dose is not None else dose_values.max()
    last_edge = np.ceil(end / step_gy) * step_gy
    bins = np.arange(0.0, last_edge + step_gy, step_gy)

    vol_hist, _ = np.histogram(dose_values, bins=bins,
                               weights=weights * voxel_vol_cc)
    cumvol = np.cumsum(vol_hist[::-1])[::-1]           # high → low
    return bins[:-1], cumvol


def dvh_table_abs(dose_values, weights,
                  voxel_vol_cc: float,
                  step: float = 0.1,
                  prescription: float | None = None,
                  max_dose: float | None = None) -> pd.DataFrame:
    d, v = _dvh_cumsum_weighted(dose_values, weights,
                                voxel_vol_cc, step, max_dose)
    if d.size == 0:
        return pd.DataFrame(columns=["Dose [Gy]", "Rel. Dose [%]", "Volume [cm³]"])
    rel = 100 * d / (prescription if prescription else d[-1])
    return pd.DataFrame({"Dose [Gy]": d,
                         "Rel. Dose [%]": rel,
                         "Volume [cm³]": v})


def compute_mask_volume(mask: np.ndarray,
                        spacing_mm: tuple[float, float, float]) -> float:
    return float(mask.sum()) * np.prod(spacing_mm) / 1000.0   # mm³ → cm³


def compute_abs_dvhs(masks: dict[str, np.ndarray],
                     dose_arr: np.ndarray,
                     voxel_vol_cc: float,
                     prescription: float | None,
                     step: float = 0.1,
                     spacing_mm: tuple[float, float, float] | None = None
                     ) -> dict[str, pd.DataFrame]:
    """
    Calculate an **absolute‑dose DVH** for every ROI.
    Each curve is padded so that its final dose bin equals the
    *global* maximum dose in the RTDOSE cube.
    """
    dvh_abs: dict[str, pd.DataFrame] = {}
    global_max_dose = float(dose_arr.max())

    for roi, mask in masks.items():
        dose_vals, weights = _sample_native_dose(mask, dose_arr)
        df = dvh_table_abs(dose_vals, weights, voxel_vol_cc,
                           step=step,
                           prescription=prescription,
                           max_dose=global_max_dose)
        dvh_abs[roi] = df

        # Optional sanity printout
        if spacing_mm is not None and not df.empty:
            v_mask = compute_mask_volume(mask, spacing_mm)
            v_dvh  = df["Volume [cm³]"].iloc[0]
            print(f"{roi:30s}: voxelised = {v_mask:7.3f} cc | "
                  f"DVH = {v_dvh:7.3f} cc | Δ = {v_dvh - v_mask:+7.3f} cc")

    return dvh_abs
# ╚═════════════════════════════════════════════════════════════════════════════╝



# ╔══════════════════════════  Small helpers for CLI  ══════════════════════════╗
def _get_prescription(rtplan) -> float | None:
    """
    Return the *total prescribed dose in Gy* if it can be found, else None.

    Search order (first match wins):
      1) DoseReferenceSequence ▸ item with DoseReferenceType == "TARGET"
         • TargetPrescriptionDose
         • DoseReferenceDose              (alternative name, some planners)
         • DeliveryMaximumDose            (IDX plans without the two above)
      2) Top‑level plan attributes
         • PrescriptionDose
         • TargetPrescriptionDose
         • DeliveryMaximumDose
      3) Derive from FractionGroupSequence[0]
         PrescriptionDosePerFraction × NumberOfFractionsPlanned
    """
    # ── 1) look inside DoseReferenceSequence for the TARGET item ────────────
    for dr in getattr(rtplan, "DoseReferenceSequence", []):
        if getattr(dr, "DoseReferenceType", "").upper() == "TARGET":
            for tag in ("TargetPrescriptionDose",
                        "DoseReferenceDose",
                        "DeliveryMaximumDose"):
                if hasattr(dr, tag):
                    try:
                        return float(getattr(dr, tag))
                    except Exception:
                        pass

    # ── 2) try top‑level plan tags ─────────────────────────────────────────
    for tag in ("PrescriptionDose",
                "TargetPrescriptionDose",
                "DeliveryMaximumDose"):
        if hasattr(rtplan, tag):
            try:
                return float(getattr(rtplan, tag))
            except Exception:
                pass

    # ── 3) derive from the first FractionGroupSequence item ───────────────
    for fg in getattr(rtplan, "FractionGroupSequence", []):
        try:
            n_frac = float(fg.NumberOfFractionsPlanned)
            dose_per_frac = float(fg.PrescriptionDosePerFraction)
            return n_frac * dose_per_frac
        except Exception:
            continue

    # ── not found ──────────────────────────────────────────────────────────
    return None
# ╚═════════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────  ROI‑level dosimetric metrics  ─────────────────────
from collections import OrderedDict


def _weighted_percentile(values: np.ndarray,
                         weights: np.ndarray,
                         q: float) -> float:
    """
    Weighted percentile (q in [0‑100]).
    """
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(q / 100.0, cdf, v)


def _weighted_mode(values: np.ndarray,
                   weights: np.ndarray,
                   bin_width: float = 0.1) -> float:
    """
    Weighted mode of *values* (Gy) with the given *weights*.

    Robust to edge cases:
      • empty input          → returns NaN
      • all values identical → returns that value
      • histogram ends up empty (e.g. zero weights) → returns weighted mean
    """
    if values.size == 0:
        return np.nan

    # If all dose values are (numerically) identical, the mode is that value
    if np.allclose(values.ptp(), 0):
        return float(values[0])

    # Build at least two bins to avoid an empty histogram
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < bin_width:               # very narrow spread
        bins = np.array([v_min, v_max + bin_width])
    else:
        bins = np.arange(v_min, v_max + bin_width, bin_width)

    hist, edges = np.histogram(values, bins=bins, weights=weights)

    if hist.size == 0 or hist.max() == 0:       # all weights zero?
        return float(np.average(values, weights=weights
                                if weights.sum() > 0 else None))

    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def compute_roi_metrics(masks: dict[str, np.ndarray],
                        dose_arr: np.ndarray,
                        voxel_vol_cc: float,
                        prescription: float | None,
                        spacing_mm: tuple[float, float, float],
                        smooth_sigma: float = SMOOTH_SIGMA_MM,
                        healthy_brain_name: str = "Brain"
                        ) -> pd.DataFrame:
    """
    Returns a table with one row per ROI and the following columns
    (NaN where a metric is not applicable):

        ROI, Volume_cc, Min_Gy, Max_Gy, Mean_Gy, Median_Gy, Mode_Gy, Std_Gy,
        D2_Gy, D50_Gy, D98_Gy, HI, CI,
        V5_cc, V10_cc, V12_cc, V18_cc, V20_cc, V23_cc, V24_cc, V25_cc, V27_cc, V30_cc
    """
    rows = []
    global_max_dose = float(dose_arr.max())

    # Pre‑compute healthy‑brain thresholds
    hb_thrs = [5, 10, 12, 18, 20, 23, 24, 25, 27, 30]

    for roi, mask in masks.items():
        dose_vals, w = _sample_native_dose(mask, dose_arr, smooth_sigma)
        if dose_vals.size == 0:
            continue

        vol_total = np.sum(w) * voxel_vol_cc
        mean = np.average(dose_vals, weights=w)
        std = np.sqrt(np.average((dose_vals - mean) ** 2, weights=w))
        median = _weighted_percentile(dose_vals, w, 50)

        d2 = _weighted_percentile(dose_vals, w, 98)
        d50 = median
        d98 = _weighted_percentile(dose_vals, w, 2)
        mode = _weighted_mode(dose_vals, w)

        # Homogeneity Index
        hi = abs((d98-d2) / prescription if prescription else np.nan)

        # Conformity Index – only meaningful for PTVs
        if prescription and ("ptv" in roi.lower()):
            v_ptv_100 = np.sum(w[dose_vals >= prescription]) * voxel_vol_cc
            v_ptv_80 = np.sum(w[dose_vals >= 0.8 * prescription]) * voxel_vol_cc
            ci = (v_ptv_100 ** 2) / (v_ptv_80 * vol_total) if v_ptv_80 > 0 else np.nan
        else:
            ci = np.nan

        row = OrderedDict([
            ("ROI", roi),
            ("Volume_cc", vol_total),
            ("Min_Gy", dose_vals.min()),
            ("Max_Gy", dose_vals.max()),
            ("Mean_Gy", mean),
            ("Median_Gy", median),
            ("Mode_Gy", mode),
            ("Std_Gy", std),
            ("D2_Gy", d2),
            ("D50_Gy", d50),
            ("D98_Gy", d98),
            ("HI", hi),
            ("CI", ci),
        ])

        # Healthy-brain Vx’s (for any ROI containing "brain" but not "brainstem")
        roi_clean = roi.strip().lower()
        if "brain" in roi_clean and "brainstem" not in roi_clean:
            for thr in hb_thrs:
                vx = np.sum(w[dose_vals >= thr]) * voxel_vol_cc
                row[f"V{thr}_cc"] = vx
        rows.append(row)

    return pd.DataFrame(rows)



import pandas as pd


# ╔══════════════════  Group‑1 metadata extraction helper  ═══════════════════╗
import os, re, numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def extract_group1_metadata(RP_data, RD_data, patient_folder_path) -> Dict[str, Any]:
    """
    Return a dict with the nine “Group‑1” plan‑level metadata items:

        PatientID │ Session date │ PrescriptionDose[Gy] │ NumberofFractions
        DoseperFraction[Gy] │ PlanningSoftware │ PlanningTechnique
        DoseCalculation Algorithm │ DoseGridSize[mm]  (z,y,x)

    *The function is resilient to missing DICOM tags and non‑standard planners –
    any unavailable item is returned as None / 'Unknown…'.*
    """
    meta: Dict[str, Any] = {
        "Patient ID":               "UnknownID",
        "Session Date":             "UnknownDate",
        "Prescription Dose [Gy]":   None,
        "Number of Fractions":      None,
        "Dose per Fraction [Gy]":   None,
        "Planning Software":        None,
        "Planning Technique":       None,
        "Dose Calculation Algorithm": None,
        "Dose Grid Size [mm]":      None,
    }

    # ── 1) Patient‑ID & session date from the folder name ─────────────────
    folder = Path(patient_folder_path).name
    if folder:
        #   • Patient‑ID = everything before first “_”
        meta["Patient ID"] = folder.split("_")[0] or meta["Patient ID"]

        #   • Any 8‑digit block interpreted as date
        m = re.search(r"(\d{8})", folder)
        if m:
            raw = m.group(1)
            for fmt in ("%d%m%Y", "%Y%m%d"):
                try:
                    meta["Session Date"] = datetime.strptime(
                        raw, fmt).strftime("%d/%m/%Y")
                    break
                except ValueError:
                    continue
            else:
                meta["Session Date"] = raw          # unknown order

    # ── 2) Prescription, n‑fractions, dose/fraction from the RTPLAN ───────
    rx = _get_prescription(RP_data)                    # helper already in file :contentReference[oaicite:2]{index=2}
    if rx is not None:
        meta["Prescription Dose [Gy]"] = float(rx)

    for fg in getattr(RP_data, "FractionGroupSequence", []):
        n_frac = getattr(fg, "NumberOfFractionsPlanned", None)
        if n_frac:
            meta["Number of Fractions"] = int(n_frac)
            break

    if meta["Prescription Dose [Gy]"] and meta["Number of Fractions"]:
        meta["Dose per Fraction [Gy]"] = (
            meta["Prescription Dose [Gy]"] / meta["Number of Fractions"]
        )

    # ── 3) Planning software / technique / algorithm ──────────────────────
    sw = getattr(RP_data, "SoftwareVersions", None)
    meta["Planning Software"] = str(sw) if sw is not None else None

    beam = next(iter(getattr(RP_data, "BeamSequence", [])), None)
    if beam:
        meta["Planning Technique"] = getattr(beam, "BeamTechnique", None)
        meta["Dose Calculation Algorithm"] = getattr(
            beam, "DoseCalculationAlgorithm", None)

    # ── 4) Dose‑grid spacing from RTDOSE ───────────────────────────────────
    try:
        spacing_x, spacing_y = map(float, RD_data.PixelSpacing)
        gfv = RD_data.GridFrameOffsetVector
        spacing_z = float(gfv[1] - gfv[0]) if len(gfv) > 1 else float(
            getattr(RD_data, "SliceThickness", np.nan))
        meta["Dose Grid Size [mm]"] = (spacing_z, spacing_y, spacing_x)
    except Exception:
        pass

    return meta
# ╚═════════════════════════════════════════════════════════════════════════════╝

import math

def _fixed_grids_for_clinical_style(max_gy: float, rx_gy: float):
    """Return (abs_dose_grid_gy step 0.1 Gy, rel_dose_grid_pct step 0.1%)."""
    abs_end = math.ceil(max_gy * 10.0) / 10.0
    abs_grid = np.round(np.arange(0.0, abs_end + 1e-9, 0.1), 1).astype(np.float32)

    rel_end = 0.0
    if rx_gy > 0:
        rel_end = math.ceil((max_gy / rx_gy) * 1000.0) / 10.0  # %Rx in 0.1%
    rel_grid = np.round(np.arange(0.0, rel_end + 1e-9, 0.1), 1).astype(np.float32)
    return abs_grid, rel_grid


def _cum_volumes_at_thresholds(dose_vals: np.ndarray, w: np.ndarray,
                               voxel_vol_cc: float, thresholds_gy: np.ndarray) -> np.ndarray:
    """Cumulative volume (cc) receiving >= each threshold (Gy)."""
    if dose_vals.size == 0:
        return np.zeros_like(thresholds_gy, dtype=np.float64)
    order = np.argsort(dose_vals)  # ascending
    dv = dose_vals[order].astype(np.float32, copy=False)
    ww = (w[order].astype(np.float32, copy=False) * float(voxel_vol_cc)).astype(np.float64)

    csum = np.cumsum(ww)       # weight for dose < dv[i]
    total = csum[-1]
    out = np.empty_like(thresholds_gy, dtype=np.float64)
    for i, thr in enumerate(thresholds_gy):
        idx = np.searchsorted(dv, float(thr), side='left')  # # voxels with dose < thr
        less = 0.0 if idx == 0 else csum[idx-1]
        out[i] = total - less
    return out


def build_dvh_tables_clinical_style(masks: dict, dose_arr: np.ndarray,
                                    voxel_vol_cc: float, rx_gy: float):
    """
    Returns:
      abs_tables: dict[roi] -> DataFrame with columns:
         ['Dose [Gy]', 'Relative dose [%]', 'Structure Volume [cm³]']
      rel_tables: dict[roi] -> DataFrame with columns:
         ['Relative dose [%]', 'Dose [Gy]', 'Ratio of Total Structure Volume [%]']
    """
    abs_tables, rel_tables = {}, {}
    dose_max = float(np.max(dose_arr)) if dose_arr.size else 0.0
    abs_grid, rel_grid = _fixed_grids_for_clinical_style(dose_max, rx_gy)
    rel_to_abs = (rel_grid / 100.0) * rx_gy if rx_gy > 0 else np.zeros_like(rel_grid)

    for roi, mask in masks.items():
        # Sample native dose inside ROI using your existing smoothing constant
        dose_vals, w = _sample_native_dose(mask, dose_arr, smooth_sigma=SMOOTH_SIGMA_MM)

        # --- Absolute sheet (cc vs Gy) + a side column for %Rx
        vols_cc_abs = _cum_volumes_at_thresholds(dose_vals, w, voxel_vol_cc, abs_grid)
        rel_col = (abs_grid / rx_gy * 100.0) if rx_gy > 0 else np.zeros_like(abs_grid, dtype=np.float32)
        abs_df = pd.DataFrame({
            "Dose [Gy]": abs_grid.astype(np.float64),
            "Relative dose [%]": rel_col.astype(np.float64),
            "Structure Volume [cm³]": vols_cc_abs.astype(np.float64),
        })
        abs_tables[roi] = abs_df

        # --- Relative sheet (% vs Gy) + % volume of structure
        vols_cc_rel = _cum_volumes_at_thresholds(dose_vals, w, voxel_vol_cc, rel_to_abs)
        total_cc = float(vols_cc_rel[0]) if vols_cc_rel.size else 0.0
        vols_pct = (vols_cc_rel / total_cc * 100.0) if total_cc > 0 else np.zeros_like(vols_cc_rel)
        rel_df = pd.DataFrame({
            "Relative dose [%]": rel_grid.astype(np.float64),
            "Dose [Gy]": rel_to_abs.astype(np.float64),
            "Ratio of Total Structure Volume [%]": vols_pct.astype(np.float64),
        })
        rel_tables[roi] = rel_df

    return abs_tables, rel_tables


def _sanitize_sheet_name(name: str, suffix: str = "") -> str:
    # Excel sheet name limit: 31 chars; avoid [/\\?*:]
    bad = '/\\?*[]:'
    for ch in bad:
        name = name.replace(ch, "_")
    nm = (name[:31 - len(suffix)] + suffix) if len(name) + len(suffix) > 31 else (name + suffix)
    return nm or "Sheet"


def export_excel_with_clinical_style_dvhs(xlsx_path: str,
                                          roi_metrics_df: pd.DataFrame,
                                          abs_tables: dict, rel_tables: dict):
    """
    Writes:
      - a 'ROI metrics' sheet (unchanged),
      - one sheet per ROI (absolute),
      - one sheet per ROI with '_rel' suffix (relative).
    """
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as ew:
        if roi_metrics_df is not None:
            roi_metrics_df.to_excel(ew, sheet_name="ROI metrics", index=False)

        # Stable order by ROI name
        for roi in sorted(abs_tables.keys()):
            ew.book.formats  # force workbook creation early (xlsxwriter quirk)
            abs_df = abs_tables[roi]
            rel_df = rel_tables.get(roi)
            # Absolute
            abs_sheet = _sanitize_sheet_name(str(roi))
            abs_df.to_excel(ew, sheet_name=abs_sheet, index=False)
            # Relative
            if rel_df is not None:
                rel_sheet = _sanitize_sheet_name(str(roi), suffix="_rel")
                rel_df.to_excel(ew, sheet_name=rel_sheet, index=False)



def _cum_volumes_at_thresholds(dose_vals, w, voxel_vol_cc, thresholds_gy):
    """
    Fast cumulative volume at dose >= threshold for an array of thresholds (Gy).
    Returns volumes in cc for each threshold.
    """
    import numpy as np
    if dose_vals.size == 0:
        return np.zeros_like(thresholds_gy, dtype=np.float64)
    order = np.argsort(dose_vals)  # ascending
    dv = dose_vals[order].astype(np.float32, copy=False)
    ww = (w[order].astype(np.float32, copy=False) * float(voxel_vol_cc)).astype(np.float64)
    csum = np.cumsum(ww)                # weight for dose < dv[i]
    total = csum[-1]
    out = np.empty_like(thresholds_gy, dtype=np.float64)
    for i, thr in enumerate(thresholds_gy):
        idx = np.searchsorted(dv, float(thr), side="left")  # # voxels with dose < thr
        less = 0.0 if idx == 0 else csum[idx - 1]
        out[i] = total - less
    return out


def plot_all_roi_dvhs(
    dvh_abs: dict[str, pd.DataFrame],
    prescription: float | None,
    *,
    x_mode: str = "dose",      # "dose" or "relative"
    y_mode: str = "volume",    # "volume" or "relative"
    ax: plt.Axes | None = None,
    linewidth: float = 1.5,

):
    """
    Draw every ROI’s cumulative DVH as a continuous line.

    Parameters
    ----------
    dvh_abs : dict[str, pd.DataFrame]
        Output of `compute_abs_dvhs()`. Each DataFrame must contain the
        columns 'Dose [Gy]' and 'Volume [cm³]'.
    prescription : float | None
        Plan prescription dose in Gy. Mandatory if x_mode == "relative".
    x_mode : {'dose', 'relative'}
        'dose'     → X‑axis in Gray (Gy).
        'relative' → X‑axis in % of prescription (values may exceed 100 %).
    y_mode : {'volume', 'relative'}
        'volume'   → Y‑axis in cubic centimetres (cm³).
        'relative' → Y‑axis in % of the ROI’s total volume.
    ax : matplotlib.axes.Axes | None
        Existing axis to plot on. If None, a new figure/axis is created.
    linewidth : float
        Line width for all ROI curves.
    legend_loc : str
        Legend location string accepted by Matplotlib.
    """
    if x_mode not in {"dose", "relative"}:
        raise ValueError("x_mode must be 'dose' or 'relative'")
    if y_mode not in {"volume", "relative"}:
        raise ValueError("y_mode must be 'volume' or 'relative'")
    if x_mode == "relative" and not prescription:
        raise ValueError("Prescription dose is required for relative x‑axis")

    # Create axis if none supplied
    if ax is None:
        _, ax = plt.subplots()

    for roi, df in dvh_abs.items():
        if df.empty:
            continue

        # -------- X‑axis values --------
        if x_mode == "dose":               # Gy
            x = df["Dose [Gy]"].to_numpy()
            xlabel = "Dose [Gy]"
        else:                              # % prescription, allow >100%
            x = df["Dose [Gy]"].to_numpy() / prescription * 100.0
            xlabel = "Relative dose [% Rx]"

        # -------- Y‑axis values --------
        if y_mode == "volume":             # cm³
            y = df["Volume [cm³]"].to_numpy()
            ylabel = "Volume [cm³]"
        else:                              # % of ROI volume
            y0 = df["Volume [cm³]"].iloc[0]        # total volume of ROI
            y = df["Volume [cm³]"].to_numpy() / y0 * 100.0
            ylabel = "Relative volume [%]"

        ax.plot(x, y, label=roi, linewidth=linewidth)

    # -------- Cosmetics --------
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)                   # always start at 0 on X
    ax.set_ylim(bottom=0)                 # always start at 0 on Y

    # Tight layout for use inside PyQt FigureCanvas
    ax.figure.tight_layout()
    return ax



def main():
    import sys, os, math
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pydicom

    if len(sys.argv) < 2:
        sys.exit("Usage:  python aligned_metrics_full.py  \"C:\\path\\to\\PATIENT_FOLDER\"")

    folder = Path(sys.argv[1]).expanduser()

    # 1) Load DICOMs & build masks ---------------------------------------------
    ct_img, _ = load_ct(str(folder))
    dose_ds, dose_raw = load_dose(str(folder))
    dose_arr = resample_dose_to_ct(dose_ds, dose_raw, ct_img)

    rs = pydicom.dcmread(_find_dcm(str(folder), "RTSTRUCT"))
    rp = pydicom.dcmread(_find_dcm(str(folder), "RTPLAN"), stop_before_pixels=True)

    rx = _get_prescription(rp)  # Rx in Gy (float)
    masks = build_roi_masks(rs, ct_img)

    sx, sy, sz = ct_img.GetSpacing()
    spacing = (sx, sy, sz)
    voxel_vol_cc = (sx * sy * sz) / 1000.0  # mm^3 -> cm^3
    dose_max = float(dose_arr.max()) if dose_arr.size else 0.0

    # 2) ROI metrics (unchanged) ------------------------------------------------
    roi_df = compute_roi_metrics(
        masks, dose_arr, voxel_vol_cc,
        rx, spacing, SMOOTH_SIGMA_MM
    )

    # 3) Absolute DVHs at 0.1 Gy (keep your existing function) -----------------
    dvh_abs = compute_abs_dvhs(
        masks, dose_arr, voxel_vol_cc, rx, spacing_mm=spacing
    )
    # We’ll rename/reorder columns when writing.

    # 4) Prepare Excel with clinical-style DVH sheets ---------------------------
    out_xlsx = folder / "abs_DVH_CTgrid.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        # ROI metrics sheet
        roi_df.to_excel(xl, sheet_name="ROI metrics", index=False)

        for roi, df_abs in dvh_abs.items():
            # ---------- ABSOLUTE sheet (0.1 Gy grid, clinical headers) ----------
            if not df_abs.empty:
                df_abs = df_abs.rename(columns={
                    "Rel. Dose [%]": "Relative dose [%]",
                    "Volume [cm³]": "Structure Volume [cm³]"
                })
                df_abs = df_abs[["Dose [Gy]", "Relative dose [%]", "Structure Volume [cm³]"]]
            abs_sheet = "".join(c for c in roi if c not in r'\/:*?[]')[:31] or "ROI"
            df_abs.to_excel(xl, sheet_name=abs_sheet, index=False)

            # ---------- RELATIVE sheet (0.1% Rx grid, clinical build) ----------
            # Build a fixed %Rx grid: 0, 0.1, 0.2, ... up to ceil(max/Rx, 0.1%)
            if rx and rx > 0:
                rel_end = math.ceil((dose_max / rx) * 1000.0) / 10.0  # %Rx, step 0.1
                rel_grid = np.round(np.arange(0.0, rel_end + 1e-9, 0.1), 1)  # [%]
                thr_gy = (rel_grid / 100.0) * rx                             # [Gy]
            else:
                # Fallback if Rx is missing/zero: keep Gy grid but mark % as NaN
                rel_grid = np.round(np.arange(0.0, math.ceil(dose_max * 10.0) / 10.0 + 1e-9, 0.1), 1)
                thr_gy = rel_grid.copy()

            # Sample dose inside ROI once
            dose_vals, w = _sample_native_dose(masks[roi], dose_arr, smooth_sigma=SMOOTH_SIGMA_MM)
            vols_cc = _cum_volumes_at_thresholds(dose_vals, w, voxel_vol_cc, thr_gy)
            total_cc = float(vols_cc[0]) if vols_cc.size else 0.0
            vols_pct = (vols_cc / max(total_cc, 1e-12)) * 100.0

            if rx and rx > 0:
                df_rel = pd.DataFrame({
                    "Relative dose [%]": rel_grid.astype(np.float64),
                    "Dose [Gy]": thr_gy.astype(np.float64),
                    "Ratio of Total Structure Volume [%]": vols_pct.astype(np.float64),
                })
            else:
                # Fallback columns if Rx unknown
                df_rel = pd.DataFrame({
                    "Relative dose [%]": np.full_like(rel_grid, np.nan, dtype=np.float64),
                    "Dose [Gy]": thr_gy.astype(np.float64),
                    "Ratio of Total Structure Volume [%]": vols_pct.astype(np.float64),
                })

            rel_name_raw = (roi[:28] + "_rel") if len(roi) > 28 else roi + "_rel"
            rel_sheet = "".join(c for c in rel_name_raw if c not in r'\/:*?[]')[:31] or "ROI_rel"
            df_rel.to_excel(xl, sheet_name=rel_sheet, index=False)

    print(f"✔ Clinical-style DVHs saved → {out_xlsx}")

if __name__ == "__main__":
    main()
# ╚═════════════════════════════════════════════════════════════════════════════╝



