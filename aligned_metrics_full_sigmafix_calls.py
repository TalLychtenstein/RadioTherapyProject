
#!/usr/bin/env python
"""
aligned_metrics_full_clean.py

Loads CT / RTSTRUCT / RTDOSE / RTPLAN from a patient folder, builds ROI masks
on the CT grid, resamples the dose to the CT grid, computes ROI‑level metrics,
and exports "clinical‑style" DVHs (absolute + relative per ROI) alongside an
"ROI metrics" sheet into a single Excel file.

Requires: numpy, pandas, pydicom, SimpleITK, scikit-image, scipy, openpyxl (or xlsxwriter)
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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



# ──────────────────────────────────────────────────────────────────────────────
# DICOM I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_dcm(folder: str, modality: str) -> str:
    """
    Return path to the first DICOM file in `folder` whose Modality equals `modality`.
    Scans only the immediate files in `folder` (not recursive).
    """
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        try:
            if pydicom.dcmread(p, stop_before_pixels=True).Modality == modality:
                return p
        except Exception:
            # Skip unreadable files
            pass
    raise FileNotFoundError(f"Could not find a DICOM with modality={modality!r} in {folder!r}")


def load_ct(folder: str) -> Tuple[sitk.Image, np.ndarray]:
    """
    Load a CT series under `folder` using SimpleITK.
    Returns (itk_image, array[Z,Y,X] in HU).
    """
    rdr = sitk.ImageSeriesReader()
    ids = rdr.GetGDCMSeriesIDs(folder) or []
    for sid in ids:
        fns = rdr.GetGDCMSeriesFileNames(folder, sid)
        if not fns:
            continue
        try:
            if pydicom.dcmread(fns[0], stop_before_pixels=True).Modality != "CT":
                continue
        except Exception:
            continue
        rdr.SetFileNames(fns)
        img = rdr.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X]
        # Convert to HU if unsigned storage (common: 0..4095)
        if arr.min() >= 0:
            arr -= 1024.0
        return img, arr
    raise RuntimeError(f"No CT series found under {folder!r}")


def load_dose(folder: str) -> Tuple[pydicom.dataset.FileDataset, np.ndarray]:
    """
    Load RTDOSE from `folder`. Returns (pydicom_ds, dose_array[Z,Y,X] in Gy).
    """
    ds = pydicom.dcmread(_find_dcm(folder, "RTDOSE"))
    dose = ds.pixel_array * ds.DoseGridScaling
    # Some RTDOSE are 4D (time). Use first frame if so.
    if dose.ndim == 4:
        dose = dose[0]
    return ds, dose


def resample_dose_to_ct(dose_ds: pydicom.dataset.FileDataset,
                        dose_arr: np.ndarray,
                        ct_img: sitk.Image) -> np.ndarray:
    """
    Wrap the raw dose array in a SimpleITK image with correct spacing/origin,
    then resample onto the CT grid with linear interpolation.
    Returns dose_on_ct as array[Z,Y,X] in Gy.
    """
    img = sitk.GetImageFromArray(dose_arr)  # [Z,Y,X]

    # DICOM PixelSpacing is [row_spacing (dy), col_spacing (dx)]
    dy, dx = map(float, dose_ds.PixelSpacing)
    # z spacing from GridFrameOffsetVector or fallback to SliceThickness
    if hasattr(dose_ds, "GridFrameOffsetVector") and len(dose_ds.GridFrameOffsetVector) > 1:
        dz = float(np.diff(dose_ds.GridFrameOffsetVector).mean())
    else:
        dz = float(getattr(dose_ds, "SliceThickness", 1.0))

    # IMPORTANT: SimpleITK expects spacing order (x, y, z)
    img.SetSpacing((dx, dy, dz))
    # Origin in patient space
    img.SetOrigin(tuple(map(float, dose_ds.ImagePositionPatient)))

    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ct_img)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    out = rf.Execute(img)
    return sitk.GetArrayFromImage(out).astype(np.float32)  # [Z,Y,X]


# ──────────────────────────────────────────────────────────────────────────────
# ROI mask building
# ──────────────────────────────────────────────────────────────────────────────

def build_roi_masks(rs: pydicom.dataset.FileDataset,
                    ct_img: sitk.Image,
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Rasterize 2D polygon contours from RTSTRUCT onto the CT grid.
    Returns dict: {roi_name -> mask[Z,Y,X] (uint8 in {0,1})}.
    Notes:
      - For each polyline we project to the nearest CT slice using the median Z‑index.
      - This ignores any cross‑slice interpolation; typical for planar slice contours.
    """
    size_x, size_y, size_z = ct_img.GetSize()     # (X, Y, Z) voxel counts
    masks: Dict[str, np.ndarray] = {}

    roi_contour_by_number = {}
    if hasattr(rs, "ROIContourSequence"):
        for rc in rs.ROIContourSequence:
            roi_contour_by_number[int(rc.ReferencedROINumber)] = rc

    for roi in getattr(rs, "StructureSetROISequence", []):
        num = int(roi.ROINumber)
        name = str(getattr(roi, "ROIName", f"ROI_{num}")).strip()

        rc = roi_contour_by_number.get(num, None)
        if rc is None or not getattr(rc, "ContourSequence", []):
            if verbose:
                print(f"⚠️  ROI '{name}' has no contours – skipped")
            continue

        m = np.zeros((size_z, size_y, size_x), np.uint8)
        for cs in rc.ContourSequence:
            pts = np.asarray(cs.ContourData, dtype=np.float64).reshape(-1, 3)
            if pts.size == 0:
                continue

            # Convert patient (x,y,z) to CT index (i,j,k) then split
            ijk = [ct_img.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
            xs, ys, zs = map(np.array, zip(*ijk))

            # Round/clip XY; use median Z slice for the polygon
            xs = np.clip(np.round(xs).astype(int), 0, size_x - 1)
            ys = np.clip(np.round(ys).astype(int), 0, size_y - 1)
            z  = int(np.clip(np.round(np.median(zs)), 0, size_z - 1))

            rr, cc = polygon(ys, xs, (size_y, size_x))
            m[z, rr, cc] = 1

        if m.any():
            masks[name] = m
        elif verbose:
            print(f"⚠️  ROI '{name}' contours lay outside the CT – skipped")

    return masks


# ──────────────────────────────────────────────────────────────────────────────
# DVH helpers
# ──────────────────────────────────────────────────────────────────────────────

# Consistent with the header now
SMOOTH_SIGMA_MM: float = 0.75


def _sample_native_dose(mask: np.ndarray,
                        spacing_mm: Tuple[float, float, float],
                        dose: np.ndarray,
                        smooth_sigma: float = SMOOTH_SIGMA_MM) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dose values and smoothed occupancy weights inside the binary `mask`.
    The smoothing produces fractional weights at edges (sub‑voxel interpolation).
    """
    if mask.shape != dose.shape:
        raise ValueError(f"Mask shape {mask.shape} must match dose shape {dose.shape}")
    if smooth_sigma > 0:
        sigma_vox = (smooth_sigma/spacing_mm[2], smooth_sigma/spacing_mm[1], smooth_sigma/spacing_mm[0])
        weight_mask = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox)
    else:
        weight_mask = mask.astype(np.float32)

    inside = mask.astype(bool)
    return dose[inside], weight_mask[inside]


def _dvh_cumsum_weighted(dose_values: np.ndarray,
                         weights: np.ndarray,
                         voxel_vol_cc: float,
                         step_gy: float = 0.1,
                         max_dose: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted differential histogram (Gy) → cumulative volume (cc) DVH.
    Returns (dose_bins_left_edges, cumulative_volumes_cc).
    """
    if dose_values.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    end = max_dose if max_dose is not None else float(dose_values.max())
    last_edge = math.ceil(end / step_gy) * step_gy
    bins = np.arange(0.0, last_edge + step_gy, step_gy, dtype=np.float32)

    vol_hist, _ = np.histogram(dose_values, bins=bins,
                               weights=weights * float(voxel_vol_cc))
    cumvol = np.cumsum(vol_hist[::-1])[::-1]
    return bins[:-1], cumvol.astype(np.float32)


def dvh_table_abs(dose_values: np.ndarray,
                  weights: np.ndarray,
                  voxel_vol_cc: float,
                  step_gy: float = 0.1,
                  prescription: Optional[float] = None,
                  max_dose: Optional[float] = None) -> pd.DataFrame:
    """
    Build an absolute DVH DataFrame with columns:
      - Dose [Gy], Rel. Dose [%], Volume [cm³]
    Rel. dose is % of prescription if available; otherwise % of max curve dose.
    """
    d, v = _dvh_cumsum_weighted(dose_values, weights, voxel_vol_cc, step_gy, max_dose=max_dose)
    if d.size == 0:
        return pd.DataFrame(columns=["Dose [Gy]", "Rel. Dose [%]", "Volume [cm³]"])

    denom = float(prescription) if prescription else float(max(d[-1], 1e-12))
    rel = (100.0 * d / denom).astype(np.float32)

    return pd.DataFrame({
        "Dose [Gy]": d.astype(np.float32),
        "Rel. Dose [%]": rel,
        "Volume [cm³]": v.astype(np.float32),
    })


def compute_mask_volume(mask: np.ndarray, spacing_mm: Tuple[float, float, float]) -> float:
    """Binary voxel count × voxel volume (mm³ → cm³)."""
    return float(mask.sum()) * float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0


def compute_abs_dvhs(masks: Dict[str, np.ndarray],
                     dose_arr: np.ndarray,
                     voxel_vol_cc: float,
                     prescription: Optional[float],
                     spacing_mm: Tuple[float, float, float],
                     step: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Compute absolute DVH tables (0.1 Gy bins) per ROI.
    """
    dvh_abs: Dict[str, pd.DataFrame] = {}
    global_max = float(dose_arr.max()) if dose_arr.size else 0.0

    for roi, mask in masks.items():
        dose_vals, weights = _sample_native_dose(mask, spacing_mm, dose_arr)
        df = dvh_table_abs(dose_vals, weights, voxel_vol_cc, step, prescription, max_dose=global_max)
        dvh_abs[roi] = df

        if spacing_mm is not None and not df.empty:
            v_mask = compute_mask_volume(mask, spacing_mm)
            v_dvh = float(df["Volume [cm³]"].iloc[0])
            print(f"{roi:30s}: voxelised = {v_mask:7.3f} cc | DVH = {v_dvh:7.3f} cc | Δ = {v_dvh - v_mask:+7.3f} cc")

    return dvh_abs


# ──────────────────────────────────────────────────────────────────────────────
# Prescription extraction
# ──────────────────────────────────────────────────────────────────────────────

def _get_prescription(rtplan: pydicom.dataset.FileDataset) -> Optional[float]:
    """
    Heuristically extract prescription dose (Gy) from RTPLAN.
    Returns None if not found.
    """
    # 1) DoseReferenceSequence (TARGET)
    try:
        for dr in getattr(rtplan, "DoseReferenceSequence", []):
            if str(getattr(dr, "DoseReferenceType", "")).upper() == "TARGET":
                for tag in ("TargetPrescriptionDose", "DeliveryMaximumDose",
                            "DeliveryWarningDose", "DeliveryUnit"):
                    if hasattr(dr, tag):
                        val = float(getattr(dr, tag))
                        if val > 0:
                            return round(val)
    except Exception:
        pass

    # 2) Common top-level fields
    for tag in ("DoseReferenceTreatmentMaxDose", "PrescriptionDescription"):
        try:
            val = float(getattr(rtplan, tag))
            if val > 0:
                return val
        except Exception:
            pass

    # 3) Fraction × number of fractions
    try:
        beams = getattr(rtplan, "BeamSequence", [])
        if beams:
            doses = []
            for b in beams:
                fracs = int(getattr(b, "NumberOfFractionsPlanned", 0) or 0)
                dose_per_frac = float(getattr(b, "FinalCumulativeMetersetWeight", 0.0) or 0.0)
                if fracs > 0 and dose_per_frac > 0:
                    doses.append(fracs * dose_per_frac)
            if doses:
                est = float(np.median(doses))
                if est > 0:
                    return est
    except Exception:
        pass

    return None


# ──────────────────────────────────────────────────────────────────────────────
# ROI‑level metrics
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile (linear interpolation on the CDF)."""
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q / 100.0, cdf, v))


def _weighted_mode(values: np.ndarray, weights: np.ndarray, bin_width: float = 0.1) -> float:
    """Weighted mode via histogram with guards for narrow/degenerate ranges."""
    if values.size == 0:
        return float("nan")
    if np.allclose(values.ptp(), 0):
        return float(values[0])
    v_min, v_max = float(values.min()), float(values.max())
    if v_max - v_min < bin_width:
        bins = np.array([v_min, v_max + bin_width], dtype=np.float32)
    else:
        bins = np.arange(v_min, v_max + bin_width, bin_width, dtype=np.float32)
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    if hist.size == 0 or hist.max() == 0:
        return float(np.average(values, weights=weights if weights.sum() > 0 else None))
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def compute_roi_metrics(masks: Dict[str, np.ndarray],
                        dose_arr: np.ndarray,
                        voxel_vol_cc: float,
                        prescription: Optional[float],
                        spacing_mm: Tuple[float, float, float],
                        smooth_sigma: float = SMOOTH_SIGMA_MM,
                        healthy_brain_name: str = "Brain") -> pd.DataFrame:
    """
    Compute per‑ROI dose/volume metrics and selected Vx (cc) for healthy brain.
    """
    rows = []
    hb_thrs = [5, 10, 12, 18, 20, 23, 24, 25, 27, 30]

    for roi, mask in masks.items():
        dose_vals, w = _sample_native_dose(mask, spacing_mm, dose_arr, smooth_sigma)
        if dose_vals.size == 0:
            continue

        vol_total = float(np.sum(w) * voxel_vol_cc)
        mean = float(np.average(dose_vals, weights=w))
        std = float(np.sqrt(np.average((dose_vals - mean) ** 2, weights=w)))
        median = _weighted_percentile(dose_vals, w, 50.0)

        d2 = _weighted_percentile(dose_vals, w, 98.0)   # high-dose tail
        d50 = median
        d98 = _weighted_percentile(dose_vals, w, 2.0)   # low-dose tail
        mode = _weighted_mode(dose_vals, w)

        hi = float(abs((d98 - d2) / prescription)) if prescription else float("nan")

        # Simple conformity index for PTVs (if name contains 'ptv')
        roi_is_ptv = ("ptv" in roi.lower())
        if prescription and roi_is_ptv:
            v_ptv_100 = float(np.sum(w[dose_vals >= prescription]) * voxel_vol_cc)
            v_ptv_80  = float(np.sum(w[dose_vals >= 0.8 * prescription]) * voxel_vol_cc)
            ci = (v_ptv_100 ** 2) / (v_ptv_80 * vol_total) if v_ptv_80 > 0 else float("nan")
        else:
            ci = float("nan")

        row = OrderedDict([
            ("ROI", roi),
            ("Volume_cc", vol_total),
            ("Min_Gy", float(dose_vals.min())),
            ("Max_Gy", float(dose_vals.max())),
            ("Mean_Gy", mean),
            ("Median_Gy", float(median)),
            ("Mode_Gy", float(mode)),
            ("Std_Gy", std),
            ("D2_Gy", float(d2)),
            ("D50_Gy", float(d50)),
            ("D98_Gy", float(d98)),
            ("HI", hi),
            ("CI", float(ci)),
        ])

        roi_clean = roi.strip().lower()
        if "brain" in roi_clean and "brainstem" not in roi_clean:
            for thr in hb_thrs:
                vx = float(np.sum(w[dose_vals >= thr]) * voxel_vol_cc)
                row[f"V{thr}_cc"] = vx

        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Clinical‑style DVH table builders + Excel export
# ──────────────────────────────────────────────────────────────────────────────

def _fixed_grids_for_clinical_style(max_gy: float, rx_gy: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (abs_grid_Gy@0.1, rel_grid_%Rx@0.1) up to rounded maxima."""
    abs_end = math.ceil(float(max_gy) * 10.0) / 10.0
    abs_grid = np.round(np.arange(0.0, abs_end + 1e-9, 0.1), 1).astype(np.float32)

    if rx_gy and rx_gy > 0:
        rel_end = math.ceil((float(max_gy) / float(rx_gy)) * 1000.0) / 10.0
    else:
        rel_end = 0.0
    rel_grid = np.round(np.arange(0.0, rel_end + 1e-9, 0.1), 1).astype(np.float32)
    return abs_grid, rel_grid


def _cum_volumes_at_thresholds(dose_vals: np.ndarray,
                               w: np.ndarray,
                               voxel_vol_cc: float,
                               thresholds_gy: np.ndarray) -> np.ndarray:
    """
    Vectorized cumulative volume (>= threshold) in cc for multiple thresholds in Gy.
    """
    if dose_vals.size == 0 or thresholds_gy.size == 0:
        return np.zeros_like(thresholds_gy, dtype=np.float32)

    order = np.argsort(dose_vals)
    v = dose_vals[order]
    wt = w[order] * float(voxel_vol_cc)

    csum = np.cumsum(wt)  # mass of voxels with dose <= v[i]
    total = float(csum[-1])

    # For each threshold t, find first index where v >= t
    idxs = np.searchsorted(v, thresholds_gy, side="left")
    less_mass = np.where(idxs > 0, csum[np.clip(idxs - 1, 0, len(csum) - 1)], 0.0)
    ge_mass = total - less_mass
    return ge_mass.astype(np.float32)


def build_dvh_tables_clinical_style(masks: Dict[str, np.ndarray],
                                    dose_arr: np.ndarray,
                                    voxel_vol_cc: float,
                                    rx_gy: Optional[float],
                                    spacing_mm: Tuple[float, float, float]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Build dicts of absolute and relative DVH tables with "clinical" headers.
    """
    abs_tables: Dict[str, pd.DataFrame] = {}
    rel_tables: Dict[str, pd.DataFrame] = {}

    dose_max = float(np.max(dose_arr)) if dose_arr.size else 0.0
    abs_grid, rel_grid = _fixed_grids_for_clinical_style(dose_max, rx_gy)
    rel_to_abs = (rel_grid / 100.0) * float(rx_gy) if rx_gy and rx_gy > 0 else np.zeros_like(rel_grid)

    for roi, mask in masks.items():
        dose_vals, w = _sample_native_dose(mask, spacing_mm, dose_arr, smooth_sigma=SMOOTH_SIGMA_MM)

        vols_cc_abs = _cum_volumes_at_thresholds(dose_vals, w, voxel_vol_cc, abs_grid)
        rel_col = (abs_grid / float(rx_gy) * 100.0) if rx_gy and rx_gy > 0 else np.zeros_like(abs_grid, dtype=np.float32)
        abs_df = pd.DataFrame({
            "Dose [Gy]": abs_grid.astype(np.float64),
            "Relative dose [%]": rel_col.astype(np.float64),
            "Structure Volume [cm³]": vols_cc_abs.astype(np.float64),
        })
        abs_tables[roi] = abs_df

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
    """Excel sheet name sanitizer (illegal chars → '_', max len 31)."""
    bad = '/\\?*[]:'
    for ch in bad:
        name = name.replace(ch, "_")
    nm = (name[:31 - len(suffix)] + suffix) if len(name) + len(suffix) > 31 else (name + suffix)
    return nm or "Sheet"


def export_excel_with_clinical_style_dvhs(xlsx_path: Path,
                                          roi_metrics_df: Optional[pd.DataFrame],
                                          abs_tables: Dict[str, pd.DataFrame],
                                          rel_tables: Dict[str, pd.DataFrame]) -> None:
    """
    Write ROI metrics + per‑ROI absolute/relative DVH sheets into one Excel.
    Uses openpyxl to avoid a hard dependency on xlsxwriter.
    """
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as ew:
        if roi_metrics_df is not None:
            roi_metrics_df.to_excel(ew, sheet_name="ROI metrics", index=False)

        for roi in sorted(abs_tables.keys(), key=lambda s: s.lower()):
            abs_df = abs_tables[roi]
            rel_df = rel_tables.get(roi)

            abs_sheet = _sanitize_sheet_name(str(roi))
            abs_df.to_excel(ew, sheet_name=abs_sheet, index=False)

            if rel_df is not None:
                rel_sheet = _sanitize_sheet_name(str(roi), suffix="_rel")
                rel_df.to_excel(ew, sheet_name=rel_sheet, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_all_roi_dvhs(dvh_abs: Dict[str, pd.DataFrame],
                      prescription: Optional[float],
                      *,
                      x_mode: str = "dose",
                      y_mode: str = "volume",
                      ax: Optional[Axes] = None,
                      linewidth: float = 1.5) -> Axes:
    """
    Plot all DVHs with flexible axes.
      x_mode: {'dose','relative'}
      y_mode: {'volume','relative'}
    """
    if x_mode not in {"dose", "relative"}:
        raise ValueError("x_mode must be 'dose' or 'relative'")
    if y_mode not in {"volume", "relative"}:
        raise ValueError("y_mode must be 'volume' or 'relative'")
    if x_mode == "relative" and not prescription:
        raise ValueError("Relative x‑axis requires a prescription dose")

    if ax is None:
        _, ax = plt.subplots()

    for roi, df in dvh_abs.items():
        if df.empty:
            continue

        if x_mode == "dose":
            x = df["Dose [Gy]"].to_numpy()
            xlabel = "Dose [Gy]"
        else:
            x = df["Dose [Gy]"].to_numpy() / float(prescription) * 100.0
            xlabel = "Relative dose [% Rx]"

        if y_mode == "volume":
            y = df["Volume [cm³]"].to_numpy()
            ylabel = "Volume [cm³]"
        else:
            y0 = float(df["Volume [cm³]"].iloc[0]) if not df.empty else 1.0
            y = df["Volume [cm³]"].to_numpy() / max(y0, 1e-12) * 100.0
            ylabel = "Relative volume [%]"

        ax.plot(x, y, label=roi, linewidth=linewidth)

    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize="small")
    ax.figure.tight_layout()
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage:  python aligned_metrics_full_sigmafix_calls.py "C:\\path\\to\\PATIENT_FOLDER"')

    folder = Path(sys.argv[1]).expanduser()
    if not folder.exists():
        sys.exit(f"Folder not found: {folder}")

    # 1) Load DICOMs & build masks
    ct_img, _ = load_ct(str(folder))
    dose_ds, dose_raw = load_dose(str(folder))
    dose_arr = resample_dose_to_ct(dose_ds, dose_raw, ct_img)

    rs = pydicom.dcmread(_find_dcm(str(folder), "RTSTRUCT"))
    rp = pydicom.dcmread(_find_dcm(str(folder), "RTPLAN"), stop_before_pixels=True)

    rx = _get_prescription(rp)
    masks = build_roi_masks(rs, ct_img)

    # Spacing (mm) and voxel volume (cc)
    spacing_x, spacing_y, spacing_z = ct_img.GetSpacing()   # (x,y,z) in mm
    spacing_mm = (float(spacing_x), float(spacing_y), float(spacing_z))
    voxel_vol_cc = (spacing_x * spacing_y * spacing_z) / 1000.0

    # 2) ROI metrics
    roi_df = compute_roi_metrics(masks, dose_arr, voxel_vol_cc, rx, spacing_mm, SMOOTH_SIGMA_MM)

    # 3) Clinical‑style DVH tables
    abs_tables, rel_tables = build_dvh_tables_clinical_style(masks, dose_arr, voxel_vol_cc, rx, spacing_mm)

    # 4) Export Excel (one file under the patient folder)
    out_xlsx = folder / "abs_DVH_CTgrid.xlsx"
    export_excel_with_clinical_style_dvhs(out_xlsx, roi_df, abs_tables, rel_tables)

    print(f"✔ Clinical‑style DVHs saved → {out_xlsx}")


if __name__ == "__main__":
    main()
