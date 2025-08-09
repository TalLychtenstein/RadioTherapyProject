#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────────────────
#  roi_dvh_backend_patched.py      ·  CT / Dose helpers  +  ROI → abs‑DVH
#  2025‑08‑01  –  fixed shape mismatch (mask vs dose)    ← NEW
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, numpy as np, pydicom, SimpleITK as sitk
from pathlib import Path
import pandas as pd
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter

# ── Optimise smooth_sigma against a clinical DVH table ──────────────────
import numpy as np

def _extract_clinical_volumes(xlsx: str) -> dict[str, float]:
    xl = pd.ExcelFile(xlsx)
    vols = {}
    for sh in xl.sheet_names:
        df = xl.parse(sh, nrows=1)
        vcol = _find_variant(df.columns, "volume")
        if vcol:
            vols[sh] = float(df[vcol].iloc[0])
    return vols


def _volume_error(dvh_abs: dict[str, pd.DataFrame],
                  ref_vols: dict[str, float]) -> float:
    """Mean absolute error (cm³) over ROIs that exist in both sets."""
    errs = []
    for roi, df in dvh_abs.items():
        if roi in ref_vols and not df.empty:
            errs.append(abs(df["Volume [cm³]"].iloc[0] - ref_vols[roi]))
    return np.mean(errs) if errs else np.inf


def tune_sigma(masks, dose_arr, voxel_vol_cc, prescription,
               spacing_mm, ref_xlsx: str,
               sigma_min=0.00, sigma_max=1.50, step=0.05,
               verbose=True) -> float:
    """
    Grid‑search σ that minimises mean |ΔVolume| to the clinical DVH export.
    Returns the best σ (float).
    """
    ref_vols = _extract_clinical_volumes(ref_xlsx)
    if not ref_vols:
        raise RuntimeError(
            f"No structure volumes found in “{ref_xlsx}”.\n"
            "Expected a column called either 'Structure Volume [cm³]' "
            "or 'Volume [cm³]' in the first row of every sheet."
        )

    best_sig, best_err = None, np.inf

    for sigma in np.arange(sigma_min, sigma_max + 1e-9, step):
        dvh_abs = compute_abs_dvhs(
            masks, dose_arr, voxel_vol_cc, prescription,
            smooth_sigma=sigma, spacing_mm=spacing_mm)

        err = _volume_error(dvh_abs, ref_vols)
        if verbose:
            print(f"σ={sigma:4.2f}→mean |ΔV|={err:6.3f}cm³")
        if err < best_err:
            best_sig, best_err = sigma, err

    # ── after the loop ────────────────────────────────────────────────
    if best_sig is None or np.isinf(best_err):
        raise RuntimeError(
                "No ROI names matched between patient DVHs and the clinical file, "
                "so σ could not be tuned. Check that ROI sheet names are identical."
            )

    print(f"🏅  Chosen σ={best_sig:4.2f}mm   (mean |ΔV|={best_err:6.3f}cm³)")
    return best_sig


# ╔════════════════════════════  DICOM HELPERS  ══════════════════════════╗
def _find_dcm(folder: str, mod: str) -> str:
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:  # faster: stop before pixel data
            if pydicom.dcmread(p, stop_before_pixels=True).Modality == mod:
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"{mod} not found in “{folder}”")

def load_ct(folder: str):
    rdr = sitk.ImageSeriesReader()
    for sid in rdr.GetGDCMSeriesIDs(folder) or []:
        fns = rdr.GetGDCMSeriesFileNames(folder, sid)
        if pydicom.dcmread(fns[0], stop_before_pixels=True).Modality != "CT":
            continue
        rdr.SetFileNames(fns)
        img = rdr.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.min() >= 0:            # unsigned CT → convert to HU
            arr -= 1024.0
        return img, arr               # (Z,Y,X)
    raise RuntimeError("CT not found")

def load_dose(folder: str):
    ds  = pydicom.dcmread(_find_dcm(folder, "RTDOSE"))
    arr = ds.pixel_array.astype(np.float32) * ds.DoseGridScaling
    if arr.ndim == 4:                # handle (1,Z,Y,X)
        arr = arr[0]
    return ds, arr                   # (Z,Y,X)

def resample_dose_to_ct(dose_ds, dose_arr, ct_img):
    """Linear‑resample RTDOSE grid → CT grid."""
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
# ╚═══════════════════════════════════════════════════════════════════════╝

# ╔══════════════════  ROI → binary mask on CT grid  ═════════════════════╗
def build_roi_masks(rs, ct_img):
    """
    Return  masks[name] -> 3‑D uint8 mask  on CT grid.
    Points outside the CT are clamped, guaranteeing full coverage.
    """
    sx, sy, sz = ct_img.GetSize()          # CT array is (X,Y,Z)
    masks = {}
    for roi in rs.StructureSetROISequence:
        num = roi.ROINumber
        rc  = next(r for r in rs.ROIContourSequence
                   if r.ReferencedROINumber == num)
        m = np.zeros((sz, sy, sx), np.uint8)
        for cs in rc.ContourSequence:
            pts = np.array(cs.ContourData).reshape(-1, 3)
            ijk = [ct_img.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
            xs, ys, zs = map(np.array, zip(*ijk))

            xs = np.clip(np.round(xs).astype(int), 0, sx - 1)
            ys = np.clip(np.round(ys).astype(int), 0, sy - 1)
            z  = int(np.clip(np.round(np.median(zs)), 0, sz - 1))

            rr, cc = polygon(ys, xs, (sy, sx))
            m[z, rr, cc] = 1
        masks[roi.ROIName] = m
    return masks
# ╚═══════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────  NEW DVH CODE  ────────────────────────────
from scipy.ndimage import gaussian_filter

def _sample_native_dose(mask: np.ndarray,
                        dose: np.ndarray,
                        smooth_sigma: float = 0.5):
    """
    Return (dose_values, occupancy_weights) for voxels that belong to *mask*.

    * We still blur the mask so that edge voxels get fractional weights,
      but **we only keep voxels where mask==1** – no halo outside the ROI.
    """
    if mask.shape != dose.shape:
        raise ValueError(
            f"Mask shape {mask.shape} and dose shape {dose.shape} differ – "
            "they have to live on the same grid."
        )

    if smooth_sigma > 0:
        weight_mask = gaussian_filter(mask.astype(np.float32),
                                      sigma=smooth_sigma)
    else:
        weight_mask = mask.astype(np.float32)

    idx = mask.astype(bool)          # ← keep strictly inside the ROI
    return dose[idx], weight_mask[idx]

def _dvh_cumsum_weighted(dose_values,
                         weights,
                         voxel_vol_cc,
                         step_gy: float = 0.1,
                         max_dose: float | None = None):
    """
    Histogram *dose_values* in *step_gy* bins, weight by *weights × voxel_vol_cc*,
    and return (dose_axis, cumulative_volume_cm³).
    If *max_dose* is given, extend the histogram so every ROI’s DVH
    reaches the same absolute maximum.
    """
    if dose_values.size == 0:
        return np.empty(0), np.empty(0)

    end = max_dose if max_dose is not None else dose_values.max()
    last_edge = np.ceil(end / step_gy) * step_gy
    bins = np.arange(0.0, last_edge + step_gy, step_gy)

    vol_hist, _ = np.histogram(dose_values, bins=bins,
                               weights=weights * voxel_vol_cc)
    cumvol = np.cumsum(vol_hist[::-1])[::-1]          # high→low
    return bins[:-1], cumvol


def dvh_table_abs(dose_values, weights,
                  voxel_vol_cc: float,
                  step: float = 0.1,
                  prescription: float | None = None,
                  max_dose: float | None = None) -> pd.DataFrame:
    """Return a 3‑column absolute DVH table (Gy, %, cm³)."""
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
    """Binary‑mask volume in cm³."""
    return float(mask.sum()) * np.prod(spacing_mm) / 1000.0  # mm³ → cm³


def compute_abs_dvhs(masks: dict[str, np.ndarray],
                     dose_arr: np.ndarray,
                     voxel_vol_cc: float,
                     prescription: float | None,
                     step: float = 0.1,
                     smooth_sigma: float = 0.89,
                     spacing_mm: tuple[float, float, float] | None = None
                     ) -> dict[str, pd.DataFrame]:
    """
    Calculate an **absolute‑dose DVH** (Gy‑axis) for every ROI.
    Each curve is padded so that its last bin equals the *global* max dose,
    matching the clinical reference export.
    """
    dvh_abs: dict[str, pd.DataFrame] = {}
    global_max_dose = float(dose_arr.max())

    for roi, mask in masks.items():
        dose_vals, weights = _sample_native_dose(mask, dose_arr, smooth_sigma)
        df = dvh_table_abs(dose_vals, weights, voxel_vol_cc,
                           step=step,
                           prescription=prescription,
                           max_dose=global_max_dose)
        dvh_abs[roi] = df

        # Optional sanity printout
        if spacing_mm is not None:
            v_mask = compute_mask_volume(mask, spacing_mm)
            v_dvh  = df["Volume [cm³]"].iloc[0] if not df.empty else 0
            print(f"{roi:30s}: voxelised = {v_mask:7.3f} cc | "
                  f"DVH = {v_dvh:7.3f} cc | Δ = {v_dvh - v_mask:+7.3f} cc")

    return dvh_abs
# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────  CLINICAL‑DVH VALIDATION  ──────────────────────────
from pathlib import Path
from scipy.interpolate import interp1d

COL_VARIANTS = {
    "dose":   {"dose [gy]", "dose[gy]", "dose (gy)"},
    "volume": {"volume [cm³]", "volume [cm^3]",
               "structure volume [cm³]", "structure volume [cm^3]",
               "volume (cm³)", "volume (cm^3)"},
}
def _find_variant(columns, key):
    """Return the actual column name that matches one of the accepted variants."""
    for c in columns:
        if c.lower().strip() in COL_VARIANTS[key]:
            return c
    return None

def _load_dvh_tables(xlsx: str) -> dict[str, pd.DataFrame]:
    """
    Return {ROI: DataFrame} for every sheet that has a Dose column and a Volume
    column, irrespective of minor spelling or unit differences.
    """
    xl = pd.ExcelFile(xlsx)
    dvhs = {}
    for roi in xl.sheet_names:
        df = xl.parse(roi)

        dcol = _find_variant(df.columns, "dose")
        vcol = _find_variant(df.columns, "volume")

        if dcol and vcol:
            dvhs[roi] = (df[[dcol, vcol]]
                         .rename(columns={dcol: "Dose [Gy]",
                                          vcol: "Volume [cm³]"})
                         .dropna())
    return dvhs


def _align_volume(my_df: pd.DataFrame, ref_df: pd.DataFrame) -> np.ndarray:
    """
    Interpolate *my_df*’s cumulative volume to the exact dose grid of *ref_df*.
    Missing values above the max dose get 0 cm³; below 0 Gy get full volume.
    """
    interp = interp1d(my_df["Dose [Gy]"],
                      my_df["Volume [cm³]"],
                      kind="linear",
                      bounds_error=False,
                      fill_value=(my_df["Volume [cm³]"].iloc[0], 0.0))
    return interp(ref_df["Dose [Gy]"])


def _d_percent(vol_curve: np.ndarray,
               dose_axis: np.ndarray,
               percent: float) -> float:
    """Return Dx% in Gy from a cumulative‑volume vector."""
    target = vol_curve[0] * (1.0 - percent / 100.0)
    return np.interp(target, vol_curve[::-1], dose_axis[::-1])


def _metrics_for_pair(my: pd.DataFrame,
                      ref: pd.DataFrame) -> dict[str, float]:
    """Metric‑level Δ for one ROI."""
    my_aligned = _align_volume(my, ref)
    vol0 = ref["Volume [cm³]"].iloc[0]
    delta_V = 100.0 * (my_aligned[0] - vol0) / vol0      # %
    d2_my  = _d_percent(my_aligned, ref["Dose [Gy]"], 2)
    d2_ref = _d_percent(ref["Volume [cm³]"].to_numpy(),
                        ref["Dose [Gy]"].to_numpy(), 2)
    rmse   = np.sqrt(np.mean((my_aligned - ref["Volume [cm³]"]) ** 2))
    maxdv  = np.max(np.abs(my_aligned - ref["Volume [cm³]"]))
    return {"ΔVolume %": delta_V,
            "ΔD2% Gy":   d2_my - d2_ref,
            "RMSE cm³":  rmse,
            "Max |ΔV| cm³": maxdv}


def _key(name: str) -> str:
    """A case‑insensitive, 31‑char Excel‑safe key for sheet comparison."""
    return name.strip().lower().replace(" ", "").replace("_", "")[:31]

def validate_clinical_dvh(computed_xlsx: str,
                          clinical_xlsx: str,
                          out_dir: str | Path = "validation results",
                          verbose: bool = True) -> pd.DataFrame:
    """
    Compare every ROI that exists in *both* files and write a tabular report.

    Returns the DataFrame so the UI layer can colour‑code and display it.
    """

    computed_raw = _load_dvh_tables(computed_xlsx)
    clinical_raw = _load_dvh_tables(clinical_xlsx)
    # normalise names → {key: (pretty_name, df)}
    computed = {_key(n): (n, df) for n, df in computed_raw.items()}
    clinical = {_key(n): (n, df) for n, df in clinical_raw.items()}

    if not clinical:
        raise RuntimeError(
            f"Could not find a usable Volume column in any sheet of "
            f"‘{Path(clinical_xlsx).name}’. "
            "Accepted spellings include: Volume [cm³], Volume [cm^3], "
            "Structure Volume [cm³], Structure Volume [cm^3]."
        )

    rows = []
    for k in sorted(set(computed) & set(clinical)):
        cname, cdf = computed[k]
        rname, rdf = clinical[k]
        row = {"ROI": cname, **_metrics_for_pair(cdf, rdf)}
        rows.append(row)
        if verbose:
            print(f"{cname:30s}  ΔVol={row['ΔVolume %']:6.2f}%  "
                  f"ΔD2%={row['ΔD2% Gy']:6.2f} Gy")

    df = pd.DataFrame(rows)

    # ── persist to Excel ─────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"{Path(computed_xlsx).stem}_vs_clinical.xlsx"
    df.to_excel(out_xlsx, index=False)
    if verbose:
        print(f"📊  Validation table → {out_xlsx}")

    return df
# ───────────────────────────────────────────────────────────────────────────────





# ╔═════════════════════  Small helpers for CLI  ════════════════════════╗
def _get_prescription(rtplan) -> float | None:
    """Return TargetPrescriptionDose if available, else None."""
    for dr in getattr(rtplan, "DoseReferenceSequence", []):
        if getattr(dr, "DoseReferenceType", "").upper() == "TARGET":
            return float(dr.TargetPrescriptionDose)
    return None
# ╚══════════════════════════════════════════════════════════════════════╝



import argparse, textwrap

def main():
    parser = argparse.ArgumentParser(
        description="Export absolute‑dose DVHs one sheet per ROI "
                    "and (optionally) auto‑tune smooth_sigma "
                    "to match a clinical DVH Excel file.")
    parser.add_argument("patient", help="Patient folder with CT / RTSTRUCT / RTDOSE / RTPLAN")
    parser.add_argument("--clinical", metavar="XLSX",
                        help="Clinical DVH export to calibrate σ")
    parser.add_argument("--sigma", type=float, default=0.89,
                        help="Smooth‑sigma (mm); ignored if --clinical is given")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Grid step for σ search (mm) with --clinical")
    # 1. extend the parser
    parser.add_argument("--validate", action="store_true",
                        help="After σ tuning and DVH export, compare with the "
                             "clinical Excel and save a validation table")
    args = parser.parse_args()

    folder = Path(args.patient).expanduser()
    if not folder.is_dir():
        raise SystemExit("❌patient folder not found")

    # 1–load DICOM objects -------------------------------------------------
    ct_img, _ = load_ct(str(folder))
    dose_ds, dose_raw = load_dose(str(folder))
    dose_arr = resample_dose_to_ct(dose_ds, dose_raw, ct_img)
    rs = pydicom.dcmread(_find_dcm(str(folder), "RTSTRUCT"))
    rp = pydicom.dcmread(_find_dcm(str(folder), "RTPLAN"),
                         stop_before_pixels=True)
    prescription = _get_prescription(rp)
    masks = build_roi_masks(rs, ct_img)

    sx, sy, sz = ct_img.GetSpacing()
    spacing_mm = (sx, sy, sz)
    voxel_vol_cc = (sx * sy * sz) / 1000.0                     # mm³ → cm³

    # 2–determine σ -------------------------------------------------------
    if args.clinical:
        sigma = tune_sigma(masks, dose_arr, voxel_vol_cc, prescription,
                           spacing_mm, args.clinical, step=args.step)
    else:
        sigma = args.sigma
        print(f"Using user‑supplied σ={sigma}")

    # 3–compute DVHs ------------------------------------------------------
    dvh_abs = compute_abs_dvhs(masks, dose_arr, voxel_vol_cc, prescription,
                               smooth_sigma=sigma, spacing_mm=spacing_mm)

    # 4–save Excel --------------------------------------------------------
    out_xlsx = folder / "abs_DVH_CTgrid.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        for roi, df in dvh_abs.items():
            df.to_excel(xl, sheet_name=roi[:31] or "ROI", index=False)

    # 5–optional: tiny report --------------------------------------------
    if args.clinical:
        report = folder / "calibration-report.txt"
        with open(report, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(f"""
                    Best smooth_sigma  : {sigma:.2f}mm

                    Compared against   : {args.clinical}
                    Patient export     : {out_xlsx.name}

                    Metric             : mean absolute error in Structure Volume (cm³)
                """).strip() + "\n")
        print(f"📄  Calibration report → {report}")



    # 2. call the routine
    if args.clinical and args.validate:
        validate_clinical_dvh(out_xlsx, args.clinical,
                              out_dir=folder / "validation results")


    print(f"✔️  Absolute DVHs saved → {out_xlsx}")

if __name__ == "__main__":
    main()
# ╚══════════════════════════════════════════════════════════════════════╝
