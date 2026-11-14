# Parametric Map Generation — Enhanced Pipeline

## Overview

The `generate_parametric_maps_enhanced.py` script generates missing **T1, T2, T2*, and 3D-reformatted T2w RARE maps** from raw Bruker PV360 MRI data for the Maraviroc Aged TBI study.

## Key Enhancements

### 1. **Complete Binary 2dseq Reading**
- Properly parses Bruker `reco` and `visu_pars` files to extract:
  - Matrix dimensions (VisuCoreSize)
  - Data type and byte order (RECO_wordtype)
  - Frame count and voxel dimensions
  - Slice positioning information
- Reads binary 2dseq data directly with correct numpy dtype
- Handles multi-frame (multi-slice, multi-echo) data correctly

**Files parsed:**
```
scan/pdata/1/reco         → Data type, word size
scan/pdata/1/visu_pars    → Matrix dims, voxel size, frame count
scan/pdata/1/2dseq        → Binary raw image data
```

### 2. **Vectorized Parametric Fitting**

#### T1 Maps (VFA FLASH)
- **Model:** Ernst equation with linearized fit
  - `S/sin(α) = E1 × S/tan(α) + M0(1-E1)`
  - where `E1 = exp(-TR/T1)`
- **Processing:**
  - Collects all FLASH scans with different flip angles
  - Extracts TR from method file
  - Performs least-squares linear regression per voxel
  - Constrains output: 100 ms < T1 < 5000 ms
- **Input:** Multiple FLASH images at different FA (e.g., 2°, 5°, 10°, 15°, 20°, 30°)
- **Output:** T1_map.nii.gz

#### T2 Maps (MSME)
- **Model:** Monoexponential decay
  - `S(TE) = S₀ × exp(-TE/T2)`
- **Processing:**
  - Extracts echo times from method file (PVM_EchoTime or EffectiveTE)
  - Reads multi-echo 2dseq data
  - Fits exponential decay curve per voxel using scipy.optimize.curve_fit
  - Constrains: 1 ms < T2 < 500 ms
- **Input:** MSME (Multi-Spin-Multi-Echo) sequence with ≥2 echoes
- **Output:** T2_map.nii.gz

#### T2* Maps (MGE)
- **Model:** Gradient-echo decay
  - `S(TE) = S₀ × exp(-TE/T2*)`
- **Processing:**
  - Similar to T2, uses multi-gradient-echo data
  - Extracts echo times from method parameters
  - Fits per-voxel exponential
  - Constrains: 0.5 ms < T2* < 200 ms (shorter than T2 at 7T)
- **Input:** MGE (Multi-Gradient-Echo) sequence with ≥2 echoes
- **Output:** T2star_map.nii.gz

### 3. **3D RARE Reformatting**

- **Input:** Raw 2D RARE slices (stacked in 2dseq)
- **Processing:**
  - Reads multi-frame 2dseq as stack of 2D images
  - Reorders from (nframes, ny, nx) to (nx, ny, nframes)
  - Builds proper 3D affine transformation using:
    - Voxel size (from visu_pars)
    - Slice thickness and positioning
  - Creates valid NIfTI geometry
- **Output:** T2w_RARE_3D.nii.gz (true 3D anatomy reference)

### 4. **Production-Ready Error Handling**

- **Logging system:**
  - Timestamped console output
  - Persistent log file: `ClaudeParameters/generation.log`
  - INFO, WARN, ERROR severity levels

- **Robust parameter parsing:**
  - Handles missing files gracefully
  - Validates array dimensions and data types
  - Falls back to default values when parameters unavailable
  - Detects and skips corrupted scan folders

- **Validation:**
  - Checks physiological T1/T2/T2* ranges
  - Validates curve fitting success before saving
  - Reports file sizes and dimensions
  - Generates comprehensive processing report

- **Fallback strategy:**
  - First tries to use pre-computed NIfTI files from:
    - `Maraviroc_DerivedMaps/T1/`, `Maraviroc_DerivedMaps/T2/`
    - Session `DerivedMaps/` folders
    - Session `T1_MFA_results/` folders
  - Falls back to raw Bruker data fitting if needed
  - Skips gracefully if no source data available

## Usage

### Installation

```bash
pip install numpy nibabel scipy
```

### Running the Script

```bash
# On macOS (where data is located):
python3 generate_parametric_maps_enhanced.py
```

The script will:

1. **Scan each cohort** for available mouse sessions
2. **Identify all Bruker sequences** in each session folder
3. **Check for existing NIfTI files** to avoid redundant processing
4. **Generate missing maps:**
   - T1 from FLASH VFA scans (or copy from existing)
   - T2 from MSME scans (or copy from existing)
   - T2* from MGE scans (always generate)
   - 3D RARE anatomical (always generate)
5. **Create output directory structure:**

```
ClaudeParameters/
├── AgedSham_Maraviroc/
│   ├── m1_cg5568_ShamMaV_SS_20260617/
│   │   ├── m1_..._T1_map.nii.gz
│   │   ├── m1_..._T2_map.nii.gz
│   │   ├── m1_..._T2star_map.nii.gz
│   │   └── m1_..._T2w_RARE_3D.nii.gz
│   ├── m2_cg5568.../
│   └── m3_cg5568.../
├── AgedSham_Vehicle/
├── AgedTBI_Maraviroc/
├── AgedTBI_Vehicle/
├── generation.log          # Detailed processing log
└── parametric_map_report.md  # Summary report
```

## Cohort Requirements

| Cohort | Missing Maps | Always Generate |
|--------|-------------|-----------------|
| **AgedSham_Maraviroc** | T1, T2 | T2*, RARE 3D |
| **AgedSham_Vehicle** | T1 | T2*, RARE 3D |
| **AgedTBI_Maraviroc** | T1, T2 | T2*, RARE 3D |
| **AgedTBI_Vehicle** | T1 | T2*, RARE 3D |

## Output Quality Checks

After generation, verify:

✓ **Dimensions:** All maps match RARE anatomical dimensions
✓ **T1 ranges:** Brain tissue 500-2000 ms (WM ~700-800, GM ~1200-1500, CSF >3000)
✓ **T2 ranges:** Brain tissue 20-80 ms (WM ~30-35, GM ~40-50, CSF >200)
✓ **T2* ranges:** Brain tissue 10-40 ms (shorter than T2)
✓ **No NaNs:** All slices contain valid data
✓ **File sizes:** Reasonable for 3D brain volume (~10-50 MB per map)

## Log Files

### `generation.log`
- Real-time processing status
- Errors and warnings with timestamps
- Per-mouse fitting results
- Useful for troubleshooting failures

### `parametric_map_report.md`
- Summary of generated files per cohort/mouse
- File sizes and dimensions
- List of successfully generated maps

## Technical Details

### Bruker Parameter Extraction

```python
# Single parameter
Method = parse_bruker_param(content, "Method")  # e.g., "RARE", "MSME", etc.

# Array parameter
echo_times = parse_bruker_array(content, "PVM_EchoTime")  # np.array([5.2, 10.4, ...])
```

### Supported Sequence Types

| Sequence | Type | Use |
|----------|------|-----|
| RARE | Anatomical | T2w reference (3D reformat) |
| MSME | Multi-echo | T2 fitting |
| FLASH | Multi-FA | T1 fitting (VFA) |
| MGE | Multi-echo | T2* fitting |

### Affine Transformation

Maps use proper 3D affine matrices for correct spatial alignment:

```python
affine = diag([vx, vy, vz, 1])  # Voxel size from visu_pars
```

This ensures compatibility with SPM, FSL, ANTs, and other analysis tools.

## Troubleshooting

### "Cannot extract VisuCoreSize"
- Check that `pdata/1/visu_pars` file exists
- Verify Bruker data is not corrupted

### "Insufficient FLASH data or missing TR"
- Verify multiple FLASH scans exist with different flip angles
- Check TR is specified in method file

### "No MGE scans found"
- MGE is required for T2* but may not exist
- Check for alternative gradient-echo sequences

### Very low T1/T2/T2* values or zeros in output
- May indicate fitting failure due to poor SNR
- Check source data quality
- Review parameter constraints in fitting functions

## Advanced Customization

To modify fitting constraints, edit in the script:

```python
# T2 fitting (line ~286)
bounds=([0, 1], [signal[0]*5, 500])  # (s0_min, t2_min), (s0_max, t2_max)

# T1 fitting (line ~328)
if 100 < t1 < 5000:  # Change physiological range if needed

# T2* fitting (line ~366)
bounds=([0, 0.5], [signal[0]*5, 200])
```

## Output Files

### NIfTI Format
All output maps are saved as compressed NIfTI files (.nii.gz):
- **Data:** 3D float32 arrays
- **Affine:** Proper spatial transformation (4×4 matrix)
- **Header:** Description field with map type and mouse ID
- **Compatible with:** FSL, SPM, ANTs, MRICron, ITK-SNAP, etc.

### File Naming Convention
```
{mouse_id}_{MAP_TYPE}_map.nii.gz

Examples:
m1_cg5568_ShamMaV_SS_20260617_T1_map.nii.gz
m1_cg5568_ShamMaV_SS_20260617_T2_map.nii.gz
m1_cg5568_ShamMaV_SS_20260617_T2star_map.nii.gz
m1_cg5568_ShamMaV_SS_20260617_T2w_RARE_3D.nii.gz
```

## Performance Notes

- **Typical processing time per mouse:** 5-15 minutes
- **Memory usage:** ~2-4 GB per mouse (depends on resolution)
- **Disk space needed:** ~500 MB - 1 GB per mouse (output only)

## References

- Bruker PV360 documentation
- Ernst equation: `M = M₀ × sin(α) × (1 - exp(-TR/T1)) / (1 - cos(α) × exp(-TR/T1))`
- T2 decay: Carr-Purcell-Meiboom-Gill (CPMG)
- T2* decay: gradient-echo (GRE)

---

**Script Version:** 2.0 (Enhanced with complete binary reading and 3D reformatting)
**Last Updated:** 2026-07-29
