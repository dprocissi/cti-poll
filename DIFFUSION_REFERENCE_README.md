# Diffusion Reference Image Generator

## Overview

The `generate_diffusion_reference_images.py` script creates **high-SNR diffusion reference images** by averaging all b-values and gradient directions from raw Bruker DTI/DWI data. These images are then **coregistered to match your existing MD, FA, and ADC maps** for seamless mask creation.

## Purpose

For manual mask generation in neuroimaging analysis, you need:
- **High signal-to-noise ratio** for clear tissue boundaries
- **Geometric alignment** with your parametric maps (MD, FA, ADC)
- **Matching spatial dimensions** for consistent mask application

This script provides exactly that by:
1. Averaging all diffusion-weighted images (all b-values + all directions)
2. Coregistering to your existing parametric maps
3. Resampling to match MD/FA/ADC dimensions and orientation

## Key Features

✓ **Reads raw Bruker DTI data** with proper parameter extraction (b-values, directions)
✓ **High-SNR averaging** across all b-values and gradient directions
✓ **Automatic coregistration** to existing MD/FA/ADC maps
✓ **Dimension matching** via affine-based resampling
✓ **Robust error handling** with detailed logging
✓ **Consistent naming:** `Mouse_{Cohort}_DiffRef.nii.gz`
✓ **Dual output:** Saved in both DiffusionReference/ and ClaudeParameters/ folders

## How It Works

### 1. DTI Data Reading

The script locates DTI/DWI scans in your raw Bruker data:
- Scans Bruker session folders for sequence identification
- Reads `method` file to confirm sequence type (DTI, DWI)
- Extracts b-values and number of gradient directions (if available)

### 2. Reference Image Generation

From raw DTI data (shape: nframes × ny × nx, where nframes = nb-values × ndirections):
```
For each slice:
  1. Average across all frames (b-values × directions)
  2. Result: Single high-SNR reference image per slice
  3. Stack slices into 3D volume
```

**Why this works:**
- All b-values contribute signal → higher SNR than single b-value
- All directions averaged → tissue contrast without directional bias
- Perfect for manual ROI tracing and mask creation

### 3. Coregistration to Parametric Maps

The script automatically:
- Loads your existing MD, FA, or ADC maps as reference
- Extracts their spatial geometry (affine matrix, dimensions)
- Resamples diffusion reference to match using scipy.ndimage.affine_transform
- Result: Identical geometry to your parametric maps

```
Reference space (from MD/FA/ADC):
  Affine matrix → spatial position/orientation
  Dimensions → (x, y, z) voxel grid

Diffusion reference → resampled to match → perfect alignment
```

## Usage

### Installation

```bash
pip install numpy nibabel scipy
```

### Running the Script

```bash
# On macOS (where data exists)
python3 generate_diffusion_reference_images.py
```

### Expected Behavior

1. **Scans each cohort** for mouse session folders
2. **Identifies DTI scans** in each session
3. **Reads raw diffusion data** from 2dseq files
4. **Averages across b-values and directions** for high SNR
5. **Loads reference maps** (MD, FA, ADC, or T2) for geometry
6. **Coregisters** diffusion reference to match reference map
7. **Saves output** in both locations:
   - `DiffusionReference/[cohort]/[mouse]/*_DiffRef.nii.gz`
   - `ClaudeParameters/[cohort]/[mouse]/*_DiffRef.nii.gz`

### Output Structure

```
DiffusionReference/
├── AgedSham_Maraviroc/
│   ├── m1_cg5568_ShamMaV_SS_20260617/
│   │   └── m1_cg5568_ShamMaV_SS_20260617_AgedSham_Maraviroc_DiffRef.nii.gz
│   ├── m2_cg5568.../
│   │   └── m2_cg5568_..._AgedSham_Maraviroc_DiffRef.nii.gz
│   └── m3_cg5568.../
├── AgedSham_Vehicle/
├── AgedTBI_Maraviroc/
├── AgedTBI_Vehicle/
├── diffusion_reference.log  # Processing log
└── diffusion_reference_report.md  # Summary

(Plus copies in ClaudeParameters/ for easy access alongside other maps)
```

## File Naming Convention

```
{mouse_id}_{cohort_name}_DiffRef.nii.gz

Examples:
m1_cg5568_ShamMaV_SS_20260617_AgedSham_Maraviroc_DiffRef.nii.gz
m2_cg0544_TBIMaV_SS_20260617_AgedTBI_Maraviroc_DiffRef.nii.gz
```

## Using Diffusion Reference for Mask Creation

### Workflow

1. **Open in visualization tool:**
   ```bash
   # Option 1: FSL
   fslview path/to/*_DiffRef.nii.gz &
   
   # Option 2: ITK-SNAP
   itksnap -g path/to/*_DiffRef.nii.gz &
   
   # Option 3: MRICron
   mricron path/to/*_DiffRef.nii.gz &
   ```

2. **Overlay parametric maps (optional but helpful):**
   ```bash
   # Load MD, FA, or ADC as overlay
   # These will be perfectly aligned due to coregistration
   ```

3. **Trace ROIs/masks** on high-SNR diffusion reference
   - Clear tissue boundaries make manual tracing easier
   - Geometric alignment ensures masks work for all maps

4. **Apply masks to all maps:**
   - Same mask coordinates apply to MD, FA, ADC, and diffusion reference
   - Perfect alignment guaranteed

## Technical Details

### Coregistration Method

Uses affine-based spatial normalization:

```python
# Reference space transformation
# Maps source coordinates to reference space
inv_source_affine = inv(source_affine[:3, :3])
transform_matrix = inv_source_affine @ reference_affine[:3, :3]
transform_offset = inv_source_affine @ (reference_affine[:3, 3] - source_affine[:3, 3])

# Resampling via scipy.ndimage.affine_transform
resampled = affine_transform(source_data,
                            inv(transform_matrix),
                            offset=-inv(transform_matrix) @ transform_offset,
                            output_shape=reference_shape,
                            order=1)  # Linear interpolation
```

### Fallback Strategy

If reference map not found:
1. Searches for MD map first (highest SNR)
2. Falls back to FA, then ADC
3. Falls back to T2 map from parametric generation
4. If still not found, uses simple zoom resampling (less accurate)

### Supported DTI Sequences

- Bruker DTI (diffusion tensor imaging)
- Multi-b-value DWI (diffusion weighted imaging)
- Any multi-echo, multi-direction gradient sequence

## Quality Checks

After generation, verify:

✓ **File created:** Check file size (should be ~10-50 MB)
✓ **NIfTI validity:** Load in FSL/ITK-SNAP without errors
✓ **Geometry matches:** Overlay with MD/FA maps → perfect alignment
✓ **High SNR:** Visual inspection shows clear tissue boundaries
✓ **No artifacts:** Check for distortions or blank regions

## Troubleshooting

### "No DTI scans found"
- Check that DTI/DWI sequence exists in session
- Verify Bruker method file has "dti" or "dwi" in Method name
- Check scan_manifest.json for sequence identification

### "Failed to read 2dseq"
- Ensure pdata/1/ folder exists with visu_pars and 2dseq
- Check file permissions (readable)
- Verify Bruker data integrity

### "No reference map found"
- Script will still generate output using simple geometry
- Quality will be lower without coregistration
- Consider running parametric_maps_enhanced.py first to generate MD/FA maps

### Dimensions don't match after coregistration
- Script logs detailed warnings to diffusion_reference.log
- Check log for specific dimension mismatches
- Resampling should still work but may indicate data issues

## Log Files

### `diffusion_reference.log`
- Real-time processing status
- Per-scan reading results
- Coregistration details
- Any errors or warnings

### `diffusion_reference_report.md`
- Summary of all generated diffusion references
- File sizes and locations
- Per-cohort and per-mouse statistics

## Performance Notes

- **Typical time per mouse:** 2-5 minutes
- **Memory usage:** ~2-3 GB per mouse
- **Disk space:** ~50-100 MB per mouse (output)

## Advanced: Customization

### Change reference map search order

Edit the `load_reference_map()` function:
```python
search_patterns = [
    os.path.join(DERIVED_BASE, "YOUR_MAP_TYPE", f"*{mouse_pattern}*"),
    # ... other patterns
]
```

### Adjust resampling interpolation order

In `resample_to_reference()`:
```python
# order=1 is linear (default)
# order=0 is nearest neighbor (faster, lower quality)
# order=3 is cubic (slower, higher quality)
affine_transform(..., order=3)  # Change as needed
```

## References

- **Diffusion Tensor Imaging:** Basser et al. (1994)
- **SNR improvement:** Averaging independent measurements
- **Affine registration:** Rigid body + scaling transformation
- **NIfTI format:** Li et al. (2016)

---

## Integration with Analysis Pipeline

These diffusion reference images fit into your workflow as:

```
Raw Bruker DTI Data
        ↓
generate_diffusion_reference_images.py
        ↓
High-SNR Diffusion Reference (coregistered)
        ↓
Manual mask creation in FSL/ITK-SNAP
        ↓
Apply masks to MD, FA, ADC maps
        ↓
Statistical analysis
```

**Script Version:** 1.0  
**Last Updated:** 2026-07-29
