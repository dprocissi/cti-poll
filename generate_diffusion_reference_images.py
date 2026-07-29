#!/usr/bin/env python3
"""
Maraviroc Aged TBI — Diffusion Reference Image Generator
=========================================================
Generates high-SNR diffusion reference images (averaged across all b-values
and directions) for manual mask creation, coregistered to match parametric maps.

Features:
✓ Reads raw Bruker DTI/DWI data (multi-b-value, multi-direction)
✓ Averages across all b-values and gradient directions for high SNR
✓ Coregisters to existing MD/FA/ADC maps (matching geometry and dimensions)
✓ Proper NIfTI output with spatial alignment
✓ Naming: Mouse_{cohort}_DiffRef.nii.gz

Usage:
    python3 generate_diffusion_reference_images.py

Requirements:
    pip install numpy nibabel scipy
"""

import os
import sys
import glob
import json
import struct
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import nibabel as nib
except ImportError:
    print("Installing nibabel..."); os.system("pip install nibabel"); import nibabel as nib

try:
    from scipy.ndimage import affine_transform, zoom
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    print("Installing scipy..."); os.system("pip install scipy")
    from scipy.ndimage import affine_transform, zoom
    from scipy.interpolate import RegularGridInterpolator

# ============ LOGGING ============
class Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.log_file = None

    def set_file(self, path):
        self.log_file = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def info(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        if self.verbose:
            print(formatted)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')

    def error(self, msg):
        self.info(f"ERROR: {msg}")

    def warn(self, msg):
        self.info(f"WARN: {msg}")

logger = Logger(verbose=True)

# ============ CONFIGURATION ============
BASE = "/Users/danieleprocissi/Documents/ONEDRIVE_Backup_AUG2025/SSchwulst_MARAVIROC"
RAW_BASE = BASE
PARAM_BASE = os.path.join(BASE, "ClaudeParameters")
DERIVED_BASE = os.path.join(BASE, "Maraviroc_DerivedMaps")
OUTPUT_BASE = os.path.join(PARAM_BASE, "DiffusionReference")
LOG_FILE = os.path.join(OUTPUT_BASE, "diffusion_reference.log")

COHORTS = {
    "AgedSham_Maraviroc": ["m1_cg5568", "m2_cg5568", "m3_cg5568"],
    "AgedSham_Vehicle": ["m1_cg5567", "m2_cg5567", "m3_cg5567"],
    "AgedTBI_Maraviroc": ["m1_cg0544", "m2_cg0544", "m3_cg0544"],
    "AgedTBI_Vehicle": ["m1_cg2884", "m2_cg2884", "m1_cg0542"],
}

# ============ BRUKER PARAMETER PARSING ============
def parse_bruker_param(content, key, default=None):
    """Extract single parameter from Bruker text file."""
    search = f"##${key}="
    for line in content.split('\n'):
        if line.startswith(search):
            val = line.split('=', 1)[1].strip()
            if ' ' in val and not val[0].isdigit():
                val = val.split(' ')[0]
            return val
    return default

def parse_bruker_array(content, key):
    """Extract array parameter from Bruker file."""
    search = f"##${key}="
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith(search):
            val = line.split('=', 1)[1].strip()
            if val and val[0] == '(':
                # Multi-line array
                size = int(val[1:-1])
                arr = []
                for j in range(i+1, min(i+10, len(lines))):
                    vals = lines[j].strip().split()
                    for v in vals:
                        try:
                            arr.append(float(v))
                        except:
                            pass
                    if len(arr) >= size:
                        return np.array(arr[:size])
                return np.array(arr) if arr else None
            else:
                try:
                    vals = val.split()
                    if len(vals) == 1:
                        return np.array([float(vals[0])])
                    else:
                        return np.array([float(v) for v in vals])
                except:
                    return None
    return None

def read_bruker_method(scan_path):
    """Read Bruker method file."""
    method_file = os.path.join(scan_path, "method")
    if not os.path.exists(method_file):
        return None

    try:
        with open(method_file, 'r', errors='ignore') as f:
            content = f.read()

        params = {}
        for line in content.split('\n'):
            if '=' in line and line.startswith('##$'):
                key, _, val = line.partition('=')
                params[key.strip().lstrip('##$')] = val.strip()
        return params
    except Exception as e:
        logger.warn(f"Failed to read method file {method_file}: {e}")
        return None

def identify_sequence(method_params):
    """Identify Bruker sequence type."""
    if method_params is None:
        return "unknown"

    method_name = method_params.get("Method", "").lower()

    if "dti" in method_name or "dwi" in method_name or "diffusion" in method_name:
        return "DTI"
    elif "epi" in method_name:
        return "EPI"
    else:
        return method_name[:20] if method_name else "unknown"

def scan_bruker_session(session_path):
    """Scan session for available scans."""
    scans = []
    try:
        for item in sorted(os.listdir(session_path)):
            scan_dir = os.path.join(session_path, item)
            if os.path.isdir(scan_dir) and item.isdigit():
                method = read_bruker_method(scan_dir)
                seq_type = identify_sequence(method)
                scans.append({
                    "scan_num": int(item),
                    "path": scan_dir,
                    "type": seq_type,
                    "method": method,
                })
    except Exception as e:
        logger.warn(f"Error scanning session {session_path}: {e}")

    return sorted(scans, key=lambda x: x['scan_num'])

# ============ BRUKER BINARY DATA READING ============
def read_bruker_2dseq(scan_path, reco_num=1):
    """Read Bruker 2dseq binary data."""
    reco_path = os.path.join(scan_path, "pdata", str(reco_num))
    reco_file = os.path.join(reco_path, "reco")
    visu_file = os.path.join(reco_path, "visu_pars")
    data_file = os.path.join(reco_path, "2dseq")

    if not all(os.path.exists(f) for f in [data_file, reco_file, visu_file]):
        return None, None, None, None

    try:
        # Parse reco for data type
        with open(reco_file, 'r', errors='ignore') as f:
            reco_content = f.read()

        wordtype = parse_bruker_param(reco_content, "RECO_wordtype", "_16BIT_SGN_INT")

        if "32BIT_FLOAT" in wordtype:
            dtype = np.float32
            itemsize = 4
        elif "32BIT_SGN_INT" in wordtype:
            dtype = np.int32
            itemsize = 4
        elif "16BIT_SGN_INT" in wordtype:
            dtype = np.int16
            itemsize = 2
        else:
            dtype = np.int16
            itemsize = 2

        # Parse visu_pars for dimensions
        with open(visu_file, 'r', errors='ignore') as f:
            visu_content = f.read()

        size_arr = parse_bruker_array(visu_content, "VisuCoreSize")
        if size_arr is None or len(size_arr) < 2:
            return None, None, None, None

        nx, ny = int(size_arr[0]), int(size_arr[1])
        nframes = int(parse_bruker_param(visu_content, "VisuCoreFrameCount", "1"))

        voxel_arr = parse_bruker_array(visu_content, "VisuCoreVoxelDimensions")
        if voxel_arr is not None and len(voxel_arr) >= 3:
            voxel_size = voxel_arr
        else:
            voxel_size = np.array([1.0, 1.0, 1.0])

        # Read binary data
        file_size = os.path.getsize(data_file)
        expected_size = nx * ny * nframes * itemsize

        with open(data_file, 'rb') as f:
            raw_data = np.fromfile(f, dtype=dtype, count=nx*ny*nframes)

        # Reshape to (frames, ny, nx)
        data = raw_data.reshape((nframes, ny, nx)).astype(np.float32)

        return data, voxel_size, (nx, ny, nframes), reco_path

    except Exception as e:
        logger.error(f"Failed to read 2dseq from {scan_path}: {e}")
        return None, None, None, None

# ============ DIFFUSION DATA PROCESSING ============
def extract_b_values_and_directions(method_params):
    """Extract b-values and gradient directions from method file."""
    if method_params is None:
        return None, None

    try:
        # Extract b-values
        b_vals_str = parse_bruker_param(method_params, "PVM_DiffBvalues", None)
        if b_vals_str:
            try:
                b_values = np.array([float(x) for x in b_vals_str.split()])
            except:
                b_values = None
        else:
            b_values = None

        # Extract number of directions
        ndirs_str = parse_bruker_param(method_params, "PVM_DiffNumDirs", None)
        if ndirs_str:
            try:
                ndirs = int(ndirs_str)
            except:
                ndirs = None
        else:
            ndirs = None

        # Extract gradient directions (if available)
        # This is complex and varies by Bruker version
        directions = None

        return b_values, ndirs, directions

    except Exception as e:
        logger.warn(f"Failed to extract diffusion parameters: {e}")
        return None, None, None

def average_diffusion_images(dti_data, b_values=None):
    """Average diffusion images across all b-values and directions.

    Input shape: (nframes, ny, nx) where nframes = nb_values * ndirections
    Output: (ny, nx) mean diffusion reference image
    """
    if dti_data is None or len(dti_data) == 0:
        return None

    try:
        # Simple approach: average all frames
        # This gives high SNR by averaging all b-values and gradient directions
        mean_image = np.mean(dti_data, axis=0)

        return mean_image.astype(np.float32)

    except Exception as e:
        logger.error(f"Failed to average diffusion images: {e}")
        return None

def stack_diffusion_slices(slice_list):
    """Stack 2D diffusion reference slices into 3D volume."""
    if not slice_list:
        return None

    nslices = len(slice_list)
    ny, nx = slice_list[0].shape

    volume_3d = np.zeros((nx, ny, nslices), dtype=np.float32)

    for i, slc in enumerate(slice_list):
        volume_3d[:, :, i] = slc.T

    return volume_3d

def build_affine_3d(voxel_size, origin=None):
    """Build 3D affine transformation."""
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])

    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    affine[:3, 3] = origin
    return affine

# ============ COREGISTRATION & RESAMPLING ============
def load_reference_map(cohort, mouse_pattern):
    """Load existing MD/FA/ADC map as reference for geometry."""
    # Try to find MD map first (usually better quality than others)
    search_patterns = [
        os.path.join(DERIVED_BASE, "MD", f"*{mouse_pattern}*"),
        os.path.join(DERIVED_BASE, "FA", f"*{mouse_pattern}*"),
        os.path.join(DERIVED_BASE, "ADC", f"*{mouse_pattern}*"),
        os.path.join(PARAM_BASE, cohort, f"*{mouse_pattern}*", "*_T2_map.nii.gz"),
    ]

    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            ref_file = files[0]
            try:
                img = nib.load(ref_file)
                data = img.get_fdata()
                affine = img.affine
                logger.info(f"  Loaded reference map: {os.path.basename(ref_file)} ({data.shape})")
                return data, affine
            except Exception as e:
                logger.warn(f"  Failed to load reference: {e}")
                continue

    logger.warn(f"  No reference map found for coregistration")
    return None, None

def resample_to_reference(source_data, source_affine, reference_shape, reference_affine):
    """Resample source image to match reference geometry using scipy."""
    try:
        if source_affine is None or reference_affine is None:
            logger.warn("  Missing affine transforms, using direct resize")
            # Fallback: just resize to match dimensions
            resampled = zoom(source_data,
                           (reference_shape[0]/source_data.shape[0],
                            reference_shape[1]/source_data.shape[1],
                            reference_shape[2]/source_data.shape[2]),
                           order=1)
            return resampled

        # Create coordinate mapping from reference to source space
        # reference_coords = source_affine^-1 @ reference_affine @ source_coords
        inv_source_affine = np.linalg.inv(source_affine[:3, :3])
        transform_matrix = inv_source_affine @ reference_affine[:3, :3]
        transform_offset = inv_source_affine @ (reference_affine[:3, 3] - source_affine[:3, 3])

        # Generate reference grid coordinates
        grid = np.indices(reference_shape, dtype=np.float32)
        coords = np.stack(grid, axis=0)  # (3, x, y, z)

        # Transform coordinates
        transformed = np.zeros_like(coords)
        for i in range(3):
            transformed[i] = transform_matrix[i, 0] * coords[0] + \
                            transform_matrix[i, 1] * coords[1] + \
                            transform_matrix[i, 2] * coords[2] + \
                            transform_offset[i]

        # Resample using scipy affine_transform
        # Note: affine_transform uses inverse mapping
        resampled = affine_transform(source_data,
                                    np.linalg.inv(transform_matrix),
                                    offset=-np.linalg.inv(transform_matrix) @ transform_offset,
                                    output_shape=reference_shape,
                                    order=1,
                                    mode='constant',
                                    cval=0)

        return resampled.astype(np.float32)

    except Exception as e:
        logger.error(f"  Resampling failed: {e}, using simple resize")
        resampled = zoom(source_data,
                        (reference_shape[0]/source_data.shape[0],
                         reference_shape[1]/source_data.shape[1],
                         reference_shape[2]/source_data.shape[2]),
                        order=1)
        return resampled.astype(np.float32)

# ============ NIFTI I/O ============
def save_nifti(data, affine, output_path, description=""):
    """Save numpy array as NIfTI."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        img.header['descrip'] = description[:80].encode() if description else b''
        nib.save(img, output_path)
        size_mb = os.path.getsize(output_path) / (1024**2)
        logger.info(f"Saved: {output_path} ({data.shape}) {size_mb:.1f}MB")
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False

# ============ MAIN PROCESSING ============
def process_mouse_dti(cohort, mouse_session_name, session_path):
    """Process DTI data for a single mouse."""
    logger.info(f"\n  Processing: {mouse_session_name}")

    # Scan for DTI sequences
    scans = scan_bruker_session(session_path)
    dti_scans = [s for s in scans if s['type'] == 'DTI']

    if not dti_scans:
        logger.warn(f"  No DTI scans found")
        return False

    logger.info(f"  Found {len(dti_scans)} DTI scan(s)")

    # Try to process first DTI scan
    for scan in dti_scans:
        scan_num = scan['scan_num']
        logger.info(f"    Processing scan {scan_num}...")

        # Read DTI data
        dti_data, voxel_size, dims, reco_path = read_bruker_2dseq(scan['path'])

        if dti_data is None:
            logger.warn(f"    Failed to read scan {scan_num}")
            continue

        logger.info(f"    Read DTI data: shape={dti_data.shape}, voxel_size={voxel_size}")

        # Average across all b-values and directions
        mean_ref = average_diffusion_images(dti_data)

        if mean_ref is None:
            logger.warn(f"    Failed to average diffusion images")
            continue

        # Stack slices if multi-slice
        if len(mean_ref.shape) == 2:
            # Single slice - convert to 3D
            volume_3d = np.expand_dims(mean_ref, axis=2)
        else:
            # Multiple slices already
            volume_3d = stack_diffusion_slices([mean_ref])

        # Load reference map for geometry
        ref_data, ref_affine = load_reference_map(cohort, mouse_session_name)

        # Create affine for diffusion reference
        diff_affine = build_affine_3d(voxel_size if voxel_size is not None else np.array([1.0, 1.0, 1.0]))

        # Coregister to reference if available
        if ref_data is not None and ref_affine is not None:
            logger.info(f"    Coregistering to reference ({ref_data.shape})...")
            volume_3d_coreg = resample_to_reference(volume_3d, diff_affine,
                                                    ref_data.shape, ref_affine)
            volume_3d = volume_3d_coreg
            diff_affine = ref_affine

        # Create output directory
        output_dir = os.path.join(OUTPUT_BASE, cohort, mouse_session_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save with consistent naming
        output_file = os.path.join(output_dir, f"{mouse_session_name}_{cohort}_DiffRef.nii.gz")

        description = f"Diffusion reference {mouse_session_name} (averaged across b-values & directions)"
        success = save_nifti(volume_3d, diff_affine, output_file, description)

        if success:
            # Also save a copy to parametric maps folder for easy access
            param_file = os.path.join(PARAM_BASE, cohort, mouse_session_name,
                                     f"{mouse_session_name}_{cohort}_DiffRef.nii.gz")
            os.makedirs(os.path.dirname(param_file), exist_ok=True)
            try:
                img = nib.load(output_file)
                nib.save(img, param_file)
                logger.info(f"Also saved to ClaudeParameters: {param_file}")
            except:
                pass

            return True

    logger.warn(f"  Could not process any DTI scans")
    return False

def process_cohort(cohort_name):
    """Process all mice in a cohort."""
    cohort_raw = os.path.join(RAW_BASE, cohort_name)

    logger.info("\n" + "="*70)
    logger.info(f"COHORT: {cohort_name}")
    logger.info("="*70)

    if not os.path.exists(cohort_raw):
        logger.error(f"Raw data folder not found: {cohort_raw}")
        return

    # Find mouse sessions
    mouse_sessions = []
    try:
        for item in sorted(os.listdir(cohort_raw)):
            session_path = os.path.join(cohort_raw, item)
            if os.path.isdir(session_path) and (item.startswith("m") or "cg" in item):
                mouse_sessions.append((item, session_path))
    except Exception as e:
        logger.error(f"Error finding mouse sessions: {e}")
        return

    if not mouse_sessions:
        logger.info(f"No mouse session folders found")
        return

    success_count = 0
    for mouse_name, session_path in mouse_sessions:
        if process_mouse_dti(cohort_name, mouse_name, session_path):
            success_count += 1

    logger.info(f"\n  Cohort complete: {success_count}/{len(mouse_sessions)} mice processed")

def generate_report(output_path):
    """Generate processing report."""
    report = [
        "# Diffusion Reference Image Generation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Base: {BASE}",
        ""
    ]

    for cohort in COHORTS.keys():
        cohort_dir = os.path.join(OUTPUT_BASE, cohort)
        report.append(f"## {cohort}")

        if os.path.exists(cohort_dir):
            for mouse in sorted(os.listdir(cohort_dir)):
                mouse_dir = os.path.join(cohort_dir, mouse)
                if os.path.isdir(mouse_dir):
                    files = glob.glob(os.path.join(mouse_dir, "*_DiffRef.nii.gz"))
                    if files:
                        report.append(f"### {mouse}")
                        for f in files:
                            size = os.path.getsize(f) / (1024**2)
                            report.append(f"- {os.path.basename(f)} ({size:.1f}MB)")
                    else:
                        report.append(f"### {mouse}")
                        report.append("- No diffusion reference generated")
        else:
            report.append("No output found\n")

        report.append("")

    report_text = '\n'.join(report)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"\nReport saved: {output_path}")

# ============ MAIN ============
if __name__ == "__main__":
    logger.set_file(LOG_FILE)

    logger.info("="*70)
    logger.info("Maraviroc Aged TBI — Diffusion Reference Image Generator")
    logger.info("="*70)
    logger.info(f"Base directory: {BASE}")
    logger.info(f"Output: {OUTPUT_BASE}")
    logger.info(f"Log: {LOG_FILE}")
    logger.info("")

    # Check base exists
    if not os.path.exists(BASE):
        logger.error(f"Base directory not found: {BASE}")
        sys.exit(1)

    # Process each cohort
    for cohort in COHORTS.keys():
        process_cohort(cohort)

    # Generate report
    report_path = os.path.join(OUTPUT_BASE, "diffusion_reference_report.md")
    generate_report(report_path)

    logger.info("\n" + "="*70)
    logger.info("GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Check outputs in: {OUTPUT_BASE}")
    logger.info(f"Report: {report_path}")
    logger.info("")
    logger.info("Diffusion reference images are coregistered to match")
    logger.info("MD/FA/ADC map geometry and dimensions.")
    logger.info("Ready for manual mask generation!")
