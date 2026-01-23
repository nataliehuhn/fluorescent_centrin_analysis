import os
import tifffile as tiff
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree


def read_image(path):
    img = tiff.imread(path)
    # Split channels explicitly
    ch1 = img[:, 0, :, :]  # channel 1
    ch2 = img[:, 1, :, :]  # channel 2
    return ch1, ch2


def drift_correction(ch1, ch2, path):
    """
    Drift-correct ch1 and ch2 using phase cross-correlation on ch2.

    Behavior:
    - If input file already ends with '_driftcorr', return data unchanged
    - Else if a drift-corrected file exists, load and return it
    - Else compute, save, and return drift-corrected data
    """

    base, ext = os.path.splitext(path)

    # ------------------------------------------------
    # Case 1: already drift-corrected input
    # ------------------------------------------------
    if base.endswith("_driftcorr"):
        print(f"Input already drift-corrected, skipping: {path}")
        return ch1, ch2

    out_path = base + "_driftcorr" + ext

    # ------------------------------------------------
    # Case 2: drift-corrected file exists on disk
    # ------------------------------------------------
    if os.path.exists(out_path):
        print(f"Loading existing drift-corrected file: {out_path}")

        stack = tiff.imread(out_path)

        # Expect Fiji hyperstack: (T, Z, C, Y, X)
        if stack.ndim != 5 or stack.shape[2] != 2:
            raise ValueError(f"Unexpected drift-corrected stack shape: {stack.shape}")

        stack = stack[:, 0]  # drop Z
        ch1_corr = stack[:, 0]
        ch2_corr = stack[:, 1]

        return ch1_corr, ch2_corr

    # ------------------------------------------------
    # Case 3: compute drift correction
    # ------------------------------------------------
    print(f"Computing drift correction for: {path}")

    reference = ch2[0]
    shifts = []

    for i in range(ch2.shape[0]):
        shift_est, _, _ = phase_cross_correlation(reference, ch2[i])
        shifts.append(shift_est)

    shifts = np.array(shifts)

    ch1_corr = np.zeros_like(ch1)
    ch2_corr = np.zeros_like(ch2)

    for i, s in enumerate(shifts):
        ch1_corr[i] = shift(ch1[i], shift=s, mode="nearest")
        ch2_corr[i] = shift(ch2[i], shift=s, mode="nearest")

    # ------------------------------------------------
    # Save for Fiji
    # ------------------------------------------------
    corrected_stack = np.stack([ch1_corr, ch2_corr], axis=-1)
    stack_5d = np.expand_dims(corrected_stack, axis=1)
    ij_stack = np.moveaxis(stack_5d, -1, 2)

    tiff.imwrite(
        out_path,
        ij_stack.astype(np.uint16),
        imagej=True,
        photometric="minisblack"
    )

    return ch1_corr, ch2_corr


def detect_centrin(ch1_corr, search_radius=20):
    coords = []

    # First frame: no prior location → global search
    frame0 = gaussian(ch1_corr[0], sigma=1, preserve_range=True)
    y0, x0 = np.unravel_index(np.argmax(frame0), frame0.shape)
    coords.append((x0, y0))

    # Subsequent frames: local search
    for frame in ch1_corr[1:]:
        frame_blur = gaussian(frame, sigma=1, preserve_range=True)
        prev_x, prev_y = coords[-1]

        # Define ROI bounds
        y_min = max(prev_y - search_radius, 0)
        y_max = min(prev_y + search_radius + 1, frame.shape[0])
        x_min = max(prev_x - search_radius, 0)
        x_max = min(prev_x + search_radius + 1, frame.shape[1])

        roi = frame_blur[y_min:y_max, x_min:x_max]

        # Find the max within the ROI
        dy, dx = np.unravel_index(np.argmax(roi), roi.shape)
        new_y = y_min + dy
        new_x = x_min + dx

        coords.append((new_x, new_y))

    return np.array(coords)


def cell_mask(ch1_corr):
    t_frames = ch1_corr.shape[0]
    height, width = ch1_corr.shape[1:]

    cell_mask = np.zeros_like(ch1_corr, dtype=bool)

    for i in range(t_frames):

        frame = ch1_corr[i]

        thr = threshold_otsu(frame)
        mask = frame > thr

        mask = remove_small_objects(mask, 300)
        mask = remove_small_holes(mask, 500)

        lbl = label(mask)
        if lbl.max() > 0:
            largest_region = 1 + np.argmax(np.bincount(lbl.flat)[1:])
            mask = lbl == largest_region
        else:
            mask = np.zeros_like(mask)

        cell_mask[i] = mask

    return cell_mask


def get_mask_center(mask_frame):
    """
    Compute the point inside the mask that has the maximum Euclidean distance to the mask edge.
    Returns (x, y) coordinates.
    """
    if mask_frame.sum() == 0:
        return None  # empty mask

    # Compute distance transform (distance to nearest background pixel)
    dist = distance_transform_edt(mask_frame)

    # Find the coordinates of the maximum distance
    max_idx = np.unravel_index(np.argmax(dist), dist.shape)
    y_center, x_center = max_idx
    return x_center, y_center


def get_mask_center_com(mask_frame):
    """
    Compute the center of mass of a single binary mask.
    Returns (x, y) coordinates or None if mask is empty.
    """
    if mask_frame.sum() == 0:
        return None

    # SciPy returns (y, x)
    y_center, x_center = center_of_mass(mask_frame)
    return float(x_center), float(y_center)


def compute_mask_centers_com(mask):
    """
    Compute the center of mass for each frame in a (T, H, W) mask tensor.
    Returns (T, 2) array with (x, y) or NaN for empty masks.
    """
    T = mask.shape[0]
    centers = np.full((T, 2), np.nan, dtype=float)

    for i in range(T):
        frame = mask[i]
        if frame.sum() == 0:
            continue

        y, x = center_of_mass(frame)
        centers[i] = [x, y]

    return centers


def compute_com_coords(mask):
    """
    Compute center of mass for each frame in a (T, H, W) binary mask.
    Return array (T, 2) with (x, y). NaN for empty masks.
    """
    T = mask.shape[0]
    coords = np.full((T, 2), np.nan, dtype=float)

    for i in range(T):
        frame = mask[i]
        if frame.sum() == 0:
            continue

        # center_of_mass returns (y, x)
        y, x = center_of_mass(frame)
        coords[i] = [x, y]

    return coords


def compute_mask_centers(mask):
    """
    Compute EDT maxima for each frame in mask (T,H,W).
    Returns (T,2) float array with (x,y) or NaN.
    """
    T = mask.shape[0]
    centers = np.full((T, 2), np.nan)

    for i in range(T):
        frame = mask[i]
        if frame.sum() == 0:
            continue

        dist = distance_transform_edt(frame)
        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        y, x = max_idx
        centers[i] = [x, y]

    return centers


def compute_deformations_max_area(ch2_corr, cell_mask, p0_grid, mag_cutoff, lk_params):
    """
    Compute deformations using a global averaged template, excluding only current frame mask.
    Returns list of deformations and list of per-frame p0.
    """
    t_frames, height, width = ch2_corr.shape
    all_deformations = []
    all_p0 = []

    avg_template = np.mean(ch2_corr, axis=0)  # np.mean(ch2_corr, axis=0)  # ch2_corr[0]
    avg_template_uint8 = np.clip(avg_template, 0, 255).astype(np.uint8)

    for i in range(t_frames):
        print(f"Processing frame {i}...")

        frame = np.clip(ch2_corr[i], 0, 255).astype(np.uint8)

        # Filter points outside current mask
        p0 = filter_points_outside_mask(p0_grid, cell_mask[i])
        all_p0.append(p0)

        if len(p0) == 0:
            all_deformations.append(np.full((0, 2), np.nan))
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            avg_template_uint8, frame, p0.astype(np.float32), None, **lk_params
        )

        deformations = p1 - p0

        exclude = err[:, 0] > 1.5 * np.nanmean(err)
        deformations[exclude] = np.nan

        magnitudes = np.linalg.norm(deformations, axis=1)
        deformations[magnitudes > mag_cutoff] = np.nan

        deformations -= np.nanmedian(deformations, axis=0)

        all_deformations.append(deformations)

    return all_deformations, all_p0


def filter_points_outside_mask(points, mask):
    """ Keeps only points where mask == False. points: Nx2 (x,y) mask: boolean mask shape (H, W) """
    x = points[:, 0].astype(int)
    y = points[:, 1].astype(int)
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    valid &= ~mask[y, x]  # only keep points NOT in mask
    return points[valid]


def compute_mtoc_speed(coords, fps):
    """
    Compute the instantaneous speed of the MTOC (brightest point)
    coords: Nx2 array of (x, y) positions
    fps: frames per second
    Returns: speed array of length N-1
    """
    coords = np.array(coords)
    # Ignore frames with NaN
    valid = ~np.isnan(coords[:,0]) & ~np.isnan(coords[:,1])
    coords_valid = coords[valid]

    # Compute displacement between consecutive frames
    displacements = np.diff(coords_valid, axis=0)
    distances = np.linalg.norm(displacements, axis=1)

    # Convert to speed (pixels per second)
    speed = distances * fps

    # To match original frame count, pad first frame with NaN
    speed_full = np.full(len(coords), np.nan)
    speed_full[1:][valid[1:]] = speed
    return speed_full


def smooth_timeseries_gaussian(data, sigma=1.5):
    """
    Safely smooth a 1D numeric timeseries using Gaussian convolution.
    Behavior:
    - Ignores NaNs
    - Returns the original data if too few valid points
    - Uses sigma to determine required minimal length
    """

    arr = np.asarray(data, float)
    smoothed = np.copy(arr)

    valid = ~np.isnan(arr)
    n_valid = valid.sum()

    # Rule of thumb: Gaussian kernel effectively spans ~6*sigma samples
    min_needed = int(np.ceil(6 * sigma))

    if n_valid < min_needed:
        return smoothed  # not enough data → return original

    # Smooth only valid part
    smoothed[valid] = gaussian_filter1d(arr[valid], sigma=sigma)

    return smoothed


def compute_spline_path(input_array, smoothing=300):
    """
    Fit spline through either mask (3D) or direct 2D points array (frames x 2).
    Returns tck, spline_points.
    """
    if input_array.ndim == 3:  # mask input
        centers = compute_mask_centers(input_array)
    elif input_array.ndim == 2 and input_array.shape[1] == 2:  # coords input
        centers = input_array
    else:
        raise ValueError("Input must be mask (3D) or coords (Nx2)")

    # Keep only valid points
    valid = ~np.isnan(centers[:, 0])
    pts = centers[valid]

    if len(pts) < 2:
        raise ValueError("Not enough points for spline fitting.")

    # Remove consecutive duplicates
    if len(pts) > 1:
        diff = np.diff(pts, axis=0)
        nonzero = ~((diff[:, 0] == 0) & (diff[:, 1] == 0))
        pts_unique = np.vstack([pts[0], pts[1:][nonzero]])
    else:
        pts_unique = pts

    k = min(3, len(pts_unique) - 1)
    tck, _ = splprep([pts_unique[:, 0], pts_unique[:, 1]], s=smoothing, k=k)

    u_sample = np.linspace(0, 1, 500)
    spline_points = np.array(splev(u_sample, tck)).T

    return tck, spline_points


def brightest_to_edt_vector_along_spline(coords, mask_centers, tck, frame_idx):
    """
    Compute vector from brightest point to EDT max (mask center),
    and determine direction along spline (+1 if along tangent, -1 if opposite)
    """
    if coords is None or len(coords) <= frame_idx:
        return None, None, None, None
    if mask_centers is None or np.any(np.isnan(mask_centers[frame_idx])):
        return None, None, None, None

    bp = coords[frame_idx]
    center = mask_centers[frame_idx]

    vector = center - bp
    distance = np.linalg.norm(vector)
    dx, dy = vector

    # Sample spline points
    u_sample = np.linspace(0,1,500)
    spline_points = np.array(splev(u_sample, tck)).T  # shape (500,2)

    # Closest point on spline
    distances_to_spline = np.linalg.norm(spline_points - center, axis=1)
    idx_closest = np.argmin(distances_to_spline)

    # Tangent along spline
    if idx_closest < len(spline_points)-1:
        tangent = spline_points[idx_closest+1] - spline_points[idx_closest]
    else:
        tangent = spline_points[idx_closest] - spline_points[idx_closest-1]

    tangent /= np.linalg.norm(tangent) + 1e-8
    vector_norm = vector / (np.linalg.norm(vector) + 1e-8)

    direction_sign = np.sign(np.dot(vector_norm, tangent))

    return dx, dy, distance, direction_sign


"""
def plot_mtoc_speed_and_deformation(
    base,
    mtoc_speed,
    vector_length,
    max10_deformations,
    time_per_frame=15
):
    # Generate two plots:
    # 1) MTOC speed vs deformation
    # 2) MTOC vector length vs deformation


    vector_length = np.asarray(vector_length, dtype=float)
    max10_deformations = np.asarray(max10_deformations, dtype=float)

    # Time axis (seconds)
    t = np.arange(len(vector_length)) * time_per_frame

    # =========================================================
    # Plot 1: MTOC speed vs deformation
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(t, mtoc_speed, color="blue", label="speed")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("speed (px/s)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t, max10_deformations, color="green", label="Top 10% deformation")
    ax2.set_ylabel("Deformation magnitude (px)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("speed and deformation over time")
    plt.tight_layout()
    plt.savefig(base + "_speed_deformation.png", dpi=300)

    # =========================================================
    # Plot 2: MTOC vector length vs deformation
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(t, vector_length, color="purple", label="MTOC–EDTmax distance")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Vector length (px)", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t, max10_deformations, color="green", label="Top 10% deformation")
    ax2.set_ylabel("Deformation magnitude (px)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("MTOC distance and deformation over time")
    plt.tight_layout()
    plt.savefig(base + "_distance_deformation.png", dpi=300, transparent=True)
"""


def plot_mtoc_speed_and_deformation(
        base,
        mtoc_speed,
        vector_length,
        deformation,
        time_per_frame=15
):
    """
    vector_length=None → skip second panel
    """

    mtoc_speed = np.asarray(mtoc_speed)
    deformation = np.asarray(deformation)

    T = len(mtoc_speed)
    t = np.arange(T) * time_per_frame

    if vector_length is None:
        # 1-plot mode
        plt.figure(figsize=(8, 4))
        plt.plot(t, mtoc_speed, label="COM speed", color="blue")
        plt.plot(t, deformation, label="Top 10% deformation", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("COM speed and cell deformation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(base + "_com_deformation.png", transparent=True)
        plt.close()
        return

    # 2-plot mode
    vector_length = np.asarray(vector_length)
    t2 = np.arange(len(vector_length)) * time_per_frame

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax[0].plot(t, mtoc_speed, label="COM speed", color="blue")
    ax[0].plot(t, deformation, label="Top 10% deformation", color="red")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(t2, vector_length, label="Vector magnitude", color="black")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Vector length")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(base + "_com_vector_deformation.png", transparent=True)
    plt.close()


def cross_correlate_single_file(
    mtoc_speed,
    deformation,
    time_per_frame,
    min_frames
):
    """
    Cross-correlate MTOC speed and deformation for a single recording.

    Returns (lag_times, corr) or (None, None) if too short.
    """

    v = np.asarray(mtoc_speed, dtype=float)
    d = np.asarray(deformation, dtype=float)

    valid = ~np.isnan(v) & ~np.isnan(d)
    v = v[valid]
    d = d[valid]

    if len(v) < min_frames:
        return None, None

    # normalize to unit variance
    v = (v - np.mean(v)) / np.std(v)
    d = (d - np.mean(d)) / np.std(d)

    corr = np.correlate(v, d, mode="full") / len(v)

    lags = np.arange(-len(v) + 1, len(v))
    lag_times = lags * time_per_frame

    return lag_times, corr


def plot_cross_correlation(lag_times, corr, path):
    plt.figure(figsize=(10,5))
    plt.plot(lag_times, corr, linewidth=2)
    plt.axvline(0, color="k", linestyle="--", alpha=0.6)

    plt.xlabel("Lag (seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation: movement vs. deformation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + "_speed_def_crosscorr_plot_com.png", dpi=300, transparent=True)
    # plt.show()
    plt.close()


def plot_single_cross_correlation(
    base,
    mtoc_speed,
    deformation,
    time_per_frame,
    min_frames=3
):
    """
    Always plot single-file cross-correlation for QC.
    """

    v = np.asarray(mtoc_speed, dtype=float)
    d = np.asarray(deformation, dtype=float)

    valid = ~np.isnan(v) & ~np.isnan(d)
    v = v[valid]
    d = d[valid]

    if len(v) < min_frames:
        print(f"Too short for cross-correlation plot: {base}")
        return

    v = (v - np.mean(v)) / np.std(v)
    d = (d - np.mean(d)) / np.std(d)

    corr = np.correlate(v, d, mode="full") / len(v)
    lags = np.arange(-len(v) + 1, len(v))
    lag_times = lags * time_per_frame

    plt.figure(figsize=(8, 4))
    plt.plot(lag_times, corr, linewidth=2)
    plt.axvline(0, color="k", linestyle="--", alpha=0.6)

    plt.xlabel("Lag (s)")
    plt.ylabel("Cross-correlation")
    plt.title("speed vs deformation")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(base + "_single_crosscorr_speed_com.png", dpi=300, transparent=True)
    # plt.show()
    plt.close()


def plot_single_cross_correlation_vector(
    base,
    vector,
    deformation,
    time_per_frame,
    min_frames=3
):
    """
    Always plot single-file cross-correlation for QC.
    """

    v = np.asarray(vector, dtype=float)
    d = np.asarray(deformation, dtype=float)

    valid = ~np.isnan(v) & ~np.isnan(d)
    v = v[valid]
    d = d[valid]

    if len(v) < min_frames:
        print(f"Too short for cross-correlation plot: {base}")
        return

    v = (v - np.mean(v)) / np.std(v)
    d = (d - np.mean(d)) / np.std(d)

    corr = np.correlate(v, d, mode="full") / len(v)
    lags = np.arange(-len(v) + 1, len(v))
    lag_times = lags * time_per_frame

    plt.figure(figsize=(8, 4))
    plt.plot(lag_times, corr, linewidth=2)
    plt.axvline(0, color="k", linestyle="--", alpha=0.6)

    plt.xlabel("Lag (s)")
    plt.ylabel("Cross-correlation")
    plt.title("MTOC-EDT distance vs deformation")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(base + "_single_crosscorr_vector.png", dpi=300, transparent=True)
    # plt.show()
    plt.close()


# ======================================================
# 1) PEAK DETECTION
# ======================================================
def detect_deformation_peaks(
        deformation,
        prominence=0.25,
        distance=10
):
    """
    Detect peaks in deformation timeseries.
    Returns array of peak indices (in full array coordinates).
    """
    deformation = np.asarray(deformation, float)
    valid = ~np.isnan(deformation)
    idx_valid = np.where(valid)[0]
    d_valid = deformation[valid]

    if len(d_valid) < 3:
        return np.array([])

    peaks, _ = find_peaks(d_valid, prominence=prominence, distance=distance)
    return idx_valid[peaks]


def compute_inward_deformation_metric(
    all_deformations,
    all_p0,
    mask,
    percentile=90,
    min_cosine=0.0
):
    """
    Compute a per-frame scalar deformation metric based on inward-pointing vectors.

    Parameters
    ----------
    all_deformations : list of (N_i, 2) arrays
        Optical flow deformation vectors per frame.
    all_p0 : list of (N_i, 2) arrays
        Grid points corresponding to deformation vectors per frame.
    mask : (T, H, W) bool or int array
        Cell mask per frame.
    percentile : float, optional
        Percentile of inward deformation magnitudes to report (default: 90).
    min_cosine : float, optional
        Minimum cosine(angle) to consider a vector inward-pointing.
        0.0   → strictly inward
        0.5   → strongly inward
        < 0.0 → include slightly tangential vectors

    Returns
    -------
    inward_metric : (T,) float array
        Per-frame inward deformation metric (NaN if undefined).
    """

    T = len(all_deformations)
    inward_metric = np.full(T, np.nan)

    for i in range(T):
        deformations = all_deformations[i]
        p0 = all_p0[i]

        if deformations.size == 0 or p0.size == 0:
            continue

        object_points = np.argwhere(mask[i])
        if object_points.size == 0:
            continue

        # Nearest mask pixel to each p0
        tree = cKDTree(object_points)
        _, idx = tree.query(p0[:, ::-1])
        nearest_points = object_points[idx][:, ::-1]

        # Direction toward the cell
        to_object = nearest_points - p0
        to_obj_mag = np.linalg.norm(to_object, axis=1)
        to_obj_mag[to_obj_mag == 0] = np.nan
        to_obj_norm = to_object / to_obj_mag[:, None]

        # Deformation vectors
        deform_mag = np.linalg.norm(deformations, axis=1)
        deform_mag[(deform_mag == 0) | np.isnan(deform_mag)] = np.nan
        deform_norm = deformations / deform_mag[:, None]

        # Cosine of angle between deformation and inward direction
        cos_angles = np.einsum("ij,ij->i", to_obj_norm, deform_norm)

        inward_mask = cos_angles > min_cosine
        if not np.any(inward_mask):
            continue

        inward_magnitudes = deform_mag[inward_mask]
        inward_magnitudes = inward_magnitudes[np.isfinite(inward_magnitudes)]

        if inward_magnitudes.size > 0:
            inward_metric[i] = np.nanpercentile(
                inward_magnitudes, percentile
            )

    return inward_metric


# ======================================================
# 2) WINDOW EXTRACTION
# ======================================================
def extract_window(signal, center_index, window):
    """
    Extract a centered window [center-window : center+window].
    Pads with NaNs when sequence is too short.
    """
    signal = np.asarray(signal, float)
    N = len(signal)
    start = max(center_index - window, 0)
    end = min(center_index + window + 1, N)

    seg = signal[start:end]

    expected = 2 * window + 1
    if len(seg) < expected:
        seg = np.pad(seg, (0, expected - len(seg)), constant_values=np.nan)

    return seg


# ======================================================
# 3) EVENT-TRIGGERED CROSS-CORRELATION PIPELINE
# ======================================================
def event_triggered_cross_correlation(
        mtoc_speed,
        deformation,
        time_per_frame,
        window=20,
        prominence=0.2,
        min_distance=10,
        min_points=3
):
    mtoc_speed = np.asarray(mtoc_speed, float)
    deformation = np.asarray(deformation, float)

    # Detect peaks
    peaks = detect_deformation_peaks(
        deformation,
        prominence=prominence,
        distance=min_distance
    )

    if len(peaks) == 0:
        return []

    results = []

    for p in peaks:
        sw = extract_window(mtoc_speed, p, window)
        dw = extract_window(deformation, p, window)

        valid = ~np.isnan(sw) & ~np.isnan(dw)
        sw = sw[valid]
        dw = dw[valid]

        if len(sw) < min_points:
            continue

        # Normalize
        s = (sw - np.mean(sw)) / (np.std(sw) + 1e-9)
        d = (dw - np.mean(dw)) / (np.std(dw) + 1e-9)

        # Full cross-correlation
        corr_full = np.correlate(s, d, mode="full") / len(s)
        lags_full = np.arange(-len(s)+1, len(s))

        # Truncate to ±window frames
        lag_mask = (lags_full >= -window) & (lags_full <= window)
        corr = corr_full[lag_mask]
        lag_times = lags_full[lag_mask] * time_per_frame

        results.append({
            "peak": p,
            "window_speed": sw,
            "window_def": dw,
            "lag_times": lag_times,
            "corr": corr,
            "seg_speed": sw,
            "seg_def": dw
        })

    return results


# ======================================================
# 4) ALIGN + AGGREGATE CROSS-CORRELATIONS
# ======================================================
def aggregate_event_triggered_events(results, window, time_per_frame):
    """
    Aggregate event-triggered windows (each peak is an event)
    and compute mean ± 95% CI across all events.

    Args:
        results: list of dicts from event_triggered_cross_correlation()
        window: half-width of the window in frames
        time_per_frame: seconds per frame

    Returns:
        dict with:
            - "t": time axis relative to peak
            - "matrix": all events (peaks) stacked
            - "mean": mean across events
            - "lower": 2.5 percentile
            - "upper": 97.5 percentile
        or None if no valid events
    """
    if len(results) == 0:
        return None

    # Only keep entries with a valid "signal"
    valid_results = [r for r in results if "signal" in r and r["signal"] is not None]
    if len(valid_results) == 0:
        return None

    n_events = len(valid_results)
    print("number of events: ", n_events)
    matrix = np.full((n_events, 2 * window + 1), np.nan)

    for i, r in enumerate(valid_results):
        sig = np.asarray(r["signal"], float)
        L = len(sig)
        if L < 2 * window + 1:
            sig = np.pad(sig, (0, 2 * window + 1 - L), constant_values=np.nan)
        matrix[i] = sig

    # Skip empty slices
    if np.all(np.isnan(matrix)):
        return None

    mean_trace = np.nanmean(matrix, axis=0)
    lower_ci = np.nanpercentile(matrix, 2.5, axis=0)
    upper_ci = np.nanpercentile(matrix, 97.5, axis=0)

    t = np.arange(-window, window + 1) * time_per_frame

    return {
        "t": t,
        "matrix": matrix,
        "mean": mean_trace,
        "lower": lower_ci,
        "upper": upper_ci
    }


def plot_event_triggered(results, summary, base_name):
    """
    Plot individual peak-triggered curves + mean + CI.
    """
    if (results is None) or (summary is None):
        print(f"No usable peaks: {base_name}")
        return

    plt.figure(figsize=(10, 6))

    # individual curves (optional: plot relative to peak)
    for r in results:
        if "lag_times" in r and r["lag_times"] is not None:
            plt.plot(r["lag_times"], r["corr"], alpha=0.3, linewidth=1)

    # confidence interval
    plt.fill_between(
        summary["t"],
        summary["lower"],
        summary["upper"],
        color="gray",
        alpha=0.3
    )

    # mean curve
    plt.plot(
        summary["t"],
        summary["mean"],
        color="blue",
        linewidth=3,
        label="Mean"
    )

    plt.axvline(0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Lag (s)")
    plt.ylabel("Cross-correlation")
    plt.title("Peak-triggered cross-correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_name + "_peak_triggered_correlation.png", dpi=300, transparent=True)
    plt.close()
