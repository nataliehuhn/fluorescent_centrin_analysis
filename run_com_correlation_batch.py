import helper_functions as hf
import video_functions as vf
import cv2
import os
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import t as t_dist

# GLOBAL SETTINGS
spacing = 15
mag_cutoff = 5
time_per_frame = 15
folder = r"Y:\nhuhn\Microscopy\Confocal\centrin_deformation_batch\deformation_events_cropped"  # \deformation_events_cropped

lk_params = dict(
    winSize=(64, 64),
    maxLevel=1,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def process_single_file(
        path,
        spacing,
        mag_cutoff,
        lk_params,
        time_per_frame
):
    base, _ = os.path.splitext(path)
    print(f"\nProcessing file: {path}")

    # load + drift correction
    ch1, ch2 = hf.read_image(path)
    ch1_corr, ch2_corr = hf.drift_correction(ch1, ch2, path)

    # cell mask
    mask = hf.cell_mask(ch1_corr)

    # deformation field extraction
    H, W = ch2_corr.shape[1:]
    p0_grid = vf.generate_grid_points((H, W), step=spacing)

    all_deformations, all_p0 = hf.compute_deformations_max_area(
        ch2_corr, mask, p0_grid, mag_cutoff, lk_params
    )

    # COM coordinates
    com_coords = hf.compute_com_coords(mask)  # (T,2)

    # COM speed
    fps = 1.0 / time_per_frame
    com_speed = hf.compute_mtoc_speed(com_coords, fps)
    com_speed = hf.smooth_timeseries_gaussian(com_speed, sigma=2.0)

    top10_deformations = hf.compute_inward_deformation_metric(
        all_deformations=all_deformations,
        all_p0=all_p0,
        mask=mask,
        percentile=95,
        min_cosine=0.0
    )

    """
    # deformation metric
    top10_deformations = np.full(len(all_deformations), np.nan)
    for i, d in enumerate(all_deformations):
        if d.size == 0:
            continue
        mags = np.linalg.norm(d, axis=1)
        mags = mags[~np.isnan(mags)]
        if len(mags) > 0:
            top10_deformations[i] = np.nanpercentile(mags, 90)

    # plotting with optional vector_length=None
    hf.plot_mtoc_speed_and_deformation(
        base,
        com_speed,
        None,
        top10_deformations,
        time_per_frame=time_per_frame
    )

    hf.plot_single_cross_correlation(
        base,
        com_speed,
        top10_deformations,
        time_per_frame=time_per_frame
    )
    """

    return com_speed, top10_deformations


def process_folder(folder_path):
    all_speeds = []
    all_deformations = []
    speed_list = []
    deformation_list = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue

        path = os.path.join(folder_path, fname)

        try:
            speed, deform = process_single_file(
                path,
                spacing=spacing,
                mag_cutoff=mag_cutoff,
                lk_params=lk_params,
                time_per_frame=time_per_frame
            )

            all_speeds.append(speed)
            all_deformations.append(deform)

            speed_list.append(speed)
            deformation_list.append(deform)

        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    return (
        np.concatenate(all_speeds),
        np.concatenate(all_deformations),
        speed_list,
        deformation_list
    )


# run batch
all_com_speed, all_top10_def, speeds_list, deformations_list = process_folder(folder)


def crop_pair_at_lag(x, y, lag):
    """
    Crop two arrays x and y at a given lag (in frames).
    Returns (x_crop, y_crop) or (None, None) if invalid.
    """
    n = min(len(x), len(y))

    if lag > 0:
        if lag >= n:
            return None, None
        return x[:n - lag], y[lag:n]

    elif lag < 0:
        lag = -lag
        if lag >= n:
            return None, None
        return x[lag:n], y[:n - lag]

    else:
        return x[:n], y[:n]


def cross_correlation_concatenated(
    speeds_list,
    deformations_list,
    time_per_frame,
    min_frames,
    normalize=True
):
    """
    For each lag:
      - crop each dataset
      - concatenate across files
      - compute cross-correlation value
    """

    max_len = max(len(s) for s in speeds_list)
    max_lag = max_len-1
    lags = np.arange(-max_lag, max_lag + 1)

    corr_values = []
    n_used = []  # optional: track effective sample size

    for lag in lags:
        xs = []
        ys = []

        for speed, deform in zip(speeds_list, deformations_list):
            if len(speed) < min_frames or len(deform) < min_frames:
                continue

            x_crop, y_crop = crop_pair_at_lag(speed, deform, lag)
            if x_crop is None:
                continue

            mask = np.isfinite(x_crop) & np.isfinite(y_crop)
            if np.sum(mask) < min_frames:
                continue

            xs.append(x_crop[mask])
            ys.append(y_crop[mask])

        if len(xs) == 0:
            corr_values.append(np.nan)
            n_used.append(0)
            continue

        X = np.concatenate(xs)
        Y = np.concatenate(ys)

        if normalize:
            X = (X - np.mean(X)) / np.std(X)
            Y = (Y - np.mean(Y)) / np.std(Y)

        c = correlate(X, Y, mode="valid")
        corr_values.append(c[0] / len(X))
        n_used.append(len(X))

    lag_times = lags * time_per_frame
    return lag_times, np.array(corr_values), np.array(n_used)


def cross_correlation_with_significance(
    speeds_list,
    deformations_list,
    time_per_frame,
    min_frames=20,
    normalize=True,
    alpha=0.05
):
    """
    Compute lag-wise cross-correlation with statistical significance.
    Returns:
      lag_times, corr_values, n_used, sig_mask (boolean array)
    """
    lag_times, corr_values, n_used = cross_correlation_concatenated(
        speeds_list,
        deformations_list,
        time_per_frame=time_per_frame,
        min_frames=min_frames,
        normalize=normalize
    )

    sig_mask = np.zeros_like(corr_values, dtype=bool)

    for i, (r, n) in enumerate(zip(corr_values, n_used)):
        if n < 4 or np.isnan(r):
            sig_mask[i] = False
            continue
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
        sig_mask[i] = p_val < alpha

    return lag_times, corr_values, n_used, sig_mask


min_frames = 20
lag_times, corr, n_eff, sig_mask = cross_correlation_with_significance(
    speeds_list,
    deformations_list,
    time_per_frame=time_per_frame,
    min_frames=min_frames,
    alpha=0.05
)


def correlation_confidence_band(r_values, n_values, alpha=0.05):
    """
    Fisher z-transform confidence interval for Pearson r.
    CI is only computed where n > 3.
    """
    lower = np.full_like(r_values, np.nan, dtype=float)
    upper = np.full_like(r_values, np.nan, dtype=float)

    valid = (n_values > 3) & np.isfinite(r_values)

    if not np.any(valid):
        return lower, upper

    z = 0.5 * np.log((1 + r_values[valid]) / (1 - r_values[valid]))
    se = 1.0 / np.sqrt(n_values[valid] - 3)

    z_crit = 1.96  # 95% CI
    z_low = z - z_crit * se
    z_high = z + z_crit * se

    lower[valid] = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
    upper[valid] = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)

    return lower, upper


min_frames = 20
lag_times, corr, n_eff, sig_mask = cross_correlation_with_significance(
    speeds_list,
    deformations_list,
    time_per_frame=time_per_frame,
    min_frames=min_frames,
    alpha=0.05
)

# Confidence interval
lower_ci, upper_ci = correlation_confidence_band(corr, n_eff)


plt.figure(figsize=(8, 5))

# Confidence band
plt.fill_between(
    lag_times,
    lower_ci,
    upper_ci,
    color="gray",
    alpha=0.3,
    label="95% CI"
)

# Correlation curve
plt.plot(
    lag_times,
    corr,
    linewidth=2,
    color="blue",
    label="Cross-correlation"
)

# Zero lag
plt.axvline(0, color='k', linestyle='--', alpha=0.6)

# Significant points
plt.scatter(
    lag_times[sig_mask],
    corr[sig_mask],
    color='red',
    s=30,
    zorder=10,
    label='p < 0.05'
)

plt.ylim(-0.6, 0.6)
plt.xlabel("Lag (s)")
plt.ylabel("Cross-correlation")
plt.title("Center of mass speed vs deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.savefig(folder + r"\ALL_FILES_concat_crosscorr_com_ci_towards_0.png",transparent=True)

'''
plt.figure(figsize=(8, 5))
plt.plot(lag_times, corr, linewidth=2, color='blue', label='Cross-correlation')
plt.axvline(0, color='k', linestyle='--', alpha=0.6)

# Highlight significant lags
plt.scatter(lag_times[sig_mask], corr[sig_mask], color='red', label='p < 0.05', zorder=10)

# plt.xlim(-300, 300)
plt.ylim(-0.6, 0.6)
plt.xlabel("Lag (s)")
plt.ylabel("Cross-correlation")
plt.title("Center of mass speed vs deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    folder + r"\ALL_FILES_concat_crosscorr_com_significant.png",
    transparent=True
)
'''
