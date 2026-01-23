import helper_functions as hf
import video_functions as vf
import cv2
import os
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL SETTINGS
spacing = 15
mag_cutoff = 5
time_per_frame = 15
folder = r"Y:\nhuhn\Microscopy\Confocal\centrin_deformation_batch"  # \deformation_events_cropped

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


# CROSS-CORRELATION ACROSS FILES
all_corrs = []
all_lags = []
min_frames = 30

for com_speed, top10_def in zip(speeds_list, deformations_list):
    lag_times, corr = hf.cross_correlate_single_file(
        com_speed,
        top10_def,
        time_per_frame,
        min_frames=min_frames
    )

    if corr is None:
        continue

    all_corrs.append(corr)
    all_lags.append(lag_times)

max_neg_lag = max(lags[0] for lags in all_lags)
max_pos_lag = min(lags[-1] for lags in all_lags)

common_lag_times = np.arange(
    max_neg_lag,
    max_pos_lag + time_per_frame,
    time_per_frame
)

corr_matrix = []

for lags, corr in zip(all_lags, all_corrs):
    interp_corr = np.interp(
        common_lag_times,
        lags,
        corr,
        left=np.nan,
        right=np.nan
    )
    corr_matrix.append(interp_corr)

corr_matrix = np.array(corr_matrix)
mean_corr = np.nanmean(corr_matrix, axis=0)
lower_ci = np.nanpercentile(corr_matrix, 2.5, axis=0)
upper_ci = np.nanpercentile(corr_matrix, 97.5, axis=0)

plt.figure(figsize=(8, 5))
plt.fill_between(
    common_lag_times,
    lower_ci,
    upper_ci,
    color="gray",
    alpha=0.3,
    label="95% CI"
)
plt.plot(common_lag_times, mean_corr, color="blue", linewidth=2, label="Mean")
plt.axvline(0, color="k", linestyle="--", alpha=0.6)

plt.xlabel("Lag (s)")
plt.ylabel("Cross-correlation")
plt.title("Cross-correlation: COM speed vs deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(folder + r"\ALL_FILES_com_boot_crosscorr_ci_min"
            + str(min_frames) + r".png", transparent=True)
plt.close()
