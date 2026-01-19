import helper_functions as hf
import video_functions as vf
import cv2
import os
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splprep, BSpline, splev, splprep

# =========================
# Global analysis parameters
# =========================

safety = 1
spacing = 15
arrow_scale = 50

lk_params = dict(
    winSize=(64, 64),
    maxLevel=1,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

folder = r"Y:\nhuhn\Microscopy\Confocal\centrin_deformation_batch\deformation_events_cropped"

mag_cutoff = 5
time_per_frame = 15  # seconds per frame

cmap = mcolors.LinearSegmentedColormap.from_list(
    "BlueGrayRed",
    [(0.0, "blue"), (0.25, "0.2"), (0.75, "0.2"), (1.0, "red")]
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

    # --- Load + drift correction ---
    ch1, ch2 = hf.read_image(path)
    ch1_corr, ch2_corr = hf.drift_correction(ch1, ch2, path)

    # --- MTOC detection ---
    coords = hf.detect_centrin(ch1_corr, search_radius=12)

    # --- MTOC speed ---
    fps = 1.0 / time_per_frame
    mtoc_speed = hf.compute_mtoc_speed(coords, fps)
    mtoc_speed = hf.smooth_timeseries_gaussian(mtoc_speed, sigma=2.0)

    # --- Cell mask ---
    mask = hf.cell_mask(ch1_corr)

    # --- EDT max centers ---
    edt_centers = hf.compute_mask_centers(mask)

    # --- MTOC → EDTmax distance ---
    vector_lengths = np.linalg.norm(edt_centers - coords, axis=1)
    vector_lengths = hf.smooth_timeseries_gaussian(vector_lengths, sigma=2.0)

    # --- Deformations ---
    H, W = ch2_corr.shape[1:]
    p0_grid = vf.generate_grid_points((H, W), step=spacing)
    all_deformations, all_p0 = hf.compute_deformations_max_area(
        ch2_corr, mask, p0_grid, mag_cutoff, lk_params
    )

    # --- Top 10% deformation ---
    top10_deformations = np.full(len(all_deformations), np.nan)
    for i, d in enumerate(all_deformations):
        if d.size == 0:
            continue
        mags = np.linalg.norm(d, axis=1)
        mags = mags[~np.isnan(mags)]
        if len(mags) > 0:
            top10_deformations[i] = np.nanpercentile(mags, 90)

    # ====================================
    # PLOTTING
    # ====================================
    hf.plot_mtoc_speed_and_deformation(
        base,
        mtoc_speed,
        vector_lengths,
        top10_deformations,
        time_per_frame=time_per_frame
    )

    hf.plot_single_cross_correlation(
        base,
        mtoc_speed,
        top10_deformations,
        time_per_frame=time_per_frame
    )

    hf.plot_single_cross_correlation_vector(
        base,
        vector_lengths,
        top10_deformations,
        time_per_frame=time_per_frame
    )

    # ====================================
    # VIDEO GENERATION
    # ====================================
    # 1) Deformation + mask center + brightest point
    video_path_1 = base + "_deformation_video.mp4"
    vf.save_deformation_video_with_mask_center(
        ch2_seq=ch2_corr,
        ch1_seq=ch1_corr,
        all_deformations=all_deformations,
        mask=mask,
        coords=coords,
        cmap=cmap,
        arrow_scale=arrow_scale,
        save_path=video_path_1,
        fps=5,
        all_p0=all_p0
    )

    # 2) Vector video with spline + EDT direction
    try:
        # EDT spline
        tck_center, spline_pts = hf.compute_spline_path(mask)
    except Exception as e:
        tck_center = None
        spline_pts = None
        print(f"Warning: Could not compute EDT spline for {path}: {e}")

    try:
        # MTOC spline
        tck_mtoc, mtoc_pts = hf.compute_spline_path(coords)
    except Exception as e:
        tck_mtoc = None
        mtoc_pts = None
        print(f"Warning: Could not compute MTOC spline for {path}: {e}")

        # Spline-based deformation video
    if spline_pts is not None:
        video_path_2 = base + "_deformation_video_vector_and_spline.mp4"
        vf.save_deformation_video_with_spline_edt(
            ch2_seq=ch2_corr,
            ch1_seq=ch1_corr,
            all_p0=all_p0,
            all_deformations=all_deformations,
            mask=mask,
            coords=coords,
            mask_centers=hf.compute_mask_centers(mask),
            tck=tck_center,
            cmap=cmap,
            arrow_scale=arrow_scale,
            fps=5,
            save_path=video_path_2,
            show_spline=True
        )

    # Last-frame overlay
    overlay_path = base + "_last_frame_spline_overlay.png"
    vf.save_last_frame_spline_overlay(
        ch2_seq=ch2_corr,
        ch1_seq=ch1_corr,
        all_p0=all_p0,
        all_deformations=all_deformations,
        mask=mask,
        coords=coords,
        tck_edt=tck_center,
        spline_pts_edt=spline_pts,
        tck_mtoc=tck_mtoc,
        spline_pts_mtoc=mtoc_pts,
        cmap=cmap,
        arrow_scale=arrow_scale,
        save_path=overlay_path
    )

    '''
    try:
        results = hf.event_triggered_cross_correlation(
            mtoc_speed,
            top10_deformations,
            time_per_frame,
            window=20,
            prominence=0.27,
            min_distance=8
        )

        # Plot each peak individually for this file
        for r in results:
            plt.figure(figsize=(6, 4))
            plt.plot(r['lag_times'], r['corr'], color='blue')
            plt.axvline(0, color='k', linestyle='--')
            plt.xlabel("Lag (s)")
            plt.ylabel("Cross-correlation")
            plt.title(f"Peak at frame {r['peak']}")
            plt.tight_layout()
            plt.savefig(base + f"_peak_{r['peak']}_crosscorr.png", dpi=300)
            plt.close()

    except Exception as e:
        print(f"Warning: Event-triggered CC failed for {path}: {e}")
    '''

    return mtoc_speed, vector_lengths, top10_deformations


def process_folder(folder_path):
    all_speeds = []
    all_vectors = []
    all_deformations = []

    speeds_list = []
    vectors_list = []
    deformations_list = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue

        path = os.path.join(folder_path, fname)

        try:
            speed, vectors, deform = process_single_file(
                path,
                spacing=spacing,
                mag_cutoff=mag_cutoff,
                lk_params=lk_params,
                time_per_frame=time_per_frame
            )

            all_speeds.append(speed)
            all_vectors.append(vectors)
            all_deformations.append(deform)

            speeds_list.append(speed)
            vectors_list.append(vectors)
            deformations_list.append(deform)

        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    return (
        np.concatenate(all_speeds),
        np.concatenate(all_vectors),
        np.concatenate(all_deformations),
        speeds_list,
        vectors_list,
        deformations_list
    )


# append all speeds, all vectors and all deformations to one array
all_mtoc_speed, all_vector_lengths, all_top10_def, speeds_list, vectors_list, deformations_list = process_folder(folder)


'''
# cross-correlation for detected peaks
# ====================================
# EVENT-TRIGGERED CROSS-CORRELATION (ALL FILES)
# ====================================
all_results = []
for speed, deform in zip(speeds_list, deformations_list):
    res = hf.event_triggered_cross_correlation(
        speed,
        deform,
        time_per_frame,
        window=20,  # same ±window
        prominence=0.27,
        min_distance=8
    )
    for r in res:
        r['signal'] = (r['seg_speed'] - np.nanmean(r['seg_speed'])) / (np.nanstd(r['seg_speed']) + 1e-9)
    all_results.extend(res)

summary = hf.aggregate_event_triggered_events(all_results, window=20, time_per_frame=time_per_frame)
hf.plot_event_triggered(all_results, summary, folder + r"\ALL_EVENTS_summary")
'''

# cross correlation of appended data - mtoc speed and deformation, mtoc-middle-distance and deformation
all_corrs = []
all_lags = []
min_frames = 20

for mtoc_speed, top10_def in zip(speeds_list, deformations_list):
    lag_times, corr = hf.cross_correlate_single_file(
        mtoc_speed,
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
plt.title("Cross-correlation: MTOC speed vs deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(folder + r"\ALL_FILES_boot_crosscorr_ci_min" + str(min_frames) + r".png", transparent=True)
# plt.show()


# cross correlation of appended data - mtoc speed and deformation, mtoc-middle-distance and deformation
all_corrs = []
all_lags = []
min_frames = 20

for vector, top10_def in zip(vectors_list, deformations_list):
    lag_times, corr = hf.cross_correlate_single_file(
        vector,
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
plt.title("Cross-correlation: Vector MTOC-Center vs deformation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(folder + r"\ALL_FILES_boot_crosscorr_vector_ci_min" + str(min_frames) + r".png", transparent=True)
# plt.show()
