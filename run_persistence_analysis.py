import os
import numpy as np
import helper_functions as hf
import video_functions as vf
import persistence_functions as pf  # contains the functions you provided
import cv2
import matplotlib.pyplot as plt

# =========================
# Global analysis parameters
# =========================

spacing = 15
mag_cutoff = 5
time_per_frame = 15  # seconds per frame

lk_params = dict(
    winSize=(64, 64),
    maxLevel=1,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

folder = r"Y:\nhuhn\Microscopy\Confocal\centrin_deformation_batch"


def process_single_file(path):
    """Process one file: compute speeds, deformations, persistence, path lengths."""

    base, _ = os.path.splitext(path)
    print(f"\nProcessing file: {path}")

    # --- Load + drift correction ---
    ch1, ch2 = hf.read_image(path)
    ch1_corr, ch2_corr = hf.drift_correction(ch1, ch2, path)

    # --- MTOC detection ---
    coords = hf.detect_centrin(ch1_corr)

    # --- MTOC speed ---
    fps = 1.0 / time_per_frame
    mtoc_speed = hf.compute_mtoc_speed(coords, fps)
    mtoc_speed = hf.smooth_timeseries_gaussian(mtoc_speed, sigma=2.0)

    # --- Cell mask ---
    mask = hf.cell_mask(ch1_corr)

    # --- EDT max centers ---
    edt_centers = np.full((mask.shape[0], 2), np.nan)
    for i in range(mask.shape[0]):
        c = hf.get_mask_center(mask[i])
        if c is not None:
            edt_centers[i] = c

    # --- Directional persistence ---
    mtoc_persistence = pf.compute_directional_persistence(coords)
    edt_persistence = pf.compute_directional_persistence(edt_centers)

    # --- Distance MTOC → EDT ---
    vector_lengths = np.linalg.norm(edt_centers - coords, axis=1)
    vector_lengths = hf.smooth_timeseries_gaussian(vector_lengths, sigma=2.0)

    # --- Deformations ---
    H, W = ch2_corr.shape[1:]
    p0_grid = vf.generate_grid_points((H, W), step=spacing)
    all_deformations, _ = hf.compute_deformations_max_area(
        ch2_corr, mask, p0_grid, mag_cutoff, lk_params
    )

    # --- Top 10% deformation ---
    top10_deformations = np.full(len(all_deformations), np.nan)
    for i, d in enumerate(all_deformations):
        if d.size == 0:
            continue
        mags = np.linalg.norm(d, axis=1)
        mags = mags[~np.isnan(mags)]
        if len(mags):
            top10_deformations[i] = np.nanpercentile(mags, 90)

    # --- Path lengths ---
    mtoc_path_lengths = pf.compute_path_lengths(coords)
    print(len(mtoc_path_lengths))
    edt_path_lengths = pf.compute_path_lengths(edt_centers)

    # --- Persistence plot (optional) ---
    # pf.plot_mtoc_and_edt_persistence(base, mtoc_persistence, edt_persistence, time_per_frame)

    # --- Return in same order as old analysis ---
    return mtoc_speed, vector_lengths, top10_deformations, mtoc_persistence, edt_persistence, mtoc_path_lengths, edt_path_lengths


def process_folder(folder_path):
    all_speeds = []
    all_vectors = []
    all_deformations = []
    mtoc_persistences = []
    edt_persistences = []

    mtoc_speed_per_sequence = []
    edt_speed_per_sequence = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue

        path = os.path.join(folder_path, fname)

        try:
            # Process single file
            all_sp, vec_l, top10def, mtoc_pers, edt_pers, mtoc_pl, edt_pl = process_single_file(path)

            # Append persistence
            mtoc_persistences.append(mtoc_pers)
            edt_persistences.append(edt_pers)

            # Append speeds, vectors, deformations as separate sequences
            all_speeds.append(all_sp)
            all_vectors.append(vec_l)
            all_deformations.append(top10def)

            # Compute mean speed per sequence (pixels/sec)
            if np.any(~np.isnan(mtoc_pl)):
                mtoc_speed_seq = np.nansum(mtoc_pl) / (np.count_nonzero(~np.isnan(mtoc_pl)) * time_per_frame)
                mtoc_speed_per_sequence.append(mtoc_speed_seq)
                # print(len(mtoc_speed_per_sequence))
                # print(mtoc_speed_per_sequence)

            if np.any(~np.isnan(edt_pl)):
                edt_speed_seq = np.nansum(edt_pl) / (np.count_nonzero(~np.isnan(edt_pl)) * time_per_frame)
                edt_speed_per_sequence.append(edt_speed_seq)
                # print(len(edt_speed_per_sequence))
                # print(edt_speed_per_sequence)

        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    # Plot mean speed per sequence as bar plot
    if mtoc_speed_per_sequence and edt_speed_per_sequence:
        print("positive!")
        pf.plot_mtoc_edt_pathlength_summary(
            folder_base=folder_path,
            mtoc_path_lengths_list=mtoc_speed_per_sequence,
            edt_path_lengths_list=edt_speed_per_sequence,
            time_per_frame=time_per_frame,
            error_type="sem"
        )
    else:
        print("No valid path length data to plot.")

    return {
        "all_speeds": all_speeds,                 # list of arrays (per sequence)
        "all_vectors": all_vectors,               # list of arrays
        "all_deformations": all_deformations,     # list of arrays
        "mtoc_persistences": mtoc_persistences,   # list of arrays
        "edt_persistences": edt_persistences,     # list of arrays
        "mtoc_speed_per_sequence": mtoc_speed_per_sequence,  # list of scalars
        "edt_speed_per_sequence": edt_speed_per_sequence     # list of scalars
    }


# Process all files in the folder
all_sp, all_vec, all_defs, mtoc_pers, edt_pers, mtoc_sp_seq, edt_sp_seq = process_folder(folder)

print("Analysis complete!")
print(f"Processed {len(mtoc_pers)} files.")

