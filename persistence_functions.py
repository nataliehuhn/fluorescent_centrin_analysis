import numpy as np
import matplotlib.pyplot as plt


def compute_directional_persistence(coords):
    """
    Directional persistence defined as cosine similarity between
    consecutive displacement vectors.

    coords: Nx2 array (x, y), may contain NaNs
    returns: array of length N with NaNs where undefined
    """

    coords = np.asarray(coords, dtype=float)

    valid = ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
    coords = coords[valid]

    if len(coords) < 3:
        return np.full(len(valid), np.nan)

    disp = np.diff(coords, axis=0)
    norms = np.linalg.norm(disp, axis=1)

    # normalize displacement vectors
    disp_norm = disp / norms[:, None]

    persistence = np.full(len(coords), np.nan)

    for i in range(1, len(disp_norm)):
        persistence[i + 1] = np.dot(disp_norm[i], disp_norm[i - 1])

    out = np.full(len(valid), np.nan)
    out[valid] = persistence
    return out


def plot_mtoc_and_edt_persistence(
    base,
    mtoc_persistence,
    edt_persistence,
    time_per_frame
):
    """
    Plot MTOC and EDT directional persistence into one plot.
    """

    t = np.arange(len(mtoc_persistence)) * time_per_frame

    plt.figure(figsize=(9, 4))

    plt.plot(
        t,
        mtoc_persistence,
        label="MTOC directional persistence",
        linewidth=2
    )

    plt.plot(
        t,
        edt_persistence,
        label="EDT center directional persistence",
        linewidth=2
    )

    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.ylim(-1.05, 1.05)

    plt.xlabel("Time (s)")
    plt.ylabel("Directional persistence (cos θ)")
    plt.title("Directional persistence: MTOC vs EDT center")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(base + "_persistence_mtoc_vs_edt.png", dpi=300, transparent=True)
    plt.close()


def compute_path_lengths(coords):
    """
    coords: (T,2) array with possible NaNs.
    Returns path length per frame: distance between frame t and t+1.
    Output length = T, with last element NaN (no step after last).
    """
    coords = np.asarray(coords, float)
    out = np.full(len(coords), np.nan)

    valid = ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
    idx = np.where(valid)[0]

    if len(idx) < 2:
        return out

    for i in range(len(idx) - 1):
        p1 = coords[idx[i]]
        p2 = coords[idx[i + 1]]
        out[idx[i + 1]] = np.linalg.norm(p2 - p1)

    return out


def plot_mtoc_edt_pathlength_summary(
        folder_base,
        mtoc_path_lengths_list,
        edt_path_lengths_list,
        time_per_frame=15,
        error_type="sem"
):
    """
    Plot mean speed (pixels per second) across sequences for MTOC and EDT trajectories,
    normalized by number of frames and time per frame.

    mtoc_path_lengths_list: list of arrays (per-frame path lengths for each file)
    edt_path_lengths_list:  list of arrays (same structure)
    time_per_frame: seconds per frame
    error_type: "std" or "sem"
    """
    # --------------------------
    # Compute mean and error across sequences
    # --------------------------
    mtoc_mean = np.nanmean(mtoc_path_lengths_list)
    edt_mean = np.nanmean(edt_path_lengths_list)

    if error_type == "std":
        mtoc_err = np.nanstd(mtoc_path_lengths_list)
        edt_err = np.nanstd(edt_path_lengths_list)
    elif error_type == "sem":
        mtoc_err = np.nanstd(mtoc_path_lengths_list) / np.sqrt(len(mtoc_path_lengths_list))
        edt_err = np.nanstd(edt_path_lengths_list) / np.sqrt(len(edt_path_lengths_list))
    else:
        raise ValueError("error_type must be 'std' or 'sem'")

    # --------------------------
    # Plot as a bar plot
    # --------------------------
    plt.figure(figsize=(2, 4))
    labels = ["MTOC", "EDT-max"]
    means = [mtoc_mean, edt_mean]
    errors = [mtoc_err, edt_err]

    plt.bar(labels, means, yerr=errors, color=["steelblue", "firebrick"], alpha=0.8, capsize=5)
    plt.ylabel("Mean speed (pixels/s)")
    plt.title("Mean speeds across all sequences")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(folder_base + "/mtoc_edt_speed_summary_bar.png", dpi=300, transparent=True)
    plt.show()
    plt.close()

    return mtoc_mean, edt_mean, mtoc_err, edt_err


