import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib import animation
from scipy.interpolate import splev
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import helper_functions as hf
import cv2


def generate_grid_points(image_shape, step=20):
    H, W = image_shape
    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return points


def save_deformation_video_with_mask_center(
    ch2_seq, ch1_seq, all_deformations, mask, coords,
    cmap, arrow_scale, save_path, fps, all_p0
):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        ax.clear()
        p0 = all_p0[i]
        deformations = all_deformations[i]

        display_frame = ch2_seq[i].copy()
        display_frame[mask[i]] = ch1_seq[i][mask[i]]
        ax.imshow(display_frame, cmap="gray", origin="lower")

        if coords is not None and len(coords) > i:
            ax.scatter(
                coords[i, 0], coords[i, 1],
                color="None", s=75, marker="o",
                edgecolors="lightblue"
            )

        center = hf.get_mask_center(mask[i])
        if center is not None:
            ax.scatter(center[0], center[1], color="red", s=60, marker="x")

        if len(p0) > 0:
            object_points = np.argwhere(mask[i])
            if len(object_points) == 0:
                colors = np.tile(np.array([[0.5, 0.5, 0.5, 1.0]]), (len(p0), 1))
            else:
                tree = cKDTree(object_points)
                _, idx = tree.query(p0[:, ::-1])
                nearest_points = object_points[idx][:, ::-1]

                to_object = nearest_points - p0
                to_obj_mag = np.linalg.norm(to_object, axis=1, keepdims=True)
                to_obj_mag[to_obj_mag == 0] = 1
                to_obj_norm = to_object / to_obj_mag

                deform_mag = np.linalg.norm(deformations, axis=1, keepdims=True)
                deform_mag[(deform_mag == 0) | np.isnan(deform_mag)] = 1
                deform_norm = deformations / deform_mag

                cos_angles = np.einsum("ij,ij->i", to_obj_norm, deform_norm)
                cos_angles[np.isnan(cos_angles)] = 0
                cos_norm = (cos_angles + 1) / 2
                colors = cmap(cos_norm)

            ax.quiver(
                p0[:, 0], p0[:, 1],
                deformations[:, 0], deformations[:, 1],
                color=colors, scale=arrow_scale, width=0.004
            )

        ax.axis("off")

    anim = animation.FuncAnimation(
        fig, update, frames=len(ch2_seq), interval=100
    )
    anim.save(save_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"Video saved to {save_path}")


def save_deformation_video_with_spline_edt(
        ch2_seq, ch1_seq, all_p0, all_deformations, mask, coords, tck, mask_centers, cmap,
        arrow_scale=50, fps=5, save_path=None, show_spline=True
):
    """
    Save video overlaying deformation vectors and EDT-center spline.
    Uses precomputed spline from splprep/splev arrays.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Precompute spline points for plotting
    if show_spline:
        u_plot = np.linspace(0, 1, 300)
        spline_plot_points = np.array(splev(u_plot, tck)).T  # shape (300,2)

    def update(i):
        ax.clear()
        p0 = all_p0[i]
        deformations = all_deformations[i]

        display_frame = ch2_seq[i].copy()
        display_frame[mask[i]] = ch1_seq[i][mask[i]]
        ax.imshow(display_frame, cmap='gray', origin='lower')

        # Plot spline path
        if show_spline:
            ax.plot(spline_plot_points[:, 0], spline_plot_points[:, 1],
                    color='white', linewidth=2, alpha=0.8)

        # Brightest point
        if coords is not None and len(coords) > i:
            ax.scatter(coords[i, 0], coords[i, 1],
                       color="None", s=75, marker='o', edgecolors='lightblue', linewidths=2)

        # Vector from brightest point to EDT max
        dx, dy, dist, direction_sign = hf.brightest_to_edt_vector_along_spline(coords, mask_centers, tck, i)
        if dx is not None:
            ax.arrow(coords[i, 0], coords[i, 1], dx, dy,
                     color='orange', width=1, head_width=5, length_includes_head=True)

        # Deformation vectors
        if len(p0) > 0:
            object_points = np.argwhere(mask[i])
            if len(object_points) == 0:
                colors = np.tile(np.array([[0.5, 0.5, 0.5, 1.0]]), (len(p0), 1))
            else:
                tree = cKDTree(object_points)
                _, nearest_idx = tree.query(p0[:, ::-1])
                nearest_points = object_points[nearest_idx][:, ::-1]
                to_object = nearest_points - p0

                to_object_mag = np.linalg.norm(to_object, axis=1, keepdims=True)
                to_object_mag[to_object_mag == 0] = 1
                to_object_norm = to_object / to_object_mag

                deformation_mag = np.linalg.norm(deformations, axis=1, keepdims=True)
                deformation_mag[deformation_mag == 0] = 1

                deformation_norm = np.zeros_like(deformations)
                valid = ~np.isnan(deformations).any(axis=1)
                deformation_norm[valid] = deformations[valid] / deformation_mag[valid]

                cos_angles = np.einsum('ij,ij->i', to_object_norm, deformation_norm)
                cos_angles[np.isnan(cos_angles)] = 0
                cos_norm = (cos_angles + 1) / 2
                colors = cmap(cos_norm)

            ax.quiver(
                p0[:, 0], p0[:, 1], deformations[:, 0], deformations[:, 1],
                color=colors, scale=arrow_scale, width=0.004
            )

        ax.axis('off')

    anim = animation.FuncAnimation(fig, update, frames=len(ch2_seq), interval=100)
    anim.save(save_path, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Video saved to {save_path}")


def save_last_frame_spline_overlay(
        ch2_seq,
        ch1_seq,
        all_p0,
        all_deformations,
        mask,
        coords,
        tck_edt,
        spline_pts_edt,
        tck_mtoc=None,
        spline_pts_mtoc=None,
        cmap=None,
        arrow_scale=50,
        save_path=None
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    # -------------------------------------------------------------------------
    # Highly distinguishable colormaps
    # -------------------------------------------------------------------------
    edt_cmap = "Reds"    # EDT = red
    mtoc_cmap = "Blues"  # MTOC = blue

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(6, 7))

    # Main image axes (as before)
    ax = fig.add_axes([0.05, 0.22, 0.90, 0.73])

    # Two separated colorbars
    cax1 = fig.add_axes([0.10, 0.17, 0.80, 0.035])   # EDT spline
    cax2 = fig.add_axes([0.10, 0.06, 0.80, 0.035])   # MTOC spline

    # -------------------------------------------------------------------------
    # Display image
    # -------------------------------------------------------------------------
    i = len(ch2_seq) - 1
    display_frame = ch2_seq[i].copy()
    display_frame[mask[i]] = ch1_seq[i][mask[i]]

    ax.imshow(display_frame, cmap='gray', origin='lower')

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # -------------------------------------------------------------------------
    # Helper to build segments
    # -------------------------------------------------------------------------
    def build_segments(points):
        return np.stack([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, 1)

    # -------------------------------------------------------------------------
    # EDT Spline (red)
    # -------------------------------------------------------------------------
    if spline_pts_edt is not None:
        pts = spline_pts_edt
        segments = build_segments(pts)
        tvals = np.linspace(0, 1, len(segments))

        lc_edt = LineCollection(
            segments,
            array=tvals,
            cmap=edt_cmap,
            norm=norm,
            linewidth=2.5
        )
        ax.add_collection(lc_edt)

        cb1 = fig.colorbar(lc_edt, cax=cax1, orientation='horizontal')
        cb1.set_label("Time along EDT spline", fontsize=10)

    # -------------------------------------------------------------------------
    # MTOC spline (blue)
    # -------------------------------------------------------------------------
    if spline_pts_mtoc is not None:
        pts = spline_pts_mtoc
        segments = build_segments(pts)
        tvals = np.linspace(0, 1, len(segments))

        lc_mtoc = LineCollection(
            segments,
            array=tvals,
            cmap=mtoc_cmap,
            norm=norm,
            linewidth=2.5
        )
        ax.add_collection(lc_mtoc)

        cb2 = fig.colorbar(lc_mtoc, cax=cax2, orientation='horizontal')
        cb2.set_label("Time along MTOC spline", fontsize=10)

    # -------------------------------------------------------------------------
    # Deformation vectors removed
    # -------------------------------------------------------------------------
    # (Intentionally removed per request)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)

    plt.close(fig)

