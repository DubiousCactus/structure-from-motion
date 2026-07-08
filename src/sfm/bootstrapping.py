from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from sfm.epipolar_geometry import (
    decompose_essential_matrix,
    select_pose_by_cheirality,
)
from sfm.data import FrameTuple, Structure


class StructureBootstrap:
    def __init__(self, frame_tuples: List[FrameTuple], K: np.ndarray) -> None:
        self.frame_tuples = frame_tuples
        assert K.shape == (3, 3)
        self.K = K

    def _compute_essential_matrix(self, F: np.ndarray):
        assert F.shape == (3, 3)
        K_rank3 = self.K.T @ F @ self.K
        U, S, Vh = np.linalg.svd(K_rank3)
        S[-1] = 0
        K_rank2 = U @ np.diag(S) @ Vh
        return K_rank2

    def init(self, inlier_observations: List[np.ndarray]) -> Structure:
        """
        Bootstrap structure by predicting the pose of the first camera pair and
        triangulating the first set of points.
        The camera pose is estimated from the Essential matrix.
        """
        assert len(inlier_observations) == len(self.frame_tuples)
        # TODO: Find the optimal image pair for bootstraping, and only compute pose
        # for it. For now we use the first image pair.
        f_tpl = self.frame_tuples[0]
        inliers = inlier_observations[0]
        assert f_tpl.fundamental_matrix is not None
        E = self._compute_essential_matrix(f_tpl.fundamental_matrix)
        f_tpl.essential_matrix = E
        cam_poses = decompose_essential_matrix(E)
        X_a = np.vstack(
            [np.array(f.pt) for f in f_tpl.frame_a_features.keypoints]
        )
        X_b = np.vstack(
            [np.array(f.pt) for f in f_tpl.frame_b_features.keypoints]
        )
        X_a = X_a[[m.queryIdx for m in f_tpl.frame_a_to_b_matches]]
        X_b = X_b[[m.trainIdx for m in f_tpl.frame_a_to_b_matches]]
        assert X_a.shape == X_b.shape
        pose_a = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ pose_a

        best_cam_pose, best_triangulated_pts = select_pose_by_cheirality(
            cam_poses, X_a[inliers], X_b[inliers], P1, self.K
        )
        if best_cam_pose is None:
            raise ValueError("Could not find a camera pose for camera B!")
        if best_triangulated_pts is None:
            raise ValueError("Could not triangulate points!")

        matches_a = np.array(
            [m.queryIdx for m in f_tpl.frame_a_to_b_matches]
        )
        matches_b = np.array(
            [m.trainIdx for m in f_tpl.frame_a_to_b_matches]
        )
        corresp_a = {
            (0, x): j
            for (x, j) in zip(
                matches_a[inliers], range(len(best_triangulated_pts))
            )
        }
        corresp_b = {
            (1, x): j
            for (x, j) in zip(
                matches_b[inliers], range(len(best_triangulated_pts))
            )
        }
        P2 = self.K @ best_cam_pose

        # Refine 3D points via non-linear triangulation
        def reproj_err(x: np.ndarray, *args, **kwargs) -> np.ndarray:
            """
            Args:
                x (np.ndarray): flattened array of shape (N*3) for N points.
            Returns the flattened vector of residuals.
            """
            n_pts = x.shape[0] // 3
            x = x.reshape(n_pts, 3)
            x_homo = np.concatenate([x, np.ones((n_pts, 1))], axis=1)
            u1, v1 = X_a[inliers].T
            u2, v2 = X_b[inliers].T
            assert u1.shape[0] == x.shape[0]
            assert u2.shape[0] == x.shape[0]
            cam_a_proj = x_homo @ P1.T  # (N, 3)
            cam_a_proj = (cam_a_proj / cam_a_proj[:, -1][:, None])[:, :2]
            cam_b_proj = x_homo @ P2.T  # (N, 3)
            cam_b_proj = (cam_b_proj / cam_b_proj[:, -1][:, None])[:, :2]
            return np.stack(
                [
                    u1 - cam_a_proj[:, 0],
                    v1 - cam_a_proj[:, 1],
                    u2 - cam_b_proj[:, 0],
                    v2 - cam_b_proj[:, 1],
                ],
                axis=1,
            ).ravel()

        n_pts = best_triangulated_pts.shape[0]
        refined_pts = least_squares(
            reproj_err,
            best_triangulated_pts.reshape(
                n_pts * 3,
            ),
            method="lm",
        ).x.reshape(n_pts, 3)
        f_tpl.cam_pose_a = pose_a
        f_tpl.cam_pose_b = best_cam_pose
        f_tpl.inliers = inliers.sum()
        f_tpl.triangulated_pts_linear = best_triangulated_pts
        f_tpl.triangulated_pts = refined_pts

        corresp = {}
        corresp.update(corresp_a)
        corresp.update(corresp_b)

        return Structure(
            points3D=refined_pts,
            correspondences=corresp,
            poses={0: pose_a, 1: best_cam_pose},
        )

    def _draw_camera_frustum(self, ax, center, R, d, w, h, color, label):
        x_axis = R[0, :]
        y_axis = R[1, :]
        z_axis = R[2, :]

        tl = center + d * z_axis - w * x_axis - h * y_axis
        tr = center + d * z_axis + w * x_axis - h * y_axis
        bl = center + d * z_axis - w * x_axis + h * y_axis
        br = center + d * z_axis + w * x_axis + h * y_axis

        for corner in [tl, tr, bl, br]:
            ax.plot(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
                color=color,
                alpha=0.5,
            )

        rect = np.array([tl, tr, br, bl, tl])
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color=color, alpha=0.8)

        forward = center + d * 1.5 * z_axis
        ax.plot(
            [center[0], forward[0]],
            [center[1], forward[1]],
            [center[2], forward[2]],
            color=color,
            linestyle="--",
            linewidth=0.5,
        )

        ax.scatter(
            [center[0]], [center[1]], [center[2]], color=color, s=50, label=label
        )

    def draw_triangulation(self):
        f_tpl = self.frame_tuples[0]
        assert f_tpl.inliers is not None
        cam_pose_a = f_tpl.cam_pose_a
        cam_pose_b = f_tpl.cam_pose_b
        assert cam_pose_a is not None and cam_pose_b is not None, (
            "Cam poses weren't bootstrapped!"
        )

        pts_linear = f_tpl.triangulated_pts_linear
        pts = f_tpl.triangulated_pts
        assert pts is not None
        print(f"Drawing {pts.shape[0]} points")
        print(pts)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        if pts_linear is not None:
            ax.scatter(
                pts_linear[:, 0],
                pts_linear[:, 2],
                s=4,
                c="orange",
                alpha=0.6,
                label="Linear triangulation",
            )
        ax.scatter(
            pts[:, 0],
            pts[:, 2],
            s=4,
            c="blue",
            alpha=0.6,
            label="Non-linear triangulation",
        )

        R_a = cam_pose_a[:, :3]
        t_a = cam_pose_a[:, 3]
        R_b = cam_pose_b[:, :3]
        t_b = cam_pose_b[:, 3]

        scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        arrow_len = max(scale * 0.1, 1.0)
        head_w = arrow_len * 0.01
        head_l = arrow_len * 0.15

        # Camera A at origin
        ax.scatter(t_a[0], t_a[2], c="red", s=50, label="Camera A")
        fwd_a = R_a[2, :]  # z-axis
        ax.arrow(
            t_a[0],
            t_a[2],
            fwd_a[0] * arrow_len,
            fwd_a[2] * arrow_len,
            head_width=head_w,
            head_length=head_l,
            fc="red",
            ec="red",
        )

        # Camera B
        ax.scatter(t_b[0], t_b[2], c="green", s=50, label="Camera B")
        fwd_b = R_b[2, :]  # z-axis in camera B's frame
        ax.arrow(
            t_b[0],
            t_b[2],
            fwd_b[0] * arrow_len,
            fwd_b[2] * arrow_len,
            head_width=head_w,
            head_length=head_l,
            fc="green",
            ec="green",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(
            "Triangulated 3D Points and Camera Frustums (seen from Y axis)"
        )
        ax.legend()
        plt.show()

    def draw_triangulation_3D(self):
        # Draw a 3D plot of the triangulated points and of the camera frustums, given
        # the camera poses
        f_tpl = self.frame_tuples[0]
        assert f_tpl.inliers is not None
        cam_pose_a = f_tpl.cam_pose_a
        cam_pose_b = f_tpl.cam_pose_b
        assert cam_pose_a is not None and cam_pose_b is not None, (
            "Cam poses weren't bootstrapped!"
        )

        pts_linear = f_tpl.triangulated_pts_linear
        pts = f_tpl.triangulated_pts
        assert pts is not None
        print(f"Drawing {pts.shape[0]} points")
        print(pts)
        center = pts.mean(axis=0)
        scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        if scale == 0:
            scale = 1.0

        # Adjusted scale to be more robust
        frustum_d = scale * 0.15
        frustum_w = scale * 0.05
        frustum_h = scale * 0.04

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        if pts_linear is not None:
            ax.scatter(
                pts_linear[:, 0],
                pts_linear[:, 1],
                pts_linear[:, 2],
                s=4,
                c="orange",
                alpha=0.6,
                label="Linear triangulation",
            )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=4,
            c="blue",
            alpha=0.6,
            label="Non-linear triangulation",
        )

        R_a = cam_pose_a[:, :3]
        t_a = cam_pose_a[:, 3]
        R_b = cam_pose_b[:, :3]
        t_b = cam_pose_b[:, 3]
        print(f"Cam pose A: R={R_a}, t={t_a}")
        print(f"Cam pose B: R={R_b}, t={t_b}")

        self._draw_camera_frustum(
            ax,
            t_a,
            R_a,
            frustum_d,
            frustum_w,
            frustum_h,
            "red",
            "Camera A",
        )
        self._draw_camera_frustum(
            ax,
            t_b,
            R_b,
            frustum_d,
            frustum_w,
            frustum_h,
            "green",
            "Camera B",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Triangulated 3D Points and Camera Frustums")
        ax.legend()
        plt.show()
