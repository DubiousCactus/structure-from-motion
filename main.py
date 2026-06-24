import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

import fqs

app = typer.Typer()


@app.command()
def extract_frames(video_path: str, output_folder: str):
    cap = cv.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    bar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)), desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
        cv.imwrite(frame_filename, frame)
        frame_count += 1
        bar.update(1)
    bar.close()
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


@dataclass
class ImageFeatures:
    keypoints: List[cv.KeyPoint]
    descriptors: List[np.ndarray]
    img_path: str


@dataclass
class FrameTuple:
    frame_a_id: int
    frame_b_id: int
    frame_a_features: ImageFeatures
    frame_b_features: ImageFeatures
    frame_a_to_b_matches: Sequence[cv.DMatch]
    fundamental_matrix: Optional[np.ndarray] = None
    essential_matrix: Optional[np.ndarray] = None
    cam_pose_a: Optional[np.ndarray] = None
    cam_pose_b: Optional[np.ndarray] = None
    inliers: Optional[int] = 0
    triangulated_pts_linear: Optional[np.ndarray] = None
    triangulated_pts: Optional[np.ndarray] = None


@dataclass
class Structure:
    points3D: np.ndarray  # List of homogeneous points
    correspondences: Dict[Tuple[int, int], int]  # Dict of (frame_id, kp_id) -> point_id
    poses: Dict[int, np.ndarray]  # Dict of frame_id -> [R|t] SO(3) pose


class EpipolarRANSAC:
    def __init__(
        self,
        frame_tuples: List[FrameTuple],
        consensus_ratio: float = 0.8,
        max_iter: int = 1000,
        threshold: float = 0.05,
    ):
        self.frame_tuples = frame_tuples
        self.frame_inliers = []
        self.frame_pair_F = []
        self.consensus_min = len(frame_tuples) * consensus_ratio
        self.max_iter = max_iter
        self.threshold = threshold

    def _compute_fundamental_matrix(
        self, kp1: np.ndarray, kp2: np.ndarray
    ) -> np.ndarray:
        # INFO: The fundamental matrix F \in R^{3x3} algebraically represents the epipolar
        # geometry that relates two sets of real-world point projections on two image
        # planes. F gives us the epipolar constraint in the form (x_{i}^{\prime})^T F x_i = 0
        assert len(kp1.shape) == 2 and kp1.shape[1] == 2
        assert len(kp2.shape) == 2 and kp2.shape[1] == 2
        assert kp1.shape[0] == kp2.shape[0], "Number of keypoints must match"
        assert kp1.shape[0] >= 8, "Number of keypoints must be 8 minimum"
        # TODO: Hartley normalization. This may incur very poor performance due to huge
        # dot products with large (u,v) coordinates.
        A = []
        for (x, y), (xp, yp) in zip(kp1, kp2):
            A.append([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1])
        A = np.asarray(A)
        if not np.linalg.matrix_rank(A) >= 8:
            raise ValueError("Bad design matrix, find better points.")
        # Using SVD we can minimize min_{||f||=1} ||Xf||, where A is the design matrix for
        # all point correspondences. The last column of V gives us the solution.
        U, S, Vh = np.linalg.svd(
            A
        )  # Vh is the Hermitian transpose of V. For a real-valued matrix A, the hermitian transpose equals the normal transpose.
        f_vec = Vh[-1, :]  # Last row of V^T is the last column of V
        # WARN: However, due to noise in the observations, our solution does not fully
        # minimize te objective, ie the last singular value is not exactly zero. This means
        # that our optimal F is not rank 2 but full rank (3)! IE due to the rank-nullity
        # theorem, the nullity of F is 0 and thus we haven't found the null space / solution
        # to Ax=0. To enforce the rank 2 constraint, we can set the smallest singular value
        # to zero and recompute F.
        F = f_vec.reshape(3, 3)
        U, S, Vh = np.linalg.svd(F)
        S[-1] = 0  # Set smallest singular value to zero
        F = U @ np.diag(S) @ Vh
        return F

    def filter(self) -> List[np.ndarray]:
        """
        Filter out outliers of a set of feature matches using RANSAC and the epipolar
        constraint as condition. For each set of candidate matches, we compute the
        Fundamental matrix and check for the epipolar constraint to be respected.

        Returns:
        inliers: List[(N,)] a list of boolean masks for the inliers of each frame
        """
        for f_tuple in self.frame_tuples:
            inliers, best_fit, best_fit_err = None, None, np.inf
            X_a = np.vstack(
                [np.array(f.pt) for f in f_tuple.frame_a_features.keypoints]
            )
            X_b = np.vstack(
                [np.array(f.pt) for f in f_tuple.frame_b_features.keypoints]
            )
            # Filter out keypoints to only keep matches:
            X_a = X_a[[m.queryIdx for m in f_tuple.frame_a_to_b_matches]]
            X_b = X_b[[m.trainIdx for m in f_tuple.frame_a_to_b_matches]]
            assert X_a.shape == X_b.shape
            n_pts = X_a.shape[0]
            print(
                f"Frame tuple ({f_tuple.frame_a_id},{f_tuple.frame_b_id}) has {n_pts} matches"
            )
            if n_pts < 8:
                warnings.warn(
                    f"Frame tuple ({f_tuple.frame_a_id},{f_tuple.frame_b_id}) does not have enough features!"
                )
                continue
            X_a_homo = np.concatenate([X_a, np.ones((n_pts, 1))], axis=1)
            X_b_homo = np.concatenate([X_b, np.ones((n_pts, 1))], axis=1)
            pbar = tqdm(range(self.max_iter), desc="Finding inliers with RANSAC...")
            for _ in pbar:
                # 1. Select hypothetical outliers
                idx = np.random.choice(n_pts, 8, replace=False)
                x_a, x_b = X_a[idx], X_b[idx]
                assert x_a.shape[0] == x_b.shape[0] == 8, "Did not sample 8 matches"
                # 2. Fit a model to these
                try:
                    F = self._compute_fundamental_matrix(x_a, x_b)
                except ValueError:
                    continue
                # 3. Test all data against this model. All data points that fit well are
                # the consensus set (i.e. inliers).
                # x^bFx^a with batch dimension:
                # TODO: Use Sampson approximation instead for scale-independent error
                err = np.abs((X_b_homo * (X_a_homo @ F.T)).sum(axis=1))
                this_inliers = err < self.threshold
                # 4. The model is reasonably good if sufficiently many points are
                # classified as part of the consensus set.
                if this_inliers.sum() < max(8, self.consensus_min):
                    continue
                this_err = err[this_inliers].mean()
                pbar.set_postfix(
                    {"#inliers": this_inliers.sum().item(), "Total err": this_err}
                )

                if this_err < best_fit_err:
                    best_fit = self._compute_fundamental_matrix(
                        X_a[this_inliers], X_b[this_inliers]
                    )
                    best_fit_err = this_err
                    inliers = this_inliers
                    # if this_err < self.threshold

            f_tuple.fundamental_matrix = best_fit
            self.frame_inliers.append(inliers)
        return self.frame_inliers

    def draw_matches(self):
        i = 0
        for f, inliers in zip(self.frame_tuples, self.frame_inliers):
            if inliers is None:
                continue
            if i == 1:
                break
            matches = f.frame_a_to_b_matches

            img1 = cv.imread(f.frame_a_features.img_path, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(f.frame_b_features.img_path, cv.IMREAD_GRAYSCALE)
            img = cv.drawMatches(
                img1,
                f.frame_a_features.keypoints,
                img2,
                f.frame_b_features.keypoints,
                matches,
                None,
                matchColor=(0, 0, 255),
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            # Draw inliers in green
            inlier_matches = [m for m, keep in zip(matches, inliers) if keep]

            img = cv.drawMatches(
                img1,
                f.frame_a_features.keypoints,
                img2,
                f.frame_b_features.keypoints,
                inlier_matches,
                img,
                matchColor=(0, 255, 0),
                flags=cv.DrawMatchesFlags_DRAW_OVER_OUTIMG,
            )

            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            i += 1


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
        # First we define the W matrix which is useful for the essential matrix
        # decomposition, where E = SR, S=UZU^T and R=UWV^T or R=UW^TV^T, where t=U[:,
        # -1] ie the translation is the last column of U.
        W = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )
        assert len(inlier_observations) == len(self.frame_tuples)
        # TODO: Find the optimal image pair for bootstraping, and only compute pose
        # for it. For now we use the first image pair.
        f_tpl = self.frame_tuples[0]
        inliers = inlier_observations[0]
        assert f_tpl.fundamental_matrix is not None
        E = self._compute_essential_matrix(f_tpl.fundamental_matrix)
        f_tpl.essential_matrix = E
        U, S_diag, Vt = np.linalg.svd(E)
        # We have four potential solutions: ([R_1|t_1], [R_1|t_2], [R_2|t_1], [R_2|t_2])
        # WARN: det(R)=1. If that is not the case, i.e. det(R)=-1, we must corect by
        # setting t = -1 and R=-R.
        t_1 = U[:, -1]
        t_2 = -U[:, -1]
        R_1 = U @ W @ Vt
        R_2 = U @ W.T @ Vt
        cam_poses = [
            np.hstack([R_1, t_1[:, None]]),
            np.hstack([R_1, t_2[:, None]]),
            np.hstack([R_2, t_1[:, None]]),
            np.hstack([R_2, t_2[:, None]]),
        ]
        # INFO: Now we find the *correct pose* using the Cheirality condition: the
        # point X st \hat{x}=KX must lie in front of both cameras. To do so, we
        # triangulate the point X from both candidate poses using **linear least
        # squares**. Then we can just check the sign of Z in the camera frame wrt to
        # its center.
        # So X is in front iff: r_3(X-C)>0 where r_3 is the third column of the
        # rotation matrix (z-axis of the camera).
        # WARN: Since all triangulated points won't satisfy this condition due to
        # noise in the correspondances, we simply take the best pose by majority
        # voting.
        # NOTE: To triangulate all points, we can solve the following homogeneous
        # linear system of equations: AX=0, where X is the matrix of row vector
        # points in homogeneous coordinates, and A is the matrix of stacked cross
        # product vectors such that AX=0 <=> x \times PX = 0. This is the cross
        # product between the observed image points and the projected points, which
        # must be 0 to indicate that both are vectors that lie in the same
        # direction, since perspective projection only matches observations up to
        # scale. We then avoid the trivial solution x=0 by minimizing ||AX|| such
        # that ||X||=1 via SVD:
        X_a = np.vstack([np.array(f.pt) for f in f_tpl.frame_a_features.keypoints])
        X_b = np.vstack([np.array(f.pt) for f in f_tpl.frame_b_features.keypoints])
        # Filter out keypoints to only keep matches:
        X_a = X_a[[m.queryIdx for m in f_tpl.frame_a_to_b_matches]]
        X_b = X_b[[m.trainIdx for m in f_tpl.frame_a_to_b_matches]]
        assert X_a.shape == X_b.shape
        best_cam_pose, largest_vote = None, 0
        best_triangulated_pts = None
        pose_a = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ pose_a
        corresp_a, corresp_b = None, None
        for pose_candidate in cam_poses:
            if np.linalg.det(pose_candidate[:, :3]) < 0:
                print("WARN: det(R)=-1, correcting")
                pose_candidate[:, :3] = -pose_candidate[:, :3]
                pose_candidate[:, -1] = -pose_candidate[:, -1]
            P2 = self.K @ pose_candidate
            triangulated_pts = np.zeros((inliers.sum(), 3))
            nb_pts_in_front = 0
            for j, ((u1, v1), (u2, v2)) in enumerate(zip(X_a[inliers], X_b[inliers])):
                # A is the matrix of stacked cross-product vectors such that
                # AX=0 <=> x \times PX = 0
                A = np.array(
                    [
                        u1 * P1[2] - P1[0],
                        v1 * P1[2] - P1[1],
                        u2 * P2[2] - P2[0],
                        v2 * P2[2] - P2[1],
                    ]
                )
                _, _, Vh = np.linalg.svd(A)
                x_star = Vh[-1, :]  # Last row of V^T is the last column of V
                if x_star[-1] != 0:
                    x_star /= x_star[-1]  # Divide by w for perspective projection
                triangulated_pts[j] = x_star[:3]

                # Cheirality check: the 3D point must lie in front of both
                # cameras, i.e. have positive depth in each camera frame.
                # Camera A sits at the world origin (pose_a = [I|0]), so its
                # depth is just X[2].
                in_front_a = x_star[2] > 0

                # Camera B has pose [R_b|t_b], meaning a world point X maps to
                # the camera-B frame as x_b = R_b X + t_b. The depth of X in
                # B's frame is therefore the third component: z_b = r_3 . X +
                # t_b[2], where r_3 is the third row of R_b (B's optical axis).
                # NOTE: t_b is NOT the camera center. The camera center in
                # world coords is C = -R_b^T t_b (obtained by solving R_b C +
                # t_b = 0). The condition r_3.(X - C) > 0 from the literature
                # is algebraically equivalent to r_3.X + t_b[2] > 0, since
                # r_3.(X - C) = r_3.X - r_3.C = r_3.X - r_3.(-R_b^T t_b) =
                # r_3.X + (r_3 R_b^T) t_b = r_3.X + t_b[2] (because r_3 R_b^T
                # picks out the third row of R_b R_b^T = I, i.e. e_3^T).
                R_b = pose_candidate[:, :3]
                t_b = pose_candidate[:, 3]
                in_front_b = (R_b[2, :] @ x_star[:3] + t_b[2]) > 0

                if in_front_a and in_front_b:
                    nb_pts_in_front += 1
            if nb_pts_in_front > largest_vote:
                best_cam_pose = pose_candidate
                largest_vote = nb_pts_in_front
                best_triangulated_pts = triangulated_pts

                matches_a = np.array(
                    [m.queryIdx for m in f_tpl.frame_a_to_b_matches]
                )  # Indices of keypoints
                matches_b = np.array(
                    [m.trainIdx for m in f_tpl.frame_a_to_b_matches]
                )  # Indices of keypoints
                corresp_a = {
                    (0, x): j
                    for (x, j) in zip(matches_a[inliers], range(len(triangulated_pts)))
                }
                corresp_b = {
                    (1, x): j
                    for (x, j) in zip(matches_b[inliers], range(len(triangulated_pts)))
                }
        if best_cam_pose is None:
            raise ValueError("Could not find a camera pose for camera B!")
        if best_triangulated_pts is None:
            raise ValueError("Could not triangulate points!")
        assert corresp_a is not None and corresp_b is not None
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
        ax.set_title("Triangulated 3D Points and Camera Frustums (seen from Y axis)")
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


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


class PerspectiveNPoint:
    def __init__(self, frame_tuples: List[FrameTuple], K: np.ndarray) -> None:
        self.frame_tuples = frame_tuples
        assert K.shape == (3, 3)
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def _solve_p3p(self, world_pts: np.ndarray, img_pts: np.ndarray):
        """
        This is the P3P implementation of Kneip et al, CVPR 2011 (https://rpg.ifi.uzh.ch/docs/CVPR11_kneip.pdf)
        """
        # TODO: Verify that all 3D and 2D points aren't colinear.
        assert world_pts.shape == (3, 3)  # TODO: Homogeneous?
        p1, p2, p3 = world_pts[0], world_pts[1], world_pts[2]
        u1, u2, u3 = img_pts[0], img_pts[1], img_pts[2]
        f1, f2, f3 = (
            normalize(self.K_inv @ u1),
            normalize(self.K_inv @ u2),
            normalize(self.K_inv @ u3),
        )
        tx = f1
        tz = normalize(np.cross(f1, f2))
        ty = np.cross(tz, tx)
        T = np.stack([tx, ty, tz])  # Stacked row-wise
        f3_tau = T @ f3
        nx = normalize(p2 - p1)
        nz = normalize(np.cross(nx, p3 - p1))
        ny = np.cross(nz, nx)
        N = np.stack([nx, ny, nz])  # Stacked row-wise
        # P3_nabla = (p1, p2, 0)^T, so:
        P3_nabla = N @ (p3 - p1)
        px = P3_nabla[0]
        py = P3_nabla[1]
        d12 = np.linalg.norm(p2 - p1)
        # The sign of b = cot Beta is given by the sign of cos Beta = f1 @ f2
        b = np.sqrt(1 / (1 - (f1 @ f2) ** 2) - 1)
        b *= np.sign(f1 @ f2)
        phi_1 = f3_tau[0] / f3_tau[2]
        phi_2 = f3_tau[1] / f3_tau[2]
        # Now we compute the factors of the polynomial (quatric):
        a4 = -(phi_2**2) * py**4 - phi_1**2 * py**4 - py**4
        a3 = (
            2 * py**3 * d12 * b
            + 2 * phi_2**2 * py**3 * d12 * b
            - 2 * phi_1 * phi_2 * py**3 * d12
        )
        a2 = (
            -(phi_2**2) * px**2 * py**2
            - phi_2**2 * py**2 * d12**2 * b**2
            - phi_2**2 * py**2 * d12**2
            + phi_2**2 * py**4
            + phi_1**2 * py**4
            + 2 * px * py**2 * d12
            + 2 * phi_1 * phi_2 * px * py**2 * d12 * b
            - phi_1**2 * px**2 * py**2
            + 2 * phi_2**2 * px * py**2 * d12
            - py**2 * d12**2 * b**2
            - 2 * px**2 * py**2
        )
        a1 = (
            2 * px**2 * py * d12 * b
            + 2 * phi_1 * phi_2 * py**3 * d12
            - 2 * phi_2**2 * py**3 * d12 * b
            - 2 * px * py * d12**2 * b
        )
        a0 = (
            -2 * phi_1 * phi_2 * px * py**2 * d12 * b
            + phi_2**2 * py**2 * d12**2
            + 2 * px**3 * d12
            - px**2 * d12**2
            + phi_2**2 * px**2 * py**2
            - px**4
            - 2 * phi_2**2 * px * py**2 * d12
            + phi_1**2 * px**2 * py**2
            + phi_2**2 * py**2 * d12**2 * b**2
        )
        # We can find the real roots of the quatric using Ferrari's closed form
        # solution:
        # NOTE: According to wikipedia, it's easier if we convert to a depressed
        # quartic, where: x^4 + bx^3 + cx^2 +dx + e = 0, with b=a3/a4, c=a2/a4, d=a1/a4,
        # e=a0/a4. Anyway, I decided to use someone else's implementation, but yeah they
        # do use the depressed quartic form!
        coeff = np.array([a4, a3, a2, a1, a0])
        if np.any(np.isnan(coeff)):
            raise ValueError("Some quartic polynomials are NaN!")
        roots = fqs.quartic_roots(coeff)
        solutions = []
        for i in range(4):
            cos_theta = roots[i]
            if abs(cos_theta.imag) > 1e-8:
                continue

            cos_theta = cos_theta.real

            if abs(cos_theta) > 1:
                continue
            # For each solution, we find the values for cot alpha:
            cot_alpha = ((phi_1 / phi_2) * px + cos_theta * py - d12 * b) / (
                (phi_1 / phi_2) * cos_theta * py - px + d12
            )
            # Compute all trigonometric forms for alpha and theta using the trignonometric
            # relationships and the restricted parameter domains:
            # NOTE: "Using the restricted domains of parameters α and θ, all appearing
            # trigonometric forms of the parameters can be directly derived from cot α
            # and cos θ using simple trigonometric relationships."
            # NOTE: "Note that θ ∈ [0; π] if f τ 3,z < 0, and θ ∈ [−π; 0] if f τ 3,z > 0,
            # where ~f τ 3 is obtained from ~f3 via (1)."
            # NOTE: "We define the free parameter α ∈ [0; π] as the angle ∠P2P1C."
            cos_alpha = cot_alpha / np.sqrt(1 + cot_alpha**2)
            sin_alpha = 1 / np.sqrt(1 + cot_alpha**2)  # Positive because α ∈ [0; π]
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
            if f3_tau[2] > 0:
                sin_theta *= -1
            # Compute Cnabla and Q for each solution:
            Cnabla = np.array(
                [
                    d12 * cos_alpha * (sin_alpha * b + cos_alpha),
                    d12 * sin_alpha * cos_theta * (sin_alpha * b + cos_alpha),
                    d12 * sin_alpha * sin_theta * (sin_alpha * b + cos_alpha),
                ]
            )
            Q = np.array(
                [
                    [-cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta],
                    [sin_alpha, -cos_alpha * cos_theta, -cos_alpha * sin_theta],
                    [0, -sin_theta, cos_theta],
                ]
            )
            # Compute the absolute camera center C and orientation R for each solution:
            C = p1 + N.T @ Cnabla
            R = N.T @ Q.T @ T
            # TODO: Cheirality check
            solutions.append((C, R))

        # TODO: Disambiguate the 4 solutions using a 4th measurement
        raise NotImplementedError("P3P not implemented yet.")

    def fit(self, structure: Structure, inlier_observations: List[np.ndarray]):
        """
        Fit camera poses to 3D-2D point correspondances via P3P and RANSAC.

        Given a calibrated pinhole camera, three 3D points x_i = (x_i, y_i, z_i),
        and corresponding homogeneous image coordinates y_i sim (u_i, v_i, 1) such
        that |y_i| = 1, then:

            lambda_i y_i = R x_i + t, i in {1, 2, 3},

        where the rotation R in SO(3) together with the translation t in R^3 define the
        pose of the camera.

        In short, a P3P solver is a function

            [R_k, t_k] = P3P(x_{1:3}, y_{1:3}).

        Depending on the configuration of the points, P3P has up to four solutions.
        """
        # Register remaining images and estimate next camera poses:
        last_frame_pts_offset = 0
        for i in range(1, len(self.frame_tuples)):  # Start at the pair after bootstrap
            # 1. Compute correspondences to existing scene points via
            # matches to previous frame:
            inliers = inlier_observations[i]
            f_tpl = self.frame_tuples[i]
            matches_a = np.array([m.queryIdx for m in f_tpl.frame_a_to_b_matches])
            matches_b = np.array([m.trainIdx for m in f_tpl.frame_a_to_b_matches])
            # Filter matches by the indices of those in common with the previous
            # pair:
            last_tpl = self.frame_tuples[i - 1]
            last_matches_b = np.array(
                [m.trainIdx for m in last_tpl.frame_a_to_b_matches]
            )
            last_inliers = inlier_observations[i - 1]
            in_common, comm1, comm2 = np.intersect1d(
                matches_a[inliers], last_matches_b[last_inliers], return_indices=True
            )  # Matches (keypoint ids) that are common to both tuples
            corresp = {}
            for j, k in enumerate(comm2):
                corresp[(i * 2 + 0, matches_a[inliers][k])] = j + last_frame_pts_offset
                corresp[(i * 2 + 1, matches_b[inliers][k])] = j + last_frame_pts_offset
            structure.correspondences.update(corresp)
            last_frame_pts_offset += len(in_common)
            # 2. Solve PnP:
            idx = np.random.choice(len(in_common), 3, replace=False)
            # TODO: Implement RANSAC loop to compute P3P on inliers.
            pts2D = None  # TODO:
            cam_pose_b = self._solve_p3p(structure.points3D[idx], pts2D)
            points3D = triangulate_pts_dlt(cam_pose_a, cam_pose_b, in_common)
            structure.points3D = np.stack([structure.points3D, points3D], axis=0)
            structure.poses[i * 2 + 1] = cam_pose_b


class BundleAdjustment:
    def adjust(self):
        raise NotImplementedError


@app.command()
def extract_and_match(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
):
    if intrinsics_path is not None and not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found at {intrinsics_path}")
    # INFO: Stage 1: feature extraction using ORB
    orb = cv.ORB_create()
    frame_features = []
    for i, frame_file in tqdm(
        enumerate(sorted(os.listdir(frames_path))), desc="Extracting features"
    ):
        if i == max_frames:
            break
        frame_path = os.path.join(frames_path, frame_file)
        img = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        frame_features.append(ImageFeatures(kp, des, frame_path))
        if debug and i == 0:
            img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()

    # INFO: Stage 2: KNN-based feature matching with FLANN
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    frame_tuples = []

    flann = cv.FlannBasedMatcher(index_params, search_params)
    for i in range(1, len(frame_features)):
        f1, f2 = frame_features[i - 1], frame_features[i]
        matches = flann.knnMatch(f1.descriptors, f2.descriptors, k=2)
        # INFO: we call knnMatch(query=f1, train=f2) and get:
        # DMatch: .query_id, .train_id, .train_img_id, .distance
        # For RANSAC -- and for the rest -- we'll want something more structured, ie:
        # Match{ .img_a_id, .img_a_feature_id, .img_b_id, .img_b_feature_id, .dist }
        # and a list of [Match], and a list of [Feature] but the ids must match.
        # Or potentially something simpler, all packaged into a frame tuple:
        # FrameTuple{ .frame_a_id, .frame_a_features, .frame_b_id, .frame_b_features, .matches_a_to_b, }

        good_matches = []
        for j, match in enumerate(matches):
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        frame_tuples.append(FrameTuple(i - 1, i, f1, f2, good_matches))

        # if debug and i < 10:
        #     # Need to draw only good matches, so create a mask
        #     matchesMask = [[0, 0] for i in range(len(matches))]
        #
        #     # ratio test as per Lowe's paper
        #     for j, (m, n) in enumerate(matches):
        #         if m.distance < 0.6 * n.distance:
        #             matchesMask[j] = [1, 0]
        #
        #     draw_params = dict(
        #         matchColor=(0, 255, 0),
        #         singlePointColor=(255, 0, 0),
        #         matchesMask=matchesMask,
        #         flags=cv.DrawMatchesFlags_DEFAULT,
        #     )
        #     img1 = cv.imread(f1.img_path, cv.IMREAD_GRAYSCALE)
        #     img2 = cv.imread(f2.img_path, cv.IMREAD_GRAYSCALE)
        #     img3 = cv.drawMatchesKnn(
        #         img1, f1.keypoints, img2, f2.keypoints, matches, None, **draw_params
        #     )
        #     plt.imshow(img3)
        #     plt.show()

    # INFO: Stage 3: RANSAC outlier removal via the epipolar constraint.
    # Now that we've got good candidate matches, we can start filtering them with
    # RANSAC and the Fundamental matrix, ie if x'TFx ~= 0 the match is good, otherwise
    # it's an outlier.
    ransac = EpipolarRANSAC(frame_tuples)
    inliers: List[np.ndarray] = ransac.filter()
    if debug:
        ransac.draw_matches()
    assert all(
        [isinstance(f_tpl.fundamental_matrix, np.ndarray) for f_tpl in frame_tuples]
    ), "Fundamental matrix not computed for all frames during RANSAC"

    # INFO: Stage 4: 2D-2D Camera pose prediction via Essential matrix decomposition and
    # point triangulation. The 3D points are a by-product of computing the pose from
    # decomposing the Essential matrix, and filtering the valid pose via the Cheirality
    # condition. Here, we only bootstrap the 3D structure.
    if intrinsics_path is None:
        # TODO: Load focal length from EXIF of images if available. If not, come up with
        # a rough initialization and optimize for it in bundle adjustment. Another way
        # is to estimate the focal length from the hommography
        # (https://imkaywu.github.io/blog/2017/10/focal-from-homography/), but this
        # assumes that the two camera centers are fixed and the caameras only undergo
        # rotations. In practice, it seems everyone uses UPnP
        # (https://openreview.net/pdf?id=PbMNl2kC0u) or a flavour of PnPf (www.researchgate.net/publication/354289451_Efficient_DLT-Based_Method_for_Solving_PnP_PnPf_and_PnPfr_Problems)
        raise NotImplementedError(
            "Intrinsics estimation not implemented yet! Please provide the intrinsics matrix"
        )
    else:
        K = np.load(intrinsics_path)[0]
    # WARN: How about scale ambiguity that comes with E-decomposition (2D-2D
    # correspondances and pose prediction)? Well, unfortunately that's just a thing of
    # monocular SfM. We just *can't recover absolute scale from images alone*. However,
    # chaining E-decomposition for each new camera *will lead to scale drift*. To remedy
    # this, we use PnP!

    bootstrap = StructureBootstrap(frame_tuples, K)
    structure = bootstrap.init(inliers)
    if debug:
        bootstrap.draw_triangulation_3D()
    # INFO: Stage 5: register all images and solve all camera poses using PnP. Given the
    # initial 3D points, register each new image in the scene using 3D-2D
    # correspondances.
    pnp = PerspectiveNPoint(frame_tuples, K)
    pnp.fit(structure, inliers)
    assert all(
        [
            isinstance(f_tpl.cam_pose_a, np.ndarray)
            and isinstance(f_tpl.cam_pose_b, np.ndarray)
            for f_tpl in frame_tuples
        ]
    ), "Camera pose not computed for all frames"

    # INFO: Stage 6: Bundle adjustment
    _ = BundleAdjustment().adjust()

    # os.makedirs(".tmp", exist_ok=True)
    # # np.savez(".tmp/matches.npz", frame_matches)
    # with open(".tmp/matches.pkl", "wb") as f:
    #     import pickle
    #
    #     pickle.dump(frame_matches, f)


if __name__ == "__main__":
    app()
