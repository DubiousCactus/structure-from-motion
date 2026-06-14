import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    triangulated_pts: Optional[np.ndarray] = None


class RANSAC:
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
            if i == 2:
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


class PosePredictor:
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

    def fit(self, inlier_observations: List[np.ndarray]):
        """
        Fit camera poses to points observed in two images using the Essential matrix.
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
        for i, f_tpl in enumerate(self.frame_tuples):
            assert f_tpl.fundamental_matrix is not None
            E = self._compute_essential_matrix(f_tpl.fundamental_matrix)
            f_tpl.essential_matrix = E
            U, S_diag, Vt = np.linalg.svd(E)
            # Ensure U and Vt are proper rotation matrices (det=1)
            if np.linalg.det(U) < 0:
                U[:, -1] *= -1
            if np.linalg.det(Vt) < 0:
                Vt[-1, :] *= -1

            # We have four potential solutions: ([R_1|t_1], [R_1|t_2], [R_2|t_1], [R_2|t_2])
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
            P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            for pose_candidate in cam_poses:
                P2 = self.K @ pose_candidate
                inliers = inlier_observations[i]
                triangulated_pts = np.zeros((inliers.sum(), 3))
                nb_pts_in_front = 0
                for j, ((u1, v1), (u2, v2)) in enumerate(
                    zip(X_a[inliers], X_b[inliers])
                ):
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
                    if pose_candidate[:, 2] @ (x_star[:3] - pose_candidate[:, 3]) > 0:
                        nb_pts_in_front += 1
                if nb_pts_in_front > largest_vote:
                    best_cam_pose = pose_candidate
                    largest_vote = nb_pts_in_front
                    best_triangulated_pts = triangulated_pts
            if best_cam_pose is None:
                raise ValueError("Could not find a camera pose for camera B!")
            f_tpl.cam_pose_a = P1
            f_tpl.cam_pose_b = best_cam_pose
            f_tpl.inliers = inlier_observations[i].sum()
            f_tpl.triangulated_pts = best_triangulated_pts

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
            )

        rect = np.array([tl, tr, br, bl, tl])
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color=color)

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
        # Draw a 3D plot of the triangulated points and of the camera frustums, given
        # the camera poses
        for f_tpl in self.frame_tuples:
            assert f_tpl.inliers is not None
            if f_tpl.inliers < 30:
                continue
            cam_pose_a = self.frame_tuples[-1].cam_pose_a
            cam_pose_b = self.frame_tuples[-1].cam_pose_b

            pts = f_tpl.triangulated_pts
            assert pts is not None
            print(f"Drawing {pts.shape[0]} points")
            print(pts)
            center = pts.mean(axis=0)
            scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            if scale == 0:
                scale = 1.0
            frustum_d = scale * 0.15
            frustum_w = scale * 0.1
            frustum_h = scale * 0.07

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=4,
                c="blue",
                alpha=0.6,
                label="Points",
            )

            R_b = cam_pose_b[:, :3]
            t_b = cam_pose_b[:, 3]

            self._draw_camera_frustum(
                ax,
                np.zeros(3),
                np.eye(3),
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
    ransac = RANSAC(frame_tuples)
    inliers: List[np.ndarray] = ransac.filter()
    if debug:
        ransac.draw_matches()
    assert all(
        [isinstance(f_tpl.fundamental_matrix, np.ndarray) for f_tpl in frame_tuples]
    ), "Fundamental matrix not computed for all frames during RANSAC"

    # INFO: Stage 4: Camera pose prediction and point triangulation
    if intrinsics_path is None:
        raise NotImplementedError(
            "Intrinsics estimation not implemented yet! Please provide the intrinsics matrix"
        )
    else:
        K = np.load(intrinsics_path)

    pose_predictor = PosePredictor(frame_tuples, K)
    pose_predictor.fit(inliers)
    if debug:
        pose_predictor.draw_triangulation()
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
