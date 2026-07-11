import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from sfm.data import FrameTuple

if TYPE_CHECKING:
    from sfm.tui import PipelineStats, SfmDisplay


def hartley_normalize(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalization of a set of 2D image points.

    Normalizes the points so that:
      1. The origin is at their centroid.
      2. The mean distance from the origin is sqrt(2).

    This prevents numerical ill-conditioning in the 8-point algorithm when
    the (u, v) pixel coordinates have large magnitudes.

    Args:
        pts: (N, 2) array of 2D points.

    Returns:
        normalized_pts: (N, 2) normalized points.
        T: (3, 3) similarity transform such that normalized_pts = (T @ pts_h).T[:, :2]
            for homogeneous pts_h. To denormalize a fundamental matrix computed
            on the normalized points, use F = T2.T @ F_norm @ T1.
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    # INFO: First we apply Hartley normalization. This is to prevent poor
    # performance due to huge dot products with large (u,v) coordinates.
    # 1. Translate the origin to the centroid of points for each image.
    # 2. Uniformly scale coordinates such that mean of distances from the origin
    # equals sqrt(2).
    # Finally, the non-normalized fundamental matrix can be recovered as F = T2^T F_normalized T1, where T1 and T2 are the normalization transforms for each image.
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    scale = np.sqrt(2) / np.linalg.norm(shifted, axis=1).mean()
    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ]
    )
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    normalized = (T @ pts_h.T).T[:, :2]
    return normalized, T


def eight_point_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Estimate the fundamental matrix from >= 8 point correspondances via the
    normalized 8-point algorithm.

    Applies Hartley normalization to both views, builds the linear design
    matrix, solves via SVD, enforces the rank-2 constraint, then denormalizes.

    Args:
        pts1: (N, 2) image points in the first view, N >= 8.
        pts2: (N, 2) matching image points in the second view.

    Returns:
        F: (3, 3) fundamental matrix with rank 2.

    Raises:
        ValueError: if the design matrix is rank-deficient (rank < 8) or if the
            ratio of the two smallest singular values indicates a degenerate
            configuration.
    """
    assert pts1.ndim == 2 and pts1.shape[1] == 2
    assert pts2.ndim == 2 and pts2.shape[1] == 2
    assert pts1.shape[0] == pts2.shape[0], "Number of keypoints must match"
    assert pts1.shape[0] >= 8, "Number of keypoints must be 8 minimum"

    n1, T1 = hartley_normalize(pts1)
    n2, T2 = hartley_normalize(pts2)

    A = np.empty((pts1.shape[0], 9))
    for i, ((x, y), (xp, yp)) in enumerate(zip(n1, n2)):
        A[i] = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1]
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Bad design matrix, find better points.")

    # Using SVD we can minimize min_{||f||=1} ||Xf||, where A is the design matrix for
    # all point correspondences. The last column of V gives us the solution.
    U, S, Vh = np.linalg.svd(
        A
    )  # Vh is the Hermitian transpose of V. For a real-valued matrix A, the hermitian transpose equals the normal transpose.
    if S[-2] / S[-1] < 1e-6:
        raise ValueError("Bad design matrix, find better points.")

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
    # Denormalize: F = T2^T F_normalized T1
    F = T2.T @ F @ T1
    return F


def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson (first-order) approximation of the geometric epipolar error.

    The Sampson distance is a scale-independent, first-order approximation of
    the true geometric distance of a point correspondence to the epipolar
    geometry defined by F. For an exact correspondence it is zero.

    Args:
        F: (3, 3) fundamental matrix.
        pts1: (N, 2) image points in the first view.
        pts2: (N, 2) matching image points in the second view.

    Returns:
        (N,) array of non-negative Sampson distances (squared, pixel^2 units).
    """
    n = pts1.shape[0]
    p1h = np.concatenate([pts1, np.ones((n, 1))], axis=1)
    p2h = np.concatenate([pts2, np.ones((n, 1))], axis=1)
    # INFO: We use Sampson approximation instead for scale-independent
    # error:
    Fx1 = p1h @ F.T  # (F x1)^T
    Ftx2 = p2h @ F  # (F^T x2)^T
    numer = (p2h * Fx1).sum(axis=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return numer / np.maximum(denom, 1e-12)


def triangulate_pts_dlt(
    X_a: np.ndarray, X_b: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    triangulated_pts = np.zeros((X_a.shape[0], 3))
    for j, ((u1, v1), (u2, v2)) in enumerate(zip(X_a, X_b)):
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
    return triangulated_pts


def decompose_essential_matrix(E: np.ndarray) -> List[np.ndarray]:
    """Decompose the essential matrix E into the four candidate camera poses.

    E = [t]_x R can be decomposed via SVD as E = U S V^T, yielding two possible
    rotations R_1 = U W V^T, R_2 = U W^T V^T and a translation direction
    t = U[:, -1] (up to sign). This gives four candidate poses [R|t].

    Each candidate's rotation is corrected so that det(R) = 1 (proper rotation):
    if det(R) < 0, both R and t are negated.

    Returns:
        A list of four (3, 4) [R|t] pose matrices:
        [R_1|t], [R_1|-t], [R_2|t], [R_2|-t] (after det correction).
    """
    assert E.shape == (3, 3)
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
    U, _, Vt = np.linalg.svd(E)
    # We have four potential solutions: ([R_1|t_1], [R_1|t_2], [R_2|t_1], [R_2|t_2])
    # WARN: det(R)=1. If that is not the case, i.e. det(R)=-1, we must corect by
    # setting t = -1 and R=-R.
    t = U[:, -1]
    R_1 = U @ W @ Vt
    R_2 = U @ W.T @ Vt
    candidates = [
        np.hstack([R_1, t[:, None]]),
        np.hstack([R_1, -t[:, None]]),
        np.hstack([R_2, t[:, None]]),
        np.hstack([R_2, -t[:, None]]),
    ]
    for pose in candidates:
        if np.linalg.det(pose[:, :3]) < 0:
            pose[:, :3] = -pose[:, :3]
            pose[:, -1] = -pose[:, -1]
    return candidates


def select_pose_by_cheirality(
    candidates: List[np.ndarray],
    X_a: np.ndarray,
    X_b: np.ndarray,
    P1: np.ndarray,
    K: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Select the camera pose that places the most triangulated points in front
    of both cameras (cheirality condition).

    For each candidate pose [R|t], triangulates the 2D correspondences and
    counts points with positive depth in both camera frames. Camera A is
    assumed to sit at the world origin ([I|0]), so its depth is simply X[2].
    Camera B's depth is r_3 @ X + t_b[2] (third row of [R|t] applied to X).

    Args:
        candidates: list of (3, 4) [R|t] pose matrices for camera B.
        X_a: (N, 2) image points in camera A.
        X_b: (N, 2) image points in camera B.
        P1: (3, 4) projection matrix of camera A (K @ [I|0]).
        K: (3, 3) camera intrinsics.

    Returns:
        (best_pose, best_triangulated_pts), or (None, None) if no candidate
        places any point in front of both cameras.
    """
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
    best_pose, best_votes = None, 0
    best_pts = None
    for pose_candidate in candidates:
        P2 = K @ pose_candidate
        pts = triangulate_pts_dlt(X_a, X_b, P1, P2)
        # Cheirality check: the 3D point must lie in front of both
        # cameras, i.e. have positive depth in each camera frame.
        # Camera A sits at the world origin (pose_a = [I|0]), so its
        # depth is just X[2].
        in_front_a = pts[:, 2] > 0
        # Camera B has pose [R_b|t_b], meaning a world point X maps to
        # the camera-B frame as x_b = R_b X + t_b. The depth of X in
        # B's frame is therefore the third component: z_b = r_3 @ X +
        # t_b[2], where r_3 is the third row of R_b (B's optical axis).
        # NOTE: t_b is NOT the camera center. The camera center in
        # world coords is C = -R_b^T t_b (obtained by solving R_b C +
        # t_b = 0). The condition r_3@(X - C) > 0 from the literature
        # is algebraically equivalent to r_3@X + t_b[2] > 0, since
        # r_3@(X - C) = r_3@X - r_3@C = r_3@X - r_3@(-R_b^T t_b) =
        # r_3@X + (r_3 R_b^T) t_b = r_3@X + t_b[2] (because r_3 R_b^T
        # picks out the third row of R_b R_b^T = I, i.e. e_3^T).
        in_front_b = (pts @ pose_candidate[2, :3] + pose_candidate[2, 3]) > 0
        n_in_front = int((in_front_a & in_front_b).sum())
        if n_in_front > best_votes:
            best_pose = pose_candidate
            best_votes = n_in_front
            best_pts = pts
    return best_pose, best_pts


class EpipolarRANSAC:
    def __init__(
        self,
        frame_tuples: List[FrameTuple],
        consensus_ratio: float = 0.6,
        max_iter: int = 2000,
        threshold: float = 6,
        display: Optional["SfmDisplay"] = None,
        stats: Optional["PipelineStats"] = None,
    ):
        self.frame_tuples = frame_tuples
        self.frame_inliers = []
        self.frame_pair_F = []
        self.consensus_ratio = consensus_ratio
        self.max_iter = max_iter
        self.threshold = threshold
        self.display = display
        self.stats = stats

    def _compute_fundamental_matrix(
        self, kp1: np.ndarray, kp2: np.ndarray
    ) -> np.ndarray:
        # INFO: The fundamental matrix F \in R^{3x3} algebraically represents the epipolar
        # geometry that relates two sets of real-world point projections on two image
        # planes. F gives us the epipolar constraint in the form (x_{i}^{\prime})^T F x_i = 0
        return eight_point_fundamental_matrix(kp1, kp2)

    def filter(self) -> List[np.ndarray]:
        """
        Filter out outliers of a set of feature matches using RANSAC and the epipolar
        constraint as condition. For each set of candidate matches, we compute the
        Fundamental matrix and check for the epipolar constraint to be respected.

        Returns:
        inliers: List[(N,)] a list of boolean masks for the inliers of each frame
        """
        for f_tuple_idx, f_tuple in enumerate(self.frame_tuples):
            inliers, best_fit, best_fit_err, best_n_inliers = None, None, np.inf, 0
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
            if self.display and self.stats:
                self.display.begin_ransac(f_tuple_idx)
            if n_pts < 8:
                warnings.warn(
                    f"Frame tuple ({f_tuple.frame_a_id},{f_tuple.frame_b_id}) does not have enough features!"
                )
                continue
            # consensus_min = int(X_a.shape[0] * self.consensus_ratio)
            consensus_min = max(8, int(self.consensus_ratio * X_a.shape[0]))
            for it in range(self.max_iter):
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
                # err = np.abs((X_b_homo * (X_a_homo @ F.T)).sum(axis=1))
                # INFO: We use Sampson approximation instead for scale-independent
                # error:
                err = sampson_distance(F, X_a, X_b)
                this_inliers = err < self.threshold
                n_inliers = this_inliers.sum()
                # 4. The model is reasonably good if sufficiently many points are
                # classified as part of the consensus set.
                if n_inliers < consensus_min:
                    continue
                this_err = err[this_inliers].mean()

                if self.display and self.stats:
                    self.display.update_ransac(
                        f_tuple_idx, it + 1, int(n_inliers), float(this_err)
                    )

                if n_inliers > best_n_inliers or (
                    n_inliers == best_n_inliers and this_err < best_fit_err
                ):
                    best_fit = self._compute_fundamental_matrix(
                        X_a[this_inliers], X_b[this_inliers]
                    )
                    best_fit_err = this_err
                    best_n_inliers = n_inliers
                    inliers = this_inliers

            f_tuple.fundamental_matrix = best_fit
            f_tuple.inliers = int(best_n_inliers) if best_n_inliers > 0 else 0
            self.frame_inliers.append(inliers)
            if self.display and self.stats:
                ratio = f_tuple.inliers / n_pts if n_pts > 0 else 0
                self.display.finish_ransac_pair(
                    f_tuple_idx,
                    f_tuple.inliers,
                    ratio,
                    float(best_fit_err) if best_fit_err < np.inf else 0.0,
                )
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
