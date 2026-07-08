import warnings
from typing import List

import numpy as np
from tqdm import tqdm

from sfm import fqs
from sfm.epipolar_geometry import triangulate_pts_dlt
from sfm.data import FrameTuple, Structure
from sfm.utils import normalize


class PerspectiveNPoint:
    def __init__(
        self,
        frame_tuples: List[FrameTuple],
        K: np.ndarray,
        ransac_inlier_threshold: float = 0.01,
        ransac_iter: int = 1000,
    ) -> None:
        self.frame_tuples = frame_tuples
        assert K.shape == (3, 3)
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.max_iter_ransac = ransac_iter
        self.inlier_threshold = ransac_inlier_threshold

    def _solve_p3p(self, world_pts: np.ndarray, img_pts: np.ndarray):
        """
        This is the P3P implementation of Kneip et al, CVPR 2011 (https://rpg.ifi.uzh.ch/docs/CVPR11_kneip.pdf)
        """
        # TODO: Verify that all 3D and 2D points aren't colinear.
        assert world_pts.shape == (4, 3)  # TODO: Homogeneous?
        assert img_pts.shape == (4, 2)  # TODO: Homogeneous?
        p1, p2, p3 = world_pts[0], world_pts[1], world_pts[2]
        u1, u2, u3 = (
            np.concatenate([img_pts[0], np.ones((1,))]),
            np.concatenate([img_pts[1], np.ones((1,))]),
            np.concatenate([img_pts[2], np.ones((1,))]),
        )
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
        roots = fqs.quartic_roots(coeff)[0]
        solutions = []
        for i in range(4):
            cos_theta = roots[i]
            if abs(cos_theta.imag) > 1e-8:
                print(
                    f"[WARN] cos_theta.imag is too large: {cos_theta.imag}. SKIPPING!"
                )
                continue

            cos_theta = cos_theta.real

            if abs(cos_theta) > 1:
                print(f"[WARN] |cos_theta| > 1: {cos_theta}. SKIPPING!")
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
            R = T.T @ Q @ N
            # Cheirality check:
            if np.all(((world_pts - C) @ R[2, :].T) > 0):
                solutions.append((C, R))

        if len(solutions) > 1:
            # INFO: Disambiguate the 4 solutions using a 4th measurement
            p4, u4 = world_pts[3], img_pts[3]
            p4_homo = np.concatenate([p4, np.ones((1,))])
            solution, best_solution_err = None, np.inf
            for candidate in solutions:
                C, R = candidate
                t = -R @ C  # TODO: Correct??
                pose = np.hstack([R, t[:, None]])
                P = self.K @ pose
                cam_b_proj = p4_homo @ P.T  # (3)
                cam_b_proj = (cam_b_proj / cam_b_proj[-1])[:2]
                reproj_err = (cam_b_proj - u4).sum() ** 2
                if reproj_err < best_solution_err:
                    solution = pose
                    best_solution_err = reproj_err
        elif len(solutions) > 0:
            C, R = solutions[0]
            t = -R @ C  # TODO: Correct??
            solution = np.hstack([R, t[:, None]])
        else:
            warnings.warn("Could not find P3P solution")
            return None

        return solution

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
        # TODO: thread and SIMD with Numba
        for i in range(1, len(self.frame_tuples)):  # Start at the pair after bootstrap
            # 1. Compute correspondences to existing scene points via
            # matches to previous frame:
            inliers = inlier_observations[i]
            f_tpl = self.frame_tuples[i]
            matches_a = np.array(
                [m.queryIdx for m in f_tpl.frame_a_to_b_matches]
            )
            # matches_b = np.array([m.trainIdx for m in f_tpl.frame_a_to_b_matches])
            # Filter matches by the indices of those in common with the previous
            # pair:
            last_tpl = self.frame_tuples[i - 1]
            last_matches_b = np.array(
                [m.trainIdx for m in last_tpl.frame_a_to_b_matches]
            )
            last_inliers = inlier_observations[i - 1]
            in_common, comm1, comm2 = np.intersect1d(
                matches_a[inliers],
                last_matches_b[last_inliers],
                return_indices=True,
            )  # Matches (keypoint ids) that are common to both tuples
            corresp = {}
            # FIXME: right now correspondances are duplicated for each tuple! shouldn't
            # we merge them? Anyway, I'm not sure we need to keep track of them, since
            # we are computing them on the fly right now.
            # for j, k in enumerate(comm2):
            #     corresp[(i * 2 + 0, matches_a[inliers][k])] = j + last_frame_pts_offset
            #     corresp[(i * 2 + 1, matches_b[inliers][k])] = j + last_frame_pts_offset
            # structure.correspondences.update(corresp)

            # 2. Solve PnP (3 points + 1 to disambiguate):
            pbar = tqdm(
                range(self.max_iter_ransac),
                desc="Finding pose with RANSAC-P3P...",
            )
            cam_pose_a = structure.poses[i * 2 - 1]
            X_a = np.vstack(
                [np.array(f.pt) for f in f_tpl.frame_a_features.keypoints]
            )
            X_b = np.vstack(
                [np.array(f.pt) for f in f_tpl.frame_b_features.keypoints]
            )
            best_solution, best_inlier_count = None, 0
            points3D = None
            for _ in pbar:
                idx = np.random.choice(len(in_common), 4, replace=False)
                # INFO: Sample 3D-2D correspondences foicr camera B:
                # pts2D_id = list(corresp.keys())[idx][1]  # (cam_id, kp_id)
                pts2D = X_b[comm2[idx]]
                pts3D = structure.points3D[last_frame_pts_offset + idx]
                cam_pose_b = self._solve_p3p(pts3D, pts2D)
                if cam_pose_b is None:
                    continue
                P1 = self.K @ cam_pose_a
                P2 = self.K @ cam_pose_b
                points3D = triangulate_pts_dlt(
                    X_a[comm1], X_b[comm2], P1, P2
                )
                x_homo = np.concatenate(
                    [points3D, np.ones((points3D.shape[0], 1))], axis=1
                )

                u, v = X_b[comm2].T
                cam_b_proj = x_homo @ P2.T  # (N, 3)
                cam_b_proj = (cam_b_proj / cam_b_proj[:, -1][:, None])[:, :2]
                proj_errors = (
                    (u - cam_b_proj[:, 0]) ** 2
                    + (v - cam_b_proj[:, 1]) ** 2
                )
                inlier_mask = proj_errors < self.inlier_threshold
                if inlier_mask.sum() > best_inlier_count:
                    best_solution = cam_pose_b
                    best_inlier_count = inlier_mask.sum()
            if points3D is None or best_solution is None:
                raise ValueError("Could not solve camera pose via P3P!")

            structure.points3D = np.vstack([structure.points3D, points3D])
            structure.poses[i * 2 + 1] = best_solution
            f_tpl.cam_pose_a = cam_pose_a
            f_tpl.cam_pose_b = best_solution
            last_frame_pts_offset += len(in_common)
