import os
from typing import TYPE_CHECKING, List, Optional

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from sfm.bootstrapping import StructureBootstrap
from sfm.bundle_adjustment import BundleAdjustment
from sfm.data import CameraDatabase, FrameTuple, ImageFeatures
from sfm.epipolar_geometry import EpipolarRANSAC
from sfm.pnp import PerspectiveNPoint

if TYPE_CHECKING:
    from sfm.tui import SfmDisplay


def match_orb_descriptors(
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    lowe_ratio: float,
) -> tuple[int, int, list[cv.DMatch]]:
    """Match ORB descriptors with exact Hamming distance and reciprocal Lowe filtering."""
    if descriptors_a is None or descriptors_b is None:
        return 0, 0, []

    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    forward_knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse_knn = matcher.knnMatch(descriptors_b, descriptors_a, k=2)

    forward_lowe = [
        pair[0]
        for pair in forward_knn
        if len(pair) == 2 and pair[0].distance < lowe_ratio * pair[1].distance
    ]
    reverse_lowe = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse_knn
        if len(pair) == 2 and pair[0].distance < lowe_ratio * pair[1].distance
    }
    mutual_matches = [
        match
        for match in forward_lowe
        if reverse_lowe.get(match.trainIdx) == match.queryIdx
    ]
    return len(forward_knn), len(forward_lowe), mutual_matches


def extract_frames_impl(
    video_path: str,
    output_folder: str,
    display: Optional["SfmDisplay"] = None,
):
    cap = cv.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if display:
        display.begin_extraction(total)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
        cv.imwrite(frame_filename, frame)
        frame_count += 1
        if display:
            display.update_extraction(frame_count, 0)
    cap.release()
    if display:
        display.finish_extraction()


def extract_and_match_impl(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
    display: Optional["SfmDisplay"] = None,
    orb_features: int = 2000,
    lowe_ratio: float = 0.8,
):
    if intrinsics_path is not None and not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found at {intrinsics_path}")
    if orb_features < 1:
        raise ValueError("orb_features must be positive")
    if not 0.0 < lowe_ratio < 1.0:
        raise ValueError("lowe_ratio must be between 0 and 1")
    # INFO: Stage 1: feature extraction using ORB
    orb = cv.ORB_create(nfeatures=orb_features, scaleFactor=1.2, nlevels=8)
    frame_files = sorted(os.listdir(frames_path))
    total_frames = len(frame_files) if max_frames is None else min(max_frames, len(frame_files))
    if display:
        display.begin_extraction(total_frames)
    frame_features = []
    for i, frame_file in enumerate(frame_files):
        if i == max_frames:
            break
        frame_path = os.path.join(frames_path, frame_file)
        img = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        frame_features.append(ImageFeatures(kp, des, frame_path))
        if display:
            display.update_extraction(i, len(kp))
        if debug and i == 0:
            img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()
    if display:
        display.finish_extraction()

    # INFO: Stage 2: exact Hamming KNN matching with reciprocal Lowe filtering.
    frame_tuples = []
    total_pairs = len(frame_features) - 1
    if display:
        display.begin_matching(total_pairs)

    for i in range(1, len(frame_features)):
        f1, f2 = frame_features[i - 1], frame_features[i]
        knn_queries, lowe_matches, mutual_matches = match_orb_descriptors(
            f1.descriptors, f2.descriptors, lowe_ratio
        )
        frame_tuples.append(FrameTuple(i - 1, i, f1, f2, mutual_matches))
        if display:
            from sfm.tui import PairStats

            display.update_matching(
                i - 1,
                PairStats(
                    pair_label=f"({i - 1},{i})",
                    features_a=len(f1.keypoints),
                    features_b=len(f2.keypoints),
                    knn_queries=knn_queries,
                    lowe_matches=lowe_matches,
                    mutual_matches=len(mutual_matches),
                ),
            )

    # INFO: Stage 3: RANSAC outlier removal via the epipolar constraint.
    ransac = EpipolarRANSAC(frame_tuples, display=display, stats=display.stats if display else None)
    inliers: List[np.ndarray] = ransac.filter()
    if debug:
        ransac.draw_matches()
    assert all(
        [isinstance(f_tpl.fundamental_matrix, np.ndarray) for f_tpl in frame_tuples]
    ), (
        "Fundamental matrix not computed for all frames during RANSAC. "
        + f"Got F list: {[f.fundamental_matrix for f in frame_tuples]}"
    )

    # INFO: Stage 4: 2D-2D Camera pose prediction via Essential matrix decomposition and
    # point triangulation. The 3D points are a by-product of computing the pose from
    # decomposing the Essential matrix, and filtering the valid pose via the Cheirality
    # condition. Here, we only bootstrap the 3D structure.
    if intrinsics_path is None:
        # TODO: Estimate the focal length from the hommography
        # (https://imkaywu.github.io/blog/2017/10/focal-from-homography/), but this
        # assumes that the two camera centers are fixed and the caameras only undergo
        # rotations. In practice, it seems everyone uses UPnP
        # (https://openreview.net/pdf?id=PbMNl2kC0u) or a flavour of PnPf (www.researchgate.net/publication/354289451_Efficient_DLT-Based_Method_for_Solving_PnP_PnPf_and_PnPfr_Problems)
        raise NotImplementedError(
            "Intrinsics estimation not implemented yet! Please provide the intrinsics matrix"
            + " by running the extract-intrinsics command."
        )
    else:
        cam_db = CameraDatabase.load(intrinsics_path)
    # WARN: How about scale ambiguity that comes with E-decomposition (2D-2D
    # correspondances and pose prediction)? Well, unfortunately that's just a thing of
    # monocular SfM. We just *can't recover absolute scale from images alone*. However,
    # chaining E-decomposition for each new camera *will lead to scale drift*. To remedy
    # this, we use PnP!

    bootstrap = StructureBootstrap(frame_tuples, cam_db, display=display, stats=display.stats if display else None)
    structure = bootstrap.init(inliers)
    if debug:
        bootstrap.draw_triangulation_3D()
    # INFO: Stage 5: register all images and solve all camera poses using PnP. Given the
    # initial 3D points, register each new image in the scene using 3D-2D
    # correspondances.
    pnp = PerspectiveNPoint(frame_tuples, cam_db, display=display, stats=display.stats if display else None)
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
