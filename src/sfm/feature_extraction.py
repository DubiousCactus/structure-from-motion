import os
from typing import List, Optional

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sfm.bootstrapping import StructureBootstrap
from sfm.bundle_adjustment import BundleAdjustment
from sfm.data import CameraDatabase, FrameTuple, ImageFeatures
from sfm.epipolar_geometry import EpipolarRANSAC
from sfm.pnp import PerspectiveNPoint


def extract_frames_impl(video_path: str, output_folder: str):
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


def extract_and_match_impl(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
):
    if intrinsics_path is not None and not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found at {intrinsics_path}")
    # INFO: Stage 1: feature extraction using ORB
    orb = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    # surf = cv.xfeatures2d.SURF_create()
    frame_features = []
    for i, frame_file in tqdm(
        enumerate(sorted(os.listdir(frames_path))), desc="Extracting features"
    ):
        if i == max_frames:
            break
        frame_path = os.path.join(frames_path, frame_file)
        img = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        # kp = orb.detect(img, None)
        # kp, des = orb.compute(img, kp)
        kp, des = orb.detectAndCompute(img, None)
        # kp, des = surf.detectAndCompute(img, None)
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
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        frame_tuples.append(FrameTuple(i - 1, i, f1, f2, good_matches))

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

    bootstrap = StructureBootstrap(frame_tuples, cam_db)
    structure = bootstrap.init(inliers)
    if debug:
        bootstrap.draw_triangulation_3D()
    # INFO: Stage 5: register all images and solve all camera poses using PnP. Given the
    # initial 3D points, register each new image in the scene using 3D-2D
    # correspondances.
    pnp = PerspectiveNPoint(frame_tuples, cam_db)
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
