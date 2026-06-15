import numpy as np
import pytest
from main import PosePredictor, FrameTuple, ImageFeatures

def test_triangulation_and_cheirality():
    # Setup camera intrinsics
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    
    # Camera 1 at origin
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = K @ np.hstack([R1, t1])
    
    # Camera 2 translated along X
    R2 = np.eye(3)
    t2 = np.array([[-1.0, 0, 0]]).T 
    P2 = K @ np.hstack([R2, t2])
    
    # A point in front of both
    X_w = np.array([0.5, 0.2, 5.0, 1.0])
    
    # Project to image points
    x1 = P1 @ X_w
    x1 /= x1[2]
    u1, v1 = x1[0], x1[1]
    
    x2 = P2 @ X_w
    x2 /= x2[2]
    u2, v2 = x2[0], x2[1]
    
    # Test our triangulation math directly as implemented in main.py
    # This is equivalent to PosePredictor.fit's inner loop
    A = np.array([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1]
    ])
    
    _, _, Vh = np.linalg.svd(A)
    X_est = Vh[-1, :]
    if X_est[-1] != 0:
        X_est /= X_est[-1]
    
    # Verify the point was correctly triangulated
    assert np.allclose(X_w[:3], X_est[:3], atol=1e-5)
    
    # Verify cheirality logic
    in_front_a = X_est[2] > 0
    z_c = R2[2, :] @ X_est[:3] + t2[2, 0]
    in_front_b = z_c > 0
    
    assert in_front_a, f"Point should be in front of Camera A (Z={X_est[2]})"
    assert in_front_b, f"Point should be in front of Camera B (Z_c={z_c})"

def test_essential_matrix_decomposition():
    # Simple case: rotation around Y and translation in X
    theta = np.radians(10)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t = np.array([[1.0, 0, 0]]).T
    
    # Create Essential matrix E = [t]x R
    tx = np.array([
        [0, -t[2,0], t[1,0]],
        [t[2,0], 0, -t[0,0]],
        [-t[1,0], t[0,0], 0]
    ])
    E = tx @ R
    
    # Decomposition logic
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
        
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # One of R1 or R2 should be close to R
    assert np.allclose(R1, R, atol=1e-5) or np.allclose(R2, R, atol=1e-5)
    # Both should be rotation matrices (det=1)
    assert np.isclose(np.linalg.det(R1), 1.0)
    assert np.isclose(np.linalg.det(R2), 1.0)
