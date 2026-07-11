# Current things I need to work on or improve

### Better bootstrapping!
The current problem that blocks the pipeline is having sufficient 2D-3D pairs for PnP.
We need to choose the pair with sufficient inliers and parallax for bootstrapping.

*So we need to change the architecture in favour of a covisibility graph*:
- [ ] **Generate candidate pairs**: match temporal neighbours + wider-baseline pairs
(i+2, i+5, etc).
- [ ] **Geometrically verify every edge**: filter by lowe's ratio, then by F/E RANSAC
hypothesis testing. Store verified inlier count, inlier ratio, median Sampson error, and parallax.
- [ ] **Choose the bootstrap edge**: require enough inliers but also large enough
parallax, because a low parallax will produce bad triangulation (near identical views
can't triangulate well). So reject pure rotation, tiny baseline and planar-dominated pairs.
- [ ] **Initialize**: structure bootstrapping.
- [ ] **Incremental registration**: choose the next frame with the most reliable 2D-3D
correspondences to the current 3D structure. Solve PnP RANSAC and triangulate.
- [ ] **Retry failed registrations later**: if PnP RANSAC gives bad results (inliers
below threshold), it's ok. We can try after we have registered more images and
triangulated more points. The thing is to properly keep track of 2D-3D correspondences.
- [ ] **Local BA frequently, global BA at the end**: jointly refine poses and points;
prune observations with high reprojection error.


Graph datastructure:
```python
@dataclass
class ImageNode:
    image_id: int
    features: ImageFeatures
    pose: np.ndarray | None = None       # [R | t] after registration
    registered: bool = False

@dataclass
class PairEdge:
    image_a: int                         # always image_a < image_b
    image_b: int
    matches: list[cv.DMatch]             # descriptor matches
    inlier_mask: np.ndarray | None       # aligned with matches
    F: np.ndarray | None
    E: np.ndarray | None
    inlier_count: int = 0
    median_sampson_px2: float = np.inf
    median_parallax_deg: float = 0.0
    usable_for_bootstrap: bool = False

    @property
    def inlier_matches(self) -> list[cv.DMatch]:
        assert self.inlier_mask is not None
        return [m for m, keep in zip(self.matches, self.inlier_mask) if keep]

@dataclass
class ViewGraph:
    nodes: dict[int, ImageNode]
    edges: dict[tuple[int, int], PairEdge]  # canonical key: (min(i,j), max(i,j))
    adjacency: dict[int, set[int]]

Observation = tuple[int, int]  # (image_id, keypoint_id)

@dataclass
class Track:
    id: int
    observations: dict[int, int]  # image_id -> keypoint_id
    point3d: np.ndarray | None = None
    reprojection_error: float = np.inf
```

-edges[(min(i, j), max(i, j))]: pair lookup and edge quality.
-adjacency[i]: candidate views for matching, triangulation, and propagation.
-observation_to_track: dict[Observation, int]: resolves (image_id, keypoint_id) to a track, allowing O(1) construction of PnP correspondences.
-tracks: dict[int, Track]: 3D landmark lifecycle and multi-view observations.

Merge tracks with a disjoint-set/union-find over observations while processing verified
inlier correspondences. Enforce the invariant that a track has at most one keypoint from
each image; skip a union that would violate it. This avoids creating invalid tracks from
conflicting pairwise matches.
