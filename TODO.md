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
