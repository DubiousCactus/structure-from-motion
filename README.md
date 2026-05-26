# SfM from scratch
My goal with this project is to strengthen my knowledge of 3D computer vision and finally
understand all about SLAM, SfM, and AR. I started this journey a long time ago, but
things kept getting in the way and I lacked the mathematical foundations to fully grasp
the concepts. I'm now coming back to it with years of computer vision, ML, and graphics
research, with the goal of building something fully functional for robotics or
entertainment in real-time applications.

## Roadmap
I will first implement a basic SfM pipeline from scratch in Python with Numpy. This is
to fully understand the subject and have a reference implementation.

### 1. Python prototypes (learning the fundamentals)

**SfM**
- [x] Feature Detection & Matching 
- [x] Estimating Fundamental Matrix (F)
- [x] RANSAC outlier rejection with F
- [x] Estimating Essential Matrix from Fundamental Matrix
- [x] Estimate Camera Pose from Essential Matrix
- [x] Check for Cheirality Condition using Triangulation
- [ ] Point refinement via non-linear triangulation
- [ ] Perspective-n-Point
- [ ] Bundle Adjustment

See my [notes](./NOTES.md).

**SLAM**
We just go from there into modern graph-based methods.

### 2. Zig real-time 3D vision library
I want to recycle [my previous C++ project](github.com/DubiousCactus/retina) into --
finally -- an AR/SLAM/SfM library with minimal dependencies. Once my python prototype is
built, I can take that along with what I built in C++ and put it all together in a Zig
library for real-time general-purpose AR/SLAM/SfM (CPU/GPU).

C++ part (the old bits) (from [my old project](github.com/DubiousCactus/retina)):
- [x] libav video decoding (ffmpeg lib)
- [x] basic math library for image operations (conv, rot, gaussian blur, etc.)
- [x] FAST
- [x] BRIEF
- [ ] ORB <-- I stopped at the last bit
- [ ] Feature matching
- [ ] SfM

Python part (SfM prototype, [this project](https://github.com/DubiousCactus/structure-from-motion)):
- [x] Opencv-based ORB detection and matching
- [x] Epipolar geometry-based match refinement with RANSAC
- [x] Camera pose estimation
- [x] Triangulation
- [x] Non-linear triangulation for point refinement
- [ ] PnP / UPnP
- [ ] Bundle adjustment


Zig part (putting it all together):
- [ ] live video decoding
- [ ] ORB implementation
- [ ] SfM
- [ ] SLAM
- [ ] abstracted dispatch layer (CPU, GPU, others)
- [ ] real-time graphics rendering
- [ ] 3D Gaussian Splatting rendering

# Literature
Here are the resources I've used for this project:
- [sfm course notes](https://cmsc426.github.io/sfm/#fundmatrix)
- [Epipolar Geometry and the Fundamental Matrix, Hartley and Zisserman](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook1/HZepipolar.pdf)
- [5-point motion estimation made easy, Li and Hartley](https://users.cecs.anu.edu.au/~hongdong/new5pt_cameraREady_ver_1.pdf-)
- [Foundations of Computer Vision](https://visionbook.mit.edu/)
