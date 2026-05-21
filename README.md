# SfM from scratch
My goal with this project is to strength my knowledge of 3D computer vision and finally
understand all about SLAM, SfM, and AR. I started this journey a long time ago, but
things kept getting in the way and I lacked the mathematical foundations to fully grasp
the concepts. I'm now coming back to it with years of computer vision, ML, and graphics
research, with the goal of building something fully functional for robotics or
entertainment in real-time applications.

## Roadmap
I will first implement a basic SfM pipeline from scratch in Python with Numpy. This is
to fully understand the subject and have a reference implementation.

### 1. Python prototypes

**SfM**
- [x] Feature Detection & Matching 
- [x] Estimating Fundamental Matrix (F)
- [x] RANSAC outlier rejection with F
- [x] Estimating Essential Matrix from Fundamental Matrix
- [ ] Estimate Camera Pose from Essential Matrix
- [ ] Check for Cheirality Condition using Triangulation
- [ ] Perspective-n-Point
- [ ] Bundle Adjustment

**SLAM**
We just go from there into modern graph-based methods.

### 2. Zig real-time 3D vision library
I want to recycle my previous C++ project (arlite) into -- finally -- an AR/SLAM/SfM
library with minimal dependencies. Once my python prototype is built, I can take that
along with what I built in C++ and put it all together in a Zig library for real-time
general-purpose AR/SLAM/SfM (CPU/GPU).

C++ part (the old bits):
- [x] libav video decoding (ffmpeg lib)
- [x] basic math library for image operations (conv, rot, gaussian blur, etc.)
- [x] FAST
- [x] BRIEF
- [ ] ORB <-- I stopped at the last bit
- [ ] Feature matching
- [ ] SfM

Python part (SfM prototype):
- [x] Opencv-based ORB detection and matching
- [x] Epipolar geometry-based match refinement with RANSAC
- [ ] Camera pose estimation
- [ ] Triangulation
- [ ] Bundle adjustment


Zig part (putting it all together):
- [ ] live video decoding
- [ ] ORB implementation
- [ ] SfM
- [ ] SLAM
- [ ] abstracted dispatch layer (CPU, GPU, others)
- [ ] real-time graphics rendering
- [ ] 3D Gaussian Splatting rendering
