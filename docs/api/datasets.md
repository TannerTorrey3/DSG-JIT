# Datasets

DSG-JIT includes light-weight loaders for common SLAM / VO datasets so you can
quickly hook **real sequences** into the sensor stack, factor graph, and
dynamic scene graph.

The goal is:

- No heavy dependencies (no OpenCV required just to list frames).
- Simple dataclasses with timestamps + file paths.
- Easy integration with `sensors.*` streams and `world.*` components.

Currently supported:

- [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset)
- [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/)

---

## `datasets.tum_rgbd`

::: datasets.tum_rgbd

---

## `datasets.kitti_odometry`

::: datasets.kitti_odometry

---