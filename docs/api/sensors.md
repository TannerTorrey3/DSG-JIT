# Sensor Modules

High-level sensor support for DSG-JIT. This package provides:

- **Lightweight measurement containers**: camera, LiDAR, IMU.
- **Streaming abstractions** (`sensors.streams`) for synthetic or file-based data.
- **Conversion utilities** (`sensors.conversion`) from raw samples → DSG-JIT measurements.
- **Sensor fusion manager** (`sensors.fusion.SensorFusionManager`) that:
  - Polls one or more sensor streams.
  - Converts them into measurements.
  - Optionally forwards them to `WorldModel` / `SceneGraphWorld` helpers.

---

## `sensors.base`

::: sensors.base

---

## `sensors.camera`

::: sensors.camera

---

## `sensors.conversion`

::: sensors.conversion

---

## `sensors.fusion`

::: sensors.fusion

---

## `sensors.imu`

::: sensors.imu

---

## `sensors.integration`

::: sensors.integration

---

## `sensors.lidar`

::: sensors.lidar

---

## `sensors.streams`

::: sensors.streams

---