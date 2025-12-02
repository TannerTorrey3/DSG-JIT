# Tutorial: Sensor Fusion Sandbox - Camera, LiDAR, and IMU Streams

**Categories:** Sensors & Fusion

## Overview

This tutorial walks through a minimal **sensor–fusion sandbox** built on top of DSG‑JIT’s `sensors.*` layer.  
The goal is to show how to:

- Define simple **synthetic sensor streams** for a camera, LiDAR, and IMU.
- Wrap them in `FunctionStream` objects so they look like real hardware.
- Register each stream with a `SensorFusionManager`.
- Attach a callback (`ToyImuIntegrator`) that consumes IMU measurements and produces a toy 1D fused pose.
- Inspect the fused pose and visualize the integrated trajectory.

This experiment focuses purely on the **sensor layer**; it does not yet create SLAM factors or update a `WorldModel` / `SceneGraphWorld`. Think of it as a clean sandbox for getting comfortable with DSG‑JIT’s sensor APIs before wiring them into the rest of the stack.

---

## 1. Synthetic Sensor “Hardware”

Instead of reading from real devices or log files, the experiment defines three **generator-style read functions**:

- `make_camera_read_fn(landmark_ids)`
- `make_lidar_read_fn()`
- `make_imu_read_fn()`

Each one returns a `read()` callable that behaves like a simple hardware driver:

- Every call to `read()` returns a **raw sample dictionary**.
- When the underlying sequence is exhausted (for the camera), it returns `None` to signal **end-of-stream**.

```python
def make_camera_read_fn(landmark_ids):
    """
    Return a generator-style read() function that yields synthetic 
    bearing measurements compatible with raw_sample_to_camera_measurement.
    """
    samples = [
        {
            "t": 0.0,
            "frame_id": 0,
            "bearings": jnp.array(
                [
                    [1.0, 0.0, 0.0],    # bearing to landmark 0
                    [1.0, 0.1, 0.0],    # bearing to landmark 1
                ],
                dtype=jnp.float32,
            ),
            "landmark_ids": landmark_ids,
            "sensor_pose": jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32),
        },
        {
            "t": 1.0,
            "frame_id": 1,
            "bearings": jnp.array(
                [
                    [1.0, 0.05, 0.0],
                    [1.0, 0.15, 0.0],
                ],
                dtype=jnp.float32,
            ),
            "landmark_ids": landmark_ids,
            "sensor_pose": jnp.array([0.5, 0.0, 0, 0, 0, 0], dtype=jnp.float32),
        },
    ]

    it = iter(samples)

    def read():
        try:
            return next(it)
        except StopIteration:
            return None  # signals end of stream

    return read
```

Key idea: the returned dictionaries are **raw samples**, not `CameraMeasurement` objects yet. They are shaped so that they can be handed to the camera converter:

```python
from dsg_jit.sensors.conversion import raw_sample_to_camera_measurement
```

The LiDAR and IMU streams follow the same pattern:

- **LiDAR**: emits a scan at 10 Hz (`t += 0.1`) with fixed ranges ≈ 5 m and angles from –45 to +45 degrees.
- **IMU**: emits at 20 Hz (`t += 0.05`) with:
  - constant acceleration `a = [0.5, 0, 0]` m/s²,
  - zero angular velocity,
  - and a scalar `dt = 0.05`.

```python
def make_lidar_read_fn() -> callable:
    t = 0.0
    def read() -> Dict[str, Any]:
        nonlocal t
        t += 0.1  # 10 Hz
        num_beams = 16
        angles = jnp.linspace(-math.pi / 4, math.pi / 4, num_beams)
        ranges = 5.0 * jnp.ones_like(angles, dtype=jnp.float32)
        return {
            "t": t,
            "frame_id": "lidar0",
            "angles": angles,
            "ranges": ranges,
            "rays": jnp.stack([angles, ranges], axis=1),
        }
    return read
```

```python
def make_imu_read_fn() -> callable:
    t = 0.0
    def read() -> Dict[str, Any]:
        nonlocal t
        t += 0.05  # 20 Hz
        dt = 0.05
        accel = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32)
        gyro = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        return {
            "t": t,
            "dt": dt,
            "frame_id": "imu0",
            "accel": accel,
            "gyro": gyro,
        }
    return read
```

These functions are deliberately simple: they isolate the **stream interface** (a `read()` function) from the rest of the system.

---

## 2. Wrapping Streams with `FunctionStream`

DSG‑JIT represents a sensor stream as an object that it can poll for new samples. For synthetic streams, the `FunctionStream` wrapper is perfect:

```python
from dsg_jit.sensors.streams import FunctionStream

landmark_ids = [0, 1]
cam_stream = FunctionStream(make_camera_read_fn(landmark_ids=landmark_ids))
lidar_stream = FunctionStream(make_lidar_read_fn())
imu_stream = FunctionStream(make_imu_read_fn())
```

Each `FunctionStream` holds onto the underlying `read()` function and exposes a standard interface that `SensorFusionManager` knows how to call.

If you later want to replace synthetic data with recorded logs, you can swap `FunctionStream` for something like `FileRangeStream` without touching the rest of the experiment.

---

## 3. From Raw Samples to Measurements

The raw dictionaries produced by the `read()` functions are converted to typed measurements via functions in `sensors.conversion`:

- `raw_sample_to_camera_measurement`
- `raw_sample_to_lidar_measurement`
- `raw_sample_to_imu_measurement`

These converters know how to interpret keys like `"t"`, `"frame_id"`, `"bearings"`, `"ranges"`, `"accel"`, `"gyro"`, and `dt`, and they produce instances of:

- `CameraMeasurement`
- `LidarMeasurement`
- `IMUMeasurement`

The fusion manager takes a **stream + converter** pair for each sensor:

```python
from dsg_jit.sensors.fusion import SensorFusionManager
from dsg_jit.sensors.conversion import (
    raw_sample_to_camera_measurement,
    raw_sample_to_lidar_measurement,
    raw_sample_to_imu_measurement,
)

fusion = SensorFusionManager()

fusion.register_sensor(
    name="cam0",
    modality="camera",
    stream=cam_stream,
    converter=raw_sample_to_camera_measurement,
)
fusion.register_sensor(
    name="lidar0",
    modality="lidar",
    stream=lidar_stream,
    converter=raw_sample_to_lidar_measurement,
)
fusion.register_sensor(
    name="imu0",
    modality="imu",
    stream=imu_stream,
    converter=raw_sample_to_imu_measurement,
)
```

Why this step? It separates **I/O concerns** (reading dictionaries from a file or device) from **measurement semantics** (what fields are present and how to interpret them for SLAM, fusion, or learning).

---

## 4. SensorFusionManager and Callbacks

Once streams and converters are registered, the fusion manager becomes the hub for **polling measurements** and dispatching them to callbacks.

```python
# Pseudo-code shape:
class SensorFusionManager:
    def register_sensor(self, name, modality, stream, converter):
        ...
    def register_callback(self, callback):
        ...
    def poll_once(self) -> int:
        # 1) poll each stream at most one sample
        # 2) convert raw samples to measurements
        # 3) invoke callbacks(meas) for all new measurements
        # 4) return the total number of new measurements
```

In this experiment, we attach a single callback: `ToyImuIntegrator`.

```python
integrator = ToyImuIntegrator(fusion)
fusion.register_callback(integrator)
```

Every time `poll_once()` retrieves a new `IMUMeasurement`, the integrator is called and updates a toy 1D pose estimate.

---

## 5. ToyImuIntegrator: A Minimal 1D IMU Filter

The `ToyImuIntegrator` class demonstrates how to **consume IMU measurements**, maintain internal state, and optionally publish a fused pose back into the fusion manager.

```python
class ToyImuIntegrator:
    """
    Very simple 1D integrator using IMU measurements.
    This is *not* a real inertial filter; it is just a toy example.
    """
    def __init__(self, fusion: SensorFusionManager) -> None:
        self.fusion = fusion
        self.x = 0.0      # position along x
        self.vx = 0.0     # velocity along x
        self.history_t: List[float] = []
        self.history_x: List[float] = []

    def __call__(self, meas: BaseMeasurement) -> None:
        if isinstance(meas, IMUMeasurement):
            a_x = float(meas.accel[0])
            dt = float(meas.dt)

            # Basic integration: v += a*dt, x += v*dt
            self.vx += a_x * dt
            self.x += self.vx * dt

            self.history_t.append(float(meas.timestamp))
            self.history_x.append(self.x)

            pose_se3 = jnp.array(
                [self.x, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=jnp.float32,
            )
            self.fusion.record_fused_pose(
                t=meas.timestamp,
                pose_se3=pose_se3,
                covariance=None,
                source_counts={"imu": 1},
            )

        elif isinstance(meas, CameraMeasurement):
            # Log camera info (timestamp + image/bearing shape)
            ...
        elif isinstance(meas, LidarMeasurement):
            # Log LiDAR mean range
            ...
```

A few details to note:

- The callback is **polymorphic**: it inspects the type of `meas` and decides what to do.
- For IMU data, it performs a **simplified double integration** along the x-axis and appends the result to a history buffer.
- It uses `fusion.record_fused_pose(...)` to publish a fused pose so it can be retrieved later via `fusion.get_latest_pose()`.

For camera and LiDAR measurements, this demo just prints a brief summary:

- Camera: timestamp and image (or bearing) shape.
- LiDAR: timestamp and mean range.

In a more advanced setup, this is exactly where you would add **DSG factors** (e.g., range factors, bearing factors, occupancy updates) or log them for later learning.

---

## 6. Running the Simulation Loop

The main function wires everything together and runs a short simulation:

```python
def main() -> None:
    # 1) Build streams
    landmark_ids = [0, 1]
    cam_stream = FunctionStream(make_camera_read_fn(landmark_ids=landmark_ids))
    lidar_stream = FunctionStream(make_lidar_read_fn())
    imu_stream = FunctionStream(make_imu_read_fn())

    # 2) Construct fusion manager and register sensors
    fusion = SensorFusionManager()
    fusion.register_sensor("cam0", "camera", cam_stream, raw_sample_to_camera_measurement)
    fusion.register_sensor("lidar0", "lidar", lidar_stream, raw_sample_to_lidar_measurement)
    fusion.register_sensor("imu0", "imu", imu_stream, raw_sample_to_imu_measurement)

    # 3) Attach toy integrator
    integrator = ToyImuIntegrator(fusion)
    fusion.register_callback(integrator)

    # 4) Run a short simulation loop
    num_steps = 100
    for step in range(num_steps):
        n_meas = fusion.poll_once()
        if n_meas == 0:
            # In a real app, you might break when all streams are exhausted.
            pass

    # 5) Inspect latest fused pose
    fused = fusion.get_latest_pose()
    if fused is not None:
        print("\n=== Latest fused pose (toy IMU integrator) ===")
        print(f"t = {fused.t}")
        print(f"pose_se3 = {fused.pose_se3}")
        print(f"source_counts = {fused.source_counts}")
    else:
        print("No fused pose produced.")
```

The key call here is:

```python
n_meas = fusion.poll_once()
```

On each iteration, `SensorFusionManager`:

1. Polls each registered `stream` at most once.
2. Converts any new raw samples into measurement objects via the provided converters.
3. Calls all registered callbacks with each measurement.

You can control the simulated duration either by `num_steps` or by breaking when `n_meas == 0` consistently.

---

## 7. Visualizing the 1D Trajectory

The integrator stores time–position pairs inside `history_t` and `history_x`. At the end of the run, the experiment uses Matplotlib to plot the estimated trajectory:

```python
if integrator.history_t:
    import matplotlib.pyplot as plt

    ts = jnp.array(integrator.history_t)
    xs = jnp.array(integrator.history_x)

    plt.figure()
    plt.plot(ts, xs, marker="o")
    plt.xlabel("time [s]")
    plt.ylabel("x position [m]")
    plt.title("Toy IMU-integrated 1D trajectory")
    plt.grid(True)
    plt.show()
```

Because the acceleration is constant, the true motion would be a **quadratic curve** in time. The simple integrator tracks this approximate shape, giving you immediate visual feedback that the IMU callback is wired correctly.

---

## 8. Where to Go Next

This sensor fusion sandbox is intentionally minimal. Once you are comfortable with:

- constructing streams,
- converting raw samples into measurements,
- using `SensorFusionManager`,
- and writing callbacks like `ToyImuIntegrator`,

you are ready to:

- Attach **measurement factors** into a `WorldModel` (e.g., range priors, bearing factors).
- Use fused poses as the backbone for **SceneGraphWorld** or **DynamicSceneGraph** trajectories.
- Swap synthetic streams with **real datasets or ROS2 bridges**.
- Extend the callback to maintain a full **state estimator** instead of a toy 1D integrator.

This experiment is meant to be your “hello world” for DSG‑JIT’s sensor fusion layer—simple enough to hack on quickly, but structured in the same way you would handle real sensor data in more advanced SLAM and learning setups.
