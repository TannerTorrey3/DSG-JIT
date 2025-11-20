

---
# Examples

This page provides **practical, end-to-end usage examples** for DSG-JIT — from constructing simple factor graphs to running differentiable optimizers, voxel pipelines, scene graphs, and hybrid SE3–voxel learning.

Each example is designed to run *as-is* inside your project using:

```bash
PYTHONPATH=dsg-jit/src python your_script.py
```

---

# 1. Minimal Example: SE(3) Odom Chain

```python
import jax.numpy as jnp
from core.types import Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import se3_additive_residual
from optimization.solvers import gauss_newton

fg = FactorGraph()
fg.register_residual("odom_se3_add", se3_additive_residual)

for i in range(3):
    fg.add_variable(Variable(id=f"pose{i}", value=jnp.zeros((6,), dtype=jnp.float32)))

fg.add_factor(Factor(
    id="f0", type="odom_se3_add",
    var_ids=["pose0", "pose1"],
    params={"measurement": jnp.array([1., 0, 0, 0, 0, 0])},
))
fg.add_factor(Factor(
    id="f1", type="odom_se3_add",
    var_ids=["pose1", "pose2"],
    params={"measurement": jnp.array([1., 0, 0, 0, 0, 0])},
))

x0, index = fg.pack_state()
objective = fg.build_objective()

x_opt = gauss_newton(objective, x0, max_iters=20)
poses = fg.unpack_state(x_opt, index)

print("Optimized poses:", poses)
```

---

# 2. Voxel Chain Optimization

```python
import jax.numpy as jnp
from core.types import Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import voxel_smoothness_residual
from optimization.solvers import gauss_newton

fg = FactorGraph()
fg.register_residual("voxel_smooth", voxel_smoothness_residual)

N = 10
for i in range(N):
    fg.add_variable(Variable(id=f"v{i}", value=jnp.array([float(i), 0., 0.])))

for i in range(N - 1):
    fg.add_factor(Factor(
        id=f"s{i}",
        type="voxel_smooth",
        var_ids=[f"v{i}", f"v{i+1}"],
        params={"offset": jnp.array([1., 0., 0.]), "weight": 1.0},
    ))

x0, index = fg.pack_state()
objective = fg.build_objective()
x_opt = gauss_newton(objective, x0)

voxels = fg.unpack_state(x_opt, index)
print(voxels)
```

---

# 3. Learnable Type Weights (log-scale training)

```python
import jax
import jax.numpy as jnp
from world.training import DSGTrainer

trainer = DSGTrainer(fg)
log_scales = jnp.zeros((1,))

loss_fn = trainer.build_type_weight_loss(["odom_se3_add"])
grad_fn = jax.grad(loss_fn, argnums=1)

for step in range(50):
    loss = loss_fn(x, log_scales)
    g = grad_fn(x, log_scales)
    log_scales -= 0.01 * g
```

---

# 4. Voxel Point Observation

```python
from slam.measurements import voxel_point_observation_residual

fg = FactorGraph()
fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

fg.add_variable(Variable(id="pose", value=jnp.zeros((6,))))
fg.add_variable(Variable(id="voxel", value=jnp.array([0., 0., 0.])))

fg.add_factor(Factor(
    id="obs0",
    type="voxel_point_obs",
    var_ids=["pose", "voxel"],
    params={"point_world": jnp.array([0.9, 0., 0.]), "weight": 1.0},
))
```

---

# 5. Scene Graph Example

```python
from world.scene_graph import SceneGraph
from optimization.solvers import gauss_newton

sg = SceneGraph()
p0 = sg.add_pose("p0", jnp.zeros((6,)))
p1 = sg.add_pose("p1", jnp.zeros((6,)))

sg.add_odom_se3_additive(p0, p1, dx=1.0)
fg = sg.to_factor_graph()

objective = fg.build_objective()
x0, index = fg.pack_state()
x_opt = gauss_newton(objective, x0)
```

---

# 6. Hybrid SE3 + Voxel Learning (DSGTrainer)

```python
import jax
import jax.numpy as jnp
from world.training import DSGTrainer
from optimization.solvers import gauss_newton

trainer = DSGTrainer(fg)
theta = trainer.init_theta(fg)

x0, index = fg.pack_state()
loss_fn = trainer.build_joint_hybrid_loss()
grad_fn = jax.grad(loss_fn)

for epoch in range(30):
    x0 = trainer.solve_state(x0, theta)
    g = grad_fn(x0, theta)
    theta = trainer.update_theta(theta, g)
```

---

# 7. Simple Visualization Example

```python
import matplotlib.pyplot as plt

poses = [fg.unpack_state(x_opt, index)[f"pose{i}"][0] for i in range(N)]
plt.plot(poses, marker="o")
plt.title("Optimized Trajectory")
plt.show()
```

---

# 8. Full Pipeline Example

```python
from world.scene_graph import SceneGraph
from world.training import DSGTrainer

sg = SceneGraph()
poses, voxels = sg.add_hybrid_chain(num_poses=50, num_voxels=500)
fg = sg.to_factor_graph()

trainer = DSGTrainer(fg)
theta = trainer.init_theta(fg)
x0, index = fg.pack_state()

for epoch in range(30):
    x0 = trainer.solve_state(x0, theta)
    theta = trainer.update_theta(theta, trainer.grad_theta(x0, theta))
```

---
