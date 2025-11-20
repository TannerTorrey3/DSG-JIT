# Contributors

This project is built around the idea that differentiable SLAM and 3D Dynamic Scene Graphs
should be accessible, hackable, and research-friendly. Thank you to everyone who helps
push DSG-JIT forward.

## Core Author

- **Tanner Kocher** – project founder and primary author of the DSG-JIT engine,
  including the Gauss–Newton solvers, SE(3) manifold utilities, voxel grid factors,
  scene graph world model, experiments, and benchmarks.

## How to Become a Contributor

Contributions are very welcome, especially around:

- New factor types (e.g., additional sensor models, semantic factors, loop closures)
- New experiments (hybrid SLAM setups, real datasets, ablations)
- Performance improvements (better JAX patterns, batching, GPU support)
- Documentation, tutorials, and example notebooks
- Packaging (PyPI, conda, Docker images)

### Basic Contribution Workflow

1. **Fork** the repository on GitHub.  
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-awesome-idea
   ```
3. Make your changes, following the project's coding style (type hints, docstrings, and tests).
4. Run the full test suite:
   ```bash
   pytest
   ```
5. (Optional) Run benchmarks and report any observed performance changes.
6. Commit and push your branch:
   ```bash
   git commit -am "Add my awesome feature"
   git push origin feature/my-awesome-idea
   ```
7. Open a **Pull Request** on GitHub describing:
   - What feature or fix you added  
   - Why it’s needed  
   - Any new factors, solvers, or experiments introduced  
   - Any benchmark performance notes  

### Code Style Expectations

- Use **type hints** in all public methods.
- Add **clear docstrings** for all new modules, classes, and functions.
- Keep tests **focused, minimal, and deterministic**.
- When adding new factor types, include:
  - At least one unit test  
  - (If applicable) a small experiment demonstrating usage  

### Communication

If you are not sure whether a feature fits the project direction, feel free to:

- Open an Issue  
- Start a discussion on GitHub  
- Suggest an experiment or benchmark for evaluation  

## Acknowledgements

DSG-JIT is inspired by prior work in differentiable SLAM, dynamic scene graphs, and 
factor-graph-based robotics systems. This project aims to complement those efforts with a 
JAX-first, research-oriented, high-performance engine.

We welcome anyone who wants to push the boundaries of differentiable robotics, mapping, 
and scene graph research.  
