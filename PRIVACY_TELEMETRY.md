# DSG-JIT Telemetry Privacy Policy

This document describes exactly what telemetry DSG-JIT collects, what it does
**not** collect, and how to control or disable telemetry.

---

## Why We Collect Telemetry

DSG-JIT collects anonymous, low-cardinality usage telemetry to:

- Measure API coverage and feature adoption
- Detect errors and performance regressions
- Understand which backends (CPU/GPU) are used
- Improve the library based on real-world usage patterns

**Telemetry is enabled by default** for the open-source version to help us
improve DSG-JIT for everyone.

---

## What We Collect

All collected data is anonymous and non-identifying:

| Category | Data Collected | Example |
|----------|----------------|---------|
| **Identity** | Anonymous install ID (UUID) | `a1b2c3d4-...` |
| | Anonymous session ID (UUID) | `e5f6g7h8-...` |
| **Package** | DSG-JIT version | `0.7.1` |
| | Python version | `3.11.7` |
| | OS type | `linux`, `darwin`, `windows` |
| | Architecture | `x86_64`, `arm64` |
| **Operations** | Operation names | `dsgjit.world.add_pose` |
| | Component names | `world`, `scene_graph`, `slam` |
| | Duration (milliseconds) | `42.7` |
| | Status | `ok`, `error` |
| **Backend** | Compute backend | `cpu`, `gpu`, `tpu` |
| **Errors** | Exception class name | `ValueError`, `ConvergenceError` |
| | Error category code | `invalid_argument`, `convergence_failure` |
| **Counts** | Bucketed counts only | `10-99`, `1k-9k` (never exact) |
| **Config** | Telemetry level | `minimal`, `standard`, `debug` |

---

## What We NEVER Collect

DSG-JIT telemetry is designed with privacy as a hard requirement. We **never**
collect:

| Category | Never Collected |
|----------|-----------------|
| **User Content** | Document text, code, prompts, embeddings |
| | Node/edge labels, attribute values |
| | Graph contents or semantic data |
| **Identifiers** | File paths, directory names |
| | Repository URLs |
| | Hostnames, usernames, IP addresses |
| | MAC addresses |
| | User-domain IDs (document IDs, object IDs) |
| **Errors** | Exception messages (only class names) |
| | Stack traces |
| | Variable values in errors |
| **Raw Values** | Exact numeric counts (always bucketed) |
| | Raw floating-point values |
| | Timestamps of user actions |

