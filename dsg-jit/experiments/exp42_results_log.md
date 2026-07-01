# Exp42 Results Log

All runs: `σ_t=0.03`, `σ_r=0.01`, `overlap=10`, `seeds=1` unless noted.
Sequences on SemanticKITTI (`~/kitti-test/sequences`).

---

## Run History

| Date | Commit | window | seqs | seeds | max-poses | ΔT | ΔR | ΔC | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-26 | bee50a3 | 50 | 02 | 1 | 200 | +53.6% | +49.1% | +51.4% | After Laplacian fix (−74% → +51%) |
| 2026-06-26 | 5f6d74b | 50 | 02 | 1 | — (all) | −5.8% | −40.2% | −23.0% | Lambda, old hash seed + no --max-poses |
| 2026-06-27 | d986b7f | 50 | 02 | 5 | 200 | +36.7% | +6.5% | +24.2% | After deterministic seed fix, 5-seed mean |
| 2026-06-27 | d986b7f | 100 | 02 | 1 | 200 | — | — | −18.4% | Window=100 seed=0 |
| 2026-06-27 | d986b7f | 100 | 02 | 5 | 200 | −10.5% | −26.3% | −84.9% | Window=100 5-seed mean |
| 2026-06-27 | d986b7f | 50 | 02+06 | 5 | 200 | 02: +36.7% / 06: −167.7% | 02: +6.5% / 06: −132.4% | 02: +21.6% / 06: −150.0% | Seq 06 total failure |
| 2026-06-29 | 41aa448 | 50 | 02+06 | 1 | — (all) | 02: +61.7% / 06: −168.8% | 02: +66.5% / 06: −82.2% | 02: +64.1% / 06: −125.5% | After so3_log bypass in outer_adam_loop |
| 2026-06-29 | c915da3 | **100** | 02+06 | 1 | — (all) | 02: −15.5% / 06: −340.9% | 02: +14.1% / 06: −106.7% | 02: −0.7% / 06: −223.8% | After relative_poses_from_mats fix; **window=100** (default, not 50) |
| 2026-06-29 | c915da3 | 50 | 02+06 | 1 | — (all) | 02: +64.6% / 06: −134.7% | 02: +67.5% / 06: −75.7% | 02: +66.1% / 06: −105.2% | Window=50; seq 06 still failing |
| 2026-06-29 | 28fc512 | **100** | 02+06 | 1 | — (all) | 02: +9.0% / 06: −307.8% | 02: +2.7% / 06: −60.0% | 02: +5.8% / 06: −183.9% | After Frobenius outer loss; **window=100** (default) |
| 2026-07-01 | d488c9a | 100 | 00,01,02,05,06,07,08,09,10 | 22 | — (all) | +38.5% | +38.3% | **+38.4%** | First 9-seq 22-seed sweep; seq 06 FIXED (+43.6%); 3 catastrophic seed failures remain |

---

## Configuration at Each Commit

| Commit | Key Change |
|---|---|
| `bee50a3` | Fix rotation Laplacian: standard graph Laplacian (−κI₃) instead of connection Laplacian (−κR̃ᵢⱼ) |
| `5f6d74b` | Fix non-deterministic seed: `hash(seq_id)` → `int(seq_id)` |
| `151523a` / `73ed603` / `d986b7f` | Align defaults with exp41: window=100, lr=1e-3, 30+20+20=70 iters |
| `41aa448` | Bypass so3_log singularity in outer_adam_loop: pass raw rotation matrices (gt_R_direct) |
| `c915da3` | Bypass so3_log singularity in relative_poses: add relative_poses_from_mats using raw SE(3) matrices |
| `28fc512` | Switch outer rotation loss: geodesic so3_log → Frobenius ‖Ra − Rb‖²_F |
| `02e6bfe` | Rename build_connection_laplacian → build_rotation_laplacian; document SE-Sync paper alignment |
| `0fa5b62` | Fix IFT bilevel: project_grad /2 (correct IFT formula) + fresh R_init per Adam step (no stale carry) |
| `983e164` | vmap over seeds: batched GPU dispatch (S, n-1, 6) instead of S sequential calls |
| `3bf9a3e` | Add result-saving: --output-dir, timestamped run dir, per-seed JSON, aggregate.json, results.txt |
| `d488c9a` | Eliminate D2H bottleneck: pure-JAX vmapped _sigma_and_precision replaces Python-loop + np.array copies |

---

## Observations

- **Seq 02** (highway, mostly straight): +33.2% ΔC at window=100, 22 seeds; was +64–67% at window=50/1-seed — multi-seed mean lower due to several near-zero seeds
- **Seq 06** (urban loop, sharp turns): **FIXED** at +43.6% ΔC, 22 seeds; was −105% to −184% through all prior fixes. IFT bilevel fix (0fa5b62) resolved it.
- **Window=100** now stable across all 9 sequences; was previously catastrophically unstable. Same bilevel fix.
- **Catastrophic failures** remain in 3 isolated seeds (seq 01 seed 09, seq 08 seed 09, seq 09 seed 10) with ΔT = −260% indicating translation divergence. Root cause unresolved.
- **Seq 06 was not a seq 06 problem** — it was a window=100 IFT gradient error that hit seq 06 hardest due to larger accumulated rotations in urban turns.
