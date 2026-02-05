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

### Bucketing

All numeric "size-like" values are bucketed before transmission to prevent
identification:

| Bucket | Range |
|--------|-------|
| `0` | 0 |
| `1-9` | 1-9 |
| `10-99` | 10-99 |
| `100-999` | 100-999 |
| `1k-9k` | 1,000-9,999 |
| `10k-99k` | 10,000-99,999 |
| `100k-999k` | 100,000-999,999 |
| `1M+` | 1,000,000+ |

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

### Technical Enforcement

Privacy is enforced at multiple levels:

1. **Allowlist**: Only explicitly approved attributes are recorded
   (see `sanitize.py`)
2. **Bucketing**: All integers are bucketed before recording
3. **String validation**: Strings must be ≤32 chars and valid identifiers
4. **Error sanitization**: Only exception class names, never messages
5. **No paths**: File operations record only `load`/`save` and format type

---

## How to Control Telemetry

### Environment Variables

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `DSGJIT_TELEMETRY` | `1`, `0` | `1` | Master on/off switch |
| `DSGJIT_TELEMETRY_LEVEL` | `minimal`, `standard`, `debug` | `standard` | Data verbosity |
| `DSGJIT_TELEMETRY_ENDPOINT` | URL or empty | `https://telemetry.ix-infra.com/v1/traces` | Where data is sent |
| `DSGJIT_TELEMETRY_SAMPLE_RATE` | `0.0`-`1.0` | `0.20` | Success span sampling rate |
| `DSGJIT_TELEMETRY_DEBUG` | `1`, `0` | `0` | Print spans to stderr |
| `DSGJIT_TELEMETRY_TAG` | string | (empty) | Custom tag for experiment/run identification |

### Reduce Data Collection

To minimize data collection while keeping telemetry enabled:

```bash
export DSGJIT_TELEMETRY_LEVEL=minimal
```

### Disable Network Export

To collect telemetry locally but not send it anywhere:

```bash
export DSGJIT_TELEMETRY_ENDPOINT=""
```

### Disable Telemetry Entirely (OSS)

For the open-source version:

```bash
export DSGJIT_TELEMETRY=0
```

This completely disables telemetry. No client is created, no spans are
recorded, and no network calls are made.

---

## DSG-JIT Pro (Future)

DSG-JIT Pro users will have additional controls:

- Full telemetry disable without environment variables
- Per-project telemetry configuration
- On-premises telemetry collection option

*These features are planned for future releases.*

---

## Data Handling

### Transport Security

- All telemetry is sent over HTTPS
- Payload format: OTLP JSON over HTTP
- Maximum payload size: 64KB
- Maximum spans per request: 200

### Rate Limiting

The telemetry endpoint enforces rate limits:

- Per-IP: 30 requests/minute, 300 requests/hour
- Per-install: 500 requests/day

### Data Retention

Telemetry data is:

- Aggregated for statistical analysis
- Not linked to individual users
- Retained for product improvement purposes only
- Not sold or shared with third parties

---

## Questions or Concerns

If you have questions about telemetry or privacy:

- Open an issue: https://github.com/tannertorrey3/DSG-JIT/issues
- Review the source: `dsg_jit/telemetry/sanitize.py`

---

## Summary

| Aspect | Status |
|--------|--------|
| Enabled by default | Yes (OSS) |
| Can be disabled | Yes (`DSGJIT_TELEMETRY=0`) |
| Collects user content | **Never** |
| Collects file paths | **Never** |
| Collects hostnames/IPs | **Never** |
| Collects error messages | **Never** (class names only) |
| Values bucketed | Always |
| Transport encrypted | Yes (HTTPS) |
