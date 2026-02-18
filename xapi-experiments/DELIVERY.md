# Delivery Guide: xAPI DATASIM Experiments

## What's Included

This package provides scripts to generate 5 scaled xAPI Statement datasets for system testing.

### Files Structure

```
xapi-experiments/
├── README.md              # Full documentation
├── SETUP.md               # Team onboarding guide
├── QUICKSTART.md          # One-page quick start
├── DELIVERY.md            # This file (delivery checklist)
├── slack_update.txt       # Slack announcement template
├── make_experiments.py    # Generates 5 experiment specs
├── run_experiments.sh     # Main runner (builds + runs DATASIM)
└── specs/                 # Generated experiment configs (committed)
    ├── tiny.json
    ├── small.json
    ├── medium.json
    ├── large.json
    └── huge.json
```

### What's NOT Included (by design)

- `xapi-experiments/outputs/` - Generated NDJSON files (large, regeneratable)
- `datasim/` - External DATASIM repo (team clones separately)

These are excluded via `.gitignore` to keep the repo lightweight.

## Delivery Checklist

### ✅ Pre-Delivery

- [x] Scripts created (`make_experiments.py`, `run_experiments.sh`)
- [x] Documentation written (`README.md`, `SETUP.md`, `QUICKSTART.md`)
- [x] `.gitignore` updated (excludes `outputs/` and `datasim/`)
- [x] Specs generated (`xapi-experiments/specs/*.json` - these ARE committed)
- [x] Scripts are executable (`chmod +x`)

### 📦 Delivery Steps

1. **Commit to repo**:
   ```bash
   git add xapi-experiments/
   git add .gitignore
   git commit -m "Add DATASIM experiments for xAPI test data generation"
   git push
   ```

2. **Share with team**:
   - Point them to `xapi-experiments/QUICKSTART.md` for immediate use
   - Share `xapi-experiments/SETUP.md` for detailed setup
   - Use `xapi-experiments/slack_update.txt` for Slack announcement

3. **Team members follow**:
   ```bash
   git pull
   git clone https://github.com/yetanalytics/datasim.git
   ./xapi-experiments/run_experiments.sh
   ```

### 🎯 What Team Gets

After setup, team members can:

- **Generate datasets on-demand**: `./xapi-experiments/run_experiments.sh`
- **Use 5 scaled datasets**: tiny (1k) → huge (100k) statements
- **Reproducible results**: Fixed seeds ensure consistent output
- **NDJSON format**: Standard format for bulk ingestion/testing

### 📊 Expected Outputs

Each team member generates locally:

| File | Size | Use Case |
|------|------|----------|
| `tiny.ndjson` | ~500KB | Quick smoke tests |
| `small.ndjson` | ~2.5MB | Light load testing |
| `medium.ndjson` | ~5MB | Medium load |
| `large.ndjson` | ~25MB | Heavy load |
| `huge.ndjson` | ~50MB | Stress testing |

**Total**: ~83MB per person (not committed, generated locally)

## Alternative: Pre-Generated Datasets

If you want to **share pre-generated datasets** instead of having everyone generate:

1. Generate once: `./xapi-experiments/run_experiments.sh`
2. Upload outputs to shared storage (S3, Google Drive, etc.)
3. Update `QUICKSTART.md` with download instructions
4. Team downloads instead of generating

**Pros**: Faster for team, consistent datasets  
**Cons**: Larger repo/artifacts, need to manage versions

## Support

- **Setup issues**: See `SETUP.md` troubleshooting section
- **Usage questions**: See `README.md` for detailed docs
- **Script problems**: Check that Java, Clojure CLI, and Python are installed
