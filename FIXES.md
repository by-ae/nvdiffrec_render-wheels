# Wheel Build Fixes

This document describes the fixes applied to the nvdiffrec_render-wheels repository build system.

## Issues Fixed

### 1. Windows Wheel Doubled Tag Bug
**Problem:** Windows wheels had doubled local version tags in filenames (e.g., `cp310+cu126torch26-cp310`)
**Cause:** PowerShell `-replace` operator replaced ALL occurrences of `-cp`, not just the first
**Fix:** Removed post-build rename logic; now set version in setup.py via NVDIFFREC_LOCAL_VERSION env var

### 2. Version Mismatch Between Filename and Metadata
**Problem:** Wheel filenames had `0.0.1+cu126torch26` but metadata showed `Version: 0.0.1`, causing pip to reject them
**Fix:** setup.py now reads NVDIFFREC_LOCAL_VERSION and sets it in package metadata, ensuring filename matches

### 3. GitHub Pages Not Deployed
**Problem:** Pages enabled but no deployment workflow existed, all URLs returned 404
**Fix:** Added deploy-pages.yml workflow that deploys wheels/ directory on push to main

### 4. Root Index Missing Wheel Links
**Problem:** Root index only had subdirectory links, not wheel links, so pip --find-links to root might not work
**Fix:** Added generate_root_index.py to aggregate all wheel links into root index.html

### 5. Unnecessary Build Triggers
**Problem:** All six build workflows triggered on every push to main
**Fix:** Added paths filters so builds only run when workflows, setup.py, or source changes

### 6. Missing Validation
**Problem:** No test to catch the doubled-tag bug
**Fix:** Enhanced validate_index.py with doubled-tag detection and added test case

## Manual Actions Required

### Old Release Assets Cleanup (Optional)
Old Windows wheel assets with doubled tags still exist in releases. To clean them up safely:

```bash
# List assets with doubled tags
gh release list | while read name rest; do
  echo "=== $name ==="
  gh release view "$name" --json assets --jq '.assets[].name' | grep -E '\+cu[0-9]+torch[0-9]+-cp[0-9]+\+cu[0-9]+torch[0-9]+-cp[0-9]+'
done

# Delete specific doubled-tag assets (example for one release)
# Only run after verifying new correctly-named assets exist
gh release view cu126-torch260 --json assets --jq '.assets[] | select(.name | test("\\+cu[0-9]+torch[0-9]+-cp[0-9]+\\+cu[0-9]+torch[0-9]+-cp[0-9]+")) | .name' | \
  xargs -I {} gh release delete-asset cu126-torch260 {} --yes
```

**Warning:** Only delete after new builds complete and publish correctly-named wheels.

### Trigger Fresh Builds
To republish all wheels with the fixes:
1. Go to Actions tab
2. Manually trigger each build-*.yml workflow via workflow_dispatch
3. Wait for builds to complete and verify new assets are correctly named
4. Pages will auto-deploy with updated indexes

### Verify Installation
After new builds deploy, test pip installation:
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/
```

## Testing Changes Locally

Run validation tests:
```bash
python3 test_validate_index.py -v
python3 test_find_wheel.py -v
python3 validate_index.py wheels/
```

Generate root index:
```bash
python3 generate_root_index.py
```

## Commit Structure

1. Fix wheel filename and metadata version mismatch
2. Add GitHub Pages deployment workflow
3. Add validation test for doubled-tag wheel filenames
4. Add root index with all wheel links for pip auto-detection
5. Document fixes and manual cleanup steps
