# nvdiffrec_render-wheels

Pre-built Python wheels for `nvdiffrec_render` (rendering utilities from [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec)).

## Requirements

- **NVIDIA GPU** with compute capability 7.5+ (RTX 20/30/40/50 series, A100, B100/B200)
- **CUDA runtime** matching your chosen wheel version (see table below)
- **PyTorch** matching your chosen wheel version
- **Platform**: Linux (x86_64) or Windows (amd64)

## Install

### Quick Install (Auto-Detect)

If you already have PyTorch installed, pip can usually auto-detect your configuration:

```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/
```

**Note:** This works best if your CUDA and PyTorch versions match one of the available configurations below. If you get "no matching distribution found", use a specific configuration URL instead.

### Install Specific Configuration

Choose the command matching your CUDA and PyTorch versions:

**CUDA 12.8 + PyTorch 2.9.1** (Linux and Windows, Python 3.10-3.13):
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu128-torch291/
```

**CUDA 12.8 + PyTorch 2.8.0** (Python 3.10-3.13):
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu128-torch280/
```

**CUDA 12.6 + PyTorch 2.8.0** (Python 3.10-3.13):
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu126-torch280/
```

**CUDA 12.6 + PyTorch 2.6.0** (Python 3.10-3.13):
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu126-torch260/
```

**CUDA 12.4 + PyTorch 2.5.1** (Python 3.10-3.12):
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu124-torch251/
```

### Not Sure Which Wheel?

Use the helper script to automatically detect your configuration and get the correct install command:

```bash
python find_wheel.py
```

Or run it directly from the repository without cloning:

```bash
curl -sL https://raw.githubusercontent.com/by-ae/nvdiffrec_render-wheels/main/find_wheel.py | python
```

The script auto-detects your Python, PyTorch, and CUDA versions. If PyTorch isn't installed yet, specify your target versions manually:

```bash
python find_wheel.py --torch 2.9.1 --cuda 12.8 --python 3.11
```

### Which Configuration Do I Need? (Manual Check)

Check your installed PyTorch and CUDA versions:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

Match the output to one of the configurations above. For example:
- Output: `PyTorch: 2.9.1` and `CUDA: 12.8` → use **cu128-torch291**
- Output: `PyTorch: 2.6.0` and `CUDA: 12.6` → use **cu126-torch260**
- Output: `PyTorch: 2.9.1` and `CUDA: 13.0` → no wheel is published yet (see below)

You can also check your Python version:
```bash
python --version
```

## Supported Configurations

| CUDA | PyTorch | Python Versions | Platforms |
|------|---------|-----------------|-----------|
| 12.8 | 2.9.1 | 3.10, 3.11, 3.12, 3.13 | Linux, Windows |
| 12.8 | 2.8.0 | 3.10, 3.11, 3.12, 3.13 | Linux, Windows |
| 12.6 | 2.8.0 | 3.10, 3.11, 3.12, 3.13 | Linux, Windows |
| 12.6 | 2.6.0 | 3.10, 3.11, 3.12, 3.13 | Linux, Windows |
| 12.4 | 2.5.1 | 3.10, 3.11, 3.12 | Linux, Windows |

Each row is a set of wheels linked from this repo's index. The build workflow installs that PyTorch version from the official index at `https://download.pytorch.org/whl/cu124`, `cu126`, or `cu128`.

## Not published yet

CUDA 13.0 + PyTorch 2.9.1 (`cu130-torch291`) has a Linux-only workflow, `build-cu130-torch291.yml`, for Python 3.10, 3.11, 3.12, and 3.13. It installs PyTorch from `https://download.pytorch.org/whl/cu130`. The GitHub release `cu130-torch291` currently has no wheel assets, and [its index page](https://by-ae.github.io/nvdiffrec_render-wheels/cu130-torch291/) has no files for pip. There is no install command until that release contains wheels.

**GPU Support:** RTX 20/30/40/50 series (SM 7.5+)

## Troubleshooting

### ERROR: No matching distribution found

This means pip couldn't find a wheel for your configuration. Common causes:

1. **CUDA version mismatch**: Your PyTorch CUDA version doesn't match any available wheel
   - Check with: `python -c "import torch; print(torch.version.cuda)"`
   - Install PyTorch with the correct CUDA version first (see [pytorch.org](https://pytorch.org))

2. **Python version not supported**: Use Python 3.10-3.13 (3.10-3.12 for CUDA 12.4)
   - Check with: `python --version`

3. **Wrong platform**: Only Linux (x86_64) and Windows (amd64) are supported

4. **CUDA 13.0 + PyTorch 2.9.1**: that release has no wheel files, so pip cannot install from its index page

5. **Typo in find-links URL**: Make sure to copy the full URL including the configuration subdirectory

**Solution**: Use a specific configuration command from the "Install Specific Configuration" section above.

### ImportError: cannot import name '_C'

Ensure your CUDA runtime version matches the wheel. Check with:
```bash
python -c "import torch; print(torch.version.cuda)"
```

If the versions don't match, reinstall with the correct configuration.

### CUDA out of memory errors

The rendering utilities require GPU memory. Reduce batch sizes or texture resolutions if you encounter OOM errors.

## Building Manually

To trigger a build for a specific configuration, go to Actions and run the corresponding workflow.

## License

The nvdiffrec rendering utilities are under NVIDIA Source Code License. See LICENSE.txt for details.
