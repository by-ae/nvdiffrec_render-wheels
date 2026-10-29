# nvdiffrec_render-wheels

Pre-built Python wheels for `nvdiffrec_render` (rendering utilities from [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec)).

## Requirements

- **NVIDIA GPU** at a compute capability the wheel was compiled for (see below)
- **CUDA runtime** matching your chosen wheel version (see table below)
- **PyTorch** matching your chosen wheel version
- **Platform**: Linux (x86_64) or Windows (amd64)

## Install

### Quick Install

The combined index lists every published wheel. Pip keeps a file when the Python, ABI, and platform tags match your interpreter, then installs the newest version. It does not look at `torch.version.cuda`. Every wheel here is version `0.0.1` plus a local label, and the newest one is `0.0.1+cu128torch29` (CUDA 12.8 + PyTorch 2.9.1, `cu128-torch291`):

```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/
```

Use that command when PyTorch is 2.9.1 built for CUDA 12.8. For any other combination, use the matching configuration below, or run `python find_wheel.py`.

Install that PyTorch build first. Every published wheel depends on `torch>=2.5.0`. If torch is missing, pip installs the PyPI build, which is not the CUDA build the wheel was compiled against.

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

The script auto-detects your Python, PyTorch, and CUDA versions. It prints an install command only when both match a published config. A nearby CUDA or PyTorch version is a different build, and that wheel will not import. If PyTorch isn't installed yet, specify your target versions manually:

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

## GPU architectures

A wheel runs only on the compute capabilities it was compiled for. Anything else fails at kernel launch with no kernel image for that device.

Wheels on the index today contain these cubins:

| CUDA wheels | Compute capabilities |
|-------------|----------------------|
| 12.4 and 12.6 | 7.5, 8.6, 8.9, 9.0 |
| 12.8 | 7.5, 8.6, 8.9, 9.0, 10.0, 12.0 |

7.5 is RTX 20-series, 8.6 is RTX 30-series, 8.9 is RTX 40-series, and 9.0 includes H100. The CUDA 12.8 wheels also include 10.0 (B100/B200) and 12.0 (RTX 50-series). Compute capability 8.0 (A100) is not in any published file. The build workflows now request 8.0 as well, so that cubin is in the next build of a configuration, not in the files pip installs today. CUDA 12.4 and 12.6 builds still do not request 10.0 or 12.0.

## Not published yet

CUDA 13.0 + PyTorch 2.9.1 (`cu130-torch291`) has a Linux-only workflow, `build-cu130-torch291.yml`, for Python 3.10, 3.11, 3.12, and 3.13. It installs PyTorch from `https://download.pytorch.org/whl/cu130`. The GitHub release `cu130-torch291` currently has no wheel assets, and [its index page](https://by-ae.github.io/nvdiffrec_render-wheels/cu130-torch291/) has no files for pip. There is no install command until that release contains wheels. That workflow requests compute capabilities 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0.

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

The wheel has to be built for the same PyTorch CUDA build you import. Check with:
```bash
python -c "import torch; print(torch.version.cuda)"
```

The root find-links URL installs `0.0.1+cu128torch29`. When `torch.version.cuda` is not `12.8`, or PyTorch is not 2.9.1, install from the matching configuration URL.

### CUDA out of memory errors

The rendering utilities require GPU memory. Reduce batch sizes or texture resolutions if you encounter OOM errors.

## Building Manually

To trigger a build for a specific configuration, go to Actions and run the corresponding workflow.

## License

The nvdiffrec rendering utilities are under NVIDIA Source Code License. See LICENSE.txt for details.
