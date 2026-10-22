# nvdiffrec_render-wheels

Pre-built Python wheels for `nvdiffrec_render` (rendering utilities from [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec)).

## Requirements

- **NVIDIA GPU** with compute capability 7.5+ (RTX 20/30/40/50 series, A100, B100/B200)
- **CUDA runtime** matching your chosen wheel version (see table below)
- **PyTorch** matching your chosen wheel version
- **Platform**: Linux (x86_64) or Windows (amd64)

## Install

```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/
```

## Supported Configurations

| CUDA | PyTorch | Python Versions |
|------|---------|-----------------|
| 12.4 | 2.5.1 | 3.10, 3.11, 3.12 |
| 12.6 | 2.6.0 | 3.10, 3.11, 3.12, 3.13 |
| 12.6 | 2.8.0 | 3.10, 3.11, 3.12, 3.13 |
| 12.8 | 2.8.0 | 3.10, 3.11, 3.12, 3.13 |
| 12.8 | 2.9.1 | 3.10, 3.11, 3.12, 3.13 |

**Platforms:** Linux (x86_64), Windows (amd64)

**GPU Support:** RTX 20/30/40/50 series (SM 7.5+)

## Troubleshooting

### ImportError: cannot import name '_C'

Ensure your CUDA runtime version matches the wheel. Check with:
```bash
python -c "import torch; print(torch.version.cuda)"
```

### CUDA out of memory errors

The rendering utilities require GPU memory. Reduce batch sizes or texture resolutions if you encounter OOM errors.

### Wheel not found during pip install

Specify the exact configuration if auto-detection fails:
```bash
pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu126-torch280/
```

## Building Manually

To trigger a build for a specific configuration, go to Actions and run the corresponding workflow.

## License

The nvdiffrec rendering utilities are under NVIDIA Source Code License. See LICENSE.txt for details.
