#!/usr/bin/env python3
"""
Validation script for nvdiffrec_render wheels.
Tests that wheel structure and metadata are correct without requiring a GPU.
"""
import sys
import os
import zipfile
import re
from pathlib import Path


def wheel_filename_version(filename):
    """Return the version component of a wheel filename, or None.

    This project's distribution name has no hyphens, so the second '-'
    separated component is the version (PEP 427), including the local
    label: nvdiffrec_render-0.0.1+cu128torch29-cp310-....whl.
    """
    if not filename.endswith('.whl'):
        return None
    parts = filename[:-4].split('-')
    if len(parts) < 5 or parts[0] != 'nvdiffrec_render' or not parts[1]:
        return None
    return parts[1]


def canonical_distribution_name(name):
    """PEP 503 name: lowercase, and collapse runs of '-', '_', and '.'."""
    return re.sub(r'[-_.]+', '-', name).lower()


def metadata_text(raw):
    """Return METADATA with LF newlines.

    bdist_wheel on Windows writes CRLF. A Version line then ends in a
    carriage return, and the line-end pattern does not treat that as
    whitespace, so the header looks absent.
    """
    return raw.replace('\r\n', '\n').replace('\r', '\n')


# Names renderutils/__init__.py imports, and the torch_bindings.cpp entry
# points those wrappers have to call. Alias assignments such as
# `_C.xfm_points` do not exist on the compiled module.
OPS_DEF_PREFIXES = (
    "def xfm_points(",
    "def xfm_vectors(",
    "def image_loss(",
    "def diffuse_cubemap(",
    "def specular_cubemap(",
    "def prepare_shading_normal(",
    "def lambert(",
    "def frostbite_diffuse(",
    "def pbr_specular(",
    "def pbr_bsdf(",
    "def _fresnel_shlick(",
    "def _ndf_ggx(",
    "def _lambda_ggx(",
    "def _masking_smith(",
)
OPS_ENTRY_POINTS = (
    "xfm_fwd(",
    "xfm_bwd(",
    "lambert_fwd(",
    "lambert_bwd(",
    "image_loss_fwd(",
    "image_loss_bwd(",
    "prepare_shading_normal_fwd(",
    "pbr_bsdf_fwd(",
    "pbr_bsdf_bwd(",
    "diffuse_cubemap_fwd(",
    "specular_cubemap_fwd(",
    "specular_bounds(",
)
OPS_ALIASES = (
    "_C.xfm_points",
    "_C.xfm_vectors",
    "_C.image_loss",
    "_C.lambert",
    "_C.pbr_bsdf",
    "_C.prepare_shading_normal",
)


def is_compiled_extension(path):
    """True for renderutils/_C.<tag>.so and renderutils/_C.<tag>.pyd."""
    name = path.rsplit("/", 1)[-1]
    parent = path[: -(len(name) + 1)] if "/" in path else ""
    if not parent.endswith("renderutils"):
        return False
    if not name.startswith("_C."):
        return False
    return name.endswith(".so") or name.endswith(".pyd")


def ops_source_error(source):
    """Return an error if ops.py is not the wrapper for the compiled entry points."""
    problems = []
    missing = [snippet for snippet in OPS_DEF_PREFIXES + OPS_ENTRY_POINTS if snippet not in source]
    if missing:
        problems.append("ops.py is missing " + ", ".join(missing))
    aliases = [alias for alias in OPS_ALIASES if alias in source]
    if aliases:
        problems.append(
            "ops.py aliases compiled names that torch_bindings.cpp does not export: "
            + ", ".join(aliases)
        )
    return "; ".join(problems)


def renderutils_init_error(source):
    if "xfm_points" not in source or "ops" not in source:
        return "renderutils/__init__.py does not import the ops API (xfm_points)"
    return ""


def validate_wheel(wheel_path):
    """Validate a wheel file's structure and metadata."""
    errors = []
    warnings = []
    
    if not os.path.exists(wheel_path):
        return [f"Wheel file not found: {wheel_path}"], []
    
    print(f"Validating: {wheel_path}")
    
    try:
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            files = zf.namelist()
            
            # Check for essential components. A stray .so anywhere in the
            # archive is not the extension users import.
            has_c_extension = any(is_compiled_extension(f) for f in files)
            has_metadata = any('METADATA' in f for f in files)
            has_wheel_metadata = any('WHEEL' in f for f in files)
            
            if not has_c_extension:
                errors.append(
                    "Missing compiled extension renderutils/_C (.so or .pyd) - "
                    "the wheel must include compiled CUDA code"
                )

            if "nvdiffrec_render/__init__.py" not in files:
                errors.append(
                    "Missing nvdiffrec_render/__init__.py - without it the wheel "
                    "is only the _C file and renderutils.ops cannot be imported"
                )
            init_name = "nvdiffrec_render/renderutils/__init__.py"
            ops_name = "nvdiffrec_render/renderutils/ops.py"
            if init_name not in files:
                errors.append(f"Missing {init_name}")
            else:
                init_error = renderutils_init_error(zf.read(init_name).decode("utf-8"))
                if init_error:
                    errors.append(init_error)
            if ops_name not in files:
                errors.append(
                    f"Missing {ops_name} - the compiled module exports xfm_fwd, "
                    "not the xfm_points name callers import"
                )
            else:
                ops_error = ops_source_error(zf.read(ops_name).decode("utf-8"))
                if ops_error:
                    errors.append(ops_error)
            
            if not has_metadata:
                errors.append("Missing METADATA file - wheel package metadata is incomplete")
            
            if not has_wheel_metadata:
                errors.append("Missing WHEEL file - wheel format metadata is incomplete")
            
            # Check wheel filename format
            wheel_name = os.path.basename(wheel_path)
            if not re.match(r'nvdiffrec_render-.*\.whl', wheel_name):
                warnings.append(f"Unusual wheel name format: {wheel_name}")
            
            # Validate CUDA/PyTorch tags if present
            if '+cu' in wheel_name:
                cuda_match = re.search(r'\+cu(\d+)', wheel_name)
                torch_match = re.search(r'torch(\d+)', wheel_name)
                
                if cuda_match:
                    cuda_ver = cuda_match.group(1)
                    print(f"  CUDA version tag: {cuda_ver}")
                else:
                    warnings.append("CUDA version tag not found in filename (expected format: +cuXXX)")
                
                if torch_match:
                    torch_ver = torch_match.group(1)
                    print(f"  PyTorch version tag: {torch_ver}")
                else:
                    warnings.append("PyTorch version tag not found in filename (expected format: torchXX)")
            
            # Check for expected package structure
            expected_paths = [
                'nvdiffrec_render/',
                'nvdiffrec_render/renderutils/',
            ]
            
            for expected in expected_paths:
                if not any(f.startswith(expected) for f in files):
                    errors.append(f"Missing expected directory structure: {expected}")
            
            # Verify METADATA content. Match the Version header itself:
            # "Version:" is a substring of "Metadata-Version:".
            # setuptools keeps the name from setup.py; the wheel package's
            # bdist_wheel writes the PEP 503 form (nvdiffrec-render).
            metadata_found = False
            for fname in files:
                if fname.endswith('METADATA'):
                    metadata = metadata_text(zf.read(fname).decode('utf-8'))
                    name_match = re.search(r'(?m)^Name:[ \t]*(\S+)[ \t]*$', metadata)
                    if name_match is None or canonical_distribution_name(name_match.group(1)) != canonical_distribution_name('nvdiffrec_render'):
                        errors.append("METADATA package name is not nvdiffrec_render")
                    version_match = re.search(r'(?m)^Version:[ \t]*(\S+)[ \t]*$', metadata)
                    if version_match is None:
                        errors.append("METADATA missing Version field")
                    else:
                        metadata_version = version_match.group(1)
                        file_version = wheel_filename_version(wheel_name)
                        if file_version is None:
                            errors.append(f"Cannot read version from wheel filename: {wheel_name}")
                        elif metadata_version != file_version:
                            errors.append(
                                f"METADATA Version {metadata_version} does not match "
                                f"filename version {file_version}"
                            )
                    metadata_found = True
                    print(f"  Package metadata validated")
                    break
            
            if not metadata_found:
                errors.append("Could not locate METADATA file in wheel")
            
            print(f"  Total files in wheel: {len(files)}")
            
    except zipfile.BadZipFile:
        errors.append("Invalid wheel file (corrupt zip) - try rebuilding the wheel")
    except Exception as e:
        errors.append(f"Unexpected error during validation: {e}")
    
    return errors, warnings


def collect_wheels(paths):
    """Expand file and directory arguments into wheel paths."""
    wheels = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(Path(path).glob('*.whl'))
            if not found:
                print(f"No wheel files found in {path}")
                sys.exit(1)
            wheels.extend(found)
        elif path.endswith('.whl'):
            wheels.append(Path(path))
        else:
            print(f"Error: {path} is not a wheel file or directory")
            sys.exit(1)
    return wheels


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: python validate_wheel.py <wheel_file_or_directory> [...]")
        print("\nValidates nvdiffrec_render wheel structure and metadata.")
        sys.exit(1)

    wheels = collect_wheels(argv)
    
    print(f"Found {len(wheels)} wheel(s) to validate\n")
    
    total_errors = 0
    total_warnings = 0
    
    for wheel in wheels:
        errors, warnings = validate_wheel(str(wheel))
        
        if errors:
            print(f"  ❌ ERRORS:")
            for err in errors:
                print(f"     - {err}")
            total_errors += len(errors)
        
        if warnings:
            print(f"  ⚠️  WARNINGS:")
            for warn in warnings:
                print(f"     - {warn}")
            total_warnings += len(warnings)
        
        if not errors and not warnings:
            print(f"  ✅ Valid")
        
        print()
    
    print(f"Validation complete: {len(wheels)} wheels, {total_errors} errors, {total_warnings} warnings")
    
    if total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
