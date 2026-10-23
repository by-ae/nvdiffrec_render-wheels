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
            
            # Check for essential components
            has_c_extension = any('.so' in f or '.pyd' in f for f in files)
            has_metadata = any('METADATA' in f for f in files)
            has_wheel_metadata = any('WHEEL' in f for f in files)
            
            if not has_c_extension:
                errors.append("Missing compiled extension (.so/.pyd) - the wheel must include compiled CUDA code")
            
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
            
            # Verify METADATA content
            metadata_found = False
            for fname in files:
                if fname.endswith('METADATA'):
                    metadata = zf.read(fname).decode('utf-8')
                    if 'Name: nvdiffrec_render' not in metadata:
                        errors.append("METADATA missing package name 'nvdiffrec_render'")
                    if 'Version:' not in metadata:
                        errors.append("METADATA missing Version field")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_wheel.py <wheel_file_or_directory>")
        print("\nValidates nvdiffrec_render wheel structure and metadata.")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        wheels = list(Path(path).glob('*.whl'))
        if not wheels:
            print(f"No wheel files found in {path}")
            sys.exit(1)
    elif path.endswith('.whl'):
        wheels = [Path(path)]
    else:
        print(f"Error: {path} is not a wheel file or directory")
        sys.exit(1)
    
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
