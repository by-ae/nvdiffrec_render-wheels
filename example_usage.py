"""
Example usage of nvdiffrec_render package.

This script demonstrates basic functionality that can be imported and used
after installing the wheel. Requires a CUDA-capable GPU to run.
"""
import sys

def check_imports():
    """Verify that all expected modules can be imported."""
    print("Checking nvdiffrec_render imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False
    
    try:
        import nvdiffrec_render.renderutils as ru
        print("✓ nvdiffrec_render.renderutils")
    except ImportError as e:
        print(f"✗ Failed to import renderutils: {e}")
        return False
    
    try:
        from nvdiffrec_render.renderutils import ops
        print("✓ nvdiffrec_render.renderutils.ops")
    except ImportError as e:
        print(f"✗ Failed to import ops: {e}")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability."""
    import torch
    
    print("\nChecking CUDA availability...")
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        print("  This package requires a CUDA-capable GPU to run")
        return False
    
    print(f"✓ CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    
    return True


def check_extensions():
    """Verify that compiled extensions are accessible."""
    print("\nChecking compiled extensions...")
    
    try:
        from nvdiffrec_render.renderutils import _C
        print("✓ Extension module _C imported successfully")
        
        # Check that key functions are available
        expected_functions = [
            'xfm_points',
            'xfm_vectors', 
            'image_loss',
            'lambert',
            'pbr_bsdf',
        ]
        
        for func_name in expected_functions:
            if hasattr(_C, func_name):
                print(f"  ✓ {func_name}")
            else:
                print(f"  ✗ {func_name} not found")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import extension: {e}")
        return False


def run_basic_test():
    """Run a basic smoke test if CUDA is available."""
    import torch
    from nvdiffrec_render.renderutils import ops
    
    print("\nRunning basic functionality test...")
    
    try:
        # Test transform operations with dummy data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy 4x4 transformation matrix
        mtx = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Create dummy 3D points
        points = torch.randn(1, 10, 3, dtype=torch.float32, device=device)
        
        # Test point transformation
        transformed = ops.xfm_points(points, mtx)
        
        if transformed.shape == points.shape:
            print("✓ Basic point transformation works")
            print(f"  Input shape: {points.shape}")
            print(f"  Output shape: {transformed.shape}")
            return True
        else:
            print(f"✗ Unexpected output shape: {transformed.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("nvdiffrec_render Installation Check")
    print("=" * 60)
    
    results = []
    
    # Check imports
    results.append(("Imports", check_imports()))
    
    # Check CUDA
    cuda_available = check_cuda()
    results.append(("CUDA", cuda_available))
    
    # Check extensions
    results.append(("Extensions", check_extensions()))
    
    # Run basic test if CUDA is available
    if cuda_available:
        results.append(("Basic test", run_basic_test()))
    else:
        print("\nSkipping basic test (no CUDA available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All checks passed!")
        return 0
    else:
        print("\n✗ Some checks failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
