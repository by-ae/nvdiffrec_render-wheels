#!/usr/bin/env python3
"""
Helper script to find the correct nvdiffrec_render wheel for your environment.
Detects Python version, PyTorch version, and CUDA version, then prints an
install command only when both match a published config and this machine
uses a platform tag that config publishes. A wheel built for a different
PyTorch, CUDA, or platform tag will not install.
"""
import sys
import os
import re
import argparse
import platform
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.request import urlopen
from urllib.error import URLError


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """
    Parse a version string like '2.9.1' or '2.9.1+cu128' into (major, minor, patch).
    Strips local version suffixes (everything after +).
    """
    # Strip local version suffix (e.g., +cu128)
    base_version = version_str.split('+')[0]
    
    # Parse major.minor.patch
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', base_version)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def parse_cuda_version(cuda_str: str) -> Tuple[int, int]:
    """
    Parse CUDA version string like '12.8' or '128' into (major, minor).
    Returns tuple for comparison.
    """
    # Handle short form like '128' (12.8)
    if '.' not in cuda_str:
        if len(cuda_str) == 3:  # e.g., '128' -> 12.8
            major = int(cuda_str[:2])
            minor = int(cuda_str[2:])
        elif len(cuda_str) == 4:  # e.g., '1308' -> 13.08 -> 13.8
            major = int(cuda_str[:2])
            minor = int(cuda_str[2:])
        else:
            raise ValueError(f"Invalid CUDA short version: {cuda_str}")
    else:
        # Handle standard form like '12.8'
        parts = cuda_str.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid CUDA version format: {cuda_str}")
        major, minor = int(parts[0]), int(parts[1])
    
    return (major, minor)


def format_cuda_version(cuda_tuple: Tuple[int, int]) -> str:
    """Format CUDA tuple (12, 8) as '12.8'."""
    return f"{cuda_tuple[0]}.{cuda_tuple[1]}"


def parse_python_range(range_str: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parse Python version range string like '3.10-3.13' into ((3, 10), (3, 13)).
    Returns None if parsing fails.
    """
    match = re.match(r'^(\d+)\.(\d+)-(\d+)\.(\d+)$', range_str.strip())
    if not match:
        return None
    
    min_major, min_minor, max_major, max_minor = match.groups()
    return ((int(min_major), int(min_minor)), (int(max_major), int(max_minor)))


def parse_configs_from_index_html(html: str) -> List[dict]:
    """
    Parse wheel configurations from the index.html table.
    
    Returns list of dicts with 'name', 'cuda', 'torch', and optionally 'python_range' keys.
    The table format is:
    <tr><td>CUDA</td><td>PyTorch</td><td>Python</td><td><a href="cuXXX-torchYYY/">...</a></td></tr>
    """
    configs = []
    platforms = platforms_by_config(html)
    
    # Find all table rows with config links
    # Pattern: <tr><td>13.0</td><td>2.9.1</td><td>3.10-3.13</td><td><a href="cu130-torch291/">cu130-torch291</a></td></tr>
    pattern = r'<tr><td>([0-9.]+)</td><td>([0-9.]+)</td><td>([0-9.-]+)</td><td><a href="([^"]+)/">'
    
    for match in re.finditer(pattern, html):
        cuda_str, torch_str, python_range_str, config_name = match.groups()
        
        try:
            # Parse CUDA version
            cuda_tuple = parse_cuda_version(cuda_str)
            
            # Parse PyTorch version
            torch_tuple = parse_version(torch_str)
            
            # Parse Python range (optional, may fail)
            python_range = parse_python_range(python_range_str)
            
            config = {
                'name': config_name,
                'cuda': cuda_tuple,
                'torch': torch_tuple,
            }
            
            if python_range:
                config['python_range'] = python_range

            config_platforms = platforms.get(config_name)
            if config_platforms:
                config['platforms'] = config_platforms
            
            configs.append(config)
            
        except (ValueError, AttributeError):
            # Skip configs we can't parse
            continue
    
    # Sort by PyTorch version (descending), then CUDA version (descending)
    configs.sort(key=lambda c: (c['torch'], c['cuda']), reverse=True)
    
    return configs


def platforms_by_config(html: str) -> dict:
    """Map a config name to the platform tags linked from an index page.

    Root-index wheel URLs look like ``.../cu128-torch291/name-linux_x86_64.whl``.
    """
    found = {}
    pattern = r'(cu\d+-torch\d+)/[^"\s<>]*-(linux_x86_64|win_amd64)\.whl'
    for match in re.finditer(pattern, html):
        name, tag = match.groups()
        found.setdefault(name, set()).add(tag)
    return found


def host_wheel_platform(system: str, machine: str) -> Optional[str]:
    """Return the wheel platform tag this machine can install, or None.

    Published wheels are tagged linux_x86_64 or win_amd64. Other machines
    get no matching file from pip.
    """
    machine_name = machine.lower()
    if system == "Linux" and machine_name in {"x86_64", "amd64"}:
        return "linux_x86_64"
    if system == "Windows" and machine_name in {"amd64", "x86_64"}:
        return "win_amd64"
    return None


def platform_refusal(host_platform: Optional[str], config: dict) -> Optional[str]:
    """Return an explanation when this machine cannot install the config."""
    available = config.get("platforms")
    if host_platform is None:
        return (
            "No published wheel matches this machine.\n"
            "Wheels are tagged linux_x86_64 or win_amd64."
        )
    if available and host_platform not in available:
        tags = ", ".join(sorted(available))
        return (
            f"{config['name']} has no {host_platform} wheel.\n"
            f"Published platform tags for that config: {tags}"
        )
    return None


def fetch_remote_index(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch the remote index.html.
    Returns the HTML content as a string, or None if the fetch fails.
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode('utf-8')
    except (URLError, OSError, UnicodeDecodeError):
        return None


def discover_configs(wheels_dir: Path) -> List[dict]:
    """
    Discover available wheel configurations by scanning wheels/*/ directories
    or by parsing wheels/index.html if it exists.
    
    Returns list of dicts with 'name', 'cuda', 'torch', and optionally 'python_range' keys.
    
    If index.html exists, prefer parsing it (includes Python ranges).
    Otherwise, fall back to directory scanning (no Python ranges).
    """
    configs = []
    
    if not wheels_dir.exists():
        return configs
    
    # Try to parse index.html first (includes Python ranges)
    index_html_path = wheels_dir / 'index.html'
    if index_html_path.exists():
        try:
            with open(index_html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            configs = parse_configs_from_index_html(html)
            if configs:
                return configs
        except (OSError, UnicodeDecodeError):
            pass
    
    # Fall back to directory scanning
    for config_dir in wheels_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == '.git':
            continue
        
        # Parse config name: cu{CUDA_SHORT}-torch{TORCH_SHORT}
        match = re.match(r'^cu(\d+)-torch(\d+)$', config_dir.name)
        if not match:
            continue
        
        cuda_short, torch_short = match.groups()
        
        # Convert short forms to full versions
        # CUDA: 128 -> (12, 8), 130 -> (13, 0)
        cuda_tuple = parse_cuda_version(cuda_short)
        
        # PyTorch: 291 -> (2, 9, 1), 2150 -> (2, 15, 0). Four digits are a
        # two-digit minor, which 2.10 and later need. The index generator
        # uses the same split.
        if len(torch_short) == 3:
            torch_tuple = (int(torch_short[0]), int(torch_short[1]), int(torch_short[2]))
        elif len(torch_short) == 4:
            torch_tuple = (int(torch_short[0]), int(torch_short[1:3]), int(torch_short[3]))
        elif len(torch_short) == 2:
            torch_tuple = (int(torch_short[0]), int(torch_short[1]), 0)
        else:
            continue
        
        config = {
            'name': config_dir.name,
            'cuda': cuda_tuple,
            'torch': torch_tuple,
        }
        index_path = config_dir / 'index.html'
        if index_path.is_file():
            try:
                page = index_path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                page = ''
            # A config page's wheel links all belong to this directory.
            tags = set(re.findall(r'-(linux_x86_64|win_amd64)\.whl', page))
            if tags:
                config['platforms'] = tags
        configs.append(config)
    
    # Sort by PyTorch version (descending), then CUDA version (descending)
    configs.sort(key=lambda c: (c['torch'], c['cuda']), reverse=True)
    
    return configs


def check_python_version_supported(python_version: Tuple[int, int], config: dict) -> bool:
    """
    Check if the Python version is supported by the config.
    Returns True if supported, False if not, or True if python_range is not available.
    """
    if 'python_range' not in config:
        # No range info available, assume supported
        return True
    
    min_version, max_version = config['python_range']
    return min_version <= python_version <= max_version


def find_matching_config(python_version: Tuple[int, int], 
                        torch_version: Optional[Tuple[int, int, int]],
                        cuda_version: Optional[Tuple[int, int]],
                        configs: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Find the exact matching config or the nearest alternative.
    
    Returns: (exact_match, nearest_match)
    - exact_match: config with exact torch and CUDA match, or None
    - nearest_match: config with same torch major.minor, highest CUDA <= local, or None
    """
    if not torch_version or not cuda_version:
        return None, None
    
    exact_match = None
    nearest_match = None
    
    torch_major_minor = (torch_version[0], torch_version[1])
    
    # Find exact match
    for config in configs:
        if config['torch'] == torch_version and config['cuda'] == cuda_version:
            exact_match = config
            break
    
    # Find nearest match (same torch major.minor, highest CUDA <= local)
    candidates = []
    for config in configs:
        config_torch_mm = (config['torch'][0], config['torch'][1])
        if config_torch_mm == torch_major_minor and config['cuda'] <= cuda_version:
            candidates.append(config)
    
    if candidates:
        # Pick the one with the highest CUDA version
        nearest_match = max(candidates, key=lambda c: c['cuda'])
    
    return exact_match, nearest_match


def detect_environment() -> Tuple[Tuple[int, int], Optional[Tuple[int, int, int]], Optional[Tuple[int, int]]]:
    """
    Detect Python version, PyTorch version, and CUDA version from the environment.
    
    Returns: (python_version, torch_version, cuda_version)
    - python_version: always present as (major, minor)
    - torch_version: (major, minor, patch) or None if torch not installed
    - cuda_version: (major, minor) or None if torch not installed or CPU-only
    """
    python_version = (sys.version_info.major, sys.version_info.minor)
    
    torch_version = None
    cuda_version = None
    
    try:
        import torch
        torch_version = parse_version(torch.__version__)
        
        # Check CUDA availability
        if torch.version.cuda is not None:
            cuda_version = parse_cuda_version(torch.version.cuda)
        
    except ImportError:
        pass
    
    return python_version, torch_version, cuda_version


def format_pip_command(config_name: str, base_url: str) -> str:
    """Format the pip install command for a given config."""
    return f"pip install nvdiffrec_render --find-links {base_url}{config_name}/"


def main():
    parser = argparse.ArgumentParser(
        description="Find the correct nvdiffrec_render wheel for your environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect from installed PyTorch
  python find_wheel.py
  
  # Override with specific versions
  python find_wheel.py --torch 2.9.1 --cuda 12.8 --python 3.11
  
  # Check compatibility without torch installed
  python find_wheel.py --torch 2.8.0 --cuda 12.6
        """
    )
    
    parser.add_argument(
        '--torch',
        type=str,
        help='PyTorch version (e.g., 2.9.1, 2.8.0)'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        help='CUDA version (e.g., 12.8, 13.0)'
    )
    parser.add_argument(
        '--python',
        type=str,
        help='Python version (e.g., 3.11, 3.12)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://by-ae.github.io/nvdiffrec_render-wheels/',
        help='Base URL for wheel index (default: GitHub Pages URL)'
    )
    parser.add_argument(
        '--wheels-dir',
        type=str,
        help='Path to wheels directory (default: ./wheels relative to script)'
    )
    
    args = parser.parse_args()
    
    # Determine wheels directory
    if args.wheels_dir:
        wheels_dir = Path(args.wheels_dir)
    else:
        script_dir = Path(__file__).parent
        wheels_dir = script_dir / 'wheels'
    
    # Discover available configurations
    configs = discover_configs(wheels_dir)
    
    # If no configs found and --wheels-dir was not explicitly given, try remote fallback
    if not configs and not args.wheels_dir:
        # Try GitHub Pages first, then raw GitHub as fallback
        remote_urls = [
            args.base_url.rstrip('/') + '/index.html',
            'https://raw.githubusercontent.com/by-ae/nvdiffrec_render-wheels/main/wheels/index.html'
        ]
        
        html = None
        for url in remote_urls:
            html = fetch_remote_index(url)
            if html:
                break
        
        if html:
            configs = parse_configs_from_index_html(html)
        
        if not configs:
            print("Error: No wheel configurations found.", file=sys.stderr)
            print(f"Local directory does not exist: {wheels_dir}", file=sys.stderr)
            if html:
                print(f"Remote index was fetched but contained no configs", file=sys.stderr)
            else:
                print(f"Could not fetch remote index from:", file=sys.stderr)
                for url in remote_urls:
                    print(f"  {url}", file=sys.stderr)
            print(file=sys.stderr)
            print("To fix this:", file=sys.stderr)
            print("1. Clone the repository: git clone https://github.com/by-ae/nvdiffrec_render-wheels", file=sys.stderr)
            print("2. Run from the repository root, or use --wheels-dir to specify the path", file=sys.stderr)
            sys.exit(1)
    elif not configs:
        print("Error: No wheel configurations found.", file=sys.stderr)
        if wheels_dir.exists():
            print(f"Checked directory: {wheels_dir}", file=sys.stderr)
        else:
            print(f"Directory does not exist: {wheels_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Detect or parse environment
    python_version, torch_version, cuda_version = detect_environment()
    
    # Override with command-line arguments
    if args.python:
        parts = args.python.split('.')
        if len(parts) < 2:
            print(f"Error: Invalid Python version format: {args.python}", file=sys.stderr)
            print("Expected format: X.Y (e.g., 3.11)", file=sys.stderr)
            sys.exit(1)
        python_version = (int(parts[0]), int(parts[1]))
    
    if args.torch:
        try:
            torch_version = parse_version(args.torch)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    if args.cuda:
        try:
            cuda_version = parse_cuda_version(args.cuda)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Display detected environment
    host_platform = host_wheel_platform(platform.system(), platform.machine())
    print("Detected environment:")
    print(f"  Python: {python_version[0]}.{python_version[1]}")
    if host_platform:
        print(f"  Platform: {host_platform}")
    else:
        print(f"  Platform: {platform.system()} {platform.machine()} (no published wheel)")
    
    if torch_version:
        torch_str = f"{torch_version[0]}.{torch_version[1]}.{torch_version[2]}"
        print(f"  PyTorch: {torch_str}")
    else:
        print("  PyTorch: not installed (use --torch to specify)")
    
    if cuda_version:
        cuda_str = format_cuda_version(cuda_version)
        print(f"  CUDA: {cuda_str}")
    elif torch_version:
        print("  CUDA: CPU-only build detected")
    
    print()
    
    # Handle CPU-only torch
    if torch_version and not cuda_version and not args.cuda:
        print("❌ Your PyTorch installation is CPU-only (no CUDA support).")
        print()
        print("nvdiffrec_render requires a CUDA-enabled PyTorch installation.")
        print()
        print("To fix this:")
        print("1. Install PyTorch with CUDA support from https://pytorch.org")
        print("2. Then run this script again, or use --cuda flag if PyTorch misreports")
        print()
        print("Example (CUDA 12.8):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)
    
    # Check if we have enough info to find a wheel
    if not torch_version:
        print("Cannot determine wheel without PyTorch version.")
        print("Either install PyTorch first, or use --torch and --cuda flags.")
        print()
        print("Example:")
        print("  python find_wheel.py --torch 2.9.1 --cuda 12.8")
        print()
        print("Available configurations:")
        for config in configs:
            cuda_str = format_cuda_version(config['cuda'])
            torch_str = f"{config['torch'][0]}.{config['torch'][1]}.{config['torch'][2]}"
            print(f"  {config['name']}: CUDA {cuda_str}, PyTorch {torch_str}")
        sys.exit(1)
    
    if not cuda_version:
        print("Cannot determine wheel without CUDA version.")
        print("Use --cuda flag to specify your CUDA version.")
        print()
        print("Example:")
        print(f"  python find_wheel.py --torch {torch_version[0]}.{torch_version[1]}.{torch_version[2]} --cuda 12.8")
        sys.exit(1)
    
    # Find matching configuration
    exact_match, nearest_match = find_matching_config(
        python_version, torch_version, cuda_version, configs
    )
    
    if exact_match:
        # Check if Python version is supported
        if not check_python_version_supported(python_version, exact_match):
            print("❌ Found matching CUDA and PyTorch, but Python version not supported.")
            print()
            cuda_str = format_cuda_version(exact_match['cuda'])
            torch_str = f"{exact_match['torch'][0]}.{exact_match['torch'][1]}.{exact_match['torch'][2]}"
            print(f"Config: {exact_match['name']} (CUDA {cuda_str}, PyTorch {torch_str})")
            
            if 'python_range' in exact_match:
                min_py, max_py = exact_match['python_range']
                print(f"Supported Python: {min_py[0]}.{min_py[1]}-{max_py[0]}.{max_py[1]}")
            
            print(f"Your Python: {python_version[0]}.{python_version[1]}")
            print()
            print("Available configurations:")
            for config in configs:
                cuda_str = format_cuda_version(config['cuda'])
                torch_str = f"{config['torch'][0]}.{config['torch'][1]}.{config['torch'][2]}"
                py_range = ""
                if 'python_range' in config:
                    min_py, max_py = config['python_range']
                    py_range = f", Python {min_py[0]}.{min_py[1]}-{max_py[0]}.{max_py[1]}"
                print(f"  {config['name']}: CUDA {cuda_str}, PyTorch {torch_str}{py_range}")
            sys.exit(1)
        
        refusal = platform_refusal(host_platform, exact_match)
        if refusal:
            print(refusal)
            print()
            sys.exit(1)

        print("✅ Found exact match!")
        print()
        print("Install command:")
        print(f"  {format_pip_command(exact_match['name'], args.base_url)}")
        print()
        sys.exit(0)
    
    # No exact match. torch.version.cuda is the CUDA build PyTorch was
    # compiled with, not a driver version that can load an older wheel.
    print("❌ No exact match found for your configuration.")
    print()
    print("The extension loads only in the PyTorch build it was compiled against.")
    print("There is no install command for a different PyTorch or CUDA version.")
    print()

    if nearest_match:
        cuda_str = format_cuda_version(nearest_match['cuda'])
        torch_str = f"{nearest_match['torch'][0]}.{nearest_match['torch'][1]}.{nearest_match['torch'][2]}"
        print(f"Closest published config is {nearest_match['name']} (CUDA {cuda_str}, PyTorch {torch_str}).")
        print("Installing that wheel into this environment will not import.")
        print()

    print("Available configurations:")
    for config in configs:
        cuda_str = format_cuda_version(config['cuda'])
        torch_str = f"{config['torch'][0]}.{config['torch'][1]}.{config['torch'][2]}"
        print(f"  {config['name']}: CUDA {cuda_str}, PyTorch {torch_str}")
    
    print()
    print("To install PyTorch with a specific CUDA version, visit:")
    print("  https://pytorch.org")
    
    sys.exit(1)


if __name__ == '__main__':
    main()
