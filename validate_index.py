#!/usr/bin/env python3
"""
Validation script for PEP 503 simple index HTML structure.
Checks that index pages are well-formed and internally consistent.
"""
import sys
import os
import re
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Set, Tuple


class IndexHTMLParser(HTMLParser):
    """Parse index.html files and extract wheel links."""
    
    def __init__(self):
        super().__init__()
        self.links = []
        self.subdirs = []
        self.in_body = False
    
    def handle_starttag(self, tag, attrs):
        if tag == 'body':
            self.in_body = True
        elif tag == 'a' and self.in_body:
            href = dict(attrs).get('href')
            if href:
                path, _fragment = split_wheel_href(href)
                if path.endswith('.whl'):
                    self.links.append(href)
                elif '/' in path or path.endswith('/'):
                    # Track subdirectory links. Hash fragments are not part of
                    # the path, so a wheel URL with #sha256=... stays a wheel.
                    self.subdirs.append(path.rstrip('/'))


def _print(line=""):
    """Print a status line when the console cannot encode every character.

    A cp1252 console raises UnicodeEncodeError on the summary marks.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding))


def split_wheel_href(href: str) -> Tuple[str, str]:
    """Split a link into path and optional URL fragment.

    PEP 503 puts the file hash in the fragment (#sha256=<hex>). The path is
    what ends with .whl; the fragment must not be treated as part of the
    filename or as a subdirectory.
    """
    path, sep, fragment = href.partition('#')
    return path, fragment if sep else ''


def wheel_href_error(href: str) -> str:
    """Return an error message for a wheel href, or '' when it is acceptable.

    A missing hash is allowed. A present hash must be a single sha256 digest,
    which is the fragment this repo's index generator writes.
    """
    path, fragment = split_wheel_href(href)
    if fragment and not re.fullmatch(r'sha256=[a-f0-9]{64}', fragment):
        return f"Invalid or unsupported hash fragment (expected #sha256=<64 hex>): {href}"
    filename = path.rsplit('/', 1)[-1]
    _ok, message = validate_wheel_filename(filename)
    return '' if _ok else message


def expected_local_tag(config_name: str):
    """Return ``cu128torch29`` for ``cu128-torch291``, or None.

    The directory keeps the PyTorch patch (torch291 is 2.9.1). The wheel's
    local version keeps major and minor only (torch29), which is the
    NVDIFFREC_LOCAL_VERSION the build writes.
    """
    match = re.fullmatch(r'cu(\d+)-torch(\d+)', config_name)
    if not match:
        return None
    cuda_short, torch_short = match.groups()
    if not cuda_short.isdigit():
        return None
    if len(torch_short) == 3 and torch_short.isdigit():
        major, minor = int(torch_short[0]), int(torch_short[1])
    elif len(torch_short) == 4 and torch_short.isdigit():
        major, minor = int(torch_short[0]), int(torch_short[1:3])
    else:
        return None
    return f"cu{int(cuda_short)}torch{major}{minor}"


def filename_local_tag(filename: str):
    """Return the local version in a wheel filename, or None."""
    if not filename.endswith('.whl'):
        return None
    parts = filename[:-4].split('-')
    if len(parts) < 2 or '+' not in parts[1]:
        return None
    local = parts[1].split('+', 1)[1]
    return local or None


def config_name_from_href(href: str):
    """Return the ``cu128-torch291`` directory in a wheel URL, or None."""
    path, _fragment = split_wheel_href(href)
    match = re.search(r'(cu\d+-torch\d+)/[^/]+\.whl$', path)
    return match.group(1) if match else None


def local_tag_error(filename: str, config_name: str) -> str:
    """Return an error when the wheel's local version is not this config's.

    Pip installs whatever file the page links. A cu126 wheel on the cu128
    page loads in the wrong PyTorch build and fails at import.
    """
    expected = expected_local_tag(config_name)
    if expected is None:
        return ''
    actual = filename_local_tag(filename)
    if actual is None:
        return f"Wheel filename has no local version for {config_name}: {filename}"
    if actual != expected:
        return f"{filename} is tagged {actual}, but {config_name} publishes {expected}"
    return ''


def wheel_link_error(href: str, config_name: str = '') -> str:
    """Return an error for a wheel href on a config page or the root index.

    ``config_name`` is the directory that owns the page. A name that is not
    a configuration is ignored, so a scratch index can still be checked for
    filename shape alone.
    """
    message = wheel_href_error(href)
    if message:
        return message
    path, _fragment = split_wheel_href(href)
    filename = path.rsplit('/', 1)[-1]
    href_config = config_name_from_href(href)
    page_config = config_name if expected_local_tag(config_name) else ''
    if href_config and page_config and href_config != page_config:
        return f"Link is under {href_config}, but this page is {page_config}: {href}"
    expected = href_config or page_config
    if not expected:
        return ''
    return local_tag_error(filename, expected)


def validate_wheel_filename(filename: str) -> Tuple[bool, str]:
    """
    Validate that a filename follows wheel naming conventions.
    Format: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    
    This project uses non-standard wheel names with CUDA+torch tags in both
    the version and build fields (e.g., 0.0.1+cu128torch29-cp310+cu128torch29-cp310).
    We validate the basic structure while allowing this deviation.
    """
    # Check basic .whl extension
    if not filename.endswith('.whl'):
        return False, f"Does not end with .whl: {filename}"
    
    # Check for nvdiffrec_render package name
    if not filename.startswith('nvdiffrec_render-'):
        return False, f"Does not start with 'nvdiffrec_render-': {filename}"
    
    # Check for doubled tag pattern (the bug we fixed)
    # Pattern: +cuXXXtorchYY-cpZZZ+cuXXXtorchYY-cpZZZ (same tag appears twice)
    doubled_tag_pattern = r'\+cu\d+torch\d+-cp\d+\+cu\d+torch\d+-cp\d+'
    if re.search(doubled_tag_pattern, filename):
        return False, f"Contains doubled tag (Windows rename bug): {filename}"
    
    # Split by '-' and check we have at least 5 parts (name-version-python-abi-platform)
    parts = filename.split('-')
    if len(parts) < 5:
        return False, f"Too few components (expected at least 5): {filename}"
    
    # Check python tag (should be cpXYZ where XYZ is python version)
    # It appears after version in this project's naming
    has_python_tag = any(part.startswith('cp3') for part in parts)
    if not has_python_tag:
        return False, f"Missing Python version tag (cp3XX): {filename}"
    
    # Check platform tag (linux_x86_64 or win_amd64)
    has_platform = any(part.endswith('linux_x86_64.whl') or part.endswith('win_amd64.whl') for part in parts)
    if not has_platform:
        return False, f"Missing or invalid platform tag (expected linux_x86_64 or win_amd64): {filename}"
    
    return True, ""


def validate_subindex(path: Path) -> Tuple[List[str], List[str]]:
    """Validate a subdirectory index.html file."""
    errors = []
    warnings = []
    
    if not path.exists():
        return [f"Index file not found: {path}"], []
    
    print(f"Validating: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = IndexHTMLParser()
        parser.feed(content)
        
        if not parser.links:
            warnings.append(f"No wheel links found in {path}")
        
        print(f"  Found {len(parser.links)} wheel link(s)")
        
        # Validate each wheel filename, optional hash, and the config tag.
        # The parent directory is the configuration when this is a real page.
        page_config = path.parent.name
        for link in parser.links:
            error_msg = wheel_link_error(link, page_config)
            if error_msg:
                errors.append(f"Invalid wheel link in {path}: {error_msg}")
        
        # Check for basic HTML structure
        if '<html>' not in content.lower() or '</html>' not in content.lower():
            errors.append(f"Missing proper HTML tags in {path}")
        
        if '<body>' not in content.lower() or '</body>' not in content.lower():
            errors.append(f"Missing body tags in {path}")
        
    except Exception as e:
        errors.append(f"Failed to parse {path}: {e}")
    
    return errors, warnings


def validate_top_index(path: Path, wheels_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate the top-level index.html and check it links to all subdirs."""
    errors = []
    warnings = []
    
    if not path.exists():
        return [f"Top-level index not found: {path}"], []
    
    print(f"Validating: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = IndexHTMLParser()
        parser.feed(content)
        
        print(f"  Found {len(parser.subdirs)} subdirectory link(s)")
        
        # Find all actual subdirectories with index.html
        actual_subdirs = set()
        for subdir in wheels_dir.iterdir():
            if subdir.is_dir() and (subdir / 'index.html').exists():
                actual_subdirs.add(subdir.name)
        
        # Check that all actual subdirs are linked in top index
        # Filter out external links (those starting with http://, https://, etc.)
        linked_subdirs = {link.rstrip('/') for link in parser.subdirs 
                         if not link.startswith(('http://', 'https://', '//'))}
        
        missing_links = actual_subdirs - linked_subdirs
        if missing_links:
            errors.append(f"Top-level index missing links to: {', '.join(sorted(missing_links))}")
        
        extra_links = linked_subdirs - actual_subdirs
        if extra_links:
            warnings.append(f"Top-level index links to non-existent subdirs: {', '.join(sorted(extra_links))}")
        
        print(f"  Actual subdirs: {sorted(actual_subdirs)}")
        print(f"  Linked subdirs: {sorted(linked_subdirs)}")

        for link in parser.links:
            error_msg = wheel_link_error(link)
            if not error_msg and config_name_from_href(link) is None:
                error_msg = f"Wheel link is not under a configuration directory: {link}"
            if error_msg:
                errors.append(f"Invalid wheel link in {path}: {error_msg}")
        
    except Exception as e:
        errors.append(f"Failed to parse {path}: {e}")
    
    return errors, warnings


def main():
    """Run all index validation checks."""
    if len(sys.argv) > 1:
        wheels_dir = Path(sys.argv[1])
    else:
        wheels_dir = Path('wheels')
    
    if not wheels_dir.exists():
        print(f"Error: wheels directory not found: {wheels_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Index HTML Validation")
    print("=" * 60)
    print()
    
    all_errors = []
    all_warnings = []
    
    # Validate top-level index
    top_index = wheels_dir / 'index.html'
    errors, warnings = validate_top_index(top_index, wheels_dir)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    print()
    
    # Validate each subdirectory index
    for subdir in sorted(wheels_dir.iterdir()):
        if subdir.is_dir():
            index_file = subdir / 'index.html'
            if index_file.exists():
                errors, warnings = validate_subindex(index_file)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
                print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_errors:
        _print(f"❌ {len(all_errors)} error(s):")
        for err in all_errors:
            _print(f"  - {err}")
    
    if all_warnings:
        _print(f"⚠️  {len(all_warnings)} warning(s):")
        for warn in all_warnings:
            _print(f"  - {warn}")
    
    if not all_errors and not all_warnings:
        _print("✅ All index files are valid")
    
    print()
    
    if all_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
