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
        
        # Validate each wheel filename and optional PEP 503 hash fragment
        for link in parser.links:
            error_msg = wheel_href_error(link)
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
            error_msg = wheel_href_error(link)
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
        print(f"❌ {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  - {err}")
    
    if all_warnings:
        print(f"⚠️  {len(all_warnings)} warning(s):")
        for warn in all_warnings:
            print(f"  - {warn}")
    
    if not all_errors and not all_warnings:
        print("✅ All index files are valid")
    
    print()
    
    if all_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
