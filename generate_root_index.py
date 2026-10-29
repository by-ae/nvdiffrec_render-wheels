#!/usr/bin/env python3
"""
Generate wheels/index.html from the per-configuration indexes.

Python ranges come from the wheel filenames that are actually linked, not
from a hardcoded table. A configuration with no valid wheel links stays
reachable (the top index must link every subdirectory) but is not listed
as installable. The root page lists every wheel anchor, including any
PEP 503 hash fragment. Pip then keeps files whose tags match and installs
the newest version; it does not compare the local label to PyTorch.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

import validate_index


def torch_from_short(torch_short: str):
    """Return '2.9.1' or '2.15.0' from a dot-stripped PyTorch tag.

    Three digits are one digit per component (2.9.1 -> 291). Four digits
    are a one-digit major, a two-digit minor, and a one-digit patch
    (2.15.0 -> 2150). Other lengths are ambiguous and rejected.
    """
    if len(torch_short) == 3:
        major, minor, patch = torch_short[0], torch_short[1], torch_short[2]
    elif len(torch_short) == 4:
        major, minor, patch = torch_short[0], torch_short[1:3], torch_short[3]
    else:
        return None
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def versions_from_dirname(name: str):
    """Return (cuda, torch) display strings for cu128-torch291, or None."""
    match = re.fullmatch(r'cu(\d+)-torch(\d+)', name)
    if not match:
        return None
    cuda_short, torch_short = match.groups()
    if len(cuda_short) not in (3, 4):
        return None
    torch = torch_from_short(torch_short)
    if torch is None:
        return None
    cuda = f"{int(cuda_short[:2])}.{int(cuda_short[2:])}"
    return cuda, torch


def python_range(filenames):
    """Return '3.10-3.13' when the cp tags form one contiguous range."""
    versions = set()
    for name in filenames:
        match = re.search(r'-cp(3\d+)-', name)
        if not match:
            raise SystemExit(f"Wheel filename has no Python tag: {name}")
        tag = match.group(1)
        versions.add((int(tag[0]), int(tag[1:])))
    ordered = sorted(versions)
    for (major, minor), (next_major, next_minor) in zip(ordered, ordered[1:]):
        if major != next_major or next_minor != minor + 1:
            raise SystemExit(
                f"Python versions are not a contiguous range: {ordered}"
            )
    low, high = ordered[0], ordered[-1]
    return f"{low[0]}.{low[1]}-{high[0]}.{high[1]}"


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href', '')
        path, _fragment = validate_index.split_wheel_href(href)
        if path.endswith('.whl'):
            self.hrefs.append(href)


def valid_wheel_links(html: str):
    """Return (filename, href) for anchors that pass the wheel-name check."""
    parser = _AnchorParser()
    parser.feed(html)
    links = []
    for href in parser.hrefs:
        if validate_index.wheel_href_error(href):
            continue
        path, _fragment = validate_index.split_wheel_href(href)
        links.append((path.rsplit('/', 1)[-1], href))
    links.sort(key=lambda item: item[0])
    return links


def collect_configs(wheels_dir: Path):
    """Return (published, unpublished) config dicts sorted by CUDA then PyTorch."""
    published = []
    unpublished = []
    for subdir in wheels_dir.iterdir():
        if not subdir.is_dir() or not (subdir / 'index.html').exists():
            continue
        versions = versions_from_dirname(subdir.name)
        if versions is None:
            raise SystemExit(f"Unrecognized configuration directory: {subdir.name}")
        cuda, torch = versions
        links = valid_wheel_links((subdir / 'index.html').read_text(encoding='utf-8'))
        entry = {
            'name': subdir.name,
            'cuda': cuda,
            'torch': torch,
            'links': links,
        }
        if links:
            entry['python'] = python_range([name for name, _href in links])
            published.append(entry)
        else:
            unpublished.append(entry)

    def sort_key(entry):
        cuda_parts = tuple(int(part) for part in entry['cuda'].split('.'))
        torch_parts = tuple(int(part) for part in entry['torch'].split('.'))
        return cuda_parts, torch_parts

    published.sort(key=sort_key)
    unpublished.sort(key=sort_key)
    return published, unpublished


def wheel_version(filename: str) -> str:
    """Return the version component of a nvdiffrec_render wheel filename."""
    stem = filename[:-4] if filename.endswith('.whl') else filename
    parts = stem.split('-')
    if len(parts) < 5 or parts[0] != 'nvdiffrec_render' or not parts[1]:
        raise SystemExit(f"Cannot read wheel version: {filename}")
    return parts[1]


def preferred_release(published):
    """Return (version, entry) for the file pip installs from the combined index.

    Pip keeps wheels whose Python, ABI, and platform tags fit, then selects
    the newest PEP 440 version. Every published file here is 0.0.1 plus one
    alphanumeric local label, and ASCII order matches that rule. Refuse other
    shapes instead of printing a version pip would not pick.
    """
    if not published:
        raise SystemExit("No published wheels to rank")
    ranked = []
    for entry in published:
        versions = {wheel_version(name) for name, _href in entry['links']}
        if len(versions) != 1:
            raise SystemExit(
                f"{entry['name']} mixes wheel versions: {sorted(versions)}"
            )
        version = next(iter(versions))
        if '+' not in version:
            raise SystemExit(f"{entry['name']} has no local version: {version}")
        public, local = version.split('+', 1)
        if public != '0.0.1' or not local.isalnum():
            raise SystemExit(
                f"Refusing to rank {version}; pip's order is only obvious for "
                "0.0.1 plus one alphanumeric local label"
            )
        ranked.append((version, entry))
    return max(ranked, key=lambda item: item[0])


def build_html(wheels_dir: Path) -> str:
    if not wheels_dir.exists():
        raise SystemExit(f"Error: {wheels_dir} directory not found")

    published, unpublished = collect_configs(wheels_dir)
    html = """<!DOCTYPE html>
<html>
<head>
    <title>nvdiffrec_render Wheels</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        a { color: #0366d6; }
        pre { background: #f6f8fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f6f8fa; }
        .wheels { font-size: 0.9em; margin-top: 30px; }
        .wheels h3 { margin-top: 20px; }
    </style>
</head>
<body>
<h1>nvdiffrec_render Wheels</h1>
<p>Pre-built Python wheels for nvdiffrec rendering utilities.</p>

<h2>Available Configurations</h2>
<table>
    <tr><th>CUDA</th><th>PyTorch</th><th>Python</th><th>Link</th></tr>
"""
    for entry in published:
        html += (
            f"    <tr><td>{entry['cuda']}</td><td>{entry['torch']}</td>"
            f"<td>{entry['python']}</td>"
            f"<td><a href=\"{entry['name']}/\">{entry['name']}</a></td></tr>\n"
        )
    html += "</table>\n"
    if published:
        version, preferred = preferred_release(published)
        html += (
            "\n<h2>Install</h2>\n"
            "<p>Pip keeps a wheel when its Python, ABI, and platform tags match "
            "this machine, then installs the newest version. It does not look at "
            "which PyTorch build is already installed. The newest file here is "
            f"<code>{version}</code> "
            f"(CUDA {preferred['cuda']}, PyTorch {preferred['torch']}, "
            f"<a href=\"{preferred['name']}/\">{preferred['name']}</a>). "
            "Open another configuration above for a different build.</p>\n"
            "<pre>pip install nvdiffrec_render --find-links "
            "https://by-ae.github.io/nvdiffrec_render-wheels/</pre>\n"
        )
    else:
        html += (
            "\n<h2>Install</h2>\n"
            "<p>No wheel files are published yet.</p>\n"
        )
    if unpublished:
        html += "\n<h2>No wheels published yet</h2>\n"
        for entry in unpublished:
            html += (
                f"<p><a href=\"{entry['name']}/\">{entry['name']}</a> "
                f"(CUDA {entry['cuda']}, PyTorch {entry['torch']}) "
                f"has an index page and no wheel files.</p>\n"
            )
    html += """
<div class="wheels">
<h2>All Wheels</h2>
<p>Every published wheel file. On the root page, pip installs only the newest compatible file named above.</p>
"""
    for entry in published:
        html += f"<h3>{entry['name']}</h3>\n"
        for filename, href in entry['links']:
            html += f'<a href="{href}">{filename}</a><br>\n'
    html += """</div>

<p style="margin-top: 30px;"><a href="https://github.com/by-ae/nvdiffrec_render-wheels">GitHub Repository</a></p>
</body>
</html>
"""
    return html


def main():
    wheels_dir = Path('wheels')
    html = build_html(wheels_dir)
    output_file = wheels_dir / 'index.html'
    output_file.write_text(html, encoding='utf-8')
    published, unpublished = collect_configs(wheels_dir)
    print(f"Generated {output_file}")
    print(f"Found {len(published)} configurations with wheels")
    if unpublished:
        print("No wheels yet: " + ", ".join(entry['name'] for entry in unpublished))


if __name__ == '__main__':
    main()
