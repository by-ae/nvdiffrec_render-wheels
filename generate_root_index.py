#!/usr/bin/env python3
"""
Generate wheels/index.html from the per-configuration indexes.

Python ranges come from the wheel filenames that are actually linked, not
from a hardcoded table. A configuration with no valid wheel links stays
reachable (the top index must link every subdirectory) but is not listed
as installable. Pip --find-links uses the wheel anchors copied from the
per-configuration pages, including any PEP 503 hash fragment.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

import validate_index


def versions_from_dirname(name: str):
    """Return (cuda, torch) display strings for cu128-torch291, or None."""
    match = re.fullmatch(r'cu(\d+)-torch(\d+)', name)
    if not match:
        return None
    cuda_short, torch_short = match.groups()
    if len(cuda_short) not in (3, 4) or len(torch_short) != 3:
        return None
    cuda = f"{int(cuda_short[:2])}.{int(cuda_short[2:])}"
    torch = f"{int(torch_short[0])}.{int(torch_short[1])}.{int(torch_short[2])}"
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

<h2>Install</h2>
<pre>pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/</pre>

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
<p>Direct links to all available wheels (for pip --find-links):</p>
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
