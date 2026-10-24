#!/usr/bin/env python3
"""
Generate the root wheels/index.html that includes links to all wheels
from all subdirectories, making --find-links to the root URL work reliably.
"""
from pathlib import Path
import sys

def main():
    wheels_dir = Path('wheels')
    if not wheels_dir.exists():
        print(f"Error: {wheels_dir} directory not found")
        sys.exit(1)
    
    # Collect all subdirectories
    subdirs = sorted([d for d in wheels_dir.iterdir() if d.is_dir() and (d / 'index.html').exists()])
    
    # Start building the HTML
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
    
    # Configuration mapping (parsed from directory names)
    config_info = {
        'cu130-torch291': ('13.0', '2.9.1', '3.10-3.13'),
        'cu128-torch291': ('12.8', '2.9.1', '3.10-3.13'),
        'cu128-torch280': ('12.8', '2.8.0', '3.10-3.13'),
        'cu126-torch280': ('12.6', '2.8.0', '3.10-3.13'),
        'cu126-torch260': ('12.6', '2.6.0', '3.10-3.13'),
        'cu124-torch251': ('12.4', '2.5.1', '3.10-3.12'),
    }
    
    for subdir in subdirs:
        name = subdir.name
        if name in config_info:
            cuda, torch, python = config_info[name]
            html += f'    <tr><td>{cuda}</td><td>{torch}</td><td>{python}</td><td><a href="{name}/">{name}</a></td></tr>\n'
    
    html += """</table>

<div class="wheels">
<h2>All Wheels</h2>
<p>Direct links to all available wheels (for pip --find-links):</p>
"""
    
    # Collect wheel links from each subdir's index
    from html.parser import HTMLParser
    
    class WheelLinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.wheel_links = []
        
        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                href = dict(attrs).get('href', '')
                if '.whl' in href:
                    self.wheel_links.append(href)
    
    for subdir in subdirs:
        index_file = subdir / 'index.html'
        parser = WheelLinkParser()
        parser.feed(index_file.read_text())
        
        if parser.wheel_links:
            html += f'<h3>{subdir.name}</h3>\n'
            for link in parser.wheel_links:
                # Keep the full GitHub release URLs
                filename = link.split('/')[-1]
                html += f'<a href="{link}">{filename}</a><br>\n'
    
    html += """</div>

<p style="margin-top: 30px;"><a href="https://github.com/by-ae/nvdiffrec_render-wheels">GitHub Repository</a></p>
</body>
</html>
"""
    
    # Write the root index
    output_file = wheels_dir / 'index.html'
    output_file.write_text(html)
    print(f"Generated {output_file}")
    print(f"Found {len(subdirs)} configurations")

if __name__ == '__main__':
    main()
