#!/usr/bin/env bash
# Generate wheels/cu<CUDA>-torch<TORCH>/index.html for one release.
#
# Wheel anchors follow PEP 503: an absolute file URL, plus a #sha256=
# fragment when the GitHub release asset digest is a sha256. Pip checks
# that fragment for --find-links pages. Assets that fail validate_index.py
# (including the old doubled-tag Windows names) are left out so a later
# index update cannot advertise wheels pip will not install.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cuda_version> <pytorch_version> <cuda_short> <torch_short>" >&2
    echo "Example: $0 12.6 2.6.0 126 260" >&2
    exit 1
fi

CUDA_VERSION="$1"
PYTORCH_VERSION="$2"
CUDA_SHORT="$3"
TORCH_SHORT="$4"

if ! [[ "$CUDA_VERSION" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    echo "Invalid CUDA version: $CUDA_VERSION" >&2
    exit 1
fi
if ! [[ "$PYTORCH_VERSION" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    echo "Invalid PyTorch version: $PYTORCH_VERSION" >&2
    exit 1
fi
if ! [[ "$CUDA_SHORT" =~ ^[0-9]+$ && "$TORCH_SHORT" =~ ^[0-9]+$ ]]; then
    echo "CUDA and PyTorch short tags must be numeric" >&2
    exit 1
fi
if [ "$CUDA_SHORT" != "${CUDA_VERSION//./}" ]; then
    echo "CUDA short tag $CUDA_SHORT does not match version $CUDA_VERSION" >&2
    exit 1
fi
if [ "$TORCH_SHORT" != "${PYTORCH_VERSION//./}" ]; then
    echo "PyTorch short tag $TORCH_SHORT does not match version $PYTORCH_VERSION" >&2
    exit 1
fi

WHEEL_DIR="wheels/cu${CUDA_SHORT}-torch${TORCH_SHORT}"
RELEASE_TAG="cu${CUDA_SHORT}-torch${TORCH_SHORT}"
REPO="${GITHUB_REPOSITORY:-by-ae/nvdiffrec_render-wheels}"
PAGES_URL="https://by-ae.github.io/nvdiffrec_render-wheels/${RELEASE_TAG}/"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

tmp_html=""
raw_file=""
filtered_file=""
links_body=""
cleanup() {
    if [ -n "$tmp_html" ]; then
        rm -f "$tmp_html"
    fi
    if [ -n "$raw_file" ]; then
        rm -f "$raw_file"
    fi
    if [ -n "$filtered_file" ]; then
        rm -f "$filtered_file"
    fi
    if [ -n "$links_body" ]; then
        rm -f "$links_body"
    fi
    return 0
}
trap cleanup EXIT

token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
have_gh=0
if command -v gh >/dev/null 2>&1 && [ -n "$token" ]; then
    have_gh=1
fi

# In Actions, writing a page with no links would replace a good index
# and pip would stop seeing the wheels. Fail instead of publishing that.
if [ "${GITHUB_ACTIONS:-}" = "true" ] && [ "$have_gh" -ne 1 ]; then
    echo "gh and GH_TOKEN (or GITHUB_TOKEN) are required to refresh the index in GitHub Actions" >&2
    exit 1
fi

mkdir -p "$WHEEL_DIR"

filtered_file=$(mktemp)
raw_file=$(mktemp)

links_note=""
if [ "$have_gh" -eq 1 ]; then
    export GH_TOKEN="$token"
    # Failure here must not truncate the published index.html.
    gh release view "$RELEASE_TAG" --repo "$REPO" --json assets \
        --jq '.assets[] | [.name, (.digest // "")] | @tsv' > "$raw_file"

    python3 - "$SCRIPT_DIR" "$raw_file" "$filtered_file" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from validate_index import validate_wheel_filename

raw_path, out_path = sys.argv[2], sys.argv[3]
kept = []
for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    name, _, digest = line.partition("\t")
    ok, message = validate_wheel_filename(name)
    if not ok:
        print(f"Skipping {name}: {message}", file=sys.stderr)
        continue
    kept.append(f"{name}\t{digest}")
kept.sort()
Path(out_path).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
PY
else
    links_note="<!-- Wheel links will be added by the workflow -->"
fi

tmp_html=$(mktemp "$WHEEL_DIR/.index.html.XXXXXX")
cat > "$tmp_html" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>nvdiffrec_render Wheels - CUDA ${CUDA_VERSION} + PyTorch ${PYTORCH_VERSION}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        a { color: #0366d6; }
        pre { background: #f6f8fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
<h1>nvdiffrec_render Wheels</h1>
<p><strong>CUDA ${CUDA_VERSION} + PyTorch ${PYTORCH_VERSION}</strong></p>
EOF

install_block="<h2>Install</h2>
<pre>pip install nvdiffrec_render --find-links ${PAGES_URL}</pre>
<h2>Available Wheels</h2>"

link_count=0
if [ "$have_gh" -eq 1 ]; then
    links_body=$(mktemp)
    while IFS=$'\t' read -r name digest; do
        [ -n "$name" ] || continue
        href="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${name}"
        hash="${digest#sha256:}"
        if [[ "$digest" == sha256:* && "$hash" =~ ^[a-f0-9]{64}$ ]]; then
            href="${href}#sha256=${hash}"
        fi
        # Names accepted by validate_wheel_filename are URL-safe and need no
        # further escaping. Quote the attribute anyway.
        printf '<a href="%s">%s</a><br>\n' "$href" "$name" >> "$links_body"
        link_count=$((link_count + 1))
    done < "$filtered_file"
    if [ "$link_count" -eq 0 ]; then
        # A pip command here 404s. Say so instead of offering the URL.
        cat >> "$tmp_html" <<EOF
<h2>No wheels published yet</h2>
<p><em>No wheel files are published for this release yet.</em></p>
EOF
    else
        printf '%s\n' "$install_block" >> "$tmp_html"
        cat "$links_body" >> "$tmp_html"
    fi
else
    printf '%s\n' "$install_block" >> "$tmp_html"
    echo "$links_note" >> "$tmp_html"
fi

cat >> "$tmp_html" <<EOF
<p style="margin-top: 20px;"><a href="../">Back to index</a></p>
</body>
</html>
EOF

mv "$tmp_html" "$WHEEL_DIR/index.html"
tmp_html=""
echo "Generated $WHEEL_DIR/index.html ($link_count wheel link(s))"
