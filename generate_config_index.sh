#!/usr/bin/env bash
# Generate index.html for a specific CUDA/PyTorch configuration
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cuda_version> <pytorch_version> <cuda_short> <torch_short>"
    echo "Example: $0 12.6 2.6.0 126 260"
    exit 1
fi

CUDA_VERSION="$1"
PYTORCH_VERSION="$2"
CUDA_SHORT="$3"
TORCH_SHORT="$4"

WHEEL_DIR="wheels/cu${CUDA_SHORT}-torch${TORCH_SHORT}"
RELEASE_TAG="cu${CUDA_SHORT}-torch${TORCH_SHORT}"
REPO="${GITHUB_REPOSITORY:-unknown/unknown}"

mkdir -p "$WHEEL_DIR"

cat > "$WHEEL_DIR/index.html" <<EOF
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
<h2>Install</h2>
<pre>pip install nvdiffrec_render --find-links https://by-ae.github.io/nvdiffrec_render-wheels/cu${CUDA_SHORT}-torch${TORCH_SHORT}/</pre>
<h2>Available Wheels</h2>
EOF

if command -v gh &> /dev/null && [ -n "${GH_TOKEN:-}" ]; then
    gh release view "$RELEASE_TAG" --json assets --jq '.assets[].name' | while read -r wheel; do
        echo "<a href=\"https://github.com/$REPO/releases/download/$RELEASE_TAG/$wheel\">$wheel</a><br>" >> "$WHEEL_DIR/index.html"
    done
else
    echo "<!-- Wheel links will be added by the workflow -->" >> "$WHEEL_DIR/index.html"
fi

cat >> "$WHEEL_DIR/index.html" <<EOF
<p style="margin-top: 20px;"><a href="../">Back to index</a></p>
</body>
</html>
EOF

echo "Generated $WHEEL_DIR/index.html"
