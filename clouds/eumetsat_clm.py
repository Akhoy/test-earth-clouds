"""EUMETSAT's own pre-computed Cloud Mask product (msg_fes:clm / msg_iodc:clm)
instead of deriving cloud/no-cloud from raw IR brightness ourselves. This is
a real algorithmic clear/cloudy classification (EUMETSAT's own product, not
a brightness threshold), so it should catch low, warm clouds that raw IR
struggles with (low cloud has little thermal contrast against the warm
surface below it).

Coverage is regional, not the full stitched global ring `mumi:worldcloudmap_
ir108` uses -- msg_fes covers -77..77 lon, msg_iodc covers -35.5..118.5 lon,
both -77..77 lat. Combined that's -77..118.5 lon, which comfortably covers
the EUMETSAT sector (confirmed in the multisat trial to need roughly
-37.6..70.35 lon). Outside that combined range the output canvas is left at
0 (no data).

Usage:
  venv/bin/python3 pipeline/eumetsat_clm.py [--width 2048] [--out mask.png]
"""
from __future__ import annotations

import argparse

import numpy as np
import requests
from io import BytesIO
from PIL import Image

WMS_URL = "https://view.eumetsat.int/geoserver/wms"

# (wms layer, west, east, south, north)
SECTORS = [
    ("msg_fes:clm", -77.0, 77.0, -77.0, 77.0),
    ("msg_iodc:clm", -35.5, 118.5, -77.0, 77.0),
]


def _fetch_sector(layer: str, west: float, east: float, south: float, north: float,
                   px_w: int, px_h: int, time: str | None = None) -> np.ndarray:
    """time: ISO slice (e.g. 2026-11-13T12:00:00.000Z); None = the layer's latest scan."""
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap",
        "layers": layer, "styles": "", "crs": "EPSG:4326",
        "bbox": f"{south},{west},{north},{east}",
        "width": str(px_w), "height": str(px_h),
        "format": "image/png",
    }
    if time:
        params["time"] = time
    resp = requests.get(WMS_URL, params=params, timeout=60)
    resp.raise_for_status()
    return np.array(Image.open(BytesIO(resp.content)).convert("RGB"))


def fetch_clm_mask(out_width: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask, covered) -- covered is True where a real CLM reading
    was written, False elsewhere (the Pacific/Americas gap -- CLM has no
    global equivalent, only the raw IR layer does)."""
    out_height = out_width // 2
    canvas = np.zeros((out_height, out_width), dtype=np.uint8)
    covered = np.zeros((out_height, out_width), dtype=bool)

    for layer, west, east, south, north in SECTORS:
        col0 = int((west + 180) / 360 * out_width)
        col1 = int((east + 180) / 360 * out_width)
        row0 = int((90 - north) / 180 * out_height)
        row1 = int((90 - south) / 180 * out_height)
        px_w, px_h = col1 - col0, row1 - row0

        rgb = _fetch_sector(layer, west, east, south, north, px_w, px_h)
        # product convention: white = cloud, green = clear land, blue = clear ocean
        is_cloud = (rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] > 200)
        canvas[row0:row1, col0:col1] = np.where(is_cloud, 255, 0).astype(np.uint8)
        covered[row0:row1, col0:col1] = True

    return canvas, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--out", default="cloud_mask_clm.png")
    args = ap.parse_args()

    mask, covered = fetch_clm_mask(args.width)
    print(f"covered {covered.mean():.1%} of globe; cloud fraction in covered sector: {(mask[covered] == 255).mean():.1%}")
    Image.fromarray(mask).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
