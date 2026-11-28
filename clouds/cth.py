"""EUMETSAT cloud-top height for one slot (msg_fes:cth 0 deg + msg_iodc:cth 45.5E), Americas-first 2048 x 4096, metres.

The WMS serves CTH only as a coloured PNG (no WCS), so heights are decoded from the layer's own legend ramp
(GetLegendGraphic: 320 m at x=148 .. 16000 m at x=632, linear): each pixel gets the height of the nearest legend
colour; transparent = no cloud top (NaN).  Nearest view wins (0 deg west of 22.75E, 45.5E east of it).  Coverage is
only ~77W..118.5E; elsewhere NaN.  Used by x1.py (X28) to confirm cloud before the cold-region boost.
(Developed as cloud_pipeline/work_1120/cth_fetch.py in the ApkWork repo.)"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import requests
from PIL import Image

URL = "https://view.eumetsat.int/geoserver/wms"
H, W = 2048, 4096


def _legend() -> tuple[np.ndarray, np.ndarray]:
    leg = np.asarray(Image.open(io.BytesIO(requests.get(URL, params=dict(
        service="WMS", version="1.3.0", request="GetLegendGraphic", format="image/png", width=640, height=80,
        layer="msg_fes:cth"), timeout=60).content)).convert("RGB"), np.float32)
    xs = np.arange(150, 631)
    return leg[18:45, xs].mean(0), 320 + (xs - 148) / (632 - 148) * (16000 - 320)


def _sector(layer, west, east, south, north, iso, ramp, ramp_h) -> np.ndarray:
    c0, c1 = int((west + 180) / 360 * W), int((east + 180) / 360 * W)
    r0, r1 = int((90 - north) / 180 * H), int((90 - south) / 180 * H)
    r = requests.get(URL, params=dict(service="WMS", version="1.3.0", request="GetMap", layers=layer, styles="",
                                      crs="EPSG:4326", bbox=f"{south},{west},{north},{east}", width=c1 - c0,
                                      height=r1 - r0, format="image/png", transparent="true", time=iso), timeout=120)
    r.raise_for_status()
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"), np.float32)
    ok = a[..., 3] > 128
    hgt = np.full(ok.shape, np.nan, np.float32)
    px = a[..., :3][ok]
    idx = np.concatenate([((px[i:i + 500000, None, :] - ramp[None]) ** 2).sum(-1).argmin(1)
                          for i in range(0, len(px), 500000)]) if len(px) else np.zeros(0, int)
    hgt[ok] = ramp_h[idx]
    out = np.full((H, W), np.nan, np.float32); out[r0:r1, c0:c1] = hgt
    return out


def fetch_height(t: datetime) -> np.ndarray:
    iso = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    ramp, ramp_h = _legend()
    fes = _sector("msg_fes:cth", -77.0, 77.0, -77.0, 77.0, iso, ramp, ramp_h)
    iodc = _sector("msg_iodc:cth", -35.5, 118.5, -77.0, 77.0, iso, ramp, ramp_h)
    lon = -180 + (np.arange(W) + 0.5) / W * 360
    use_iodc = (lon > 22.75)[None, :] & ~np.isnan(iodc) | np.isnan(fes)
    return np.where(use_iodc, iodc, fes).astype(np.float32)
