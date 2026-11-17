"""Debug outputs written next to each published mask:

  _debug_sources.png          which part of the mask produced each pixel's cloud (colour) x mask value (brightness)
  _debug_cloud_mask_land.png  the mask with every land area drawn at one faint level, so all continents read alike
  _sources.txt                every URL requested to build the mask (Himawari's hundreds of byte-range reads of one
                              file are listed once, with their count)
"""
from __future__ import annotations

from collections import OrderedDict
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image, ImageDraw

_seen: "OrderedDict[str, int]" = OrderedDict()


def record_urls() -> None:
    """Patch requests so every HTTP request (requests.get and Sessions alike) is logged with its query string."""
    orig = requests.sessions.Session.request

    def request(self, method, url, params=None, **kw):
        full = url + ("?" + urlencode(params) if params else "")
        key = f"{method.upper()} {full}"; _seen[key] = _seen.get(key, 0) + 1
        return orig(self, method, url, params=params, **kw)

    requests.sessions.Session.request = request


def write_urls(path) -> None:
    lines = [f"{k}" + (f"   (x{n})" if n > 1 else "") for k, n in _seen.items()]
    path.write_text("\n".join(lines) + "\n")


CLASSES = [("clear (haze / tone floor only)", (40, 40, 40)),
           ("IR 10.8 (EUMETSAT IR mosaic)", (90, 160, 255)),
           ("visible (EUMETSAT natural colour, blue channel)", (255, 230, 90)),
           ("thin layer placed by the cold-land rule", (200, 120, 255))]
SAT_COLOURS = [(255, 110, 110), (255, 160, 60), (80, 220, 120), (60, 210, 210), (255, 120, 200)]


def source_map(parts: dict, per: dict, mask: np.ndarray) -> Image.Image:
    """parts: visir._compute_mask's components (Greenwich-first); per: each satellite's faded flags (Greenwich-first);
    mask: final uint8 mask (Americas-first).  Returns an annotated 2048-wide image with a legend and shares."""
    ir, vis, day, thin = parts["ir"], parts["vis"], parts["day"], parts["thin"]
    names = list(per)
    if "lack" in parts:
        satw = [per[n] * ((1 - day) + day * parts["lack"]) for n in names]
        src = np.argmax(np.stack([parts["cold"]] + satw), 0)
    else:
        src = np.zeros(ir.shape, np.int64)
    irp = np.maximum(ir, thin); visc = day * np.clip(vis - irp, 0, None)
    cls = np.ones(ir.shape, np.int8)
    cls[thin > ir] = 3 + src[thin > ir]
    cls[visc > irp] = 2
    cls[irp + visc < 0.05] = 0
    cls = np.roll(cls, cls.shape[1] // 2, 1)                       # -> Americas-first like the mask
    classes = CLASSES + [(f"thin layer placed by {n}", SAT_COLOURS[i % len(SAT_COLOURS)]) for i, n in enumerate(names)]
    lut = np.array([c for _, c in classes], np.float32) / 255
    rgb = lut[cls] * (0.35 + 0.65 * (mask.astype(np.float32) / 255)[..., None])
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((2048, 1024), Image.LANCZOS)
    H = cls.shape[0]; band = np.abs(90 - (np.arange(H) + 0.5) / H * 180) < 62
    leg = Image.new("RGB", (2048, 30 + 45 * ((len(classes) + 2) // 3)), (0, 0, 0)); d = ImageDraw.Draw(leg)
    for i, (n, c) in enumerate(classes):
        x, y = 10 + (i % 3) * 680, 10 + (i // 3) * 45
        d.rectangle((x, y, x + 28, y + 28), fill=c)
        d.text((x + 38, y + 8), f"{n}  {np.mean(cls[band] == i):.1%}", fill=(230, 230, 230))
    out = Image.new("RGB", (2048, 1024 + leg.height)); out.paste(img, (0, 0)); out.paste(leg, (0, 1024))
    return out


def land_debug(mask: np.ndarray, land_level: float = 0.22) -> Image.Image:
    """Every land pixel drawn at the same faint level (max with the mask), coastline a little brighter."""
    from global_land_mask import globe
    from scipy.ndimage import binary_erosion
    H, W = mask.shape
    lat = 90 - (np.arange(H) + 0.5) / H * 180; lon = -180 + (np.arange(W) + 0.5) / W * 360
    land = globe.is_land(np.clip(lat, -89.9, 89.9)[:, None], lon[None, :])
    coast = land & ~binary_erosion(land, iterations=2)
    lvl = np.where(coast, land_level + 0.15, np.where(land, land_level, 0.0))
    return Image.fromarray(np.maximum(mask, (lvl * 255).astype(np.uint8)))
