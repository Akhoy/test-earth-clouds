"""The 5 geostationary cloud masks for one 3-hour slot, as one 0..1 'a mask says cloud' map (Americas-first,
lon -180 at column 0) that visir.py uses to decide where its thin cloud layer goes.

  EUMETSAT CLM   Meteosat 0 deg and 45.5E (EUMETSAT WMS)
  GOES-East      ABI-L2-ACMF clear sky mask, NOAA open data on AWS (noaa-goes19), first file of the hour
  GOES-West      same, noaa-goes18
  Himawari-9     AHI-L2-FLDK-Clouds CloudMask, NOAA open data on AWS (noaa-himawari9).  Only the CloudMask chunks are
                 range-read (~3 MB of a ~350 MB file).  Its fixed lat/lon grid is read once from one full file and kept
                 in cache/h09_latlon.npz (restored by the workflow's cache step).

Each mask is faded by viewing angle (full within 55 deg of the sub-point, 0 at 70: near the limb every mask calls
nearly everything cloud) and its edges softened ~2 px; the result is the max over the five."""
from __future__ import annotations

import logging
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, uniform_filter

from eumetsat_clm import SECTORS, _fetch_sector

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).parent.parent / "cache"
H09_BUCKET = "https://noaa-himawari9.s3.amazonaws.com"
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


# ---------- GOES (ACMF, GOES-R fixed grid) ----------
def _goes_file(sat: str, t: datetime) -> bytes:
    prefix = f"ABI-L2-ACMF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
    xml = session.get(f"https://noaa-{sat}.s3.amazonaws.com/?list-type=2&prefix={prefix}", timeout=60).text
    keys = sorted(re.findall(r"<Key>([^<]+)</Key>", xml))
    if not keys:
        raise RuntimeError(f"{sat}: nothing under {prefix}")
    r = session.get(f"https://noaa-{sat}.s3.amazonaws.com/{keys[0]}", timeout=600)
    r.raise_for_status()
    return r.content


def _goes_equirect(blob: bytes, W: int) -> np.ndarray:
    """GOES BCM (0 clear, 1 cloudy) reprojected to equirect: 1 cloudy, 0 clear, NaN not seen."""
    import io
    with h5py.File(io.BytesIO(blob), "r") as f:
        v = f["BCM"]; raw = v[()].astype(np.float32)
        fill = v.attrs.get("_FillValue", None)
        p = f["goes_imager_projection"].attrs
        g = lambda k: float(np.ravel(p[k])[0])
        req, rpol, hp, lon0 = g("semi_major_axis"), g("semi_minor_axis"), g("perspective_point_height"), g("longitude_of_projection_origin")
        def axis(n):
            a = f[n]; d = a[()].astype(np.float64)
            return d * float(np.ravel(a.attrs.get("scale_factor", [1.0]))[0]) + float(np.ravel(a.attrs.get("add_offset", [0.0]))[0])
        x, y = axis("x"), axis("y")
    bad = ~np.isfinite(raw) | (raw < 0) | (raw > 1)
    if fill is not None:
        bad |= raw == float(np.ravel(fill)[0])
    c = np.where(bad, np.nan, (raw >= 1).astype(np.float32))
    Hs = hp + req; H_ = W // 2
    lat = np.radians(90 - (np.arange(H_) + 0.5) / H_ * 180)[:, None]
    lon = np.radians(-180 + (np.arange(W) + 0.5) / W * 360)[None, :]
    e2 = (req ** 2 - rpol ** 2) / req ** 2
    phic = np.arctan(rpol ** 2 / req ** 2 * np.tan(lat))
    rc = rpol / np.sqrt(1 - e2 * np.cos(phic) ** 2)
    dl = lon - np.radians(lon0)
    sx = Hs - rc * np.cos(phic) * np.cos(dl); sy = -rc * np.cos(phic) * np.sin(dl); sz = rc * np.sin(phic) * np.ones_like(dl)
    seen = Hs * (Hs - sx) >= sy ** 2 + (req ** 2 / rpol ** 2) * sz ** 2
    xs = np.arcsin(np.clip(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2), -1, 1)); ys = np.arctan(sz / sx)
    ix = np.rint((xs - x[0]) / (x[1] - x[0])).astype(np.int64); iy = np.rint((ys - y[0]) / (y[1] - y[0])).astype(np.int64)
    ok = seen & (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    out = np.full((H_, W), np.nan, np.float32)
    out[ok] = c[iy[ok], ix[ok]]
    return out


# ---------- Himawari-9 (CMSK, range-read) ----------
def _get_range(url: str, lo: int, hi: int) -> bytes:
    for attempt in range(4):
        try:
            r = session.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=60)
            if r.status_code == 206 and len(r.content) == hi - lo + 1:
                return r.content
            raise IOError(f"bad range reply {r.status_code} len {len(r.content)}")
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


class _RangeFile:
    """Read-only seekable file over HTTP range requests with a small block cache (h5py metadata reads)."""
    BLOCK = 16 << 10

    def __init__(self, url):
        self.url, self.pos, self.blocks = url, 0, {}
        r = session.head(url, timeout=60); r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def _block(self, i):
        if i not in self.blocks:
            self.blocks[i] = _get_range(self.url, i * self.BLOCK, min((i + 1) * self.BLOCK, self.size) - 1)
        return self.blocks[i]

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = bytearray()
        while len(out) < n:
            p = self.pos + len(out); off = p % self.BLOCK
            out += self._block(p // self.BLOCK)[off:off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True


def _h09_url(t: datetime) -> str:
    prefix = f"AHI-L2-FLDK-Clouds/{t:%Y/%m/%d/%H}00/"
    xml = session.get(f"{H09_BUCKET}/?list-type=2&prefix={prefix}", timeout=60).text
    keys = [k for k in re.findall(r"<Key>([^<]+)</Key>", xml) if "CMSK" in k]
    if not keys:
        raise RuntimeError(f"no Himawari CMSK file under {prefix}")
    return f"{H09_BUCKET}/{keys[0]}"


def _h09_cloudmask(url: str) -> np.ndarray:
    rf = _RangeFile(url)
    with h5py.File(rf, "r") as f:
        d = f["CloudMask"]
        shape, chunk, fill = d.shape, d.chunks, d.fillvalue
        index = sorted((d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks())), key=lambda c: c.byte_offset)
    groups = [[index[0]]]
    for c in index[1:]:
        last = groups[-1][-1]
        if c.byte_offset - (last.byte_offset + last.size) <= (4 << 10): groups[-1].append(c)
        else: groups.append([c])
    out = np.full(shape, fill, np.int8)
    def fetch(g):
        lo = g[0].byte_offset; blob = _get_range(url, lo, g[-1].byte_offset + g[-1].size - 1)
        for c in g:
            a = np.frombuffer(zlib.decompress(blob[c.byte_offset - lo:c.byte_offset - lo + c.size]), np.int8).reshape(chunk)
            r, q = c.chunk_offset
            out[r:r + chunk[0], q:q + chunk[1]] = a[:shape[0] - r, :shape[1] - q]
    with ThreadPoolExecutor(64) as ex:
        list(ex.map(fetch, groups))
    return out


def _h09_latlon(url: str) -> tuple[np.ndarray, np.ndarray]:
    """The fixed full-disk lat/lon grid: read once from one full file (~350 MB, first run only), then cached."""
    path = CACHE_DIR / "h09_latlon.npz"
    if not path.exists():
        logger.info("Himawari lat/lon not cached: downloading one full file once ...")
        tmp = CACHE_DIR / "h09_full.nc"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with session.get(url, stream=True, timeout=1200) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(8 << 20): fh.write(chunk)
        with h5py.File(tmp, "r") as f:
            np.savez(path, lat=f["Latitude"][()], lon=f["Longitude"][()])
        tmp.unlink()
    g = np.load(path)
    return g["lat"], g["lon"]


def _h09_equirect(t: datetime, W: int) -> np.ndarray:
    """4-level CloudMask (clear, prob clear, prob cloudy, cloudy) averaged into map cells, 1-cell gaps closed."""
    url = _h09_url(t)
    cm = _h09_cloudmask(url).astype(np.float32); la, lo = _h09_latlon(url)
    ok = (cm >= 0) & (cm <= 3) & (la > -90) & (lo >= -180)
    ang = np.degrees(np.arccos(np.clip(np.cos(np.radians(la)) * np.cos(np.radians(lo - 140.7)), -1, 1)))
    ok &= ang < 78
    c = np.choose(np.clip(cm[ok], 0, 3).astype(int), [0.0, 0.25, 0.75, 1.0])
    H_ = W // 2
    r = np.clip(((90 - la[ok]) / 180 * H_).astype(int), 0, H_ - 1); q = np.clip(((lo[ok] + 180) / 360 * W).astype(int), 0, W - 1)
    s = np.zeros((H_, W)); n = np.zeros((H_, W))
    np.add.at(s, (r, q), c); np.add.at(n, (r, q), 1)
    hole = n == 0
    s = np.where(hole, uniform_filter(s, 3, mode="wrap"), s); n = np.where(hole, uniform_filter(n, 3, mode="wrap"), n)
    return np.where(n > 0.05, s / np.maximum(n, 1e-9), np.nan).astype(np.float32)


# ---------- EUMETSAT CLM ----------
def _clm_equirect(sector, t: datetime, W: int) -> np.ndarray:
    layer, west, east, south, north = sector; H_ = W // 2
    c0, c1 = int((west + 180) / 360 * W), int((east + 180) / 360 * W)
    r0, r1 = int((90 - north) / 180 * H_), int((90 - south) / 180 * H_)
    for attempt in range(4):
        try:
            rgb = _fetch_sector(layer, west, east, south, north, c1 - c0, r1 - r0, time=t.strftime("%Y-%m-%dT%H:%M:%S.000Z")).astype(np.int16)
            break
        except Exception:
            if attempt == 3: raise
            time.sleep(5 * (attempt + 1))
    out = np.full((H_, W), np.nan, np.float32)
    out[r0:r1, c0:c1] = np.where(rgb.max(-1) > 20, (rgb.min(-1) > 200).astype(np.float32), np.nan)
    return out


def _view_fade(lon0: float, W: int) -> np.ndarray:
    H_ = W // 2
    lat = np.radians(90 - (np.arange(H_) + 0.5) / H_ * 180)[:, None]
    lon = np.radians(-180 + (np.arange(W) + 0.5) / W * 360)[None, :]
    ang = np.degrees(np.arccos(np.clip(np.cos(lat) * np.cos(lon - np.radians(lon0)), -1, 1)))
    t = np.clip((ang - 55) / 15, 0, 1)
    return 1 - t * t * (3 - 2 * t)


def fetch_flags(t: datetime, W: int = 4096) -> tuple[np.ndarray, list[str]]:
    """Max over the 5 masks of (cloud flag, softened ~2 px) x viewing-angle fade.  Americas-first.  A source that
    fails is skipped and reported; returns (flags, names of sources used)."""
    jobs = {
        "Meteosat 0 CLM": (lambda: _clm_equirect(SECTORS[0], t, W), 0.0),
        "Meteosat 45.5E CLM": (lambda: _clm_equirect(SECTORS[1], t, W), 45.5),
        "GOES-East": (lambda: _goes_equirect(_goes_file("goes19", t), W), -75.2),
        "GOES-West": (lambda: _goes_equirect(_goes_file("goes18", t), W), -137.0),
        "Himawari-9": (lambda: _h09_equirect(t, W), 140.7),
    }
    flags = np.zeros((W // 2, W), np.float32); used = []
    with ThreadPoolExecutor(5) as ex:
        futures = {name: ex.submit(fn) for name, (fn, _) in jobs.items()}
        for name, fut in futures.items():
            try:
                m = fut.result()
            except Exception:
                logger.exception("%s mask failed, skipped", name); continue
            sigma = 2.3 * W / 4096
            flags = np.maximum(flags, gaussian_filter(np.nan_to_num(m), sigma, mode="wrap") * _view_fade(jobs[name][1], W))
            used.append(name)
    return flags, used
