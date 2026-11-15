"""Cloud-density mask from EUMETSAT's IR + visible layers.

IR covers the whole globe, day and night. Wherever the visible layer has
signal (the daylit side) the brighter of the two is used, over land and
ocean alike. Visible is faded out inside the washed-out rim of its own
footprint so the hand-over to IR-only has no visible join.

On the night side, where the IR is an even cold gray that hides the land,
a very thin layer of matteason/live-cloud-maps' IR + dust cloud score is
added on top (see THIN_LAYER_OPACITY below); high terrain is excluded.

All layers are fetched as two halves (lon 0->180 and -180->0) with the
same bboxes and joined the same way, so they share one pixel grid.

Two corrections on top:
- Snow/ice (NASA MODIS monthly climatology) raises the threshold a pixel has
  to clear to count as cloud, because cold snow reads as mid-gray cloud in
  IR. It never zeroes the mask, so bright cloud over snow survives.
- Beyond EUMETSAT's view-angle cutoff (a little past +-81 deg) there is no
  data; the mask fades to clear toward both poles identically.

Not wired into run_pipeline.py. `fetch_cloud_mask` returns `(mask,
still_gap)` like its siblings; still_gap is always all-zero.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d, percentile_filter

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "eumetsat_visible_ir_last_good.npy"

WMS_URL = "https://view.eumetsat.int/geoserver/ows"
LAYER_VISIBLE = "mumi:wideareacoverage_rgb_natural"
LAYER_IR = "mumi:worldcloudmap_ir108"
LAYER_DUST = "mumi:wideareacoverage_rgb_dust"

# bbox = minLat,minLon,maxLat,maxLon (EUMETSAT's WMS 1.3.0 + EPSG:4326 uses
# lat,lon axis order -- NOT lon,lat. lon,lat order silently returns a
# plausible-looking but wrongly-scaled crop instead of an error.)
BBOX_EAST = (-90, 0, 90, 180)     # lon 0 -> 180
BBOX_WEST = (-90, -180, 90, 0)    # lon -180 -> 0

# Beyond the mosaic's view-angle cutoff every pixel is a flat white fill.
NODATA_THRESHOLD = 254.0
# Degrees of latitude, inside the no-data edge, over which the mask fades
# to clear. See _polar_fade.
POLAR_FADE_DEG = 8.0

# IR gray levels mapped to clear (0) and fully cloudy (1).
IR_CLEAR = 72.0
IR_CLOUD = 160.0
# Gray levels above the local clear level that still count as clear, so
# leftover background noise does not show as thin film.
IR_MARGIN = 4.0
# <1 lifts mid-level (warmer, lower) cloud toward opaque white, the way it
# looks in a daylight photo, instead of leaving it semi-transparent.
IR_GAMMA = 0.65
# Unsharp-mask strength applied to the IR before scoring, to bring out
# fine cloud texture.
IR_DETAIL_GAIN = 0.7
# IR_CLEAR is the floor; the clear level is raised locally up to
# IR_CLEAR_MAX over cold surfaces (high latitudes, night-time land, high
# terrain). Lower = more of that cold gray shows as cloud.
IR_CLEAR_MAX = 130.0
CLEAR_SKY_WINDOW_DEG = 25.0
CLEAR_SKY_PERCENTILE = 10

# Where the IR is an even cold gray that hides the land, the crisp scoring
# above shows nothing. There, on the night side only (visible light shows
# such cloud properly by day), a very thin layer is added from
# matteason/live-cloud-maps' IR + dust recipe: IR stretched IR_CLEAR..
# HIDDEN_IR_CLOUD and brightened (HIDDEN_IR_GAMMA), screen-blended with a
# cloud signal from EUMETSAT's "dust" composite, and brightened again
# (HIDDEN_FINAL_GAMMA) -- his values -- then scaled to at most
# THIN_LAYER_OPACITY so it reads as a faint haze instead of his solid white
# sheet. (Dust is built from IR channels only and its blue channel is the
# same 10.8um temperature, so over cold ground it says "cold", not
# "cloud": checked against visible, the full recipe overstates cover.)
# A pixel counts as "land hidden" when its whole region is cold (broad
# clear level above HIDDEN_REGION_LEVEL), it is itself at least
# HIDDEN_COLDER_BY gray levels colder than the clearest ground within
# HIDDEN_REFERENCE_WINDOW_DEG, and it is not high terrain, which is cold
# without any cloud.
THIN_LAYER_OPACITY = 0.6
HIDDEN_IR_CLOUD = 178.0
HIDDEN_IR_GAMMA = 1.46
HIDDEN_IR_WEIGHT = 0.77
HIDDEN_FINAL_GAMMA = 2.0
# cloud_pipeline tuning (2026-11-15): thin layer textured from IR (range in IR grey levels, depth 0..1); a 30% blend
# with a 1 px blur (slightly less sharp); haze = 8% of the cloud field blurred ~12 px (at 4096 wide), added on top.
THIN_TEXTURE_RANGE = 24.0
SEA_HAZE = 0.0   # off (user, 2026-11-15). Tried 0.02, 0.04; 2026-11-15: 0.09 / 0.14 / 0.10 too much (grey, "slightly purple"); user: "a tiny bit"
NASA_TONE_IN = [0.0, 0.008, 0.02, 0.047, 0.09, 0.133, 0.18, 0.231, 0.278, 0.329, 0.38, 0.443, 0.502, 0.565, 0.655,
                0.765, 0.886, 0.988, 1.0]
NASA_TONE_OUT = [0.06, 0.098, 0.133, 0.169, 0.208, 0.247, 0.29, 0.337, 0.384, 0.431, 0.482, 0.537, 0.6, 0.663, 0.722,
                 0.78, 0.831, 0.886, 0.92]
THIN_TEXTURE_DEPTH = 0.6
SOFTEN_BLEND = 0.3
HAZE_OPACITY = 0.08
HIDDEN_REGION_LEVEL = 86.0
HIDDEN_COLDER_BY = 16.0
HIDDEN_REFERENCE_WINDOW_DEG = 40.0
# NASA GIBS' SRTM colour-index layer; its red channel is near 0-40 over
# lowland and ~190 over anything mountainous (checked: Ukraine 0, Germany
# 30, Alps/Iran/Tibet/Mongolia 186-201).
ELEVATION_LAYER = "SRTM_Color_Index"
ELEVATION_HIGH = (60.0, 140.0)


# The visible layer is a "natural colour" composite whose blue channel is
# the 0.6um reflectance: cloud (water or ice) is bright in it, while clear
# ocean, vegetation and most bare ground stay below VISIBLE_CLEAR (checked
# against live data: clear land p90 ~0.36-0.44, cloud p50 ~0.6-0.7).
VISIBLE_CLEAR = 0.32
VISIBLE_CLOUD = 0.80
VISIBLE_GAMMA = 0.8
# The visible mosaic's outer rim (low sun) is washed out and over-bright.
# Visible is faded in over this fraction of the canvas width inside it.
VISIBLE_EDGE_FADE = 0.045

# NASA GIBS' MODIS monthly snow-cover climatology. Monthly, not daily: a
# single day's retrieval is mostly cloud-obscured gaps.
SNOW_ICE_LAYER = "MODIS_Terra_L3_Snow_Cover_Monthly_Average_Pct"
SNOW_ICE_SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
# The cloud threshold reaches SNOW_ICE_MAX_THRESHOLD where monthly snow
# cover is at or above SNOW_ICE_FULL_FRACTION.
SNOW_ICE_FULL_FRACTION = 0.5
SNOW_ICE_MAX_THRESHOLD = 0.55


def _get_wms_image(url: str, params: dict, attempts: int = 4, base_delay: float = 2.0) -> np.ndarray:
    """GET a WMS image with exponential backoff. EUMETSAT's GeoServer
    transiently 502/503s or returns a non-image body with a 200; both clear
    up on a later attempt."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(base_delay * (2 ** (attempt - 1)))
        try:
            resp = requests.get(url, params=params, timeout=90)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return np.array(img)
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            logger.warning("WMS fetch attempt %d/%d failed: %s", attempt + 1, attempts, exc)
    raise last_exc


def _fetch_layer(layer: str, bbox: tuple[float, float, float, float], size: int, time_iso: str | None = None) -> np.ndarray:
    """time_iso pins an exact 3-hourly scan (e.g. 2026-11-15T03:00:00.000Z) so these layers match the satellite
    masks of the same slot; None lets the server pick its latest slice."""
    min_lat, min_lon, max_lat, max_lon = bbox
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "format": "image/png",
        "crs": "EPSG:4326",
        "bbox": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "width": str(size),
        "height": str(size),
    }
    if time_iso:
        params["time"] = time_iso
    return _get_wms_image(WMS_URL, params)


def _stitch(east: np.ndarray, west: np.ndarray) -> np.ndarray:
    """East-then-west: column 0 = lon 0, the antimeridian is at the centre,
    and the prime meridian is at the canvas's wrap edges. Both halves are
    crops of one mosaic, so plain concatenation lines up pixel-for-pixel."""
    return np.concatenate([east, west], axis=1)


def _month_for_snow_climatology(now: datetime) -> datetime:
    """The last fully completed month -- GIBS returns no data for the
    month still in progress."""
    last_of_prev_month = now.replace(day=1) - timedelta(days=1)
    return last_of_prev_month.replace(day=15)


def _fetch_snow_ice_fraction(width: int, height: int) -> np.ndarray:
    """Monthly snow/ice-cover fraction (0..1) on this module's canvas,
    cached per calendar month. Falls back to an older cached month, then to
    all-zero, rather than failing the whole mask."""
    target = _month_for_snow_climatology(datetime.now(timezone.utc))
    cache_file = CACHE_DIR / f"snow_ice_fraction_{width}x{height}_{target:%Y%m}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    try:
        params = {
            "REQUEST": "GetSnapshot",
            "TIME": target.strftime("%Y-%m-%d"),
            "BBOX": "-90,-180,90,180",
            "CRS": "EPSG:4326",
            "LAYERS": SNOW_ICE_LAYER,
            "WRAP": "day",
            "FORMAT": "image/png",
            "WIDTH": str(width),
            "HEIGHT": str(height),
        }
        resp = requests.get(SNOW_ICE_SNAPSHOT_URL, params=params, timeout=90)
        resp.raise_for_status()
        rgba = np.array(Image.open(BytesIO(resp.content)).convert("RGBA")).astype(np.float32)
        # Colour ramp runs white (0%) to pure red (100%), so green falls
        # monotonically with snow cover; transparent means no snow.
        fraction = np.where(rgba[..., 3] > 0, 1.0 - rgba[..., 1] / 255.0, 0.0)
        # The snapshot is lon -180..180; roll it onto the lon 0-first canvas.
        fraction = np.roll(np.clip(fraction, 0.0, 1.0), -(width // 2), axis=1)
    except Exception:
        logger.exception("NASA snow/ice climatology fetch failed")
        cached_any = sorted(CACHE_DIR.glob(f"snow_ice_fraction_{width}x{height}_*.npy"))
        if cached_any:
            return np.load(cached_any[-1])
        return np.zeros((height, width), dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, fraction)
    for stale in CACHE_DIR.glob(f"snow_ice_fraction_{width}x{height}_*.npy"):
        if stale != cache_file:
            stale.unlink(missing_ok=True)
    return fraction


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _visible_weight(vis: np.ndarray) -> np.ndarray:
    """0..1 weight for the visible channel: 0 outside its daylit footprint
    and at the footprint's rim, rising smoothly to 1 well inside it. Derived
    from the image itself, so there is no sun-angle calculation to drift
    out of step with the mosaic's own time."""
    lit = ((vis.max(axis=-1) > 10) & (vis.min(axis=-1) < NODATA_THRESHOLD)).astype(np.float32)
    sigma = VISIBLE_EDGE_FADE * vis.shape[1]
    blurred = gaussian_filter(lit, sigma=sigma, mode=("nearest", "wrap"))
    # Starts well inside the rim: the visible image is dim under a low sun.
    return _smoothstep((blurred - 0.7) / 0.28)


def _polar_fade(ir_gray: np.ndarray) -> np.ndarray:
    """0..1 weight that is 0 in each pole's no-data cap and rises to 1 over
    POLAR_FADE_DEG of latitude inside it. Same treatment for both poles.
    The no-data edge is a sawtooth, so the fade follows a smoothed line that
    sits at or inside the outermost point of that edge."""
    h, w = ir_gray.shape
    valid = ir_gray < NODATA_THRESHOLD
    fade_rows = POLAR_FADE_DEG / 180.0 * h
    rows = np.arange(h, dtype=np.float32)[:, None]

    def edge(first_valid: np.ndarray) -> np.ndarray:
        widened = maximum_filter1d(first_valid.astype(np.float32), size=max(3, w // 80), mode="wrap")
        return gaussian_filter1d(widened, sigma=w / 160, mode="wrap")[None, :]

    north = edge(np.argmax(valid, axis=0))
    south = edge(np.argmax(valid[::-1], axis=0))
    from_north = (rows - north) / fade_rows
    from_south = ((h - 1 - rows) - south) / fade_rows
    return _smoothstep(from_north) * _smoothstep(from_south)


def _regional_percentile(ir_gray: np.ndarray, percentile: float, window_deg: float) -> np.ndarray:
    """Smoothed `percentile` of the IR over a `window_deg`-wide
    neighbourhood, computed on a small copy and scaled back up."""
    h, w = ir_gray.shape
    small_w = 256
    small_h = max(1, round(h * small_w / w))
    small = np.array(Image.fromarray(ir_gray).resize((small_w, small_h), Image.BILINEAR))
    size = max(3, round(window_deg / 360.0 * small_w))
    # percentile_filter takes one edge mode for both axes; wrap longitude by hand.
    padded = np.pad(small, ((0, 0), (size, size)), mode="wrap")
    level = percentile_filter(padded, percentile, size=size, mode="nearest")[:, size:-size]
    level = gaussian_filter(level, sigma=size / 3, mode=("nearest", "wrap"))
    return np.array(Image.fromarray(level).resize((w, h), Image.BICUBIC))


def _fetch_elevation(width: int, height: int) -> np.ndarray:
    """Terrain-height proxy (see ELEVATION_LAYER) on this module's canvas,
    cached indefinitely. All-zero (no terrain exclusion) if it cannot be
    fetched and was never cached."""
    cache_file = CACHE_DIR / f"elevation_proxy_{width}x{height}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    try:
        params = {
            "REQUEST": "GetSnapshot", "TIME": "2026-10-31", "BBOX": "-90,-180,90,180",
            "CRS": "EPSG:4326", "LAYERS": ELEVATION_LAYER, "WRAP": "day",
            "FORMAT": "image/png", "WIDTH": str(width), "HEIGHT": str(height),
        }
        resp = requests.get(SNOW_ICE_SNAPSHOT_URL, params=params, timeout=90)
        resp.raise_for_status()
        red = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))[..., 0].astype(np.float32)
    except Exception:
        logger.exception("elevation fetch failed; high terrain will not be excluded this run")
        return np.zeros((height, width), dtype=np.float32)
    # The snapshot is lon -180..180; roll it onto the lon 0-first canvas.
    elevation = np.roll(red, -(width // 2), axis=1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, elevation)
    return elevation


def _hidden_land_weight(ir_gray: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """0..1: how strongly a pixel counts as "land hidden under even cold
    gray" (see HIDDEN_REGION_LEVEL and friends)."""
    wrap = ("nearest", "wrap")
    w = ir_gray.shape[1]
    region = _regional_percentile(ir_gray, CLEAR_SKY_PERCENTILE, CLEAR_SKY_WINDOW_DEG)
    clearest = _regional_percentile(ir_gray, 2, HIDDEN_REFERENCE_WINDOW_DEG)
    colder_by = gaussian_filter(ir_gray, sigma=w / 900, mode=wrap) - clearest
    low, high = ELEVATION_HIGH
    highland = gaussian_filter(_smoothstep((elevation - low) / (high - low)), sigma=w / 400, mode=wrap)
    return (
        _smoothstep((region - HIDDEN_REGION_LEVEL) / 10.0)
        * _smoothstep((colder_by - HIDDEN_COLDER_BY) / 14.0)
        * (1.0 - highland)
    )


def _ir_dust_score(ir_gray: np.ndarray, dust: np.ndarray) -> np.ndarray:
    """matteason's IR + dust cloud score, 0..1. In the dust composite the
    red channel carries cloud detail and the inverted blue channel is a
    cold-surface mask; red is masked by it and half of it is added back."""
    dust_f = dust.astype(np.float32) / 255.0
    cold = 1.0 - dust_f[..., 2]
    dust_score = 1.0 - (1.0 - dust_f[..., 0] * cold) * (1.0 - 0.5 * cold)
    ir_score = np.clip((ir_gray - IR_CLEAR) / (HIDDEN_IR_CLOUD - IR_CLEAR), 0, 1) ** (1 / HIDDEN_IR_GAMMA)
    combined = 1.0 - (1.0 - dust_score) * (1.0 - ir_score * HIDDEN_IR_WEIGHT)
    return combined ** (1 / HIDDEN_FINAL_GAMMA)


def _clear_sky_level(ir_gray: np.ndarray) -> np.ndarray:
    """Per-pixel IR gray level of cloud-free ground/sea. Cold surfaces read
    well above IR_CLEAR and would otherwise show as a flat gray film.
    Capped so a region that is overcast across the whole neighbourhood is
    not mistaken for ground."""
    level = _regional_percentile(ir_gray, CLEAR_SKY_PERCENTILE, CLEAR_SKY_WINDOW_DEG)
    return np.clip(level, IR_CLEAR, IR_CLEAR_MAX)


def _sea_weight(height: int, width: int) -> np.ndarray:
    """(cloud_pipeline) 1 over open sea, exactly 0 on every land pixel; the feather (~2 px) ramps up offshore only, so
    land is never touched.  Greenwich-first canvas like the rest of this module (column 0 = lon 0)."""
    from global_land_mask import globe
    lat = 90 - (np.arange(height) + 0.5) / height * 180
    lon = (np.arange(width) + 0.5) / width * 360
    lon = np.where(lon > 180, lon - 360, lon)
    land = globe.is_land(np.clip(lat, -89.9, 89.9)[:, None], lon[None, :]).astype(np.float32)
    return (1.0 - gaussian_filter(land, sigma=width / 2048, mode=("nearest", "wrap"))) * (1.0 - land)


def _compute_mask(
    vis: np.ndarray, ir: np.ndarray, dust: np.ndarray, snow_ice_fraction: np.ndarray, elevation: np.ndarray,
    flags: np.ndarray | None = None,
) -> np.ndarray:
    """flags (cloud_pipeline): optional 0..1 map of 'a geostationary cloud mask says cloud' (CLM x2, GOES-E/W,
    Himawari; viewing-angle faded, same canvas as vis).  When given it replaces the IR+dust thin layer."""
    blue = vis[..., 2].astype(np.float32) / 255.0
    vis_score = np.clip((blue - VISIBLE_CLEAR) / (VISIBLE_CLOUD - VISIBLE_CLEAR), 0, 1) ** VISIBLE_GAMMA

    ir_gray = ir.astype(np.float32).mean(axis=-1)
    clear = _clear_sky_level(ir_gray) + IR_MARGIN
    detail = ir_gray - gaussian_filter(ir_gray, sigma=ir_gray.shape[1] / 700, mode=("nearest", "wrap"))
    ir_sharp = ir_gray + IR_DETAIL_GAIN * detail
    ir_score = np.clip((ir_sharp - clear) / (IR_CLOUD - clear), 0, 1) ** IR_GAMMA

    daylight = _visible_weight(vis)
    if flags is None:
        thin = THIN_LAYER_OPACITY * _hidden_land_weight(ir_gray, elevation) * (1.0 - daylight) * _ir_dust_score(ir_gray, dust)
    else:
        # The satellite masks decide WHERE the thin layer goes (in addition to the cold-land rule); the dust + IR score
        # decides how it LOOKS (painting the binary flags directly read as flat, comic-like fills).  By day only where
        # visible shows no cloud nearby.
        lack = np.clip(1.0 - gaussian_filter(vis_score * daylight, sigma=ir_gray.shape[1] / 1000, mode=("nearest", "wrap")) / 0.25, 0, 1)
        where = np.maximum(_hidden_land_weight(ir_gray, elevation) * (1.0 - daylight), flags * ((1.0 - daylight) + daylight * lack))
        thin = THIN_LAYER_OPACITY * where * _ir_dust_score(ir_gray, dust)
    # (cloud_pipeline copy) The dust product is coarse, so the thin layer was a smooth texture-less fill that read as
    # blotches on the phone.  Give it the IR's own local structure: same cloud, real texture.
    ir_tex = np.clip(0.5 + (ir_gray - gaussian_filter(ir_gray, sigma=ir_gray.shape[1] / 500, mode=("nearest", "wrap"))) / THIN_TEXTURE_RANGE, 0, 1)
    thin = thin * (1.0 - THIN_TEXTURE_DEPTH + THIN_TEXTURE_DEPTH * 2.0 * ir_tex).clip(0, 1)
    ir_score = np.maximum(ir_score, thin)

    # Inside the daylit footprint the brighter of the two wins; the weight
    # fades visible out toward the footprint's rim, leaving IR alone.
    final = ir_score + daylight * np.clip(vis_score - ir_score, 0, None)

    # Feathered so the correction does not trace hard coastlines.
    snow = gaussian_filter(snow_ice_fraction, sigma=ir_gray.shape[1] / 500, mode=("nearest", "wrap"))
    threshold = np.clip(snow / SNOW_ICE_FULL_FRACTION, 0, 1) * SNOW_ICE_MAX_THRESHOLD
    final = np.clip((final - threshold) / (1.0 - threshold), 0, 1)

    # (cloud_pipeline copy) Very slightly softer cloud edges, then a faint wide haze for the soft look the user wants.
    final = (1.0 - SOFTEN_BLEND) * final + SOFTEN_BLEND * gaussian_filter(final, sigma=ir_gray.shape[1] / 4096, mode=("nearest", "wrap"))
    haze = HAZE_OPACITY * gaussian_filter(final, sigma=ir_gray.shape[1] / 330, mode=("nearest", "wrap"))
    final = final + haze * (1.0 - final)
    # (cloud_pipeline) Tone matched to the NASA VIIRS mask the user calls the gold standard (gh-pages
    # 20261115T062515Z): 21-quantile match of this output (5-sat v4, 03Z) at |lat|<55, ends smoothed.  It lifts clear
    # areas to a faint haze and keeps the brightest cloud just under white: subtle and soft, not black-to-white.
    final = np.interp(final, NASA_TONE_IN, NASA_TONE_OUT)
    # (cloud_pipeline) Sea haze: a light veil over open sea only, so the phone's navy sea reads lighter (user's Himawari
    # reference: sea median RGB (54,86,111) vs phone (49,69,90); the mask can only blend toward white, ~0.09 needed).
    if SEA_HAZE > 0:
        final = final + SEA_HAZE * _sea_weight(*final.shape) * (1.0 - final)
    final = final * _polar_fade(ir_gray)
    return (final * 255).astype(np.uint8)


def fetch_cloud_mask(width: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask, still_gap); still_gap is always all-zero. Falls back
    to the last successfully computed mask on any fetch/decode failure."""
    half_size = width // 2

    try:
        vis_east = _fetch_layer(LAYER_VISIBLE, BBOX_EAST, half_size)
        vis_west = _fetch_layer(LAYER_VISIBLE, BBOX_WEST, half_size)
        ir_east = _fetch_layer(LAYER_IR, BBOX_EAST, half_size)
        ir_west = _fetch_layer(LAYER_IR, BBOX_WEST, half_size)

        dust_east = _fetch_layer(LAYER_DUST, BBOX_EAST, half_size)
        dust_west = _fetch_layer(LAYER_DUST, BBOX_WEST, half_size)

        vis = _stitch(vis_east, vis_west)
        ir = _stitch(ir_east, ir_west)
        dust = _stitch(dust_east, dust_west)
        elevation = _fetch_elevation(width, vis.shape[0])
        snow_ice_fraction = _fetch_snow_ice_fraction(width, vis.shape[0])

        mask = _compute_mask(vis, ir, dust, snow_ice_fraction, elevation)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_FILE, mask)
    except Exception:
        logger.exception("EUMETSAT visible+IR fetch failed")
        if CACHE_FILE.exists():
            logger.warning("falling back to last cached cloud mask")
            mask = np.load(CACHE_FILE)
        else:
            raise

    still_gap = np.zeros(mask.shape, dtype=np.float32)
    return mask, still_gap
