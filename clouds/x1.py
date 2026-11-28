"""X28 cloud mask (2026-11-28; was X1, chosen 2026-11-20).  Pure visible by day, matteason's IR + dust look elsewhere.

X28 = X1 with three changes (developed in the ApkWork repo, cloud_pipeline/work_1120/x1_slot.py modes x14 / x25 / x28):
  * Day: PURE visible from the slot's own 3-hourly EUMETSAT natural-colour mosaic -- v5's visible score alone (no IR
    max, no mask-placed thin layer), measured against each region's own clear-ground brightness so sand is not cloud.
  * Night (X19): matteason's formula with a lower clear-ground IR level (-CLEAR_DROP) and a looser classifier gate;
    where Meteosat CLM says cloud the CLM replaces the gate.
  * Night, cold regions (X25): where the clear ground reads cold in IR (frozen land, sea ice) AND EUMETSAT's cloud-top
    height (cth.py) reports a cloud top, the cloud is made bolder and the nearest satellite's mask cloud is drawn.

Original X1 description:

  * Inside EUMETSAT's daylit visible footprint: the frozen v5 checkpoint (visir_v5.py, unchanged), its thin cloud layer
    placed by the GOES-East/West + Himawari masks only.  Meteosat CLM is NOT used to place cloud (it drew flat grey
    sheets over Europe/Africa).
  * Everywhere else (night, no visible): matteason's formula on IR + dust above the local clear ground,
    matt = sqrt(screen(dust_m, 0.77 * ir_m)), gated by a logistic classifier that each slot learns from the satellites'
    own clear/cloud verdicts (nearest view, all 5 masks incl. Meteosat) on night pixels; lowered over high ground.
  * Meteosat sun-glint spots over the sea use the night layer instead of visible.
  * The polar no-data cap (no IR/dust) is filled softly from the surrounding cloud.

Developed and checked against NASA VIIRS (inside NASA coverage) in the ApkWork repo, cloud_pipeline/work_1120/x1_slot.py;
this is the same code for one slot, sun from the slot time."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter as gf
from sklearn.linear_model import LogisticRegression

import visir as V                 # helpers (clear-ground level, regional percentile, constants)
import visir_v5 as V5             # frozen v5 checkpoint (2026-11-15)

H, W = 2048, 4096
WRAP = ("nearest", "wrap")
CLEAR_DROP = 8.0                  # (X19) clear-ground IR level lowered: warm low cloud was cut to 0 before the gate
GATE_LO, GATE_HI = 0.05, 0.30     # (X19) classifier gate ramp (X1: 0.10 .. 0.45)
COLD_LO, COLD_HI = 95.0, 115.0    # (X25) clear-ground IR level ramp that marks a cold region
BOLD_GAMMA, MASK_BASE = 0.65, 0.55  # (X25) cold-region boost: gamma on the X19 cloud; mask-cloud floor
HGT_LO, HGT_HI = 500.0, 4000.0    # (X25) cloud-top height ramp (m) that confirms cloud for the boost
CLOUD_SCALE = 0.85    # (2026-11-20) user: clouds too bright on the phone; thickest cloud now ~76% white, not 90%
SUN_REF, SUN_GAIN_MAX = 0.40, 2.5          # (2026-11-28) low-sun correction of the daytime photo
SEA_MASK = Path(__file__).parent.parent / "assets" / "sea_mask_4096x2048.npz"   # Americas-first, 1 = sea


def _A(x):                        # Greenwich-first <-> Americas-first
    return np.roll(x, W // 2, 1)


def _ss(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _sun(t: datetime):
    """Solar declination and subsolar longitude (radians), NOAA approximation."""
    hour = t.hour + t.minute / 60
    g = 2 * np.pi / 365 * (t.timetuple().tm_yday - 1 + (hour - 12) / 24)
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) \
        + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    return dec, np.radians(((-15 * (hour - 12) - eot / 4) + 180) % 360 - 180)


def compute_mask(t: datetime, L: dict, per: dict, clear: np.ndarray, cloud: np.ndarray,
                 sources: dict | None = None) -> np.ndarray:
    """L: EUMETSAT vis/ir/dust/snow/elev (Greenwich-first, run.fetch_visir_layers); per, clear, cloud: sat_masks.fetch_flags
    output (Americas-first).  Returns the uint8 mask, Americas-first (as the tiles expect).  sources: filled with
    what source_map() needs."""
    vis, ir, dust_u8, snow, elev_g = L["vis"], L["ir"], L["dust"], L["snow"], L["elev"]
    clear_v, cloud_v = clear.astype(np.float32), cloud.astype(np.float32)
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    lon = -180 + (np.arange(W) + 0.5) / W * 360
    LAT, LON = np.radians(lat)[:, None], np.radians(lon)[None, :]
    dec, sl = _sun(t)
    cosz = np.sin(LAT) * np.sin(dec) + np.cos(LAT) * np.cos(dec) * np.cos(LON - sl)
    a_day = _ss((cosz - 0.15) / (0.40 - 0.15))
    sea = np.load(SEA_MASK)["a"].astype(bool)

    # ---- night layer: matteason IR + dust, gated by a per-slot classifier (Americas-first)
    irg = _A(ir.astype(np.float32).mean(-1)); dust = _A(dust_u8).astype(np.float32) / 255
    ground_ir = V._clear_ground_level(irg, clear_v > 0.5, cell=32, fill_cells=3)
    clear_ir = np.maximum(np.clip(V._regional_percentile(irg, 10, 25.0), V.IR_CLEAR, V.IR_CLEAR_MAX),
                          np.where(ground_ir > 0, ground_ir + 10, 0)) - CLEAR_DROP
    rel = np.where(ground_ir > 0, irg - ground_ir, irg - clear_ir) / 50.0
    tex = np.sqrt(gf((irg - gf(irg, 2.0, mode=WRAP)) ** 2, 3.0, mode=WRAP)) / 10.0
    F = np.stack([dust[..., 0], dust[..., 1], dust[..., 2], rel, tex, dust[..., 0] * dust[..., 1], dust[..., 1] ** 2,
                  rel ** 2], -1).reshape(-1, 8)
    lab = np.where(cloud_v > 0.5, 1, np.where(clear_v > 0.5, 0, -1))
    trn = (a_day < 0.1) & (lab >= 0) & (np.abs(lat) < 70)[:, None]
    if trn.sum() >= 2000 and len(np.unique(lab[trn])) == 2:
        idx = np.random.default_rng(0).choice(np.flatnonzero(trn), min(300000, int(trn.sum())), replace=False)
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(F[idx], lab.ravel()[idx])
        p_cloud = clf.predict_proba(F)[:, 1].reshape(H, W).astype(np.float32)
    else:                                                         # no mask labels: ungated
        p_cloud = np.ones((H, W), np.float32)
    del F
    scr = lambda a, b: 1 - (1 - a) * (1 - b)                     # noqa: E731
    cold = 1 - dust[..., 2]
    ir_m = np.clip((irg - clear_ir) / (178.0 - 72.0), 0, 1) ** (1 / 1.46)
    dust_raw = scr(dust[..., 0] * cold, 0.5 * cold)
    dust_g = V._clear_ground_level(dust_raw, clear_v > 0.5, cell=32)
    dust_m = np.clip((dust_raw - dust_g) / np.maximum(1 - dust_g, 0.2), 0, 1)
    matt_raw = np.sqrt(scr(dust_m, 0.77 * ir_m)).astype(np.float32)          # ungated matt formula
    matt = matt_raw * _ss((p_cloud - GATE_LO) / (GATE_HI - GATE_LO))
    high = _ss((_A(elev_g) - V.ELEVATION_HIGH[0]) / (V.ELEVATION_HIGH[1] - V.ELEVATION_HIGH[0]))
    matt = (matt * (1 - 0.8 * high * (1 - gf(cloud_v, 1.5, mode=WRAP)))).astype(np.float32)

    # ---- Meteosat 0 / 45.5E sun glint over the sea: 1 = visible usable, 0 inside the spot
    unit = lambda la, lo: np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),   # noqa: E731
                                    np.sin(la) * np.ones_like(lo)], -1)
    n = unit(LAT, LON); sun = unit(dec, sl)
    refl = 2 * (n * sun).sum(-1)[..., None] * n - sun
    subs = (0.0, 45.5); near = np.argmin([-np.cos(LAT) * np.cos(LON - np.radians(s)) for s in subs], 0)
    glint = np.ones((H, W), np.float32)
    for i, s in enumerate(subs):
        v = unit(0.0, np.radians(s)) * 6.62 - n; v /= np.linalg.norm(v, axis=-1, keepdims=True)
        ga = np.degrees(np.arccos(np.clip((refl * v).sum(-1), -1, 1)))
        glint = np.where(near == i, _ss((ga - 15.0) / (35.0 - 15.0)), glint)
    del n, refl
    glint = np.where(sea & (cosz > 0), glint, 1.0).astype(np.float32)

    # ---- v5 by day (thin layer placed by GOES-E/W + Himawari only), glint spots taken out of its visible footprint
    flags = _A(np.max([np.nan_to_num(np.asarray(per[k], np.float32)) for k in per if not k.startswith("Meteosat")]
                      or [np.zeros((H, W), np.float32)], 0))
    v5parts = {}
    vw = V5._visible_weight
    V5._visible_weight = lambda x: vw(x) * _A(glint)
    try:
        m5 = _A(V5._compute_mask(vis, ir, dust_u8, snow, elev_g, flags, _A(clear_v > 0.5), v5parts).astype(np.float32) / 255)
        dl = _A(V5._visible_weight(vis))
    finally:
        V5._visible_weight = vw
    # ---- X28 day: pure visible (v5's desert-aware visible score + its snow / soften / haze / NASA-tone steps)
    # Clear-ground brightness per region where the satellite masks say clear.  (2026-11-28) On high ground IR is not
    # required to agree: cold plateau ground (Tibet, Andes) reads cloudy in IR, so it got no reference of its own,
    # borrowed a darker lowland one and its bare ground drew as ~0.15 cloud (washed-out Himalaya).
    # (2026-11-28) Low-sun correction: the photo is raw reflectance, so thick cloud under a low sun reads dark (median
    # 0.22 at cosz 0.1, 0.38 at 0.2 vs ~0.65 at cosz >= 0.4, 28 Nov 06Z) and drew as a half-transparent milky veil
    # (Ukraine / south Russia in the morning).  Scale the photo up to the cosz-0.4 level, at most x2.5.
    cz_g = np.roll(cosz, -W // 2, 1)                                              # Greenwich-first
    sun_gain = np.clip(SUN_REF / np.maximum(cz_g, 0.05), 1.0, SUN_GAIN_MAX)[..., None].astype(np.float32)
    vis_n = np.clip(np.asarray(vis, np.float32) * sun_gain, 0, 255)
    blue = vis_n[..., 2] / 255.0
    high_g = np.asarray(elev_g, np.float32) > V.ELEVATION_HIGH[0]
    clear_px = _A(clear_v > 0.5) & ((v5parts["ir"] < 0.08) | high_g) & (vw(vis) > 0.3)
    thr = np.maximum(V5.VISIBLE_CLEAR, V._clear_ground_level(blue, clear_px) + V.VISIBLE_GROUND_MARGIN)
    fv = np.clip((blue - thr) / np.maximum(V5.VISIBLE_CLOUD - thr, 0.15), 0, 1) ** V5.VISIBLE_GAMMA   # Greenwich-first
    # (2026-11-28) Snow is not cloud: in the natural-colour RGB (R = 1.6 um, B = 0.6 um) snow is cyan (bright B, dark
    # R) and water cloud white.  Ice cloud is cyan too, so only where the satellite masks say clear.
    red = vis_n[..., 0] / 255.0
    snowy = _ss((blue - red - 0.15) / 0.15) * _A(gf(clear_v, 1.5, mode=WRAP))
    fv = fv * (1 - snowy)
    sn = gf(np.asarray(snow, np.float32), W / 500, mode=WRAP)
    th = np.clip(sn / V5.SNOW_ICE_FULL_FRACTION, 0, 1) * V5.SNOW_ICE_MAX_THRESHOLD
    fv = np.clip((fv - th) / (1.0 - th), 0, 1)
    fv = (1.0 - V5.SOFTEN_BLEND) * fv + V5.SOFTEN_BLEND * gf(fv, W / 4096, mode=WRAP)
    fv = fv + V5.HAZE_OPACITY * gf(fv, W / 330, mode=WRAP) * (1.0 - fv)
    # (2026-11-28) The NASA tone curve lifts clear (0) to 0.06: a faint veil over all daylit ground (washed-out look,
    # night clear stays 0).  Drop that floor so clear day ground is clear; the rest of the curve is unchanged.
    t0 = V5.NASA_TONE_OUT[0]
    mvis = _A(np.clip((np.interp(fv, V5.NASA_TONE_IN, V5.NASA_TONE_OUT) - t0) / (1 - t0), 0, 1).astype(np.float32))

    # ---- X19 night: Meteosat CLM replaces the classifier gate in the IR-only area
    nightw = _ss((0.2 - _A(vw(vis))) / 0.2)
    clm = np.max([np.nan_to_num(np.asarray(per[k], np.float32)) for k in per if k.startswith("Meteosat")]
                 or [np.zeros((H, W), np.float32)], 0)
    clm_cloud = _ss(gf(clm, 2.0, mode=WRAP) / 0.8) * nightw * matt_raw
    night = np.maximum(matt, clm_cloud)
    by_clm = clm_cloud > matt + 0.02                          # source map: CLM lifted the gate here
    by_cold = np.zeros((H, W), bool)
    # ---- X25: cold regions, only where EUMETSAT's cloud-top height confirms a top: bolder + nearest mask cloud
    hgt = L.get("cth")
    if hgt is not None:
        has_top = gf(_ss((np.nan_to_num(np.asarray(hgt, np.float32), nan=0.0) - HGT_LO) / (HGT_HI - HGT_LO)), 1.5,
                     mode=WRAP)
        cold_w = gf(_ss((clear_ir - COLD_LO) / (COLD_HI - COLD_LO)), 3.0, mode=WRAP) * nightw * has_top
        bold = np.clip(night, 0, 1) ** (1 - (1 - BOLD_GAMMA) * cold_w)
        mask_cloud = np.maximum(_ss(gf(cloud_v, 2.0, mode=WRAP) / 0.8), has_top) * nightw \
            * (MASK_BASE + (1 - MASK_BASE) * matt_raw)
        boosted = np.maximum(bold, cold_w * mask_cloud)
        by_cold = boosted > night + 0.02                      # source map: cold-region boost added cloud here
        night = boosted
    night = night.astype(np.float32)
    final = dl * mvis + (1 - dl) * night

    # ---- polar no-data cap: soft normalised-convolution fill from the surrounding cloud
    has_ir = (np.asarray(ir).max(-1) < 250) & (np.asarray(dust_u8).max(-1) > 0)
    has = gf(_A(has_ir).astype(np.float32), 10.0, mode=WRAP) > 0.99
    has |= dl > 0.5
    fill = np.full((H, W), np.nan, np.float32)
    for s in (6, 16, 40, 100, 250):
        wsum = gf(has.astype(np.float32), s, mode=WRAP)
        val = gf(np.where(has, final, 0).astype(np.float32), s, mode=WRAP) / np.maximum(wsum, 1e-6)
        fill = np.where(np.isnan(fill) & (wsum > 0.05), val, fill)
    fill = np.where(np.isnan(fill), final[has].mean(), fill)
    keep = gf(has.astype(np.float32), 8.0, mode=WRAP) * has
    final = np.clip(keep * final + (1 - keep) * fill, 0, 1)
    if sources is not None:
        sources.update(_classify(final, dl, keep, glint, _A(vw(vis)), by_clm, by_cold))
    return (final * CLOUD_SCALE * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- source map (published as _debug_sources.png)
SOURCE_NAMES = ["visible (EUMETSAT 3-hourly)", "night: matt IR+dust", "night: Meteosat CLM gate lifted",
                "cold-region boost (cloud top)", "glint spot -> night layer", "polar no-data fill"]
SOURCE_COLOURS = np.array([[255, 220, 0], [170, 60, 255], [0, 160, 255], [0, 230, 60], [255, 255, 255],
                           [255, 140, 0]], np.float32)


def _classify(final, dl, keep, glint, vis_footprint, by_clm, by_cold) -> dict:
    """Which source drew each cloudy pixel (Americas-first); -1 = clear (final < 0.15)."""
    cloudy = final > 0.15
    cls = np.full(final.shape, -1, np.int8)
    day = dl > 0.5
    cls[day & cloudy] = 0
    cls[~day & cloudy] = 1
    cls[~day & cloudy & by_clm] = 2
    cls[~day & cloudy & by_cold] = 3
    cls[(vis_footprint > 0.5) & (glint < 0.5) & cloudy] = 4
    cls[(keep < 0.5) & cloudy] = 5
    return {"cls": cls, "final": final}


def _font(size: int):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


NASA_NOON_SUN_MIN = 10.5   # deg: NASA VIIRS true colour has no daytime image where the noon sun stays this low


def nasa_nodata(t: datetime) -> np.ndarray:
    """Where NASA's VIIRS true-colour day has no data: the polar-night cap, rows whose noon sun elevation is below
    NASA_NOON_SUN_MIN on the slot's day (Americas-first 4096x2048, NASA's grid).  Fitted to NASA's real no-data area:
    99.8% agreement on 18 Nov (local nasa18.jpg) and on 20 Nov (main feed's _debug_cloud_mask.png).  No download."""
    dec, _ = _sun(t)
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    return np.repeat((90 - np.abs(lat - np.degrees(dec)) < NASA_NOON_SUN_MIN)[:, None], W, 1)


def source_map(sources: dict, t: datetime, nodata: np.ndarray | None = None):
    """Colour per source, brighter where the cloud is denser; legend with each source's share.  nodata: NASA's no-data
    mask; those pixels are black and left out of the shares."""
    from PIL import Image, ImageDraw
    cls, final = sources["cls"], sources["final"]
    img = np.full(cls.shape + (3,), 45.0, np.float32)
    for i, c in enumerate(SOURCE_COLOURS):
        m = cls == i
        img[m] = c * (0.55 + 0.45 * final[m, None])
    inside = np.ones(cls.shape, bool) if nodata is None else ~nodata
    img[~inside] = 0
    img = Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((2048, 1024))
    f26, f18 = _font(26), _font(18)
    leg = Image.new("RGB", (2048, 320), (0, 0, 0)); d = ImageDraw.Draw(leg)
    dec, sl = _sun(t)
    d.text((20, 8), f"X28, {t:%d %b %Y %H:%M}Z  (subsolar {np.degrees(sl):.0f}E, dec {np.degrees(dec):.1f})",
           fill=(255, 255, 255), font=f26)
    items = [(n, tuple(int(v) for v in c), np.mean(cls[inside] == i) * 100) for i, (n, c) in enumerate(zip(SOURCE_NAMES, SOURCE_COLOURS))]
    items.append(("clear (mask < 0.15)", (45, 45, 45), np.mean(cls[inside] == -1) * 100))
    for i, (n, c, pct) in enumerate(items):
        x, y = 20 + (i % 3) * 680, 52 + (i // 3) * 50
        d.rectangle([x, y, x + 40, y + 36], fill=c, outline=(255, 255, 255))
        d.text((x + 52, y + 4), f"{n}  {pct:.1f}%", fill=c if c != (45, 45, 45) else (190, 190, 190), font=f26)
    d.text((20, 290), "black = no NASA data (excluded);  % = share of the area NASA covers" if nodata is not None
           else "% = share of the whole map", fill=(170, 170, 170), font=f18)
    out = Image.new("RGB", (2048, 1344)); out.paste(img, (0, 0)); out.paste(leg, (0, 1024))
    return out
