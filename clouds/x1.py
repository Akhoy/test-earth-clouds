"""X1 cloud mask (chosen 2026-11-20): the v5 look by day, matteason's IR + dust look elsewhere.

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
CLOUD_SCALE = 0.85    # (2026-11-20) user: clouds too bright on the phone; thickest cloud now ~76% white, not 90%
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
                          np.where(ground_ir > 0, ground_ir + 10, 0))
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
    matt = np.sqrt(scr(dust_m, 0.77 * ir_m)) * _ss((p_cloud - 0.10) / (0.45 - 0.10))
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
    final = dl * m5 + (1 - dl) * matt

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
        sources.update(_classify(final, dl, keep, glint, _A(vw(vis)), v5parts, per))
    return (final * CLOUD_SCALE * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- source map (published as _debug_sources.png)
SOURCE_NAMES = ["v5 visible (EUMETSAT)", "v5 IR", "v5 thin: cold-land rule", "v5 thin: GOES-East mask",
                "v5 thin: GOES-West mask", "v5 thin: Himawari mask", "matt IR+dust (no visible)", "glint spot -> matt",
                "polar no-data fill"]
SOURCE_COLOURS = np.array([[255, 220, 0], [0, 160, 255], [200, 200, 200], [255, 40, 40], [255, 80, 220], [0, 230, 60],
                           [170, 60, 255], [255, 255, 255], [255, 140, 0]], np.float32)


def _classify(final, dl, keep, glint, vis_footprint, p, per) -> dict:
    """Which source drew each cloudy pixel (Americas-first); -1 = clear (final < 0.15)."""
    vis_w = _A(p["day"] * p["vis"] > np.maximum(p["ir"], p["thin"]))
    thin_w = _A(p["thin"] > p["ir"]) & ~vis_w
    per_g = [_A(np.nan_to_num(np.asarray(per[k], np.float32))) for k in ("GOES-East", "GOES-West", "Himawari-9")
             if k in per]
    who = _A(np.argmax(np.stack([p["coldl"]] + per_g), 0))
    cloudy = final > 0.15
    cls = np.full(final.shape, -1, np.int8)
    pv5 = dl > 0.5
    cls[pv5 & cloudy] = 1
    sel = pv5 & cloudy & thin_w
    cls[sel] = 2 + who[sel]
    cls[pv5 & cloudy & vis_w] = 0
    cls[~pv5 & cloudy] = 6
    cls[(vis_footprint > 0.5) & (glint < 0.5) & cloudy] = 7
    cls[(keep < 0.5) & cloudy] = 8
    return {"cls": cls, "final": final}


def _font(size: int):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def source_map(sources: dict, t: datetime):
    """Colour per source, brighter where the cloud is denser; legend with each source's share of the map."""
    from PIL import Image, ImageDraw
    cls, final = sources["cls"], sources["final"]
    img = np.full(cls.shape + (3,), 45.0, np.float32)
    for i, c in enumerate(SOURCE_COLOURS):
        m = cls == i
        img[m] = c * (0.55 + 0.45 * final[m, None])
    img = Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((2048, 1024))
    f26, f18 = _font(26), _font(18)
    leg = Image.new("RGB", (2048, 320), (0, 0, 0)); d = ImageDraw.Draw(leg)
    dec, sl = _sun(t)
    d.text((20, 8), f"X1, {t:%d %b %Y %H:%M}Z  (subsolar {np.degrees(sl):.0f}E, dec {np.degrees(dec):.1f})",
           fill=(255, 255, 255), font=f26)
    items = [(n, tuple(int(v) for v in c), np.mean(cls == i) * 100) for i, (n, c) in enumerate(zip(SOURCE_NAMES, SOURCE_COLOURS))]
    items.append(("clear (mask < 0.15)", (45, 45, 45), np.mean(cls == -1) * 100))
    for i, (n, c, pct) in enumerate(items):
        x, y = 20 + (i % 3) * 680, 52 + (i // 3) * 50
        d.rectangle([x, y, x + 40, y + 36], fill=c, outline=(255, 255, 255))
        d.text((x + 52, y + 4), f"{n}  {pct:.1f}%", fill=c if c != (45, 45, 45) else (190, 190, 190), font=f26)
    d.text((20, 290), "% = share of the whole map; brighter = denser cloud", fill=(170, 170, 170), font=f18)
    out = Image.new("RGB", (2048, 1344)); out.paste(img, (0, 0)); out.paste(leg, (0, 1024))
    return out
