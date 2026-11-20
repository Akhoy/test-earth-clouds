"""Build and publish one cloud mask for the latest complete 3-hour slot.

  1. slot = the latest 00/03/.../21Z scan at least ~1.5 h old (EUMETSAT and NOAA need that long to publish);
     if EUMETSAT does not have it yet, the slot before.
  2. the 5 satellite cloud masks for that slot (sat_masks.py) and EUMETSAT's IR mosaic, natural-colour visible and
     dust for the SAME slot (visir.py); x1.py builds the grey mask (v5 by day, matteason IR + dust elsewhere).
  3. 96 DDS tiles (6 cube faces x 4x4) + root.json, in the app's cloudMap format, into OUT/.

Usage: python clouds/run.py --out site --base-url https://<owner>.github.io/<repo>"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import visir as V                                              # noqa: E402
import x1 as X1                                                # noqa: E402
from sat_masks import fetch_flags                              # noqa: E402
from cubemap import equirect_to_cubemap, FACE_ORDER            # noqa: E402
from dxt1 import encode_tile_to_dds                            # noqa: E402
import debug                                                   # noqa: E402

W = 4096
FACE_SIZE, TILE_SIZE = 1024, 256
log = logging.getLogger("clouds")


def latest_slot(now: datetime) -> datetime:
    t = now - timedelta(hours=1, minutes=30)
    return t.replace(hour=t.hour - t.hour % 3, minute=0, second=0, microsecond=0)


def seed_static_cache() -> None:
    """NASA's snapshot API (terrain proxy, monthly snow climatology) often does not answer GitHub's runners.  Seed
    visir's cache from the copies in assets/ so a failed fetch falls back to them instead of to 'no terrain/snow'.
    The snow fetch still runs first each month and wins when it succeeds."""
    V.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for src in (Path(__file__).parent.parent / "assets").glob("*.npz"):
        dst = V.CACHE_DIR / (src.stem + ".npy")
        if not dst.exists():
            np.save(dst, np.load(src)["a"])


def fetch_visir_layers(t: datetime) -> dict:
    iso = t.strftime("%Y-%m-%dT%H:%M:%S.000Z"); h = W // 2
    jobs = [(k, lay, bb) for k, lay in (("vis", V.LAYER_VISIBLE), ("ir", V.LAYER_IR), ("dust", V.LAYER_DUST))
            for bb in (V.BBOX_EAST, V.BBOX_WEST)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: V._fetch_layer(j[1], j[2], h, iso), jobs))
    L = {k: V._stitch(res[2 * i], res[2 * i + 1]) for i, k in enumerate(("vis", "ir", "dust"))}
    if (L["ir"].max(-1) < 10).mean() > 0.5:                     # an unpublished scan comes back blank
        raise RuntimeError(f"EUMETSAT IR for {iso} is mostly empty")
    L["elev"] = V._fetch_elevation(W, h)
    L["snow"] = V._fetch_snow_ice_fraction(W, h)
    return L


def shade_of(gray: np.ndarray) -> np.ndarray:
    """(test repo, 2026-11-18 option 5) matteason's edge shading: cloud colour = 0.5 + 0.5 * blurred mask (3 px at
    4096), so thin edges and small cloud read grey and thick cores white.  Goes in the tiles' green channel; the
    patched earth.frag paints day cloud with it instead of pure white."""
    g = gaussian_filter(gray.astype(np.float32) / 255.0, 3.0 * gray.shape[1] / 4096, mode=("nearest", "wrap"))
    return ((0.5 + 0.5 * g) * 255).round().astype(np.uint8)


def write_tiles(gray: np.ndarray, dest: Path) -> int:
    """R = B = cloud mask (alpha), G = shade_of(mask) (colour)."""
    dest.mkdir(parents=True, exist_ok=True)
    faces = equirect_to_cubemap(np.stack([gray, shade_of(gray), gray], -1), FACE_SIZE)
    n = 0
    for name, face in faces.items():
        idx = FACE_ORDER.index(name)
        for row in range(FACE_SIZE // TILE_SIZE):
            for col in range(FACE_SIZE // TILE_SIZE):
                tile = face[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE]
                (dest / f"{idx}_2_{col}_{row}.dds").write_bytes(encode_tile_to_dds(tile)); n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--slot", help="YYYYmmddTHH (default: latest complete)")
    ap.add_argument("--skip-if-current", metavar="ROOT_JSON_URL",
                    help="exit without building when this published root.json already points at the latest slot")
    ap.add_argument("--raw", help="npz: load the fetched inputs from here if it exists, else save them there (local tuning)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_static_cache()
    debug.record_urls()

    if a.raw and Path(a.raw).exists():
        R = np.load(a.raw, allow_pickle=True)
        t = datetime.fromisoformat(str(R["slot"])); L = {k: R[k] for k in ("vis", "ir", "dust", "elev", "snow")}
        per = R["per"].item(); used = list(per); flags = np.max(np.stack(list(per.values())), 0)
        clear = R["clear"] if "clear" in R.files else None
        cloud = R["cloud"] if "cloud" in R.files else None
        log.info("loaded inputs for %s from %s", t, a.raw)
        return finish(a, t, L, flags, used, per, clear, cloud)
    if a.skip_if_current and not a.slot:
        import requests
        try:
            cur = requests.get(a.skip_if_current, timeout=30).json()["baseUrl"].rstrip("/").split("/")[-1]
        except Exception:
            cur = None
        want = f"{latest_slot(datetime.now(timezone.utc)):%Y%m%dT%H}Z"
        if cur == want:
            log.info("published root.json already at %s: nothing to do", want); return 0
    first = datetime.strptime(a.slot, "%Y%m%dT%H").replace(tzinfo=timezone.utc) if a.slot else latest_slot(datetime.now(timezone.utc))
    for t in (first, first - timedelta(hours=3)):
        try:
            log.info("slot %s: fetching EUMETSAT IR / visible / dust ...", t.strftime("%Y-%m-%d %HZ"))
            L = fetch_visir_layers(t); break
        except Exception as exc:
            log.warning("slot %s not usable: %r", t.strftime("%HZ"), exc)
    else:
        log.error("no usable EUMETSAT slot"); return 1

    flags, used, per, clear, cloud = fetch_flags(t, W)
    log.info("satellite masks used: %s", ", ".join(used) or "none")
    if a.raw:
        np.savez(a.raw, slot=t.isoformat(), per=np.array(per, dtype=object), clear=clear, cloud=cloud, **L)
    return finish(a, t, L, flags, used, per, clear, cloud)


def finish(a, t, L, flags, used, per, clear=None, cloud=None) -> int:
    # (2026-11-20) X1 look: v5 by day + matteason IR/dust elsewhere (x1.py).  per/clear/cloud stay Americas-first.
    if clear is None or cloud is None:
        raise RuntimeError("X1 needs the satellites' clear/cloud verdicts (re-fetch the raw inputs)")
    sources = {}
    mask = X1.compute_mask(t, L, per, clear, cloud, sources)     # Americas-first, as the tiles expect

    out = Path(a.out); vid = f"{t:%Y%m%dT%H}Z"
    vdir = out / "versions" / vid
    n = write_tiles(mask, vdir)
    Image.fromarray(mask).save(vdir / "_debug_cloud_mask.png")
    debug.land_debug(mask).save(vdir / "_debug_cloud_mask_land.png")
    X1.source_map(sources, t).save(vdir / "_debug_sources.jpg", quality=90)
    debug.write_urls(vdir / "_sources.txt")
    (vdir / "_info.json").write_text(json.dumps({"slot": t.isoformat(), "built": datetime.now(timezone.utc).isoformat(),
                                                 "masks": used, "mean": round(float(mask.mean()) / 255, 4)}, indent=1))
    (out / "root.json").write_text(json.dumps({"baseUrl": f"{a.base_url.rstrip('/')}/versions/{vid}/"}))
    log.info("wrote %d tiles -> %s; root.json -> %s", n, vdir, vid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
