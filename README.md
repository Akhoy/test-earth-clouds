# test-earth-clouds

Test feed for the Earth live wallpaper's cloud layer (`com.breel.wallpapers18`). It runs for a few days before
anything moves to the main repo, [pixel-earth-wallpaper-live-clouds](https://github.com/Akhoy/pixel-earth-wallpaper-live-clouds).

A GitHub Actions job builds one grey cloud mask for each 3-hour satellite slot. It tries hourly, because GitHub's
schedule often skips runs, and does nothing when the latest slot is already published. It publishes
the mask to GitHub Pages in the app's `cloudMap` format: `root.json` plus 96 DDS tiles under `versions/<slot>/`.
The test APK reads `https://akhoy.github.io/test-earth-clouds/root.json`.

## How the mask is made
- **Cloud shapes and texture** come from EUMETSAT's global IR 10.8 mosaic and its natural-colour visible imagery,
  in `clouds/visir.py`. Visible is scored from the blue channel, and IR against a local clear-sky level. Inside
  the daylit area the brighter of the two is kept.
- **A thin cloud layer** covers cloud that IR can't see. Its look comes from EUMETSAT's dust product combined with
  IR texture.
- **Where that thin layer goes** is decided by 5 geostationary cloud masks, in `clouds/sat_masks.py`: EUMETSAT CLM
  for both Meteosat discs, GOES-East, GOES-West and Himawari-9. Each mask is faded toward its satellite's limb. By
  day the masks only add cloud where visible shows almost none.
- **The finish** is a slight softening, a faint wide haze, and a tone curve matched to NASA VIIRS true-colour
  cloud masks.

## Files next to each mask (`versions/<slot>/`)
- `_debug_cloud_mask.png`: the published mask.
- `_debug_cloud_mask_land.png`: the same mask with every continent drawn at one faint level, for orientation.
- `_debug_sources.png`: which part produced each pixel's cloud (IR, visible, the thin layer placed by each satellite
  or by the cold-land rule), with each part's share of the map.
- `_sources.txt`: every URL requested to build it.
- `_info.json`: slot, build time, masks used, mean.

## Run locally
```
pip install -r requirements.txt
python clouds/run.py --out site --base-url http://localhost:8765 [--slot YYYYmmddTHH]
```
The first run downloads one full Himawari file (~350 MB) to cache its fixed lat/lon grid in `cache/`.
