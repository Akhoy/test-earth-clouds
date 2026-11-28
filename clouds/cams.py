"""CAMS total aerosol optical depth (550 nm) for one slot, Americas-first 2048 x 4096 (2026-11-28, haze veil).

Source: Copernicus Atmosphere Data Store, dataset cams-global-atmospheric-composition-forecasts (global, 0.4 deg,
hourly lead times, runs at 00Z and 12Z, published several hours after the run).  The newest run that is out is used,
with the lead time that lands on the slot.  Credentials: cdsapi reads CDSAPI_URL / CDSAPI_KEY from the environment
(GitHub secret ADS_KEY in the workflow) or ~/.cdsapirc locally.  Checked against NASA VIIRS true colour (27 Nov): where
CAMS AOD > 0.4 on clear sky, NASA is clearly brighter than where AOD < 0.15 (land +0.19 vs +0.02 Africa/Europe).
(Developed in the ApkWork repo, cloud_pipeline/work_1120/cams/.)"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

log = logging.getLogger(__name__)
H, W = 2048, 4096
DATASET = "cams-global-atmospheric-composition-forecasts"


def fetch_aod(t: datetime) -> np.ndarray:
    import cdsapi
    import netCDF4

    c = cdsapi.Client(quiet=True, progress=False)
    run = t.replace(hour=t.hour // 12 * 12, minute=0, second=0, microsecond=0)
    last = None
    for back in range(4):                                       # this run, then up to 1.5 days older
        r = run - timedelta(hours=12 * back)
        lead = int((t - r).total_seconds() // 3600)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aod.nc"
            try:
                c.retrieve(DATASET, {"date": f"{r:%Y-%m-%d}/{r:%Y-%m-%d}", "time": [f"{r:%H}:00"],
                                     "leadtime_hour": [str(lead)], "type": ["forecast"],
                                     "variable": ["total_aerosol_optical_depth_550nm"], "data_format": "netcdf"}, str(out))
            except Exception as exc:                            # run not published yet: try the previous one
                last = exc
                continue
            d = netCDF4.Dataset(out)
            a = np.asarray(d.variables["aod550"][:], np.float32).reshape(-1, *d.variables["aod550"].shape[-2:])[0]
            lon = np.asarray(d.variables["longitude"][:])
            d.close()
        a = np.roll(a, -int(np.searchsorted(lon, 180.0)), 1)   # lon 0..360 -> -180..180 (Americas-first)
        log.info("CAMS AOD: run %s +%dh", r.strftime("%Y-%m-%d %HZ"), lead)
        return zoom(a, (H / a.shape[0], W / a.shape[1]), order=1).astype(np.float32)
    raise RuntimeError(f"no CAMS run available for {t:%Y-%m-%d %HZ}: {last!r}")
