"""Download/refresh the DB-IP City Lite database (no account or key needed).

DB-IP publishes a new file each month at a predictable URL. This fetches the
current month's file (falling back to last month's around the turn of a month,
before the new one is published) and atomically swaps it into place, so a
running app never sees a half-written database.

Run monthly:  python -m src.data.update_geoip
Attribution (CC-BY 4.0): IP geolocation by DB-IP.com
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

URL_TEMPLATE = "https://download.db-ip.com/free/dbip-city-lite-{y}-{m:02d}.mmdb.gz"


def update() -> bool:
    target = Path(settings.geoip_db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    today = date.today()
    months = [today, today.replace(day=1) - timedelta(days=1)]  # current, previous

    for d in months:
        url = URL_TEMPLATE.format(y=d.year, m=d.month)
        tmp = target.with_suffix(".mmdb.tmp")
        try:
            logger.info(f"Downloading {url}")
            # Custom User-Agent: the CDN 403s urllib's default 'Python-urllib'.
            req = urllib.request.Request(
                url, headers={"User-Agent": "realestatesearch-geoip-updater/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp, \
                 gzip.open(resp, "rb") as gz, open(tmp, "wb") as out:
                shutil.copyfileobj(gz, out)
            tmp.replace(target)  # atomic on the same filesystem
            logger.info(f"GeoIP database updated: {target} ({target.stat().st_size:,} bytes)")
            return True
        except Exception as e:  # noqa: BLE001 — try the previous month before giving up
            logger.warning(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    logger.error("GeoIP update failed for current and previous month")
    return False


if __name__ == "__main__":
    sys.exit(0 if update() else 1)
