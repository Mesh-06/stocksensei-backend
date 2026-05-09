"""
cron.py — Daily retraining cron job for Stock Sensei AI
Runs every weekday at 6:30 PM IST (13:30 UTC) via Railway cron.
Re-fine-tunes all registered stocks with the latest market data.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

REGISTRY_PATH = Path("./models/registry.json")


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def main():
    logging.info("=" * 55)
    logging.info("  Stock Sensei AI — Daily Retraining Cron")
    logging.info(f"  Started: {datetime.now().isoformat()}")
    logging.info("=" * 55)

    reg = load_registry()
    if not reg:
        logging.warning("Registry is empty — no stocks to update. "
                        "Analyze at least one stock first via /analyze.")
        return

    # Import here so failures are caught cleanly
    try:
        from main import run_fine_tune
    except Exception as e:
        logging.error(f"Failed to import run_fine_tune: {e}")
        raise

    tickers = list(reg.keys())
    logging.info(f"Stocks to update: {tickers}")

    results = {}
    for i, ticker in enumerate(tickers, 1):
        logging.info(f"\n[{i}/{len(tickers)}] Updating {ticker}...")
        try:
            result = run_fine_tune(ticker, force_retrain=True)
            chg = result.get("change_pct_day1", 0)
            sig = result.get("signal", "?")
            logging.info(f"  ✓ {ticker}  signal={sig}  day1_chg={chg:+.2f}%")
            results[ticker] = "updated"
        except Exception as e:
            logging.error(f"  ✗ {ticker}  ERROR: {e}")
            results[ticker] = f"error: {e}"

    logging.info("\n" + "=" * 55)
    logging.info("  Summary")
    logging.info("=" * 55)
    updated = [t for t, s in results.items() if s == "updated"]
    failed  = [t for t, s in results.items() if s != "updated"]
    logging.info(f"  Updated : {len(updated)}  {updated}")
    if failed:
        logging.warning(f"  Failed  : {len(failed)}  {failed}")
    logging.info(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
