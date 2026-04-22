#!/usr/bin/env python3
"""
Morning Sine Scanner (Polygon, VWAP touch only)

Purpose:
- Find stocks where price touches VWAP during the pre-market session.
- Output JSON for frontend / GitHub Actions use.

Design choices:
- Intentionally simple
- No HH/HL logic
- No extra filters, indicators, trend gates, gap rules, or complex scoring logic
- Uses Polygon/Massive 1-minute aggregates, then resamples to 15-minute candles
- Looks only at the PRE-MARKET window: 4:00 AM to 8:30 AM America/New_York
- Includes volume fields in the API response

Environment:
- Set POLYGON_API_KEY

Dependencies:
  pip install requests pandas numpy
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
NY_TZ = "America/New_York"
POLYGON_BASE = "https://api.polygon.io"


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    if any(c not in df.columns for c in REQUIRED_COLS):
        return pd.DataFrame()

    df = df[REQUIRED_COLS].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df.sort_index().dropna(subset=["Open", "High", "Low", "Close"])


def read_tickers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out


def json_sanitize(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


def debug_print(enabled: bool, *args):
    if enabled:
        print(*args)


def download_bars_polygon_1m(
    ticker: str,
    lookback_days: int,
    api_key: str,
    asof_date: Optional[str] = None,
    debug: bool = False,
) -> pd.DataFrame:
    import random
    import time

    def is_retryable(status_code: int, text: str) -> bool:
        msg = (text or "").lower()
        return status_code in (429, 500, 502, 503, 504) or "rate" in msg

    if not api_key:
        raise ValueError("Missing POLYGON_API_KEY")

    if asof_date:
        target_day = pd.Timestamp(asof_date, tz=NY_TZ)
    else:
        target_day = pd.Timestamp.now(tz=NY_TZ)

    start_day = (target_day - pd.Timedelta(days=max(1, int(lookback_days)))).date().isoformat()
    end_day = target_day.date().isoformat()

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper().strip()}/range/1/minute/{start_day}/{end_day}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    debug_print(debug, f"\n=== DOWNLOAD DEBUG (1m): {ticker} ===")
    debug_print(debug, "Polygon URL:", url)
    debug_print(debug, "Params:", params)

    max_retries = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            debug_print(debug, f"HTTP status: {resp.status_code}")

            if resp.status_code != 200:
                debug_print(debug, "Response text:", resp.text[:1000])
                if is_retryable(resp.status_code, resp.text) and attempt < max_retries - 1:
                    sleep_s = min(120.0, base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0))
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"polygon_http_{resp.status_code}: {resp.text[:300]}")

            payload = resp.json()
            results = payload.get("results", [])

            debug_print(debug, "Results count:", len(results))
            debug_print(debug, "Payload status:", payload.get("status"))
            debug_print(debug, "Request id:", payload.get("request_id"))
            debug_print(debug, "Ticker returned:", payload.get("ticker"))

            if not results:
                return pd.DataFrame(columns=REQUIRED_COLS)

            df = pd.DataFrame(results).copy()
            df["Datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.set_index("Datetime")
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            out = normalize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]])

            debug_print(debug, "Normalized 1m rows:", len(out))
            if not out.empty:
                debug_print(debug, "RAW 1m INDEX HEAD:")
                debug_print(debug, out.index[:10])
                debug_print(debug, "RAW 1m INDEX TAIL:")
                debug_print(debug, out.index[-10:])

            return out

        except Exception as e:
            last_err = e
            debug_print(debug, f"Download attempt {attempt + 1} failed:", repr(e))
            if attempt == max_retries - 1:
                raise
            sleep_s = min(120.0, base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0))
            time.sleep(sleep_s)

    raise last_err


def resample_to_15m(df_1m: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    if df_1m is None or df_1m.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    dfx = df_1m.copy()
    dfx.index = dfx.index.tz_convert(NY_TZ)

    out = (
        dfx.resample("15min", label="left", closed="left")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    debug_print(debug, "Resampled 15m rows:", len(out))
    if not out.empty:
        debug_print(debug, "15m INDEX HEAD:")
        debug_print(debug, out.index[:10])
        debug_print(debug, "15m INDEX TAIL:")
        debug_print(debug, out.index[-10:])

    return out


def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    vol = out["Volume"].fillna(0.0).astype(float)
    pv = typical * vol
    cum_vol = vol.cumsum()
    cum_pv = pv.cumsum()
    out["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
    return out


def extract_latest_premarket_window(
    df: pd.DataFrame,
    premarket_start: str,
    premarket_end: str,
    debug: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        debug_print(debug, "Session extract: empty raw dataframe")
        return pd.DataFrame()

    dfx = df.copy()
    if dfx.index.tz is None:
        dfx.index = dfx.index.tz_localize(NY_TZ)
    else:
        dfx.index = dfx.index.tz_convert(NY_TZ)

    start_h, start_m = map(int, premarket_start.split(":"))
    end_h, end_m = map(int, premarket_end.split(":"))

    all_days = sorted(pd.Index(dfx.index.date).unique())
    debug_print(debug, "All ET dates found:", all_days)

    if not all_days:
        return pd.DataFrame()

    latest_day = all_days[-1]
    debug_print(debug, "LATEST DAY FOUND:", latest_day)

    day_df = dfx[dfx.index.date == latest_day].copy()
    debug_print(debug, "Rows on latest ET day:", len(day_df))

    if day_df.empty:
        return pd.DataFrame()

    mask = []
    for ts in day_df.index:
        mins = ts.hour * 60 + ts.minute
        in_range = (mins >= start_h * 60 + start_m) and (mins < end_h * 60 + end_m)
        mask.append(in_range)

    sess = day_df.loc[np.array(mask, dtype=bool)].copy()
    debug_print(debug, "SESSION BARS:", len(sess))

    if not sess.empty:
        debug_print(debug, "SESSION HEAD:")
        debug_print(debug, sess[["Open", "High", "Low", "Close", "Volume"]].head(10).to_string())
        debug_print(debug, "SESSION TAIL:")
        debug_print(debug, sess[["Open", "High", "Low", "Close", "Volume"]].tail(10).to_string())

    if sess.empty:
        return pd.DataFrame()

    return add_session_vwap(sess)


def detect_vwap_touch(
    df: pd.DataFrame,
    vwap_touch_tolerance_pct: float,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        debug_print(debug, "detect_vwap_touch: empty session")
        return None

    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    vwaps = df["VWAP"].values.astype(float)
    vols = df["Volume"].fillna(0).values.astype(float)

    touch_bar = None
    touch_close = None
    touch_vwap = None
    touch_distance_pct = None

    for i in range(len(df)):
        low_i = float(lows[i])
        high_i = float(highs[i])
        close_i = float(closes[i])
        vwap_i = float(vwaps[i]) if np.isfinite(vwaps[i]) else np.nan
        if not np.isfinite(vwap_i) or vwap_i <= 0:
            continue

        candle_touches_vwap = low_i <= vwap_i <= high_i
        close_near_vwap_pct = abs(close_i - vwap_i) / vwap_i * 100.0
        near_vwap = close_near_vwap_pct <= float(vwap_touch_tolerance_pct)

        debug_print(
            debug,
            f"Bar {i}: low={low_i:.4f} high={high_i:.4f} close={close_i:.4f} "
            f"vwap={vwap_i:.4f} touch={candle_touches_vwap} near={near_vwap} "
            f"dist_pct={close_near_vwap_pct:.4f}"
        )

        if candle_touches_vwap or near_vwap:
            touch_bar = i
            touch_close = close_i
            touch_vwap = vwap_i
            touch_distance_pct = close_near_vwap_pct
            break

    if touch_bar is None:
        debug_print(debug, "Pattern failed at VWAP touch stage")
        return None

    debug_print(debug, f"VWAP touch found at bar {touch_bar}")

    total_volume = float(np.nansum(vols))
    avg_bar_volume = float(np.nanmean(vols)) if len(vols) else 0.0
    current_close = float(closes[-1])
    current_vwap = float(vwaps[-1]) if np.isfinite(vwaps[-1]) else np.nan

    score = 0.0
    if touch_distance_pct is not None:
        score += max(0.0, float(vwap_touch_tolerance_pct) - float(touch_distance_pct))

    return {
        "pattern_detected": True,
        "bars_used": int(len(df)),
        "pivot_strength": None,
        "session_date": str(df.index[-1].date()),
        "premarket_start": str(df.index[0]),
        "premarket_end": str(df.index[-1]),
        "hh_ok": None,
        "hl_ok": None,
        "prev_pivot_high_bar": None,
        "last_pivot_high_bar": None,
        "prev_pivot_low_bar": None,
        "last_pivot_low_bar": None,
        "prev_pivot_high": None,
        "last_pivot_high": None,
        "prev_pivot_low": None,
        "last_pivot_low": None,
        "vwap_pullback_bar": int(touch_bar),
        "vwap_pullback_time": str(df.index[touch_bar]),
        "vwap_at_pullback": safe_float(touch_vwap),
        "close_at_pullback": safe_float(touch_close),
        "close_to_vwap_pct": safe_float(touch_distance_pct),
        "current_close": safe_float(current_close),
        "current_vwap": safe_float(current_vwap),
        "current_close_vs_vwap_pct": safe_float(
            ((current_close - current_vwap) / current_vwap * 100.0)
            if np.isfinite(current_vwap) and current_vwap > 0 else None
        ),
        "premarket_volume": safe_float(total_volume),
        "avg_15m_volume": safe_float(avg_bar_volume),
        "last_15m_volume": safe_float(vols[-1] if len(vols) else None),
        "rank_score": safe_float(score),
    }


def scan(
    tickers: List[str],
    lookback_days: int,
    max_bars: int,
    vwap_touch_tolerance_pct: float,
    premarket_start: str,
    premarket_end: str,
    only_matches: bool,
    api_key: str,
    asof_date: Optional[str],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        t = ticker.strip().upper()
        if not t:
            continue

        debug_print(debug, f"\n================ {t} ================")

        try:
            df_1m = download_bars_polygon_1m(
                t,
                lookback_days=lookback_days,
                api_key=api_key,
                asof_date=asof_date,
                debug=debug,
            )

            if df_1m.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": "15m",
                        "pattern_detected": False,
                        "error": "no_polygon_bars_found",
                    })
                continue

            df_15m = resample_to_15m(df_1m, debug=debug)
            if df_15m.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": "15m",
                        "pattern_detected": False,
                        "error": "no_resampled_15m_bars_found",
                    })
                continue

            sess = extract_latest_premarket_window(
                df_15m,
                premarket_start=premarket_start,
                premarket_end=premarket_end,
                debug=debug,
            )

            if sess.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": "15m",
                        "pattern_detected": False,
                        "error": "no_premarket_window_found",
                    })
                continue

            sess = sess.tail(int(max_bars)).copy()

            info = detect_vwap_touch(
                df=sess,
                vwap_touch_tolerance_pct=vwap_touch_tolerance_pct,
                debug=debug,
            )

            if info is None:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": "15m",
                        "pattern_detected": False,
                        "bars_available": int(len(sess)),
                        "session_date": str(sess.index[-1].date()) if len(sess) else None,
                    })
                continue

            rows.append({
                "ticker": t,
                "tf": "15m",
                "bars_available": int(len(sess)),
                **info,
            })

        except Exception as e:
            if not only_matches:
                rows.append({
                    "ticker": t,
                    "tf": "15m",
                    "pattern_detected": False,
                    "error": str(e),
                })
            debug_print(debug, "Unhandled error:", repr(e))

    rows.sort(key=lambda r: float(r.get("rank_score", 0.0) or 0.0), reverse=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morning Sine Scanner")
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers_file", default="")
    p.add_argument("--out", default="output/morning_sine.json")

    p.add_argument("--lookback_days", type=int, default=2, help="Small recent fetch window for Polygon 1m aggregates")
    p.add_argument("--max_bars", type=int, default=40, help="Keep last N premarket bars after session filter")

    p.add_argument("--premarket_start", default="04:00", help="America/New_York")
    p.add_argument("--premarket_end", default="08:30", help="America/New_York")
    p.add_argument("--vwap_touch_tolerance_pct", type=float, default=0.30, help="Allow close to be this %% away from VWAP if the candle does not directly touch it")
    p.add_argument("--date", default="", help="Optional ET date like 2026-04-22. If omitted, uses current ET date.")
    p.add_argument("--only_matches", action="store_true", help="Write only matching tickers")
    p.add_argument("--debug", action="store_true", help="Print verbose debug logs to console")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("POLYGON_API_KEY", "").strip()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    payload: Dict[str, Any] = {
        "meta": {
            "scanner": "morning_sine_scanner",
            "interval": "15m",
            "lookback_days": int(args.lookback_days),
            "max_bars": int(args.max_bars),
            "premarket_timezone": NY_TZ,
            "premarket_start": args.premarket_start,
            "premarket_end": args.premarket_end,
            "pivot_strength": None,
            "vwap_touch_tolerance_pct": float(args.vwap_touch_tolerance_pct),
            "date": args.date or None,
            "only_matches": bool(args.only_matches),
            "debug": bool(args.debug),
            "lastUpdatedTs": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "total": 0,
            "matched": 0,
        },
        "data": [],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not api_key:
        payload["counts"]["total"] = len(tickers)
        payload["data"] = [{
            "ticker": None,
            "tf": "15m",
            "pattern_detected": False,
            "error": "Missing POLYGON_API_KEY",
        }]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)
        print("Missing POLYGON_API_KEY")
        return 2

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print("No tickers provided.")
        return 2

    rows = scan(
        tickers=tickers,
        lookback_days=args.lookback_days,
        max_bars=args.max_bars,
        vwap_touch_tolerance_pct=args.vwap_touch_tolerance_pct,
        premarket_start=args.premarket_start,
        premarket_end=args.premarket_end,
        only_matches=bool(args.only_matches),
        api_key=api_key,
        asof_date=(args.date.strip() if args.date else None),
        debug=bool(args.debug),
    )

    payload["counts"]["total"] = len(tickers)
    payload["counts"]["matched"] = sum(1 for r in rows if r.get("pattern_detected"))
    payload["data"] = rows

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)

    print(f"Wrote {payload['counts']['matched']} matches to {args.out}")
    if rows:
        preview_cols = [
            "ticker",
            "tf",
            "pattern_detected",
            "session_date",
            "vwap_pullback_time",
            "close_to_vwap_pct",
            "premarket_volume",
            "rank_score",
            "error",
        ]
        dfp = pd.DataFrame(rows)
        preview_cols = [c for c in preview_cols if c in dfp.columns]
        if preview_cols:
            print(dfp[preview_cols].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
