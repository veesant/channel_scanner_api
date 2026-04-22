#!/usr/bin/env python3
"""
Morning Sine Scanner

Purpose:
- Find a simple pre-market 15-minute pattern:
  1) HH + HL structure inside the pre-market window
  2) price pulls back to VWAP inside the same pre-market window
- Output JSON for frontend / GitHub Actions use

Important design choices:
- Intentionally simple
- No extra filters, indicators, trend gates, gap rules, or complex scoring logic
- Looks only at the 15-minute PRE-MARKET window: 4:00 AM to 9:00 AM America/New_York
- Includes volume fields in the API response

Dependencies:
  pip install yfinance pandas numpy
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
NY_TZ = "America/New_York"


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None



def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    if any(c not in df.columns for c in REQUIRED_COLS):
        return pd.DataFrame()

    df = df[REQUIRED_COLS].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
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


# -----------------------------
# Download
# -----------------------------

def download_bars(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame:
    import random
    import time

    def is_rate_limited(err) -> bool:
        msg = str(err).lower()
        return (
            "too many requests" in msg
            or "rate limited" in msg
            or "429" in msg
        )

    if interval != "15m":
        raise ValueError("interval must be 15m")

    # For 15m data, yfinance is usually more reliable with a short `period`
    # than with explicit start/end dates. We only need the latest pre-market
    # session, so a tiny recent window is enough.
    period_days = max(1, int(lookback_days))
    period = f"{period_days}d"

    max_retries = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker.upper().strip(),
                period=period,
                interval="15m",
                auto_adjust=True,
                prepost=True,
                progress=False,
                threads=False,
            )
            return normalize_ohlcv(raw)
        except Exception as e:
            last_err = e
            if not is_rate_limited(e) or attempt == max_retries - 1:
                raise
            sleep_s = min(120.0, base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0))
            time.sleep(sleep_s)

    raise last_err


# -----------------------------
# Pivot logic
# -----------------------------

def pivot_points(values: np.ndarray, left: int, right: int, mode: str) -> List[Tuple[int, float]]:
    n = len(values)
    if n == 0:
        return []

    left = max(1, int(left))
    right = max(1, int(right))
    out: List[Tuple[int, float]] = []

    for i in range(left, n - right):
        w = values[i - left : i + right + 1]
        v = values[i]
        if not np.isfinite(v):
            continue

        if mode == "low":
            m = np.nanmin(w)
            if np.isfinite(m) and v == m and np.sum(w == m) == 1:
                out.append((i, float(v)))
        elif mode == "high":
            m = np.nanmax(w)
            if np.isfinite(m) and v == m and np.sum(w == m) == 1:
                out.append((i, float(v)))
        else:
            raise ValueError("mode must be 'low' or 'high'")

    return out


# -----------------------------
# Session helpers
# -----------------------------

def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    vol = out["Volume"].fillna(0.0).astype(float)
    pv = typical * vol
    cum_vol = vol.cumsum()
    cum_pv = pv.cumsum()
    out["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
    return out



def extract_latest_premarket_window(df: pd.DataFrame, premarket_start: str, premarket_end: str, current_only: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    dfx = df.copy()
    dfx.index = dfx.index.tz_convert(NY_TZ)

    start_h, start_m = map(int, premarket_start.split(":"))
    end_h, end_m = map(int, premarket_end.split(":"))

    all_days = sorted(pd.Index(dfx.index.date).unique())

    if current_only:
        ny_now = pd.Timestamp.now(tz=NY_TZ)
        all_days = [d for d in all_days if d == ny_now.date()]

    for day in reversed(all_days):
        day_df = dfx[dfx.index.date == day].copy()
        if day_df.empty:
            continue

        mask = []
        for ts in day_df.index:
            mins = ts.hour * 60 + ts.minute
            in_range = (mins >= start_h * 60 + start_m) and (mins < end_h * 60 + end_m)
            mask.append(in_range)
        sess = day_df.loc[np.array(mask, dtype=bool)].copy()
        if not sess.empty:
            return add_session_vwap(sess)

    return pd.DataFrame()


# -----------------------------
# Morning Sine logic
# -----------------------------

def detect_morning_sine(
    df: pd.DataFrame,
    pivot_strength: int,
    vwap_touch_tolerance_pct: float,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None

    if len(df) < max(5, pivot_strength * 2 + 3):
        return None

    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    vwaps = df["VWAP"].values.astype(float)
    vols = df["Volume"].fillna(0).values.astype(float)

    ph = pivot_points(highs, pivot_strength, pivot_strength, "high")
    pl = pivot_points(lows, pivot_strength, pivot_strength, "low")

    if len(ph) < 2 or len(pl) < 2:
        return None

    last_two_highs = ph[-2:]
    last_two_lows = pl[-2:]

    prev_high_bar, prev_high = last_two_highs[0]
    last_high_bar, last_high = last_two_highs[1]
    prev_low_bar, prev_low = last_two_lows[0]
    last_low_bar, last_low = last_two_lows[1]

    hh_ok = bool(last_high > prev_high)
    hl_ok = bool(last_low > prev_low)
    if not (hh_ok and hl_ok):
        return None

    structure_bar = max(last_high_bar, last_low_bar)

    touch_bar = None
    touch_close = None
    touch_vwap = None
    touch_distance_pct = None

    for i in range(structure_bar, len(df)):
        low_i = float(lows[i])
        high_i = float(highs[i])
        close_i = float(closes[i])
        vwap_i = float(vwaps[i]) if np.isfinite(vwaps[i]) else np.nan
        if not np.isfinite(vwap_i) or vwap_i <= 0:
            continue

        candle_touches_vwap = low_i <= vwap_i <= high_i
        close_near_vwap_pct = abs(close_i - vwap_i) / vwap_i * 100.0
        near_vwap = close_near_vwap_pct <= float(vwap_touch_tolerance_pct)

        if candle_touches_vwap or near_vwap:
            touch_bar = i
            touch_close = close_i
            touch_vwap = vwap_i
            touch_distance_pct = close_near_vwap_pct
            break

    if touch_bar is None:
        return None

    total_volume = float(np.nansum(vols))
    avg_bar_volume = float(np.nanmean(vols)) if len(vols) else 0.0
    current_close = float(closes[-1])
    current_vwap = float(vwaps[-1]) if np.isfinite(vwaps[-1]) else np.nan

    score = 0.0
    score += (last_high - prev_high)
    score += (last_low - prev_low)
    if touch_distance_pct is not None:
        score += max(0.0, float(vwap_touch_tolerance_pct) - float(touch_distance_pct))

    return {
        "pattern_detected": True,
        "bars_used": int(len(df)),
        "pivot_strength": int(pivot_strength),
        "session_date": str(df.index[-1].date()),
        "premarket_start": str(df.index[0]),
        "premarket_end": str(df.index[-1]),
        "hh_ok": bool(hh_ok),
        "hl_ok": bool(hl_ok),
        "prev_pivot_high_bar": int(prev_high_bar),
        "last_pivot_high_bar": int(last_high_bar),
        "prev_pivot_low_bar": int(prev_low_bar),
        "last_pivot_low_bar": int(last_low_bar),
        "prev_pivot_high": safe_float(prev_high),
        "last_pivot_high": safe_float(last_high),
        "prev_pivot_low": safe_float(prev_low),
        "last_pivot_low": safe_float(last_low),
        "vwap_pullback_bar": int(touch_bar),
        "vwap_pullback_time": str(df.index[touch_bar]),
        "vwap_at_pullback": safe_float(touch_vwap),
        "close_at_pullback": safe_float(touch_close),
        "close_to_vwap_pct": safe_float(touch_distance_pct),
        "current_close": safe_float(current_close),
        "current_vwap": safe_float(current_vwap),
        "current_close_vs_vwap_pct": safe_float(((current_close - current_vwap) / current_vwap * 100.0) if np.isfinite(current_vwap) and current_vwap > 0 else None),
        "premarket_volume": safe_float(total_volume),
        "avg_15m_volume": safe_float(avg_bar_volume),
        "last_15m_volume": safe_float(vols[-1] if len(vols) else None),
        "rank_score": safe_float(score),
    }


# -----------------------------
# Scan
# -----------------------------

def scan(
    tickers: List[str],
    interval: str,
    lookback_days: int,
    max_bars: int,
    pivot_strength: int,
    vwap_touch_tolerance_pct: float,
    premarket_start: str,
    premarket_end: str,
    only_matches: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        t = ticker.strip().upper()
        if not t:
            continue

        try:
            df = download_bars(t, interval=interval, lookback_days=lookback_days)
            if df.empty:
                continue

            sess = extract_latest_premarket_window(
                df,
                premarket_start=premarket_start,
                premarket_end=premarket_end,
                current_only=True,
            )
            if sess.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": interval,
                        "pattern_detected": False,
                        "error": "no_premarket_window_found",
                    })
                continue

            sess = sess.tail(int(max_bars)).copy()
            info = detect_morning_sine(
                df=sess,
                pivot_strength=pivot_strength,
                vwap_touch_tolerance_pct=vwap_touch_tolerance_pct,
            )

            if info is None:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": interval,
                        "pattern_detected": False,
                        "bars_available": int(len(sess)),
                        "session_date": str(sess.index[-1].date()) if len(sess) else None,
                    })
                continue

            rows.append({
                "ticker": t,
                "tf": interval,
                "bars_available": int(len(sess)),
                **info,
            })
        except Exception as e:
            if not only_matches:
                rows.append({
                    "ticker": t,
                    "tf": interval,
                    "pattern_detected": False,
                    "error": str(e),
                })

    rows.sort(key=lambda r: float(r.get("rank_score", 0.0) or 0.0), reverse=True)
    return rows


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morning Sine Scanner")
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers_file", default="")
    p.add_argument("--out", default="output/morning_sine.json")

    p.add_argument("--interval", default="15m", help="Must be 15m")
    p.add_argument("--lookback_days", type=int, default=2, help="Recent days to fetch from Yahoo; kept small because only today's premarket is needed")
    p.add_argument("--max_bars", type=int, default=40, help="Keep last N premarket bars after session filter")

    p.add_argument("--premarket_start", default="04:00", help="America/New_York")
    p.add_argument("--premarket_end", default="08:30", help="America/New_York")
    p.add_argument("--pivot_strength", type=int, default=1, help="Small value keeps HH/HL detection simple and sensitive")
    p.add_argument("--vwap_touch_tolerance_pct", type=float, default=0.30, help="Allow close to be this %% away from VWAP if the candle does not directly touch it")

    p.add_argument("--only_matches", action="store_true", help="Write only matching tickers")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    args = parse_args()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    payload: Dict[str, Any] = {
        "meta": {
            "scanner": "morning_sine_scanner",
            "interval": args.interval,
            "lookback_days": int(args.lookback_days),
            "max_bars": int(args.max_bars),
            "premarket_timezone": NY_TZ,
            "premarket_start": args.premarket_start,
            "premarket_end": args.premarket_end,
            "pivot_strength": int(args.pivot_strength),
            "vwap_touch_tolerance_pct": float(args.vwap_touch_tolerance_pct),
            "only_matches": bool(args.only_matches),
            "lastUpdatedTs": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "total": 0,
            "matched": 0,
        },
        "data": [],
    }

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print("No tickers provided.")
        return 2

    rows = scan(
        tickers=tickers,
        interval=args.interval,
        lookback_days=args.lookback_days,
        max_bars=args.max_bars,
        pivot_strength=args.pivot_strength,
        vwap_touch_tolerance_pct=args.vwap_touch_tolerance_pct,
        premarket_start=args.premarket_start,
        premarket_end=args.premarket_end,
        only_matches=bool(args.only_matches),
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
            "hh_ok",
            "hl_ok",
            "vwap_pullback_time",
            "close_to_vwap_pct",
            "premarket_volume",
            "rank_score",
        ]
        dfp = pd.DataFrame(rows)
        preview_cols = [c for c in preview_cols if c in dfp.columns]
        if preview_cols:
            print(dfp[preview_cols].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
