#!/usr/bin/env python3
"""
Double Bottom Scanner

Purpose:
- Find a clean double bottom (W pattern) in the last N bars
- Main use case: monthly timeframe
- Optional use case: weekly timeframe
- Output JSON for frontend / GitHub Actions use

Design choices:
- Intentionally simple shape-only logic
- No trend filters, indicators, SMA gates, or other complex conditions
- Detects a pivot-low -> pivot-high -> pivot-low sequence inside the last pattern window
- Requires the two bottoms to be reasonably close and the middle peak to be clearly above them
- Requires price to have lifted after the second bottom

Dependencies:
  pip install yfinance pandas numpy
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_COLS = ["Open", "High", "Low", "Close"]


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index().dropna()


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

def download_bars(ticker: str, tf: str, lookback_days: int) -> pd.DataFrame:
    import random
    import time

    def is_rate_limited(err) -> bool:
        msg = str(err).lower()
        return (
            "too many requests" in msg
            or "rate limited" in msg
            or "429" in msg
        )

    interval = tf.lower().strip()
    if interval not in ("1mo", "1wk"):
        raise ValueError("tf must be 1mo or 1wk")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))

    max_retries = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker.upper().strip(),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return normalize_ohlc(raw)
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
# Double bottom logic
# -----------------------------

def detect_double_bottom(
    df: pd.DataFrame,
    pattern_bars: int,
    pivot_strength: int,
    bottom_tolerance_pct: float,
    min_bounce_pct: float,
    min_rise_after_second_bottom_pct: float,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None

    window = df.tail(int(pattern_bars)).copy()
    if len(window) < max(7, int(pattern_bars) - 1):
        return None

    lows = window["Low"].values.astype(float)
    highs = window["High"].values.astype(float)
    closes = window["Close"].values.astype(float)

    pl = pivot_points(lows, pivot_strength, pivot_strength, "low")
    ph = pivot_points(highs, pivot_strength, pivot_strength, "high")

    if len(pl) < 2 or len(ph) < 1:
        return None

    best: Optional[Dict[str, Any]] = None

    for li in range(len(pl) - 1):
        i1, low1 = pl[li]
        i2, low2 = pl[li + 1]
        if i2 <= i1 + 1:
            continue

        mid_highs = [(j, h) for (j, h) in ph if i1 < j < i2]
        if not mid_highs:
            continue

        j_mid, mid_peak = max(mid_highs, key=lambda x: x[1])

        if low1 <= 0 or low2 <= 0:
            continue

        lower_bottom = min(low1, low2)
        higher_bottom = max(low1, low2)
        bottom_similarity_pct = ((higher_bottom - lower_bottom) / lower_bottom) * 100.0
        bounce_from_bottom_pct = ((mid_peak - higher_bottom) / higher_bottom) * 100.0 if higher_bottom > 0 else 0.0

        if bottom_similarity_pct > float(bottom_tolerance_pct):
            continue
        if bounce_from_bottom_pct < float(min_bounce_pct):
            continue

        # Keep it simple: price should have lifted after second bottom.
        close_now = float(closes[-1])
        rise_after_second_bottom_pct = ((close_now - low2) / low2) * 100.0 if low2 > 0 else 0.0
        if rise_after_second_bottom_pct < float(min_rise_after_second_bottom_pct):
            continue

        # Optional cleanliness score: second bottom should happen after the middle peak,
        # and the current price should not be sitting below the middle of the W.
        midpoint = (mid_peak + higher_bottom) / 2.0
        recovered_halfway = close_now >= midpoint

        score = 0.0
        score += max(0.0, float(bottom_tolerance_pct) - bottom_similarity_pct)
        score += bounce_from_bottom_pct
        score += rise_after_second_bottom_pct
        score += 2.0 if recovered_halfway else 0.0

        rec = {
            "pattern_detected": True,
            "bars_used": int(len(window)),
            "pivot_strength": int(pivot_strength),
            "first_bottom_bar": int(i1),
            "middle_peak_bar": int(j_mid),
            "second_bottom_bar": int(i2),
            "first_bottom_date": str(window.index[i1].date()),
            "middle_peak_date": str(window.index[j_mid].date()),
            "second_bottom_date": str(window.index[i2].date()),
            "first_bottom": safe_float(low1),
            "middle_peak": safe_float(mid_peak),
            "second_bottom": safe_float(low2),
            "bottom_similarity_pct": safe_float(bottom_similarity_pct),
            "bounce_from_bottom_pct": safe_float(bounce_from_bottom_pct),
            "rise_after_second_bottom_pct": safe_float(rise_after_second_bottom_pct),
            "current_close": safe_float(close_now),
            "recovered_halfway": bool(recovered_halfway),
            "rank_score": safe_float(score),
        }

        if best is None or float(rec["rank_score"] or 0.0) > float(best["rank_score"] or 0.0):
            best = rec

    return best


# -----------------------------
# Scan
# -----------------------------

def scan(
    tickers: List[str],
    tf: str,
    lookback_days: int,
    max_bars: int,
    pattern_bars: int,
    pivot_strength: int,
    bottom_tolerance_pct: float,
    min_bounce_pct: float,
    min_rise_after_second_bottom_pct: float,
    only_matches: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        t = ticker.strip().upper()
        if not t:
            continue

        try:
            df = download_bars(t, tf=tf, lookback_days=lookback_days)
            if df.empty:
                continue
            df = df.tail(int(max_bars)).copy()
            info = detect_double_bottom(
                df=df,
                pattern_bars=pattern_bars,
                pivot_strength=pivot_strength,
                bottom_tolerance_pct=bottom_tolerance_pct,
                min_bounce_pct=min_bounce_pct,
                min_rise_after_second_bottom_pct=min_rise_after_second_bottom_pct,
            )

            if info is None:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": tf,
                        "pattern_detected": False,
                        "bars_available": int(len(df)),
                    })
                continue

            rows.append({
                "ticker": t,
                "tf": tf,
                "bars_available": int(len(df)),
                **info,
            })
        except Exception as e:
            if not only_matches:
                rows.append({
                    "ticker": t,
                    "tf": tf,
                    "pattern_detected": False,
                    "error": str(e),
                })

    rows.sort(key=lambda r: float(r.get("rank_score", 0.0) or 0.0), reverse=True)
    return rows


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Double Bottom Scanner")
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers_file", default="")
    p.add_argument("--out", default="output/double_bottom.json")

    p.add_argument("--tf", default="1mo", help="1mo or 1wk")
    p.add_argument("--lookback_days", type=int, default=2200, help="Recommended: ~6 years for monthly, ~4 years for weekly")
    p.add_argument("--max_bars", type=int, default=120)

    p.add_argument("--pattern_bars", type=int, default=12, help="Inspect only the last N bars for the W shape")
    p.add_argument("--pivot_strength", type=int, default=1, help="Small value keeps the pattern simple and sensitive")
    p.add_argument("--bottom_tolerance_pct", type=float, default=8.0, help="Max %% difference allowed between the two bottoms")
    p.add_argument("--min_bounce_pct", type=float, default=6.0, help="Middle peak must be at least this %% above the higher bottom")
    p.add_argument("--min_rise_after_second_bottom_pct", type=float, default=3.0, help="Current close must be at least this %% above second bottom")

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
            "scanner": "double_bottom_scanner",
            "tf": args.tf,
            "lookback_days": int(args.lookback_days),
            "max_bars": int(args.max_bars),
            "pattern_bars": int(args.pattern_bars),
            "pivot_strength": int(args.pivot_strength),
            "bottom_tolerance_pct": float(args.bottom_tolerance_pct),
            "min_bounce_pct": float(args.min_bounce_pct),
            "min_rise_after_second_bottom_pct": float(args.min_rise_after_second_bottom_pct),
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
        tf=args.tf,
        lookback_days=args.lookback_days,
        max_bars=args.max_bars,
        pattern_bars=args.pattern_bars,
        pivot_strength=args.pivot_strength,
        bottom_tolerance_pct=args.bottom_tolerance_pct,
        min_bounce_pct=args.min_bounce_pct,
        min_rise_after_second_bottom_pct=args.min_rise_after_second_bottom_pct,
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
            "first_bottom_date",
            "second_bottom_date",
            "bottom_similarity_pct",
            "bounce_from_bottom_pct",
            "rise_after_second_bottom_pct",
            "rank_score",
        ]
        dfp = pd.DataFrame(rows)
        preview_cols = [c for c in preview_cols if c in dfp.columns]
        if preview_cols:
            print(dfp[preview_cols].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
