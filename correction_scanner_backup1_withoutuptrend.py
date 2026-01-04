#!/usr/bin/env python3
"""
Actionable-after-correction scanner (structure-first, TradingView-like)

Goal:
Find stocks that had a correction and are now actionable:
- Strong move -> pullback/correction -> structure holds -> rebound starts

Fixes vs prior version:
- trend_ok does NOT require SMA50 slope >= 0 (too strict; INTC failed)
- support_held is based on structure anchor close, NOT SMA50 floor (INTC failed)
- configurable support tolerance (default 1%)

Examples:
  python correction_actionable_scanner.py --tickers INTC --tf 1d --out debug.json
  python correction_actionable_scanner.py --tickers_file data/nasdaq100.txt --tf 1d --only_actionable --out api/actionable/nas.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close"]


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) and len(c) else str(c) for c in df.columns]

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


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = normalize_ohlc(df)
    if df.empty:
        return df
    return (
        df.resample(rule)
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
          .dropna()
    )


def download_bars(ticker: str, tf: str, lookback_days: int, threads: bool = True) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    t = ticker.upper().strip()
    tf_l = tf.lower().strip()

    if tf_l in ("4h", "2h"):
        raw = yf.download(
            t,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="60m",
            auto_adjust=True,
            progress=False,
            threads=threads,
        )
        raw = normalize_ohlc(raw)
        if raw.empty:
            return pd.DataFrame()
        rule = "4h" if tf_l == "4h" else "2h"
        return resample_ohlc(raw, rule)

    interval = "1d" if tf_l in ("1d", "d", "day") else tf_l
    raw = yf.download(
        t,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=threads,
    )
    return normalize_ohlc(raw)


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def read_tickers_file(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out


def compute_actionable(
    df: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    min_pullback_pct: float,
    swing_window: int,
    support_lookback: int,
    rebound_bars: int,
    support_tolerance_pct: float,
) -> Dict[str, Any]:
    close = df["Close"]
    high = df["High"]

    close_now = float(close.iloc[-1])

    s_fast = sma(close, sma_fast)
    s_slow = sma(close, sma_slow)

    sma_fast_now = safe_float(s_fast.iloc[-1])
    sma_slow_now = safe_float(s_slow.iloc[-1])

    # Trend filter: price above SMA slow (no slope requirement)
    trend_ok = bool(sma_slow_now is not None and close_now > sma_slow_now)

    # Recent swing high for pullback measurement
    swing_window = int(max(20, swing_window))
    recent_high = float(high.tail(swing_window).max())
    pullback_pct = (recent_high - close_now) / recent_high * 100.0 if recent_high > 0 else 0.0
    had_correction = pullback_pct >= float(min_pullback_pct)

    # Where did the recent high occur?
    recent_slice = df.tail(swing_window)
    idx_high = recent_slice["High"].idxmax()
    after_high = df.loc[idx_high:]

    # Correction low using CLOSE (not wicks)
    correction_close_low = float(after_high["Close"].min()) if len(after_high) else float(close.tail(swing_window).min())

    # Anchor support: lowest CLOSE in the period BEFORE the swing high (structure)
    support_lookback = int(max(20, support_lookback))
    before_high = df.loc[:idx_high].tail(support_lookback)
    anchor_support_close = float(before_high["Close"].min()) if len(before_high) else float(close.tail(support_lookback).min())

    # Support held: correction closes did not break anchor (allow small tolerance)
    tol = float(max(0.0, support_tolerance_pct)) / 100.0
    support_floor = anchor_support_close
    support_floor_adj = support_floor * (1.0 - tol)
    support_held = bool(correction_close_low >= support_floor_adj)

    # Rebound confirmation:
    close_above_fast = bool(sma_fast_now is not None and close_now > sma_fast_now)

    rebound_bars = int(max(3, rebound_bars))
    recent_closes = close.tail(rebound_bars).values
    net_up = bool(recent_closes[-1] > recent_closes[0])
    rebound_ok = bool(close_above_fast and net_up)

    actionable = bool(trend_ok and had_correction and support_held and rebound_ok)

    dist_to_support = None
    if support_floor > 0:
        dist_to_support = abs(close_now - support_floor) / support_floor

    return {
        "close": close_now,
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "sma_fast_now": safe_float(sma_fast_now),
        "sma_slow_now": safe_float(sma_slow_now),
        "trend_ok": bool(trend_ok),

        "recent_swing_high": safe_float(recent_high),
        "pullback_from_swing_high_pct": safe_float(pullback_pct),
        "had_correction": bool(had_correction),

        "anchor_support_close": safe_float(anchor_support_close),
        "support_tolerance_pct": float(support_tolerance_pct),
        "support_floor": safe_float(support_floor),
        "support_floor_adj": safe_float(support_floor_adj),
        "correction_close_low": safe_float(correction_close_low),
        "support_held": bool(support_held),

        "close_above_sma_fast": bool(close_above_fast),
        "rebound_ok": bool(rebound_ok),

        "actionable_after_correction": bool(actionable),
        "dist_to_support": safe_float(dist_to_support),
    }


def scan(
    tickers: List[str],
    tf: str,
    lookback_days: int,
    max_bars: int,
    sma_fast: int,
    sma_slow: int,
    min_pullback_pct: float,
    swing_window: int,
    support_lookback: int,
    rebound_bars: int,
    support_tolerance_pct: float,
    only_actionable: bool,
    threads: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue

        df = download_bars(t, tf=tf, lookback_days=lookback_days, threads=threads)
        if df.empty:
            continue

        df = df.tail(int(max_bars)).copy()
        if len(df) < max(120, sma_slow + 10):
            continue

        info = compute_actionable(
            df=df,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            min_pullback_pct=min_pullback_pct,
            swing_window=swing_window,
            support_lookback=support_lookback,
            rebound_bars=rebound_bars,
            support_tolerance_pct=support_tolerance_pct,
        )

        if only_actionable and not info["actionable_after_correction"]:
            continue

        # score: actionable + bigger pullback + closer to anchor support
        score = 0.0
        score += 1.0 if info["actionable_after_correction"] else 0.0
        pb = info["pullback_from_swing_high_pct"] or 0.0
        score += min(pb / 10.0, 0.8)

        d = info["dist_to_support"]
        if d is not None:
            score += max(0.0, 0.5 - d)

        results.append({"ticker": t, "tf": tf, "bars_used": int(len(df)), **info, "rank_score": float(score)})

    results.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)

    # one row per ticker
    seen = set()
    out = []
    for r in results:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actionable-after-correction scanner -> JSON")
    p.add_argument("--tickers", nargs="*", default=[], help="Tickers list")
    p.add_argument("--tickers_file", default="", help="One ticker per line")

    p.add_argument("--tf", default="1d", help="1d or 4h recommended")
    p.add_argument("--lookback_days", type=int, default=260)
    p.add_argument("--max_bars", type=int, default=300)

    p.add_argument("--sma_fast", type=int, default=20)
    p.add_argument("--sma_slow", type=int, default=50)

    p.add_argument("--min_pullback_pct", type=float, default=3.0)
    p.add_argument("--swing_window", type=int, default=80)
    p.add_argument("--support_lookback", type=int, default=80)
    p.add_argument("--rebound_bars", type=int, default=4)
    p.add_argument("--support_tolerance_pct", type=float, default=1.0, help="Allow small undercut of anchor support (in %)")

    p.add_argument("--only_actionable", action="store_true")
    p.add_argument("--out", default="output.json")
    p.add_argument("--no_threads", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "tf": args.tf,
            "lookback_days": args.lookback_days,
            "max_bars": args.max_bars,
            "sma_fast": args.sma_fast,
            "sma_slow": args.sma_slow,
            "min_pullback_pct": args.min_pullback_pct,
            "swing_window": args.swing_window,
            "support_lookback": args.support_lookback,
            "rebound_bars": args.rebound_bars,
            "support_tolerance_pct": args.support_tolerance_pct,
            "only_actionable": bool(args.only_actionable),
        },
        "count": 0,
        "data": [],
    }

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("No tickers provided.")
        return 2

    data = scan(
        tickers=tickers,
        tf=args.tf,
        lookback_days=args.lookback_days,
        max_bars=args.max_bars,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        min_pullback_pct=args.min_pullback_pct,
        swing_window=args.swing_window,
        support_lookback=args.support_lookback,
        rebound_bars=args.rebound_bars,
        support_tolerance_pct=args.support_tolerance_pct,
        only_actionable=bool(args.only_actionable),
        threads=not args.no_threads,
    )

    payload["count"] = len(data)
    payload["data"] = data

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(data)} rows to {args.out}")
    if data:
        dfp = pd.DataFrame(data)[
            ["ticker", "tf", "actionable_after_correction", "pullback_from_swing_high_pct",
             "trend_ok", "support_held", "rebound_ok", "rank_score"]
        ].head(25)
        print(dfp.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
