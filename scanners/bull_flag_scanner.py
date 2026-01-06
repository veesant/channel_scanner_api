#!/usr/bin/env python3
"""
Bull Flag Scanner (1D)

Bull flag definition (rule-based):
1) Flagpole: strong up move in last POLE_BARS bars (default 10)
   - % gain >= min_pole_gain_pct (default 10%)
   - slope positive
2) Flag: controlled pullback / consolidation in last FLAG_BARS bars (default 8)
   - pullback from pole high between flag_pullback_min_pct .. flag_pullback_max_pct (default 1%..8%)
   - flag range is relatively tight compared to pole range
   - current close is still above SMA20 and SMA50 (trend intact)
3) "Near breakout": current close within near_breakout_pct of pole high (default 3%)

Outputs:
- JSON with diagnostics
- Prints top rows summary to console
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Utilities
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
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close"]].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index().dropna()


def download_daily(ticker: str, lookback_days: int, threads: bool = True) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    raw = yf.download(
        ticker.upper().strip(),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=threads,
    )
    return normalize_ohlc(raw)


def read_tickers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def linreg_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom != 0 else 0.0


# -----------------------------
# Bull flag logic
# -----------------------------

@dataclass
class Config:
    pole_bars: int = 10
    flag_bars: int = 8

    min_pole_gain_pct: float = 10.0  # pole strength threshold
    near_breakout_pct: float = 3.0   # close must be within this % of pole high

    flag_pullback_min_pct: float = 1.0
    flag_pullback_max_pct: float = 8.0

    # flag should be "tight": flag range <= tight_ratio * pole range
    tight_ratio: float = 0.60

    sma_fast: int = 20
    sma_slow: int = 50


def evaluate_bull_flag(df: pd.DataFrame, cfg: Config) -> Optional[Dict[str, Any]]:
    need = max(cfg.sma_slow + 20, cfg.pole_bars + cfg.flag_bars + 5)
    if df is None or df.empty or len(df) < need:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    close_now = float(close.iloc[-1])
    sma20 = safe_float(sma(close, cfg.sma_fast).iloc[-1])
    sma50 = safe_float(sma(close, cfg.sma_slow).iloc[-1])

    # Trend filter: keep it simple and strong
    if sma20 is None or sma50 is None:
        return None
    trend_ok = bool(close_now > sma20 and sma20 > sma50)

    if not trend_ok:
        return None

    # Split windows: pole followed by flag
    w = df.tail(cfg.pole_bars + cfg.flag_bars).copy()
    pole = w.iloc[: cfg.pole_bars]
    flag = w.iloc[cfg.pole_bars :]

    pole_start = float(pole["Close"].iloc[0])
    pole_end = float(pole["Close"].iloc[-1])
    pole_high = float(pole["High"].max())
    pole_low = float(pole["Low"].min())

    if pole_start <= 0:
        return None

    pole_gain_pct = (pole_end - pole_start) / pole_start * 100.0
    pole_slope = linreg_slope(pole["Close"].values.astype(float))
    pole_range = pole_high - pole_low

    pole_ok = bool(pole_gain_pct >= cfg.min_pole_gain_pct and pole_slope > 0 and pole_range > 0)

    if not pole_ok:
        return None

    # Flag characteristics
    flag_high = float(flag["High"].max())
    flag_low = float(flag["Low"].min())
    flag_range = flag_high - flag_low

    # Pullback measured from pole_high to current close
    pullback_pct = (pole_high - close_now) / pole_high * 100.0 if pole_high > 0 else 0.0
    pullback_ok = bool(cfg.flag_pullback_min_pct <= pullback_pct <= cfg.flag_pullback_max_pct)

    # Tightness: consolidation range smaller than pole range
    tight_ok = bool(flag_range <= cfg.tight_ratio * pole_range)

    # Not broken: flag low should not undercut SMA20 too badly
    flag_hold_ok = bool(flag_low >= sma20 * 0.98)  # allow small undercut

    # Near breakout: close within near_breakout_pct of pole high
    near_breakout = bool(pullback_pct <= cfg.near_breakout_pct)

    flag_ok = bool(pullback_ok and tight_ok and flag_hold_ok)

    if not flag_ok:
        return None

    # Rank: prefer closer to breakout, stronger pole, tighter flag
    score = 0.0
    score += 1.0
    score += min(1.0, pole_gain_pct / 20.0)          # stronger pole
    score += max(0.0, (cfg.near_breakout_pct - pullback_pct) / max(0.01, cfg.near_breakout_pct))  # closer to breakout
    score += max(0.0, (cfg.tight_ratio - (flag_range / pole_range)) / max(0.01, cfg.tight_ratio)) # tighter flag

    return {
        "close": safe_float(close_now),
        "sma20": safe_float(sma20),
        "sma50": safe_float(sma50),
        "trend_ok": True,

        "pole_bars": cfg.pole_bars,
        "pole_gain_pct": safe_float(pole_gain_pct),
        "pole_high": safe_float(pole_high),
        "pole_range": safe_float(pole_range),

        "flag_bars": cfg.flag_bars,
        "flag_range": safe_float(flag_range),
        "pullback_from_pole_high_pct": safe_float(pullback_pct),
        "pullback_ok": bool(pullback_ok),
        "tight_ok": bool(tight_ok),
        "flag_hold_ok": bool(flag_hold_ok),
        "near_breakout": bool(near_breakout),

        "pattern": "bull_flag",
        "rank_score": float(score),
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bull Flag Scanner (1D)")
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--tickers_file", default="")
    p.add_argument("--out", default="bull_flags.json")

    p.add_argument("--lookback_days", type=int, default=520)
    p.add_argument("--max_bars", type=int, default=320)
    p.add_argument("--no_threads", action="store_true")

    # pattern parameters
    p.add_argument("--pole_bars", type=int, default=10)
    p.add_argument("--flag_bars", type=int, default=8)
    p.add_argument("--min_pole_gain_pct", type=float, default=10.0)
    p.add_argument("--near_breakout_pct", type=float, default=3.0)
    p.add_argument("--flag_pullback_min_pct", type=float, default=1.0)
    p.add_argument("--flag_pullback_max_pct", type=float, default=8.0)
    p.add_argument("--tight_ratio", type=float, default=0.60)

    p.add_argument("--sma_fast", type=int, default=20)
    p.add_argument("--sma_slow", type=int, default=50)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    cfg = Config(
        pole_bars=args.pole_bars,
        flag_bars=args.flag_bars,
        min_pole_gain_pct=args.min_pole_gain_pct,
        near_breakout_pct=args.near_breakout_pct,
        flag_pullback_min_pct=args.flag_pullback_min_pct,
        flag_pullback_max_pct=args.flag_pullback_max_pct,
        tight_ratio=args.tight_ratio,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
    )

    payload: Dict[str, Any] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "tf": "1d",
        "params": asdict(cfg),
        "count": 0,
        "data": [],
    }

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("No tickers provided.")
        return 2

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        df = download_daily(t, lookback_days=args.lookback_days, threads=not args.no_threads)
        if df.empty:
            continue
        df = df.tail(int(args.max_bars)).copy()

        info = evaluate_bull_flag(df, cfg)
        if info is None:
            continue
        rows.append({"ticker": t, "tf": "1d", "bars_used": int(len(df)), **info})

    rows.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)
    payload["count"] = len(rows)
    payload["data"] = rows

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(rows)} bull flags to {args.out}")
    if rows:
        dfp = pd.DataFrame(rows)[
            ["ticker", "close", "pole_gain_pct", "pullback_from_pole_high_pct", "near_breakout", "rank_score"]
        ].head(25)
        print(dfp.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
