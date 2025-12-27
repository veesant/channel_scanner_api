#!/usr/bin/env python3
"""
Weekly Regression Channel Scanner (US stocks) -> JSON output (for GitHub Actions)

- Downloads OHLC via yfinance
- Resamples to weekly candles (W-FRI)
- Fits linear regression on log(Close)
- Builds channel: midline ± k * std(residuals)
- Filters for "clean" channels
- Classifies trend: Ascending / Sideways / Descending (annualized trend %)
- Flags: near support/resistance (pos)
- Supports:
    --only_in_channel   (current close must still be inside channel)
    --only_ascending    (output only Ascending)
    ONE ROW PER TICKER  (keeps best lookback by rank_score)
- Writes JSON to --out (default: output.json)

Example:
  python channel_scanner.py --tickers_file nasdaq100.txt --only_in_channel --only_ascending --out output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Weekly candles
# ----------------------------
def weekly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"].resample("W-FRI").first()
    h = df["High"].resample("W-FRI").max()
    l = df["Low"].resample("W-FRI").min()
    c = df["Close"].resample("W-FRI").last()
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()


# ----------------------------
# Trend classification + zones
# ----------------------------
def annualized_trend_pct_from_slope(slope: float) -> float:
    return (np.exp(slope * 52.0) - 1.0) * 100.0


def classify_trend(annual_trend_pct: float, sideways_threshold_pct: float = 8.0) -> str:
    if annual_trend_pct >= sideways_threshold_pct:
        return "Ascending"
    if annual_trend_pct <= -sideways_threshold_pct:
        return "Descending"
    return "Sideways"


def support_resistance_flags(
    pos: float, support_cutoff: float = 0.20, resistance_cutoff: float = 0.80
) -> tuple[bool, bool, str]:
    near_support = pos <= support_cutoff
    near_resistance = pos >= resistance_cutoff
    if near_support:
        zone = "Near Support"
    elif near_resistance:
        zone = "Near Resistance"
    else:
        zone = "Mid Channel"
    return near_support, near_resistance, zone


# ----------------------------
# Regression channel scoring
# ----------------------------
def regression_channel_score(
    w: pd.DataFrame, lookback: int = 52, k: float = 2.0
) -> Optional[Dict[str, float]]:
    w = w.dropna()
    if len(w) < lookback + 5:
        return None

    w2 = w.iloc[-lookback:].copy()
    y = np.log(w2["Close"].values)
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat

    resid_std = float(resid.std(ddof=1))
    if resid_std == 0 or not np.isfinite(resid_std):
        return None

    upper = yhat + k * resid_std
    lower = yhat - k * resid_std

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    inside = float(np.mean((y >= lower) & (y <= upper)))

    band = upper - lower
    near_upper = float(np.mean((upper - y) <= 0.10 * band))
    near_lower = float(np.mean((y - lower) <= 0.10 * band))

    pos = float((y[-1] - lower[-1]) / (upper[-1] - lower[-1]))

    return {
        "slope": float(slope),
        "r2": float(r2),
        "inside": float(inside),
        "near_upper": float(near_upper),
        "near_lower": float(near_lower),
        "pos": float(pos),
        "k": float(k),
        "lookback": float(lookback),
    }


# ----------------------------
# Scan tickers
# ----------------------------
def scan_channels(
    tickers: Iterable[str],
    years: int = 3,
    lookbacks: Tuple[int, ...] = (26, 39, 52),
    k: float = 2.0,
    min_r2: float = 0.65,
    min_inside: float = 0.92,
    min_touch: float = 0.03,
    only_in_channel: bool = True,
    pos_min: float = 0.05,
    pos_max: float = 0.95,
    sideways_threshold_pct: float = 8.0,
    support_cutoff: float = 0.20,
    resistance_cutoff: float = 0.80,
    threads: bool = True,
) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=True,
        group_by="ticker",
        threads=threads,
        progress=False,
    )

    results: List[Dict[str, Any]] = []

    for t in tickers:
        try:
            df = data[t].dropna()
            if df.empty:
                continue

            w = weekly_ohlc(df)
            if w.empty:
                continue

            for lb in lookbacks:
                s = regression_channel_score(w, lookback=int(lb), k=float(k))
                if not s:
                    continue

                if only_in_channel and not (pos_min <= s["pos"] <= pos_max):
                    continue

                if not (
                    s["r2"] >= min_r2
                    and s["inside"] >= min_inside
                    and s["near_upper"] >= min_touch
                    and s["near_lower"] >= min_touch
                ):
                    continue

                annual_trend_pct = annualized_trend_pct_from_slope(s["slope"])
                trend_type = classify_trend(annual_trend_pct, sideways_threshold_pct=sideways_threshold_pct)
                near_support, near_resistance, action_zone = support_resistance_flags(
                    s["pos"], support_cutoff=support_cutoff, resistance_cutoff=resistance_cutoff
                )

                results.append(
                    {
                        "ticker": t,
                        **s,
                        "annual_trend_pct": float(annual_trend_pct),
                        "trend_type": trend_type,
                        "near_support": bool(near_support),
                        "near_resistance": bool(near_resistance),
                        "action_zone": action_zone,
                    }
                )
        except Exception:
            continue

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["rank_score"] = (out["r2"] * 0.5) + (out["inside"] * 0.4) + ((1.0 - out["pos"]) * 0.1)
    out = out.sort_values(["rank_score"], ascending=False).reset_index(drop=True)
    return out


# ----------------------------
# CLI
# ----------------------------
def read_tickers_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s.split()[0].upper())
    return tickers


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan US stocks for weekly regression channels (JSON output).")
    p.add_argument("--tickers", nargs="*", default=[], help="List of tickers (space-separated).")
    p.add_argument("--tickers_file", default="", help="Text file with one ticker per line.")
    p.add_argument("--years", type=int, default=3, help="How many years of data to fetch.")
    p.add_argument("--lookbacks", nargs="*", type=int, default=[26, 39, 52], help="Lookback windows in weeks.")
    p.add_argument("--k", type=float, default=2.0, help="Channel width multiplier (std residuals).")
    p.add_argument("--min_r2", type=float, default=0.65, help="Minimum R² for regression fit.")
    p.add_argument("--min_inside", type=float, default=0.92, help="Minimum fraction of closes inside channel.")
    p.add_argument("--min_touch", type=float, default=0.03, help="Minimum fraction of weeks near each band.")
    p.add_argument("--out", default="output.json", help="Output JSON path.")

    p.add_argument("--only_in_channel", action="store_true", help="Require current close to be inside the channel.")
    p.add_argument("--pos_min", type=float, default=0.05, help="Minimum pos (0=lower, 1=upper).")
    p.add_argument("--pos_max", type=float, default=0.95, help="Maximum pos (0=lower, 1=upper).")

    p.add_argument("--only_ascending", action="store_true", help="Output only Ascending channels.")

    p.add_argument(
        "--sideways_threshold_pct",
        type=float,
        default=8.0,
        help="Annualized trend %% threshold to classify Asc/Desc vs Sideways.",
    )
    p.add_argument("--support_cutoff", type=float, default=0.20, help="pos <= this is Near Support.")
    p.add_argument("--resistance_cutoff", type=float, default=0.80, help="pos >= this is Near Resistance.")

    p.add_argument("--no_threads", action="store_true", help="Disable yfinance threading.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])

    if not tickers:
        print("No tickers provided. Use --tickers or --tickers_file.", file=sys.stderr)
        payload = {"updated_utc": pd.Timestamp.utcnow().isoformat(), "data": []}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return 2

    df = scan_channels(
        tickers=tickers,
        years=args.years,
        lookbacks=tuple(int(x) for x in args.lookbacks),
        k=args.k,
        min_r2=args.min_r2,
        min_inside=args.min_inside,
        min_touch=args.min_touch,
        only_in_channel=args.only_in_channel,
        pos_min=args.pos_min,
        pos_max=args.pos_max,
        sideways_threshold_pct=args.sideways_threshold_pct,
        support_cutoff=args.support_cutoff,
        resistance_cutoff=args.resistance_cutoff,
        threads=not args.no_threads,
    )

    # Optional: keep only Ascending
    if args.only_ascending and not df.empty:
        df = df[df["trend_type"] == "Ascending"].reset_index(drop=True)

    # ONE ROW PER TICKER: keep best lookback by rank_score
    if not df.empty:
        df = (
            df.sort_values("rank_score", ascending=False)
              .drop_duplicates(subset="ticker", keep="first")
              .reset_index(drop=True)
        )

    payload = {
        "updated_utc": pd.Timestamp.utcnow().isoformat(),
        "params": {
            "years": args.years,
            "lookbacks": list(map(int, args.lookbacks)),
            "k": args.k,
            "min_r2": args.min_r2,
            "min_inside": args.min_inside,
            "min_touch": args.min_touch,
            "only_in_channel": bool(args.only_in_channel),
            "pos_min": args.pos_min,
            "pos_max": args.pos_max,
            "only_ascending": bool(args.only_ascending),
            "sideways_threshold_pct": args.sideways_threshold_pct,
            "support_cutoff": args.support_cutoff,
            "resistance_cutoff": args.resistance_cutoff,
        },
        "count": int(len(df)),
        "data": df.to_dict(orient="records") if not df.empty else [],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {payload['count']} rows to: {args.out}")
    if payload["count"] > 0:
        # Print a small preview
        print(df[["ticker", "lookback", "trend_type", "pos", "action_zone", "rank_score"]].head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
