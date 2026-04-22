#!/usr/bin/env python3
"""
Morning Sine Scanner

Purpose:
- Pre-market (America/New_York, default 04:00–09:00) scan for US-style extended
  hours: price repeatedly interacting with session VWAP (anchored at the first
  bar in that window), with closes tending closer to VWAP through the window.
- HH / HL from simple pivots are computed for context only — they do not gate a match.

Match idea (see reference charts):
- VWAP touch: wick spans VWAP (Low <= VWAP <= High) or |Close - VWAP| / VWAP
  within a small tolerance.
- Convergence (on by default): mean distance Close↔VWAP in the second half of
  the window is tighter than in the first half (or the session was already tight
  on VWAP throughout).
- Optional: require at least N touch bars (default 2).

Dependencies:
  pip install yfinance pandas numpy

Data providers (bars only):
  - yfinance (--data-source yfinance, or auto with no API key)
  - Polygon.io REST aggregates (--data-source polygon, or auto with POLYGON_API_KEY)
    Get a key: https://polygon.io/dashboard (sign up → Dashboard → API Keys).
    Lower Polygon tiers often return no minute bars or no extended-hours slice; unless you
    pass --no-polygon-fallback-yfinance, the scanner refetches from yfinance when Polygon
    has no usable 4am–9am ET data (see JSON field bars_provider on each row).
    Manual test (PowerShell): $env:POLYGON_API_KEY="your_key"; python scanners/morning_sine_scanner.py --tickers AAPL --data-source auto --lookback_days 5 --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
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

POLYGON_REST_BASE = "https://api.polygon.io"


def polygon_ticker(yahoo_like: str) -> str:
    """Map Yahoo-style symbols to Polygon stock tickers (US-focused lists)."""
    u = yahoo_like.upper().strip().split()[0]
    if u.endswith(".NS") or u.endswith(".NSE"):
        u = u[: u.rfind(".")]
    elif u.endswith(".BO"):
        u = u[: u.rfind(".")]
    u = u.replace("-", ".")
    return u


def _polygon_fetch_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "morning_sine_scanner/1.0 (channel_scanner_api)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _polygon_append_api_key(url: str, api_key: str) -> str:
    if "apiKey=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urllib.parse.urlencode({'apiKey': api_key})}"


def _polygon_bar_time_unit(sample_t: Any) -> str:
    """Polygon `t` is usually Unix ms; detect ns/s mis-encoding."""
    try:
        tv = int(float(sample_t))
    except (TypeError, ValueError):
        return "ms"
    if tv > 10**15:
        return "ns"
    if tv < 10**11:
        return "s"
    return "ms"


def download_bars_polygon(ticker: str, interval: str, lookback_days: int, api_key: str) -> pd.DataFrame:
    if interval not in ("15m", "30m"):
        raise ValueError("interval must be 15m or 30m")
    mult = 15 if interval == "15m" else 30
    sym = polygon_ticker(ticker)
    if not sym:
        return pd.DataFrame()

    # Free / Basic Polygon plans often return sparse intraday; ask for more calendar history.
    eff_lookback = max(int(lookback_days), 7)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=eff_lookback)
    from_ms = int(start.timestamp() * 1000)
    to_ms = int(end.timestamp() * 1000)

    q = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key.strip(),
    }
    path_sym = urllib.parse.quote(sym, safe="")
    first = (
        f"{POLYGON_REST_BASE}/v2/aggs/ticker/{path_sym}/range/{mult}/minute/"
        f"{from_ms}/{to_ms}?{urllib.parse.urlencode(q)}"
    )

    bars: List[Dict[str, Any]] = []
    url: Optional[str] = first
    max_retries = 8
    base_sleep = 2.0

    while url:
        for attempt in range(max_retries):
            try:
                data = _polygon_fetch_json(url)
                status = str(data.get("status", "") or "")
                if status == "ERROR":
                    err_msg = data.get("error") or data.get("message") or str(data)
                    raise RuntimeError(f"Polygon API error: {err_msg}")
                for r in data.get("results") or []:
                    bars.append(r)
                url = data.get("next_url")
                if url:
                    url = _polygon_append_api_key(str(url), api_key.strip())
                else:
                    url = None
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    time.sleep(min(120.0, base_sleep * (2**attempt) + random.uniform(0.0, 1.0)))
                    continue
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                msg = f"Polygon HTTP {e.code}: {e.reason}"
                if detail:
                    msg = f"{msg} | body: {detail}"
                raise RuntimeError(msg) from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Polygon network error: {e}") from e

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    if not {"o", "h", "l", "c", "t"}.issubset(df.columns):
        return pd.DataFrame()

    vol = df["v"] if "v" in df.columns else pd.Series(0.0, index=df.index)
    t_unit = _polygon_bar_time_unit(df["t"].iloc[0])
    out = pd.DataFrame(
        {
            "Open": df["o"].astype(float),
            "High": df["h"].astype(float),
            "Low": df["l"].astype(float),
            "Close": df["c"].astype(float),
            "Volume": vol.astype(float),
        },
        index=pd.to_datetime(df["t"], unit=t_unit, utc=True),
    )
    return normalize_ohlcv(out)


def download_bars(
    ticker: str,
    interval: str,
    lookback_days: int,
    *,
    data_source: str = "yfinance",
    polygon_api_key: str = "",
) -> pd.DataFrame:
    def is_rate_limited(err: BaseException) -> bool:
        msg = str(err).lower()
        return "too many requests" in msg or "rate limited" in msg or "429" in msg

    if interval not in ("15m", "30m"):
        raise ValueError("interval must be 15m or 30m")

    if data_source == "polygon":
        return download_bars_polygon(ticker, interval, lookback_days, polygon_api_key)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))

    max_retries = 6
    base_sleep = 2.0
    last_err: Optional[BaseException] = None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker.upper().strip(),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
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
            sleep_s = min(120.0, base_sleep * (2**attempt) + random.uniform(0.0, 1.0))
            time.sleep(sleep_s)

    raise last_err  # type: ignore[misc]


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


def find_latest_rising_pair(pivots: List[Tuple[int, float]]) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
    """
    Find the most recent pair of pivots where the later pivot is higher than the earlier one.
    Returns: prev_bar, prev_value, last_bar, last_value
    """
    if len(pivots) < 2:
        return None, None, None, None

    for i in range(len(pivots) - 1, 0, -1):
        prev_bar, prev_val = pivots[i - 1]
        last_bar, last_val = pivots[i]
        if last_val > prev_val:
            return int(prev_bar), float(prev_val), int(last_bar), float(last_val)

    return None, None, None, None


# -----------------------------
# Session helpers
# -----------------------------

def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    vol = out["Volume"].fillna(0.0).astype(float)
    if float(vol.sum()) <= 0.0:
        # Yahoo often reports 0 volume in extended hours; true VWAP is undefined.
        # Use expanding mean of typical price as a stand-in "price magnet" line.
        out["VWAP"] = typical.expanding().mean()
        out["vwap_mode"] = "hlc3_expanding_mean"
    else:
        pv = typical * vol
        cum_vol = vol.cumsum()
        cum_pv = pv.cumsum()
        out["VWAP"] = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
        out["vwap_mode"] = "session_vwap"
    return out


def extract_latest_premarket_window(
    df: pd.DataFrame,
    premarket_start: str,
    premarket_end: str,
    strict_today: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    dfx = df.copy()
    dfx.index = dfx.index.tz_convert(NY_TZ)

    ny_today = pd.Timestamp.now(tz=NY_TZ).date()

    start_h, start_m = map(int, premarket_start.split(":"))
    end_h, end_m = map(int, premarket_end.split(":"))
    start_mins = start_h * 60 + start_m
    end_mins = end_h * 60 + end_m

    def session_for_date(d) -> pd.DataFrame:
        day_df = dfx[dfx.index.date == d].copy()
        if day_df.empty:
            return pd.DataFrame()
        mask = []
        for ts in day_df.index:
            mins = ts.hour * 60 + ts.minute
            in_range = (mins >= start_mins) and (mins < end_mins)
            mask.append(in_range)
        sess = day_df.loc[np.array(mask, dtype=bool)].copy()
        return sess

    if strict_today:
        sess = session_for_date(ny_today)
        return add_session_vwap(sess) if not sess.empty else pd.DataFrame()

    # Prefer calendar "today" when it has premarket bars; else most recent day in history.
    dates_try = [ny_today]
    for d in sorted({ts.date() for ts in dfx.index}, reverse=True):
        if d not in dates_try:
            dates_try.append(d)

    for d in dates_try:
        sess = session_for_date(d)
        if not sess.empty:
            return add_session_vwap(sess)

    return pd.DataFrame()


# -----------------------------
# Morning Sine logic
# -----------------------------

def _bar_touches_vwap(
    low_i: float,
    high_i: float,
    close_i: float,
    vwap_i: float,
    vwap_touch_tolerance_pct: float,
) -> Tuple[bool, float]:
    """Wick spans VWAP or close within tolerance; returns (touch, |close-vwap| as % of vwap)."""
    if not np.isfinite(vwap_i) or vwap_i <= 0:
        return False, float("nan")
    close_near_vwap_pct = abs(close_i - vwap_i) / vwap_i * 100.0
    candle_touches_vwap = low_i <= vwap_i <= high_i
    near_vwap = close_near_vwap_pct <= float(vwap_touch_tolerance_pct)
    return bool(candle_touches_vwap or near_vwap), float(close_near_vwap_pct)


def detect_morning_sine(
    df: pd.DataFrame,
    pivot_strength: int,
    vwap_touch_tolerance_pct: float,
    min_vwap_touches: int,
    convergence_required: bool,
    convergence_late_vs_early_ratio: float,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None

    if len(df) < max(3, pivot_strength * 2 + 1):
        return None

    vwap_mode = str(df["vwap_mode"].iloc[0]) if "vwap_mode" in df.columns else "session_vwap"

    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    vwaps = df["VWAP"].values.astype(float)
    vols = df["Volume"].fillna(0).values.astype(float)

    ph = pivot_points(highs, pivot_strength, pivot_strength, "high")
    pl = pivot_points(lows, pivot_strength, pivot_strength, "low")

    prev_high_bar, prev_high, last_high_bar, last_high = find_latest_rising_pair(ph)
    prev_low_bar, prev_low, last_low_bar, last_low = find_latest_rising_pair(pl)

    hh_ok = bool(prev_high is not None and last_high is not None and last_high > prev_high)
    hl_ok = bool(prev_low is not None and last_low is not None and last_low > prev_low)

    touch_bars: List[int] = []
    dist_close_vwap_pct: List[float] = []

    for i in range(len(df)):
        low_i = float(lows[i])
        high_i = float(highs[i])
        close_i = float(closes[i])
        vwap_i = float(vwaps[i]) if np.isfinite(vwaps[i]) else float("nan")
        touch, d_pct = _bar_touches_vwap(
            low_i, high_i, close_i, vwap_i, vwap_touch_tolerance_pct
        )
        dist_close_vwap_pct.append(d_pct if np.isfinite(d_pct) else float("nan"))
        if touch:
            touch_bars.append(i)

    touch_count = len(touch_bars)
    if touch_count < int(min_vwap_touches):
        return None

    dist_arr = np.array(dist_close_vwap_pct, dtype=float)
    valid = np.isfinite(dist_arr) & np.isfinite(vwaps.astype(float))
    if not np.any(valid):
        return None

    n = len(df)
    mid = max(1, n // 2)
    early = dist_arr[:mid]
    late = dist_arr[mid:]
    early_mean = float(np.nanmean(early)) if np.size(early) else float("nan")
    late_mean = float(np.nanmean(late)) if np.size(late) else float("nan")

    tol = float(vwap_touch_tolerance_pct)
    # Already hugging VWAP whole window — treat as converged.
    if np.isfinite(early_mean) and early_mean <= tol:
        convergence_ok = True
    elif np.isfinite(early_mean) and np.isfinite(late_mean) and early_mean > 1e-12:
        convergence_ok = late_mean <= early_mean * float(convergence_late_vs_early_ratio)
    elif np.isfinite(late_mean) and np.isfinite(dist_arr[-1]):
        last_d = float(dist_arr[-1])
        convergence_ok = last_d <= tol
    else:
        convergence_ok = False

    if convergence_required and not convergence_ok:
        return None

    last_touch_bar = touch_bars[-1]
    touch_close = float(closes[last_touch_bar])
    touch_vwap = float(vwaps[last_touch_bar])
    touch_distance_pct = abs(touch_close - touch_vwap) / touch_vwap * 100.0 if touch_vwap > 0 else float("nan")

    total_volume = float(np.nansum(vols))
    avg_bar_volume = float(np.nanmean(vols)) if len(vols) else 0.0
    current_close = float(closes[-1])
    current_vwap = float(vwaps[-1]) if np.isfinite(vwaps[-1]) else np.nan

    score = float(touch_count) * 10.0
    if np.isfinite(early_mean) and np.isfinite(late_mean) and early_mean > 0:
        score += max(0.0, (early_mean - late_mean) / early_mean * 5.0)
    if np.isfinite(touch_distance_pct):
        score += max(0.0, tol - float(touch_distance_pct))
    if hh_ok:
        score += 1.0
    if hl_ok:
        score += 1.0
    if prev_high is not None and last_high is not None:
        score += float(last_high - prev_high) / max(float(last_high), 1e-9) * 2.0
    if prev_low is not None and last_low is not None:
        score += float(last_low - prev_low) / max(float(last_low), 1e-9) * 2.0

    return {
        "pattern_detected": True,
        "bars_used": int(len(df)),
        "pivot_strength": int(pivot_strength),
        "session_date": str(df.index[-1].date()),
        "premarket_start": str(df.index[0]),
        "premarket_end": str(df.index[-1]),
        "vwap_mode": vwap_mode,
        "hh_ok": bool(hh_ok),
        "hl_ok": bool(hl_ok),
        "prev_pivot_high_bar": int(prev_high_bar) if prev_high_bar is not None else None,
        "last_pivot_high_bar": int(last_high_bar) if last_high_bar is not None else None,
        "prev_pivot_low_bar": int(prev_low_bar) if prev_low_bar is not None else None,
        "last_pivot_low_bar": int(last_low_bar) if last_low_bar is not None else None,
        "prev_pivot_high": safe_float(prev_high),
        "last_pivot_high": safe_float(last_high),
        "prev_pivot_low": safe_float(prev_low),
        "last_pivot_low": safe_float(last_low),
        "vwap_touch_count": int(touch_count),
        "first_touch_bar": int(touch_bars[0]),
        "vwap_pullback_bar": int(last_touch_bar),
        "vwap_pullback_time": str(df.index[last_touch_bar]),
        "vwap_at_pullback": safe_float(touch_vwap),
        "close_at_pullback": safe_float(touch_close),
        "close_to_vwap_pct": safe_float(touch_distance_pct),
        "mean_close_vwap_dist_early_pct": safe_float(early_mean),
        "mean_close_vwap_dist_late_pct": safe_float(late_mean),
        "close_converging_to_vwap": bool(convergence_ok),
        "current_close": safe_float(current_close),
        "current_vwap": safe_float(current_vwap),
        "current_close_vs_vwap_pct": safe_float(
            ((current_close - current_vwap) / current_vwap * 100.0)
            if np.isfinite(current_vwap) and current_vwap > 0
            else None
        ),
        "premarket_volume": safe_float(total_volume),
        "avg_15m_volume": safe_float(avg_bar_volume),
        "last_15m_volume": safe_float(vols[-1] if len(vols) else None),
        "rank_score": safe_float(score),
    }


# -----------------------------
# Scan
# -----------------------------


def fetch_bars_for_morning_sine(
    ticker: str,
    interval: str,
    lookback_days: int,
    data_source_mode: str,
    polygon_api_key: str,
    polygon_fallback_yfinance: bool,
    premarket_start: str,
    premarket_end: str,
    strict_today_session: bool,
) -> Tuple[pd.DataFrame, str]:
    """
    Return (ohlcv_index_utc_df, bars_provider).
    bars_provider explains which feed supplied the rows eventually used downstream.
    """
    mode = (data_source_mode or "auto").strip().lower()
    key = (polygon_api_key or "").strip()

    def _premarket_slice(d: pd.DataFrame) -> pd.DataFrame:
        return extract_latest_premarket_window(
            d,
            premarket_start=premarket_start,
            premarket_end=premarket_end,
            strict_today=strict_today_session,
        )

    if mode == "yfinance":
        df = download_bars(
            ticker,
            interval=interval,
            lookback_days=lookback_days,
            data_source="yfinance",
            polygon_api_key="",
        )
        return df, "yfinance"

    if mode == "polygon":
        if not key:
            raise ValueError("Polygon selected but no API key: set POLYGON_API_KEY or pass --polygon-api-key")
        df = download_bars_polygon(ticker, interval, lookback_days, key)
        sess = _premarket_slice(df)
        if (df.empty or sess.empty) and polygon_fallback_yfinance:
            df2 = download_bars(
                ticker,
                interval=interval,
                lookback_days=lookback_days,
                data_source="yfinance",
                polygon_api_key="",
            )
            return df2, "yfinance_fallback_polygon_no_usable_bars_or_premarket"
        return df, "polygon"

    # auto
    if key:
        df = download_bars_polygon(ticker, interval, lookback_days, key)
        sess = _premarket_slice(df)
        if df.empty or sess.empty:
            if polygon_fallback_yfinance:
                df2 = download_bars(
                    ticker,
                    interval=interval,
                    lookback_days=lookback_days,
                    data_source="yfinance",
                    polygon_api_key="",
                )
                return df2, "yfinance_fallback_polygon_no_usable_bars_or_premarket"
        return df, "polygon"

    df = download_bars(
        ticker,
        interval=interval,
        lookback_days=lookback_days,
        data_source="yfinance",
        polygon_api_key="",
    )
    return df, "yfinance"


def scan(
    tickers: List[str],
    interval: str,
    lookback_days: int,
    max_bars: int,
    pivot_strength: int,
    vwap_touch_tolerance_pct: float,
    min_vwap_touches: int,
    convergence_required: bool,
    convergence_late_vs_early_ratio: float,
    premarket_start: str,
    premarket_end: str,
    strict_today_session: bool,
    only_matches: bool,
    data_source_mode: str,
    polygon_api_key: str,
    polygon_fallback_yfinance: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for ticker in tickers:
        t = ticker.strip().upper()
        if not t:
            continue

        try:
            df, bars_provider = fetch_bars_for_morning_sine(
                ticker=t,
                interval=interval,
                lookback_days=lookback_days,
                data_source_mode=data_source_mode,
                polygon_api_key=polygon_api_key,
                polygon_fallback_yfinance=polygon_fallback_yfinance,
                premarket_start=premarket_start,
                premarket_end=premarket_end,
                strict_today_session=strict_today_session,
            )
            if df.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": interval,
                        "pattern_detected": False,
                        "bars_provider": bars_provider,
                        "error": "no_bars_downloaded",
                    })
                continue

            sess = extract_latest_premarket_window(
                df,
                premarket_start=premarket_start,
                premarket_end=premarket_end,
                strict_today=strict_today_session,
            )
            if sess.empty:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": interval,
                        "pattern_detected": False,
                        "bars_provider": bars_provider,
                        "error": "no_premarket_window_found",
                    })
                continue

            sess = sess.tail(int(max_bars)).copy()
            info = detect_morning_sine(
                df=sess,
                pivot_strength=pivot_strength,
                vwap_touch_tolerance_pct=vwap_touch_tolerance_pct,
                min_vwap_touches=min_vwap_touches,
                convergence_required=convergence_required,
                convergence_late_vs_early_ratio=convergence_late_vs_early_ratio,
            )

            if info is None:
                if not only_matches:
                    rows.append({
                        "ticker": t,
                        "tf": interval,
                        "pattern_detected": False,
                        "bars_provider": bars_provider,
                        "bars_available": int(len(sess)),
                        "session_date": str(sess.index[-1].date()) if len(sess) else None,
                    })
                continue

            rows.append({
                "ticker": t,
                "tf": interval,
                "bars_available": int(len(sess)),
                "bars_provider": bars_provider,
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

    p.add_argument("--interval", default="15m", choices=["15m", "30m"], help="Intraday bar size (yfinance or Polygon minute aggregates)")
    p.add_argument(
        "--lookback_days",
        type=int,
        default=2,
        help="Calendar days of history to request (small is enough for latest pre-market session)",
    )
    p.add_argument(
        "--data-source",
        dest="data_source",
        choices=["auto", "yfinance", "polygon"],
        default="auto",
        help="auto: Polygon when POLYGON_API_KEY or --polygon-api-key is set, otherwise yfinance",
    )
    p.add_argument(
        "--polygon-api-key",
        dest="polygon_api_key",
        default="",
        help="Optional; prefer environment variable POLYGON_API_KEY (safer than argv)",
    )
    p.add_argument(
        "--no-polygon-fallback-yfinance",
        dest="no_polygon_fallback_yfinance",
        action="store_true",
        help="When using Polygon (or auto with a key), do not refetch from yfinance if Polygon has no minute bars or no 4am–9am ET slice (lower tiers often omit intraday/extended)",
    )
    p.add_argument("--max_bars", type=int, default=40, help="Keep last N premarket bars after session filter")

    p.add_argument("--premarket_start", default="04:00", help="America/New_York (inclusive)")
    p.add_argument("--premarket_end", default="09:00", help="America/New_York (exclusive bar open time, same as extract filter)")
    p.add_argument("--pivot_strength", type=int, default=1, help="Pivot width for optional HH/HL reporting (does not filter matches)")
    p.add_argument("--vwap_touch_tolerance_pct", type=float, default=0.30, help="Allow close to be this %% away from VWAP if the candle does not directly touch it")
    p.add_argument("--min_vwap_touches", type=int, default=2, help="Minimum pre-market bars that touch/near VWAP")
    p.add_argument(
        "--no_convergence_required",
        action="store_true",
        help="Disable the default rule that mean |Close-VWAP|%% tightens from first half to second half of the window",
    )
    p.add_argument(
        "--convergence_late_vs_early_ratio",
        type=float,
        default=0.97,
        help="Match convergence when mean(late_dist) <= mean(early_dist) * this ratio (unless early is already within tolerance)",
    )
    p.add_argument(
        "--strict_today_session",
        action="store_true",
        help="Only use America/New_York calendar-today premarket; default uses latest day in the download that has premarket bars",
    )

    p.add_argument("--only_matches", action="store_true", help="Write only matching tickers")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    args = parse_args()

    polygon_key = (args.polygon_api_key or os.environ.get("POLYGON_API_KEY", "") or "").strip()
    if args.data_source == "polygon" and not polygon_key:
        print(
            "Polygon selected but no API key: set POLYGON_API_KEY or pass --polygon-api-key",
            file=sys.stderr,
        )
        return 2

    polygon_fallback_yfinance = not bool(getattr(args, "no_polygon_fallback_yfinance", False))

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
            "min_vwap_touches": int(args.min_vwap_touches),
            "convergence_required": not bool(args.no_convergence_required),
            "convergence_late_vs_early_ratio": float(args.convergence_late_vs_early_ratio),
            "strict_today_session": bool(args.strict_today_session),
            "only_matches": bool(args.only_matches),
            "data_source": args.data_source,
            "polygon_configured": bool(polygon_key),
            "polygon_yfinance_fallback_enabled": bool(polygon_fallback_yfinance),
            "lastUpdatedTs": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "total": 0,
            "matched": 0,
        },
        "data": [],
    }

    if not tickers:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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
        min_vwap_touches=args.min_vwap_touches,
        convergence_required=not args.no_convergence_required,
        convergence_late_vs_early_ratio=args.convergence_late_vs_early_ratio,
        premarket_start=args.premarket_start,
        premarket_end=args.premarket_end,
        strict_today_session=bool(args.strict_today_session),
        only_matches=bool(args.only_matches),
        data_source_mode=args.data_source,
        polygon_api_key=polygon_key,
        polygon_fallback_yfinance=polygon_fallback_yfinance,
    )

    payload["counts"]["total"] = len(tickers)
    payload["counts"]["matched"] = sum(1 for r in rows if r.get("pattern_detected"))
    payload["data"] = rows

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)

    print(f"Wrote {payload['counts']['matched']} matches to {args.out}")
    if rows:
        preview_cols = [
            "ticker",
            "tf",
            "bars_provider",
            "pattern_detected",
            "session_date",
            "hh_ok",
            "hl_ok",
            "vwap_touch_count",
            "close_converging_to_vwap",
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