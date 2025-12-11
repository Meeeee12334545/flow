#!/usr/bin/env python3
import io
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events


class AnomalyTracker:
    """Track anomaly codes for depth and velocity measurements."""

    def __init__(self, index: pd.Index):
        self._depth = {idx: set() for idx in index}
        self._velocity = {idx: set() for idx in index}

    def add(self, mask: pd.Series, code: str, field: str):
        if mask is None or code is None:
            return
        target = self._depth if field == "depth" else self._velocity
        target_index = pd.Index(target.keys())
        if isinstance(mask, pd.Series):
            mask_iter = mask.reindex(target_index).fillna(False)
        else:
            mask_iter = pd.Series(mask, index=target_index)
        for idx in mask_iter[mask_iter].index:
            target[idx].add(code)

    def extend(self, indices: pd.Index, code: str, field: str):
        target = self._depth if field == "depth" else self._velocity
        for idx in indices:
            if idx in target:
                target[idx].add(code)

    def series(self, field: str) -> pd.Series:
        target = self._depth if field == "depth" else self._velocity
        return pd.Series(
            [";".join(sorted(codes)) if codes else "" for codes in target.values()],
            index=pd.Index(target.keys()),
        )

    def merge_code(self, mask: pd.Series, code: str):
        self.add(mask, code, "depth")
        self.add(mask, code, "velocity")


def infer_sampling_seconds(timestamps: pd.Series) -> float:
    """Infer dominant sampling interval in seconds from a timestamp series."""
    diffs = timestamps.sort_values().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 60.0
    return float(diffs.median())


def seconds_to_pandas_rule(seconds: float) -> str:
    seconds = max(1.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))}S"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(round(minutes))}min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{int(round(hours))}H"
    days = hours / 24.0
    return f"{max(1, int(round(days)))}D"


# =========================
# Core processing functions
# =========================

def load_data_from_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(buffer)

    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: timestamp, depth, velocity")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Identify columns by name hints
    timestamp_col = None
    depth_col = None
    velocity_col = None

    for c in df.columns:
        if timestamp_col is None and ("time" in c or "date" in c):
            timestamp_col = c
        elif depth_col is None and "depth" in c:
            depth_col = c
        elif velocity_col is None and ("vel" in c or "speed" in c):
            velocity_col = c

    cols = list(df.columns)
    if timestamp_col is None:
        timestamp_col = cols[0]
    if depth_col is None:
        depth_col = cols[1]
    if velocity_col is None:
        velocity_col = cols[2]

    df = df[[timestamp_col, depth_col, velocity_col]]
    df.columns = ["timestamp", "depth", "velocity"]

    # Parse timestamps (dd/mm/yyyy hh:mm)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_rainfall_from_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    """Load rainfall time series (timestamp, mm) from CSV buffer."""
    df = pd.read_csv(buffer)
    if df.shape[1] < 2:
        raise ValueError("Rainfall CSV must have at least timestamp and rainfall columns")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    timestamp_col = None
    rain_col = None
    for c in df.columns:
        if timestamp_col is None and ("time" in c or "date" in c):
            timestamp_col = c
        elif rain_col is None and ("rain" in c or "mm" in c or "precip" in c):
            rain_col = c

    if timestamp_col is None:
        timestamp_col = df.columns[0]
    if rain_col is None:
        rain_col = df.columns[1]

    df = df[[timestamp_col, rain_col]].rename(columns={timestamp_col: "timestamp", rain_col: "rainfall_mm"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def combine_rainfall_series(rainfall_frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Combine multiple rainfall gauges by averaging coincident timestamps."""
    valid_frames = [f for f in rainfall_frames if f is not None and not f.empty]
    if not valid_frames:
        return None

    combined = pd.concat(valid_frames, axis=0, ignore_index=True)
    combined = combined.dropna(subset=["timestamp"]).copy()
    agg = (
        combined.groupby("timestamp")
        ["rainfall_mm"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return agg


def derive_weather_context(
    flow_df: pd.DataFrame,
    rainfall_df: Optional[pd.DataFrame],
    rolling_window_hours: float = 6.0,
    wet_threshold_mm: float = 0.2,
) -> Optional[Dict[str, pd.Series]]:
    """Derive wet/dry classification and rainfall summaries aligned to flow data."""
    if rainfall_df is None or rainfall_df.empty:
        return None

    timestamps = flow_df["timestamp"].sort_values().reset_index(drop=False)
    freq_seconds = infer_sampling_seconds(flow_df["timestamp"])
    freq_rule = seconds_to_pandas_rule(freq_seconds)

    rain = rainfall_df.set_index("timestamp").sort_index()
    rain_resampled = rain["rainfall_mm"].resample(freq_rule).sum().fillna(0.0)

    flow_times = timestamps["timestamp"]
    rain_aligned = rain_resampled.reindex(flow_times, method="ffill").fillna(0.0)

    window_steps = max(1, int(round((rolling_window_hours * 3600.0) / max(freq_seconds, 1.0))))
    rain_window = rain_aligned.rolling(window=window_steps, min_periods=1).sum()
    wet_mask = rain_window > wet_threshold_mm

    is_weekend = flow_times.dt.weekday >= 5
    condition_label = []
    for dry, weekend in zip(~wet_mask, is_weekend):
        if weekend and dry:
            condition_label.append("weekend_dry")
        elif weekend and not dry:
            condition_label.append("weekend_wet")
        elif (not weekend) and dry:
            condition_label.append("weekday_dry")
        else:
            condition_label.append("weekday_wet")

    context = {
        "rainfall_mm": pd.Series(rain_aligned.values, index=flow_df.index),
        "rainfall_window_mm": pd.Series(rain_window.values, index=flow_df.index),
        "is_wet_weather": pd.Series(wet_mask.values, index=flow_df.index),
        "is_weekend": pd.Series(is_weekend.values, index=flow_df.index),
        "condition_label": pd.Series(condition_label, index=flow_df.index),
    }
    return context


def detect_flatlines(series: pd.Series, min_run: int = 10) -> pd.Series:
    """Flag long runs of identical values (sensor stuck)."""
    s = series.copy()
    groups = (s != s.shift()).cumsum()
    group_sizes = groups.value_counts()
    bad_groups = group_sizes[group_sizes >= min_run].index
    flat_mask = groups.isin(bad_groups)
    flat_mask = flat_mask & s.notna()
    return flat_mask


def robust_spike_mask(series: pd.Series, window: int = 7, k: float = 6.0) -> pd.Series:
    """Spike detection using rolling median & MAD (Hampel-style)."""
    s = series.astype(float)
    med = s.rolling(window=window, center=True, min_periods=3).median()
    diff = (s - med).abs()
    mad = diff.rolling(window=window, center=True, min_periods=3).median()
    mad_repl = mad.replace(0, np.nan)
    z = diff / mad_repl
    spike_mask = z > k
    spike_mask = spike_mask.fillna(False)
    return spike_mask


def detect_gradient_anomalies(
    timestamps: pd.Series,
    series: pd.Series,
    k: float = 8.0,
    window: int = 5,
) -> pd.Series:
    """Flag points where rate of change deviates strongly from typical behaviour."""
    if series.isna().all() or len(series) < 5:
        return pd.Series(False, index=series.index)

    time_diff = timestamps.diff().dt.total_seconds().replace(0, np.nan)
    gradient = series.diff() / time_diff
    gradient = gradient.rolling(window=window, center=True, min_periods=2).median()

    grad_med = gradient.median()
    grad_mad = (gradient - grad_med).abs().median()
    if grad_mad and grad_mad > 0:
        z = (gradient - grad_med).abs() / (grad_mad / 0.6745)
    else:
        std = gradient.std()
        if std and std > 0:
            z = (gradient - grad_med).abs() / std
        else:
            return pd.Series(False, index=series.index)

    mask = z > k
    mask = mask.reindex(series.index, fill_value=False)
    return mask.fillna(False)


def compute_froude_number(depth: pd.Series, velocity: pd.Series) -> pd.Series:
    """Compute Froude number for each observation with safety for low depth."""
    g = 9.80665
    depth_eff = depth.clip(lower=1e-4)
    velo_eff = velocity.clip(lower=0.0)
    froude = velo_eff / np.sqrt(g * depth_eff)
    froude[depth.isna() | velocity.isna()] = np.nan
    return froude


def detect_froude_anomalies(
    depth: pd.Series,
    velocity: pd.Series,
    froude_max: float,
    froude_min: float = 0.0,
) -> pd.Series:
    """Flag velocities that imply implausible Froude numbers."""
    froude = compute_froude_number(depth, velocity)
    mask_high = froude > froude_max
    mask_low = froude < froude_min
    mask = (mask_high | mask_low).fillna(False)
    return mask


def _compute_pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute monotone cubic slopes using the Fritsch-Carlson method."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    slopes = np.zeros_like(y)

    def _edge_slope(delta0, delta1, h0, h1):
        m = ((2 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if m * delta0 <= 0:
            return 0.0
        if delta0 * delta1 < 0 and abs(m) > 2 * abs(delta0):
            return 2 * delta0
        return m

    if n == 2:
        slopes[:] = delta[0]
        return slopes

    slopes[0] = _edge_slope(delta[0], delta[1], h[0], h[1])
    slopes[-1] = _edge_slope(delta[-1], delta[-2], h[-1], h[-2])

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            slopes[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    return slopes


def _pchip_eval(
    x: np.ndarray,
    y: np.ndarray,
    slopes: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Evaluate monotone cubic Hermite spline at new x values."""
    x_new = np.asarray(x_new, dtype=float)
    result = np.empty_like(x_new)

    for idx, xi in enumerate(x_new):
        if xi <= x[0]:
            result[idx] = y[0] + (xi - x[0]) * slopes[0]
            continue
        if xi >= x[-1]:
            result[idx] = y[-1] + (xi - x[-1]) * slopes[-1]
            continue

        j = np.searchsorted(x, xi) - 1
        j = np.clip(j, 0, len(x) - 2)
        h = x[j + 1] - x[j]
        t = (xi - x[j]) / h

        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2

        result[idx] = (
            h00 * y[j]
            + h10 * h * slopes[j]
            + h01 * y[j + 1]
            + h11 * h * slopes[j + 1]
        )

    return result


def make_monotone_cubic_interpolator(
    x: np.ndarray,
    y: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return callable monotone cubic interpolator for (x, y) data."""
    order = np.argsort(x)
    x_ord = np.asarray(x, dtype=float)[order]
    y_ord = np.asarray(y, dtype=float)[order]

    unique_mask = np.concatenate(([True], np.diff(x_ord) > 0))
    x_unique = x_ord[unique_mask]
    y_unique = y_ord[unique_mask]

    if len(x_unique) < 2:
        def _flat(_x_new):
            return np.full_like(np.asarray(_x_new, dtype=float), y_unique[0])

        return _flat

    slopes = _compute_pchip_slopes(x_unique, y_unique)

    def _interp(x_new: np.ndarray) -> np.ndarray:
        return _pchip_eval(x_unique, y_unique, slopes, np.asarray(x_new, dtype=float))

    return _interp


def monotone_time_fill(series: pd.Series) -> pd.Series:
    """Fill missing values using monotone cubic interpolation over time."""
    if series.isna().sum() == 0:
        return series

    mask = series.notna()
    if mask.sum() < 4:
        return series

    x = series.index.view(np.int64) / 1e9
    x_known = x[mask]
    y_known = series[mask].astype(float).values

    unique, idx = np.unique(x_known, return_index=True)
    x_known = x_known[idx]
    y_known = y_known[idx]

    if len(x_known) < 4:
        return series

    interpolator = make_monotone_cubic_interpolator(x_known, y_known)

    missing_idx = np.where(~mask)[0]
    if len(missing_idx) == 0:
        return series

    x_missing = x[~mask]
    y_missing = interpolator(x_missing)

    filled = series.copy()
    filled.iloc[missing_idx] = y_missing
    return filled


def long_true_runs(mask: pd.Series, min_run: int) -> pd.Series:
    """
    Given a boolean Series, return True where there are contiguous runs
    of True values with length >= min_run.
    """
    mask = mask.fillna(False)
    groups = (mask != mask.shift()).cumsum()
    group_sizes = groups[mask].value_counts()
    bad_groups = group_sizes[group_sizes >= min_run].index
    return mask & groups.isin(bad_groups)


def build_rating_curve(
    depth: pd.Series,
    velocity: pd.Series,
    good_mask: pd.Series,
    n_bins: int = 30,
):
    """
    Build a non-linear rating curve v = f(depth) using bin-median approach.
    Returns:
        depth_knots, vel_knots, v_expected(d), d_expected(v)
    """
    mask = good_mask & depth.notna() & velocity.notna() & (depth > 0)
    d_good = depth[mask]
    v_good = velocity[mask]

    if len(d_good) < 10:
        return None, None, None, None

    d_min, d_max = float(d_good.min()), float(d_good.max())
    if d_max <= d_min:
        return None, None, None, None

    bins = np.linspace(d_min, d_max, n_bins + 1)
    bin_ids = np.digitize(d_good, bins) - 1

    depth_knots = []
    vel_knots = []

    for b in range(n_bins):
        in_bin = bin_ids == b
        if not np.any(in_bin):
            continue
        d_bin = d_good[in_bin]
        v_bin = v_good[in_bin]
        depth_knots.append(float(d_bin.median()))
        vel_knots.append(float(v_bin.median()))

    if len(depth_knots) < 3:
        return None, None, None, None

    depth_knots = np.array(depth_knots)
    vel_knots = np.array(vel_knots)

    order = np.argsort(depth_knots)
    depth_knots = depth_knots[order]
    vel_knots = vel_knots[order]

    # Enforce non-negative, non-decreasing velocity with depth and smooth
    vel_knots = np.maximum(vel_knots, 0.0)
    vel_knots = np.maximum.accumulate(vel_knots)

    v_interp = make_monotone_cubic_interpolator(depth_knots, vel_knots)
    d_interp = make_monotone_cubic_interpolator(vel_knots, depth_knots)

    def v_expected(d):
        d_arr = np.asarray(d, dtype=float)
        v_vals = v_interp(d_arr)
        return np.clip(v_vals, a_min=0.0, a_max=None)

    def d_expected(v):
        v_arr = np.asarray(v, dtype=float)
        d_vals = d_interp(v_arr)
        return np.clip(d_vals, a_min=depth_knots[0], a_max=depth_knots[-1])

    return depth_knots, vel_knots, v_expected, d_expected


def detect_diurnal_baseline_anomalies(
    df: pd.DataFrame,
    depth_flag: pd.Series,
    velocity_flag: pd.Series,
    diurnal_freq: str = "15min",
    diurnal_min_samples: int = 5,
    k: float = 5.0,
    min_run_bins: int = 6,
):
    """
    Detect abnormal baseline shifts vs typical diurnal pattern.

    Implementation note:
    - We KEEP the original row index (0..N-1) everywhere to avoid index
      alignment problems.
    - Time-of-day bins are handled via a column, not the index.
    """
    if df.empty:
        return depth_flag, velocity_flag

    data = df[["timestamp", "depth", "velocity"]].copy()
    data["depth_flag"] = depth_flag.values
    data["velocity_flag"] = velocity_flag.values

    data = data.sort_values("timestamp")
    data["tod_bin"] = data["timestamp"].dt.floor(diurnal_freq).dt.time

    # ---- Depth baseline anomalies ----
    if "depth" in data.columns:
        mask_good_d = data["depth_flag"] & data["depth"].notna()
        if mask_good_d.any():
            grp_d = data.loc[mask_good_d].groupby("tod_bin")["depth"]
            depth_stats = grp_d.agg(
                median="median",
                mad=lambda x: (x - x.median()).abs().median(),
                count="count",
            )
            depth_stats = depth_stats[depth_stats["count"] >= diurnal_min_samples]

            if not depth_stats.empty:
                med_map = depth_stats["median"]
                scale_map = (depth_stats["mad"] / 0.6745).replace(0, np.nan)

                med_series = data["tod_bin"].map(med_map)
                scale_series = data["tod_bin"].map(scale_map)

                z = (data["depth"] - med_series).abs() / scale_series
                candidate = (
                    data["depth_flag"]
                    & med_series.notna()
                    & scale_series.notna()
                    & z.gt(k)
                )

                baseline_mask = long_true_runs(candidate, min_run_bins)
                # Depth is generally trusted; we *do not* flip depth_flag here.
                depth_flag = depth_flag  # explicit, no change

    # ---- Velocity baseline anomalies ----
    if "velocity" in data.columns:
        mask_good_v = data["velocity_flag"] & data["velocity"].notna()
        if mask_good_v.any():
            grp_v = data.loc[mask_good_v].groupby("tod_bin")["velocity"]
            vel_stats = grp_v.agg(
                median="median",
                mad=lambda x: (x - x.median()).abs().median(),
                count="count",
            )
            vel_stats = vel_stats[vel_stats["count"] >= diurnal_min_samples]

            if not vel_stats.empty:
                med_map_v = vel_stats["median"]
                scale_map_v = (vel_stats["mad"] / 0.6745).replace(0, np.nan)

                med_series_v = data["tod_bin"].map(med_map_v)
                scale_series_v = data["tod_bin"].map(scale_map_v)

                z_v = (data["velocity"] - med_series_v).abs() / scale_series_v
                candidate_v = (
                    data["velocity_flag"]
                    & med_series_v.notna()
                    & scale_series_v.notna()
                    & z_v.gt(k)
                )

                baseline_mask_v = long_true_runs(candidate_v, min_run_bins)
                velocity_flag = velocity_flag & (~baseline_mask_v)

    return depth_flag, velocity_flag


def apply_quality_checks(
    df: pd.DataFrame,
    pipe_diam_m: Optional[float] = None,
    flatline_run: int = 10,
    spike_window: int = 7,
    spike_k: float = 6.0,
    diurnal_freq_baseline: str = "15min",
    diurnal_min_samples_baseline: int = 5,
    depth_min_meas: Optional[float] = None,
    depth_max_meas: Optional[float] = None,
    vel_min_meas: Optional[float] = None,
    vel_max_meas: Optional[float] = None,
    froude_max: float = 1.5,
    gradient_k: float = 8.0,
    slope: Optional[float] = None,
    roughness_n: Optional[float] = None,
    hydraulic_tolerance: float = 0.35,
    full_pipe_capacity_lps: Optional[float] = None,
    capacity_tolerance: float = 0.1,
    condition_labels: Optional[pd.Series] = None,
    rainfall_context: Optional[Dict[str, pd.Series]] = None,
    manual_depth_mask: Optional[pd.Series] = None,
    manual_velocity_mask: Optional[pd.Series] = None,
):
    """
    Add depth_flag and velocity_flag columns (True = good).
    Includes:
      - basic plausibility checks (distribution + pipe geometry)
      - measured min/max limits (if provided)
      - flatline detection
      - spike detection (Hampel/MAD)
      - rating-curve consistency (hydraulic), **biasing blame to velocity**
      - diurnal baseline anomaly detection
    """
    depth = df["depth"]
    vel = df["velocity"]
    timestamps = df["timestamp"]

    depth_flag = pd.Series(True, index=df.index)
    vel_flag = pd.Series(True, index=df.index)
    tracker = AnomalyTracker(df.index)

    # NaNs
    depth_missing = depth.isna()
    vel_missing = vel.isna()
    tracker.add(depth_missing, "missing_value", "depth")
    tracker.add(vel_missing, "missing_value", "velocity")
    depth_flag &= ~depth_missing
    vel_flag &= ~vel_missing

    # Depth plausibility
    d_pos = depth[depth > 0]
    if len(d_pos) > 0:
        d_p99 = d_pos.quantile(0.99)
        depth_max_limit = d_p99 * 1.5
        depth_max_limit = max(depth_max_limit, d_p99 + 0.5)
    else:
        depth_max_limit = np.inf

    depth_negative = depth < 0
    tracker.add(depth_negative, "plausibility_limit", "depth")
    depth_flag &= ~depth_negative
    if pipe_diam_m is not None and pipe_diam_m > 0:
        over_full_depth = depth > pipe_diam_m * 1.05
        tracker.add(over_full_depth, "plausibility_limit", "depth")
        depth_flag &= ~over_full_depth
        depth_max_limit = min(depth_max_limit, pipe_diam_m * 1.05)
    else:
        exceed_limit = depth > depth_max_limit
        tracker.add(exceed_limit, "plausibility_limit", "depth")
        depth_flag &= ~exceed_limit

    # Measured depth limits
    if depth_min_meas is not None:
        below_measured = depth < depth_min_meas
        tracker.add(below_measured, "measured_min_limit", "depth")
        depth_flag &= ~below_measured
    if depth_max_meas is not None:
        above_measured = depth > depth_max_meas
        tracker.add(above_measured, "measured_max_limit", "depth")
        depth_flag &= ~above_measured

    # Velocity plausibility
    v_abs = vel.abs()[vel.notna()]
    if len(v_abs) > 0:
        v_p99 = v_abs.quantile(0.99)
        vel_max_limit = v_p99 * 1.5
        vel_max_limit = max(vel_max_limit, v_p99 + 0.5)
    else:
        vel_max_limit = np.inf

    # No negative velocities allowed in QC
    non_positive_vel = vel <= 0.0
    tracker.add(non_positive_vel, "negative_or_zero_velocity", "velocity")
    vel_flag &= ~non_positive_vel

    over_vel_limit = vel.abs() > vel_max_limit
    tracker.add(over_vel_limit, "plausibility_limit", "velocity")
    vel_flag &= ~over_vel_limit

    # Measured velocity limits
    if vel_min_meas is not None:
        below_vel_min = vel < vel_min_meas
        tracker.add(below_vel_min, "measured_min_limit", "velocity")
        vel_flag &= ~below_vel_min
    if vel_max_meas is not None:
        above_vel_max = vel > vel_max_meas
        tracker.add(above_vel_max, "measured_max_limit", "velocity")
        vel_flag &= ~above_vel_max

    # Flatlines
    depth_flat = detect_flatlines(depth, min_run=flatline_run)
    vel_flat = detect_flatlines(vel, min_run=flatline_run)
    tracker.add(depth_flat, "flatline_sensor_stuck", "depth")
    tracker.add(vel_flat, "flatline_sensor_stuck", "velocity")
    depth_flag &= ~depth_flat
    vel_flag &= ~vel_flat

    # Spikes
    depth_spike = robust_spike_mask(depth, window=spike_window, k=spike_k)
    vel_spike = robust_spike_mask(vel, window=spike_window, k=spike_k)
    tracker.add(depth_spike, "spike_outlier", "depth")
    tracker.add(vel_spike, "spike_outlier", "velocity")
    depth_flag &= ~depth_spike
    vel_flag &= ~vel_spike

    # Rapid gradient anomalies (sudden jumps inconsistent with sampling)
    depth_grad = detect_gradient_anomalies(timestamps, depth, k=gradient_k)
    vel_grad = detect_gradient_anomalies(timestamps, vel, k=gradient_k)
    tracker.add(depth_grad, "gradient_outlier", "depth")
    tracker.add(vel_grad, "gradient_outlier", "velocity")
    depth_flag &= ~depth_grad
    vel_flag &= ~vel_grad

    # Froude-based hydraulic plausibility
    if froude_max is not None and froude_max > 0:
        froude_bad = detect_froude_anomalies(depth, vel, froude_max=froude_max)
        tracker.add(froude_bad, "froude_outlier", "velocity")
        vel_flag &= ~froude_bad

    # Manning-based hydraulic expectation
    hydraulic_velocity = manning_velocity(depth, pipe_diam_m, slope, roughness_n)
    df["velocity_manning"] = hydraulic_velocity
    if hydraulic_velocity.notna().any() and hydraulic_tolerance is not None:
        tolerance = max(0.05, hydraulic_tolerance)
        hydro_mask = vel.notna() & hydraulic_velocity.notna()
        upper = hydraulic_velocity * (1 + tolerance)
        lower = hydraulic_velocity * max(0.0, 1 - tolerance)
        lower = lower.clip(lower=0.0)
        hydro_high = hydro_mask & (vel > upper)
        hydro_low = hydro_mask & (vel < lower)
        hydro_bad = hydro_high | hydro_low
        tracker.add(hydro_bad, "hydraulic_outlier", "velocity")
        vel_flag &= ~hydro_bad

    # Capacity check
    capacity_lps = full_pipe_capacity_lps
    if capacity_lps is None:
        capacity_lps = theoretical_full_pipe_capacity_lps(pipe_diam_m, slope, roughness_n)

    if capacity_lps is not None and pipe_diam_m is not None and pipe_diam_m > 0:
        area = circular_area_from_depth(depth, pipe_diam_m)
        flow_lps_est = area * vel * 1000.0
        df["flow_est_lps"] = flow_lps_est
        overload = flow_lps_est > capacity_lps * (1 + max(0.0, capacity_tolerance))
        tracker.add(overload, "capacity_exceeded", "velocity")
        vel_flag &= ~overload

    manual_depth_series: Optional[pd.Series] = None
    manual_velocity_series: Optional[pd.Series] = None
    if manual_depth_mask is not None:
        manual_depth_series = manual_depth_mask.reindex(df.index)
        manual_depth_series = manual_depth_series.where(~manual_depth_series.isna(), depth_flag)
        manual_depth_series = manual_depth_series.astype(bool)
    if manual_velocity_mask is not None:
        manual_velocity_series = manual_velocity_mask.reindex(df.index)
        manual_velocity_series = manual_velocity_series.where(~manual_velocity_series.isna(), vel_flag)
        manual_velocity_series = manual_velocity_series.astype(bool)

    depth_flag_auto = depth_flag.copy()
    vel_flag_auto = vel_flag.copy()

    if manual_depth_series is not None:
        override_mask = manual_depth_series != depth_flag
        tracker.add(override_mask & manual_depth_series, "manual_override_good", "depth")
        tracker.add(override_mask & (~manual_depth_series), "manual_override_bad", "depth")
        depth_flag = manual_depth_series

    if manual_velocity_series is not None:
        override_mask_v = manual_velocity_series != vel_flag
        tracker.add(override_mask_v & manual_velocity_series, "manual_override_good", "velocity")
        tracker.add(override_mask_v & (~manual_velocity_series), "manual_override_bad", "velocity")
        vel_flag = manual_velocity_series

    # Rating curve consistency: depth is generally trusted, so we only
    # use the curve to mark velocity outliers here.
    combined_good = depth_flag & vel_flag
    d_knots, v_knots, v_expected, d_expected = build_rating_curve(
        depth, vel, combined_good
    )

    cluster_curves: Dict[str, Dict[str, Callable]] = {}
    if condition_labels is not None:
        unique_labels = [lab for lab in condition_labels.dropna().unique()]
        for label in unique_labels:
            mask_cluster = combined_good & condition_labels.eq(label)
            if mask_cluster.sum() < 10:
                continue
            d_k, v_k, v_fn, d_fn = build_rating_curve(depth, vel, mask_cluster)
            if v_fn is not None and d_fn is not None:
                cluster_curves[str(label)] = {
                    "v": v_fn,
                    "d": d_fn,
                }

    if v_expected is not None:
        # Velocity vs expected from depth
        v_pred = pd.Series(v_expected(depth), index=df.index)
        if cluster_curves and condition_labels is not None:
            for label, funcs in cluster_curves.items():
                mask_label = condition_labels.astype(str) == label
                if mask_label.any():
                    v_pred.loc[mask_label] = funcs["v"](depth[mask_label])
        v_resid = vel - v_pred
        mad = v_resid.abs().median()
        scale = mad / 0.6745 if mad > 0 else v_resid.std()
        if scale and scale > 0:
            z_v = (v_resid.abs() / scale)
            curve_outliers_v = z_v > 4.0
            tracker.add(curve_outliers_v, "rating_curve_mismatch", "velocity")
            vel_flag &= ~curve_outliers_v

        # Depth consistency check where velocity is reliable
        if d_expected is not None:
            d_pred = pd.Series(d_expected(np.clip(vel, a_min=0.0, a_max=None)), index=df.index)
            if cluster_curves and condition_labels is not None:
                for label, funcs in cluster_curves.items():
                    mask_label = condition_labels.astype(str) == label
                    if mask_label.any():
                        d_pred.loc[mask_label] = funcs["d"](
                            np.clip(vel[mask_label], a_min=0.0, a_max=None)
                        )
            d_resid = depth - d_pred
            d_resid_good = d_resid[combined_good]
            mad_d = d_resid_good.abs().median()
            scale_d = mad_d / 0.6745 if mad_d > 0 else d_resid_good.std()
            if scale_d and scale_d > 0:
                z_d = (d_resid.abs() / scale_d)
                curve_outliers_d = z_d > 4.5
                tracker.add(curve_outliers_d, "rating_curve_mismatch", "depth")
                depth_flag &= ~curve_outliers_d

    # Diurnal baseline anomalies
    depth_flag_pre = depth_flag.copy()
    vel_flag_pre = vel_flag.copy()
    depth_flag, vel_flag = detect_diurnal_baseline_anomalies(
        df,
        depth_flag,
        vel_flag,
        diurnal_freq=diurnal_freq_baseline,
        diurnal_min_samples=diurnal_min_samples_baseline,
        k=5.0,
        min_run_bins=6,
    )
    tracker.add(depth_flag_pre & (~depth_flag), "diurnal_anomaly", "depth")
    tracker.add(vel_flag_pre & (~vel_flag), "diurnal_anomaly", "velocity")

    df["depth_flag"] = depth_flag
    df["velocity_flag"] = vel_flag
    df["depth_flag_auto"] = depth_flag_auto
    df["velocity_flag_auto"] = vel_flag_auto
    df["depth_anomaly_codes"] = tracker.series("depth")
    df["velocity_anomaly_codes"] = tracker.series("velocity")
    return d_knots, v_knots, v_expected, d_expected, tracker


def compute_diurnal_profiles(
    df: pd.DataFrame,
    use_depth_flag: bool = True,
    use_velocity_flag: bool = True,
    diurnal_freq: str = "15min",
    diurnal_min_samples: int = 5,
):
    """
    Build diurnal (time-of-day) median profiles for depth and velocity
    using only "good" data on other days.
    """
    df = df.copy()
    df = df.set_index("timestamp")

    tod_index = df.index.floor(diurnal_freq).time
    df["tod_bin"] = tod_index

    depth_profile = None
    vel_profile = None

    if use_depth_flag and "depth_flag" in df.columns:
        mask_d = df["depth_flag"] & df["depth"].notna()
        if mask_d.any():
            grp_d = df.loc[mask_d].groupby("tod_bin")["depth"]
            depth_profile = grp_d.agg(median="median", count="count")

    if use_velocity_flag and "velocity_flag" in df.columns:
        mask_v = df["velocity_flag"] & df["velocity"].notna()
        if mask_v.any():
            grp_v = df.loc[mask_v].groupby("tod_bin")["velocity"]
            vel_profile = grp_v.agg(median="median", count="count")

    if depth_profile is not None:
        depth_profile = depth_profile[depth_profile["count"] >= diurnal_min_samples]
    if vel_profile is not None:
        vel_profile = vel_profile[vel_profile["count"] >= diurnal_min_samples]

    return depth_profile, vel_profile


def apply_diurnal_infill(
    df: pd.DataFrame,
    depth_profile: Optional[pd.DataFrame],
    vel_profile: Optional[pd.DataFrame],
    diurnal_freq: str = "15min",
    anomaly_tracker: Optional[AnomalyTracker] = None,
):
    """
    Use diurnal profiles to fill remaining NaNs in depth_clean / velocity_clean
    based on time-of-day medians from other days.
    """
    if depth_profile is None and vel_profile is None:
        return df

    df = df.copy()
    df = df.set_index("timestamp")

    tod_index = df.index.floor(diurnal_freq).time
    df["tod_bin"] = tod_index

    if depth_profile is not None:
        missing_depth = df["depth_clean"].isna()
        if missing_depth.any():
            depth_median_map = depth_profile["median"]
            diurnal_depth_vals = df["tod_bin"].map(depth_median_map)
            use_diurnal_depth = missing_depth & diurnal_depth_vals.notna()
            df.loc[use_diurnal_depth, "depth_clean"] = diurnal_depth_vals[use_diurnal_depth]
            df.loc[use_diurnal_depth, "depth_source"] = "diurnal_profile"
            if anomaly_tracker is not None:
                anomaly_tracker.add(use_diurnal_depth, "diurnal_infill", "depth")

    if vel_profile is not None:
        missing_vel = df["velocity_clean"].isna()
        if missing_vel.any():
            vel_median_map = vel_profile["median"]
            diurnal_vel_vals = df["tod_bin"].map(vel_median_map)
            use_diurnal_vel = missing_vel & diurnal_vel_vals.notna()
            df.loc[use_diurnal_vel, "velocity_clean"] = diurnal_vel_vals[use_diurnal_vel]
            df.loc[use_diurnal_vel, "velocity_source"] = "diurnal_profile"
            if anomaly_tracker is not None:
                anomaly_tracker.add(use_diurnal_vel, "diurnal_infill", "velocity")

    df = df.drop(columns=["tod_bin"])
    df = df.reset_index()
    return df


def interpolate_signals(
    df: pd.DataFrame,
    v_expected,
    d_expected,
    anomaly_tracker: Optional[AnomalyTracker] = None,
    diurnal_freq: str = "15min",
    diurnal_min_samples: int = 5,
    depth_min_meas: Optional[float] = None,
    depth_max_meas: Optional[float] = None,
    vel_min_meas: Optional[float] = None,
    vel_max_meas: Optional[float] = None,
) -> pd.DataFrame:
    """
        Build depth_clean / velocity_clean using, in order:
            1) rating curve infill (physics-based, **depth→velocity especially**)
            2) diurnal (time-of-day) median profiles across good days
            3) monotone Hermite interpolation to conserve hydraulic shape
            4) time interpolation as a fallback

    Measured min/max depth & velocity (if provided) are enforced at the end,
    and a global minimum of 0.01 m/s is applied to velocity_clean.
    """
    df = df.copy()
    tracker = anomaly_tracker or AnomalyTracker(df.index)

    depth_profile, vel_profile = compute_diurnal_profiles(
        df,
        use_depth_flag=True,
        use_velocity_flag=True,
        diurnal_freq=diurnal_freq,
        diurnal_min_samples=diurnal_min_samples,
    )

    df = df.set_index("timestamp")

    df["depth_clean"] = df["depth"]
    df["velocity_clean"] = df["velocity"]
    df["depth_source"] = np.where(df["depth_flag"], "raw", "bad")
    df["velocity_source"] = np.where(df["velocity_flag"], "raw", "bad")

    depth_bad = ~df["depth_flag"]
    vel_bad = ~df["velocity_flag"]

    # 1) Rating-curve-based infill
    if v_expected is not None and d_expected is not None:
        # Preferred path: good depth, bad velocity → infer velocity from depth.
        mask_rating_vel = vel_bad & (~depth_bad) & df["depth_clean"].notna()
        if mask_rating_vel.any():
            df.loc[mask_rating_vel, "velocity_clean"] = v_expected(
                df.loc[mask_rating_vel, "depth_clean"]
            )
            df.loc[mask_rating_vel, "velocity_source"] = "rating_from_depth"
            tracker.add(mask_rating_vel, "rating_curve_infill", "velocity")

        # Secondary: good velocity, bad depth (less common, but allowed).
        mask_rating_depth = depth_bad & (~vel_bad) & df["velocity_clean"].notna()
        if mask_rating_depth.any():
            df.loc[mask_rating_depth, "depth_clean"] = d_expected(
                df.loc[mask_rating_depth, "velocity_clean"]
            )
            df.loc[mask_rating_depth, "depth_source"] = "rating_from_velocity"
            tracker.add(mask_rating_depth, "rating_curve_infill", "depth")

    df = df.reset_index()

    # 2) Diurnal profile infill
    df = apply_diurnal_infill(
        df,
        depth_profile=depth_profile,
        vel_profile=vel_profile,
        diurnal_freq=diurnal_freq,
        anomaly_tracker=tracker,
    )

    # 3) Hydraulic shape-preserving interpolation (monotone cubic)
    df = df.set_index("timestamp")

    depth_before = df["depth_clean"].copy()
    depth_monotone = monotone_time_fill(df["depth_clean"])
    depth_monotone_mask = depth_before.isna() & depth_monotone.notna()
    df.loc[depth_monotone_mask, "depth_clean"] = depth_monotone[depth_monotone_mask]
    df.loc[
        depth_monotone_mask & (df["depth_source"] == "bad"),
        "depth_source",
    ] = "monotone_interp"
    tracker.add(depth_monotone_mask, "shape_preserving_infill", "depth")

    vel_before = df["velocity_clean"].copy()
    vel_monotone = monotone_time_fill(df["velocity_clean"])
    vel_monotone_mask = vel_before.isna() & vel_monotone.notna()
    df.loc[vel_monotone_mask, "velocity_clean"] = vel_monotone[vel_monotone_mask]
    df.loc[
        vel_monotone_mask & (df["velocity_source"] == "bad"),
        "velocity_source",
    ] = "monotone_interp"
    tracker.add(vel_monotone_mask, "shape_preserving_infill", "velocity")

    # 4) Time interpolation fallback
    depth_before_time = df["depth_clean"].copy()
    df["depth_clean"] = df["depth_clean"].interpolate(method="time", limit_direction="both")
    filled_depth_time = depth_before_time.isna() & df["depth_clean"].notna()
    df.loc[
        filled_depth_time & (df["depth_source"] == "bad"),
        "depth_source",
    ] = "time_interp"
    tracker.add(filled_depth_time, "time_interpolation", "depth")

    vel_before_time = df["velocity_clean"].copy()
    df["velocity_clean"] = df["velocity_clean"].interpolate(method="time", limit_direction="both")
    filled_vel_time = vel_before_time.isna() & df["velocity_clean"].notna()
    df.loc[
        filled_vel_time & (df["velocity_source"] == "bad"),
        "velocity_source",
    ] = "time_interp"
    tracker.add(filled_vel_time, "time_interpolation", "velocity")

    df = df.reset_index()

    # 5) Enforce measured & global bounds on final cleaned series
    # Depth
    df["depth_clean"] = df["depth_clean"].clip(
        lower=depth_min_meas,
        upper=depth_max_meas,
    )

    # Velocity: never zero or negative
    eff_vel_min = 0.01  # global min
    if vel_min_meas is not None:
        eff_vel_min = max(eff_vel_min, vel_min_meas)
    df["velocity_clean"] = df["velocity_clean"].clip(
        lower=eff_vel_min,
        upper=vel_max_meas,
    )

    depth_codes = tracker.series("depth")
    velocity_codes = tracker.series("velocity")
    df["depth_anomaly_codes"] = depth_codes.values
    df["velocity_anomaly_codes"] = velocity_codes.values

    return df


def circular_area_from_depth(depth: pd.Series, pipe_diam_m: float) -> pd.Series:
    """Wetted area of partially full circular pipe."""
    R = pipe_diam_m / 2.0
    if R <= 0:
        return pd.Series(np.nan, index=depth.index)

    h = depth.clip(lower=0.0, upper=2 * R)
    x = (R - h) / R
    x = x.clip(lower=-1.0, upper=1.0)
    theta = 2.0 * np.arccos(x)
    area = (R ** 2 / 2.0) * (theta - np.sin(theta))
    return pd.Series(area, index=depth.index)


def circular_wetted_perimeter(depth: pd.Series, pipe_diam_m: float) -> pd.Series:
    """Wetted perimeter (m) of a partially full circular pipe."""
    R = pipe_diam_m / 2.0
    if R <= 0:
        return pd.Series(np.nan, index=depth.index)

    h = depth.clip(lower=0.0, upper=2 * R)
    x = (R - h) / R
    x = x.clip(lower=-1.0, upper=1.0)
    theta = 2.0 * np.arccos(x)
    perimeter = theta * R
    perimeter = perimeter.where(h > 0.0, other=0.0)
    return pd.Series(perimeter, index=depth.index)


def manning_velocity(
    depth: pd.Series,
    pipe_diam_m: Optional[float],
    slope: Optional[float],
    roughness_n: Optional[float],
) -> pd.Series:
    """Compute expected velocity (m/s) from Manning's equation."""
    if pipe_diam_m is None or slope is None or roughness_n is None:
        return pd.Series(np.nan, index=depth.index)
    if pipe_diam_m <= 0 or slope <= 0 or roughness_n <= 0:
        return pd.Series(np.nan, index=depth.index)

    area = circular_area_from_depth(depth, pipe_diam_m)
    perimeter = circular_wetted_perimeter(depth, pipe_diam_m)
    hydraulic_radius = area / perimeter.replace(0.0, np.nan)
    velocity = (1.0 / roughness_n) * (hydraulic_radius ** (2.0 / 3.0)) * np.sqrt(slope)
    velocity = velocity.replace([np.inf, -np.inf], np.nan)
    velocity = velocity.where(area.notna() & perimeter.notna(), np.nan)
    return velocity


def theoretical_full_pipe_capacity_lps(
    pipe_diam_m: Optional[float],
    slope: Optional[float],
    roughness_n: Optional[float],
) -> Optional[float]:
    """Return theoretical full-pipe capacity in L/s when metadata is provided."""
    if pipe_diam_m is None or slope is None or roughness_n is None:
        return None
    if pipe_diam_m <= 0 or slope <= 0 or roughness_n <= 0:
        return None

    radius = pipe_diam_m / 2.0
    area = np.pi * radius ** 2
    hydraulic_radius = pipe_diam_m / 4.0
    velocity = (1.0 / roughness_n) * (hydraulic_radius ** (2.0 / 3.0)) * np.sqrt(slope)
    flow_m3s = area * velocity
    return float(flow_m3s * 1000.0)


def add_flow_columns(df: pd.DataFrame, pipe_diam_m: Optional[float]) -> pd.DataFrame:
    if pipe_diam_m is None or pipe_diam_m <= 0:
        return df

    df = df.copy()
    area = circular_area_from_depth(df["depth_clean"], pipe_diam_m)
    df["area_m2"] = area
    df["flow_m3s"] = df["area_m2"] * df["velocity_clean"]
    df["flow_lps"] = df["flow_m3s"] * 1000.0
    return df


def assign_quality_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive human-readable quality classes from flags/source:
      good_raw, raw_bad, inferred_rating, inferred_diurnal, inferred_time, missing
    """
    df = df.copy()

    # Depth quality
    depth_quality = np.full(len(df), "missing", dtype=object)

    is_nan_d = df["depth_clean"].isna()
    depth_quality[is_nan_d] = "missing"

    is_raw_good_d = df["depth_flag"] & (df["depth_source"] == "raw")
    depth_quality[is_raw_good_d] = "good_raw"

    is_raw_bad_d = (~df["depth_flag"]) & (df["depth_source"] == "bad")
    depth_quality[is_raw_bad_d] = "raw_bad"

    is_rating_d = df["depth_source"] == "rating_from_velocity"
    depth_quality[is_rating_d] = "inferred_rating"

    is_diurnal_d = df["depth_source"] == "diurnal_profile"
    depth_quality[is_diurnal_d] = "inferred_diurnal"

    is_time_d = df["depth_source"] == "time_interp"
    depth_quality[is_time_d] = "inferred_time"

    is_shape_d = df["depth_source"] == "monotone_interp"
    depth_quality[is_shape_d] = "inferred_shape"

    df["depth_quality"] = depth_quality

    # Velocity quality
    velocity_quality = np.full(len(df), "missing", dtype=object)

    is_nan_v = df["velocity_clean"].isna()
    velocity_quality[is_nan_v] = "missing"

    is_raw_good_v = df["velocity_flag"] & (df["velocity_source"] == "raw")
    velocity_quality[is_raw_good_v] = "good_raw"

    is_raw_bad_v = (~df["velocity_flag"]) & (df["velocity_source"] == "bad")
    velocity_quality[is_raw_bad_v] = "raw_bad"

    is_rating_v = df["velocity_source"] == "rating_from_depth"
    velocity_quality[is_rating_v] = "inferred_rating"

    is_diurnal_v = df["velocity_source"] == "diurnal_profile"
    velocity_quality[is_diurnal_v] = "inferred_diurnal"

    is_time_v = df["velocity_source"] == "time_interp"
    velocity_quality[is_time_v] = "inferred_time"

    is_shape_v = df["velocity_source"] == "monotone_interp"
    velocity_quality[is_shape_v] = "inferred_shape"

    df["velocity_quality"] = velocity_quality

    return df


def qc_stats(df: pd.DataFrame, pipe_diam_m: Optional[float]):
    """
    Return both raw QC stats and post-processing coverage stats,
    plus the depth–velocity relationship on raw & cleaned data.
    """
    n = len(df)
    d_good = df["depth_flag"].sum()
    v_good = df["velocity_flag"].sum()

    depth_usable = int(df["depth_clean"].notna().sum()) if "depth_clean" in df.columns else 0
    velocity_usable = int(df["velocity_clean"].notna().sum()) if "velocity_clean" in df.columns else 0
    depth_usable_pct = float(depth_usable / n * 100) if n > 0 else 0.0
    velocity_usable_pct = float(velocity_usable / n * 100) if n > 0 else 0.0

    flow_usable = 0
    flow_usable_pct = 0.0
    if pipe_diam_m is not None and "flow_lps" in df.columns:
        flow_usable = int(df["flow_lps"].notna().sum())
        flow_usable_pct = float(flow_usable / n * 100) if n > 0 else 0.0

    stats = {
        "total_points": int(n),
        # raw QC
        "depth_good": int(d_good),
        "depth_good_pct": float(d_good / n * 100 if n > 0 else 0),
        "velocity_good": int(v_good),
        "velocity_good_pct": float(v_good / n * 100 if n > 0 else 0),
        # post-processed coverage
        "depth_usable": depth_usable,
        "depth_usable_pct": depth_usable_pct,
        "depth_missing_final": int(n - depth_usable),
        "velocity_usable": velocity_usable,
        "velocity_usable_pct": velocity_usable_pct,
        "velocity_missing_final": int(n - velocity_usable),
        "flow_usable": flow_usable,
        "flow_usable_pct": flow_usable_pct,
        "flow_missing_final": int(n - flow_usable) if pipe_diam_m is not None and "flow_lps" in df.columns else None,
        # sources & quality
        "depth_source_counts": df["depth_source"].value_counts().to_dict(),
        "velocity_source_counts": df["velocity_source"].value_counts().to_dict(),
        "depth_quality_counts": df["depth_quality"].value_counts().to_dict(),
        "velocity_quality_counts": df["velocity_quality"].value_counts().to_dict(),
    }

    # Depth–velocity relationship on cleaned data
    if "depth_clean" in df.columns and "velocity_clean" in df.columns:
        mask_clean = df["depth_clean"].notna() & df["velocity_clean"].notna()
        if mask_clean.sum() >= 10:
            d = df.loc[mask_clean, "depth_clean"].values
            v = df.loc[mask_clean, "velocity_clean"].values
            corr = float(np.corrcoef(d, v)[0, 1])
            slope = float(np.polyfit(d, v, 1)[0])
            stats["depth_velocity_relationship_clean"] = {
                "corr": corr,
                "slope_mps_per_m": slope,
            }

    # Depth–velocity relationship on raw data
    if "depth" in df.columns and "velocity" in df.columns:
        mask_raw = df["depth"].notna() & df["velocity"].notna()
        if mask_raw.sum() >= 10:
            d0 = df.loc[mask_raw, "depth"].values
            v0 = df.loc[mask_raw, "velocity"].values
            corr0 = float(np.corrcoef(d0, v0)[0, 1])
            slope0 = float(np.polyfit(d0, v0, 1)[0])
            stats["depth_velocity_relationship_raw"] = {
                "corr": corr0,
                "slope_mps_per_m": slope0,
            }

    if pipe_diam_m is not None and "flow_lps" in df.columns:
        good_flow = df["flow_lps"].dropna()
        if not good_flow.empty:
            stats["flow_stats"] = {
                "min_lps": float(good_flow.min()),
                "max_lps": float(good_flow.max()),
                "mean_lps": float(good_flow.mean()),
                "p95_lps": float(good_flow.quantile(0.95)),
            }

    if "depth_clean" in df.columns and "velocity_clean" in df.columns:
        froude = compute_froude_number(df["depth_clean"], df["velocity_clean"])
        froude = froude.dropna()
        if not froude.empty:
            stats["froude_stats"] = {
                "min": float(froude.min()),
                "median": float(froude.median()),
                "p95": float(froude.quantile(0.95)),
                "max": float(froude.max()),
            }
    return stats


def process_dataset(
    df: pd.DataFrame,
    pipe_diam_m: Optional[float],
    flatline_run: int,
    spike_window: int,
    spike_k: float,
    diurnal_freq: str,
    diurnal_min_samples: int,
    depth_min_meas: Optional[float],
    depth_max_meas: Optional[float],
    vel_min_meas: Optional[float],
    vel_max_meas: Optional[float],
    froude_max: float,
    gradient_k: float,
    slope: Optional[float] = None,
    roughness_n: Optional[float] = None,
    hydraulic_tolerance: float = 0.35,
    full_pipe_capacity_lps: Optional[float] = None,
    capacity_tolerance: float = 0.1,
    condition_labels: Optional[pd.Series] = None,
    rainfall_context: Optional[Dict[str, pd.Series]] = None,
    manual_depth_mask: Optional[pd.Series] = None,
    manual_velocity_mask: Optional[pd.Series] = None,
):
    d_knots, v_knots, v_expected, d_expected, tracker = apply_quality_checks(
        df,
        pipe_diam_m=pipe_diam_m,
        flatline_run=flatline_run,
        spike_window=spike_window,
        spike_k=spike_k,
        diurnal_freq_baseline=diurnal_freq,
        diurnal_min_samples_baseline=diurnal_min_samples,
        depth_min_meas=depth_min_meas,
        depth_max_meas=depth_max_meas,
        vel_min_meas=vel_min_meas,
        vel_max_meas=vel_max_meas,
        froude_max=froude_max,
        gradient_k=gradient_k,
        slope=slope,
        roughness_n=roughness_n,
        hydraulic_tolerance=hydraulic_tolerance,
        full_pipe_capacity_lps=full_pipe_capacity_lps,
        capacity_tolerance=capacity_tolerance,
        condition_labels=(
            condition_labels
            if condition_labels is not None
            else rainfall_context.get("condition_label") if rainfall_context else None
        ),
        manual_depth_mask=manual_depth_mask,
        manual_velocity_mask=manual_velocity_mask,
    )

    if v_expected is None or d_expected is None:
        v_expected = None
        d_expected = None

    df_clean = interpolate_signals(
        df,
        v_expected,
        d_expected,
        anomaly_tracker=tracker,
        diurnal_freq=diurnal_freq,
        diurnal_min_samples=diurnal_min_samples,
        depth_min_meas=depth_min_meas,
        depth_max_meas=depth_max_meas,
        vel_min_meas=vel_min_meas,
        vel_max_meas=vel_max_meas,
    )
    if rainfall_context is not None:
        for key, series in rainfall_context.items():
            df_clean[key] = series.values
    if rainfall_context is not None and "condition_label" in rainfall_context:
        df_clean["dwf_wwf"] = np.where(
            rainfall_context["is_wet_weather"],
            "wet_weather",
            "dry_weather",
        )
    df_clean = add_flow_columns(df_clean, pipe_diam_m)
    df_clean = assign_quality_labels(df_clean)
    stats = qc_stats(df_clean, pipe_diam_m)
    return df_clean, stats


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Sewer Depth & Velocity QC",
    layout="wide",
)

st.title("Sewer Depth & Velocity QC + Flow (A × V = Q)")

st.markdown(
    """
Upload gravity sewer **depth & velocity** data (CSV) with:
- **Column A**: timestamp (`dd/mm/yyyy hh:mm`)
- **Column B**: depth (m)
- **Column C**: velocity (m/s)

This app applies:
- Hydrometric-style QC: flatlines, spikes (MAD), physical & measured limits,
  rating-curve consistency (biasing blame to velocity), diurnal baseline checks.
- Multi-layer infill: depth-driven rating curve → diurnal profiles → time interpolation.
- Hydraulic integrity checks: Froude screening, gradient outlier detection, and
    shape-preserving monotone interpolation for high-fidelity infill.
- Optional open-channel flow calculation **Q = A × V** for circular pipes.
- Clear visibility of **good vs bad vs inferred** data,
  including depth–velocity scatter plots.
"""
)

with st.sidebar:
    st.header("Settings")

    pipe_diam_m = st.number_input(
        "Pipe diameter (m, circular, optional)",
        min_value=0.0,
        value=0.0,
        step=0.05,
        help="Set to 0 for no flow calculation",
    )
    pipe_diam_m = pipe_diam_m if pipe_diam_m > 0 else None

    st.markdown("---")
    st.caption("Advanced QC parameters")

    flatline_run = st.number_input(
        "Flatline run length (points)",
        min_value=3,
        value=10,
        step=1,
        help="Min consecutive identical values to treat as sensor stuck",
    )
    spike_window = st.number_input(
        "Spike detection window (points)",
        min_value=3,
        value=7,
        step=1,
        help="Rolling window size for spike detection",
    )
    spike_k = st.number_input(
        "Spike sensitivity k (MAD multiplier)",
        min_value=2.0,
        value=6.0,
        step=0.5,
        help="Higher = less sensitive to spikes",
    )

    froude_max = st.number_input(
        "Max Froude number",
        min_value=0.5,
        value=1.5,
        step=0.1,
        help="Flag velocity when V/sqrt(g*depth) exceeds this limit.",
    )

    gradient_k = st.number_input(
        "Gradient anomaly z-threshold",
        min_value=3.0,
        value=8.0,
        step=0.5,
        help="Higher = less sensitive to sudden jumps between samples.",
    )

    st.markdown("---")
    st.caption("Measured bounds (optional, from field/inspection)")

    use_depth_min = st.checkbox("Constrain min depth", value=False)
    if use_depth_min:
        depth_min_meas = st.number_input(
            "Measured min depth (m)",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )
    else:
        depth_min_meas = None

    use_depth_max = st.checkbox("Constrain max depth", value=False)
    if use_depth_max:
        depth_max_meas = st.number_input(
            "Measured max depth (m)",
            min_value=0.0,
            value=1.0,
            step=0.01,
        )
    else:
        depth_max_meas = None

    use_vel_min = st.checkbox("Constrain min velocity", value=False)
    if use_vel_min:
        vel_min_meas = st.number_input(
            "Measured min velocity (m/s)",
            value=0.1,
            step=0.01,
        )
    else:
        vel_min_meas = None

    use_vel_max = st.checkbox("Constrain max velocity", value=False)
    if use_vel_max:
        vel_max_meas = st.number_input(
            "Measured max velocity (m/s)",
            value=1.0,
            step=0.01,
        )
    else:
        vel_max_meas = None

    st.markdown("---")
    st.caption("Diurnal profile settings")

    diurnal_freq = st.selectbox(
        "Diurnal bin size",
        options=["1min", "5min", "15min", "30min", "60min"],
        index=1,
        help="Time-of-day bin size for diurnal medians.",
    )

    diurnal_min_samples = st.number_input(
        "Min good samples per diurnal bin",
        min_value=3,
        value=5,
        step=1,
        help="Minimum number of good points needed for a diurnal median.",
    )

uploaded_file = st.file_uploader(
    "Upload CSV file", type=["csv"], accept_multiple_files=False
)

run_button = st.button("Run analysis", disabled=uploaded_file is None)

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "pipe_diam_m" not in st.session_state:
    st.session_state.pipe_diam_m = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "last_run_params" not in st.session_state:
    st.session_state.last_run_params = None
if "manual_depth_mask" not in st.session_state:
    st.session_state.manual_depth_mask = None
if "manual_velocity_mask" not in st.session_state:
    st.session_state.manual_velocity_mask = None

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
else:
    if run_button:
        try:
            buffer = io.BytesIO(uploaded_file.getvalue())
            raw_df = load_data_from_buffer(buffer)

            st.success(f"Loaded {len(raw_df)} rows from `{uploaded_file.name}`")

            df_clean, stats = process_dataset(
                raw_df,
                pipe_diam_m=pipe_diam_m,
                flatline_run=int(flatline_run),
                spike_window=int(spike_window),
                spike_k=float(spike_k),
                diurnal_freq=diurnal_freq,
                diurnal_min_samples=int(diurnal_min_samples),
                depth_min_meas=depth_min_meas,
                depth_max_meas=depth_max_meas,
                vel_min_meas=vel_min_meas,
                vel_max_meas=vel_max_meas,
                froude_max=float(froude_max),
                gradient_k=float(gradient_k),
            )

            st.session_state.df_clean = df_clean
            st.session_state.pipe_diam_m = pipe_diam_m
            st.session_state.raw_df = raw_df.copy()
            st.session_state.manual_depth_mask = None
            st.session_state.manual_velocity_mask = None
            st.session_state.last_run_params = {
                "pipe_diam_m": pipe_diam_m,
                "flatline_run": int(flatline_run),
                "spike_window": int(spike_window),
                "spike_k": float(spike_k),
                "diurnal_freq": diurnal_freq,
                "diurnal_min_samples": int(diurnal_min_samples),
                "depth_min_meas": depth_min_meas,
                "depth_max_meas": depth_max_meas,
                "vel_min_meas": vel_min_meas,
                "vel_max_meas": vel_max_meas,
                "froude_max": float(froude_max),
                "gradient_k": float(gradient_k),
            }

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

if st.session_state.df_clean is not None:
    df_clean = st.session_state.df_clean
    pipe_diam_m_saved = st.session_state.pipe_diam_m

    stats = qc_stats(df_clean, pipe_diam_m_saved)

    st.subheader("QC Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total points", stats["total_points"])
        st.metric(
            "Depth good (raw %)",
            f'{stats["depth_good_pct"]:.1f}%',
            stats["depth_good"],
        )
        st.metric(
            "Depth usable (final %)",
            f'{stats["depth_usable_pct"]:.2f}%',
            stats["depth_usable"],
        )

    with col2:
        st.metric(
            "Velocity good (raw %)",
            f'{stats["velocity_good_pct"]:.1f}%',
            stats["velocity_good"],
        )
        st.metric(
            "Velocity usable (final %)",
            f'{stats["velocity_usable_pct"]:.2f}%',
            stats["velocity_usable"],
        )
        if pipe_diam_m_saved is not None and "flow_lps" in df_clean.columns:
            st.metric(
                "Flow usable (final %)",
                f'{stats["flow_usable_pct"]:.2f}%',
                stats["flow_usable"],
            )

    st.markdown(
        f"""
**Post-processing coverage**

- Depth usable: **{stats["depth_usable_pct"]:.2f}%**  
- Velocity usable: **{stats["velocity_usable_pct"]:.2f}%**  
""" + (
            f'- Flow usable: **{stats["flow_usable_pct"]:.2f}%**\n'
            if pipe_diam_m_saved is not None and "flow_lps" in df_clean.columns
            else ""
        )
    )

    if "froude_stats" in stats:
        fr = stats["froude_stats"]
        st.markdown(
            f"""
**Froude number (cleaned data)**
- Min: **{fr["min"]:.3f}**
- Median: **{fr["median"]:.3f}**
- 95th percentile: **{fr["p95"]:.3f}**
- Max: **{fr["max"]:.3f}**
"""
        )

    # Depth–velocity relationship summary (raw vs cleaned)
    st.markdown("### Depth–Velocity Relationship (raw vs cleaned)")

    if "depth_velocity_relationship_raw" in stats:
        rel0 = stats["depth_velocity_relationship_raw"]
        st.markdown(
            f"""
**Raw data**  
- Correlation coefficient: **{rel0["corr"]:.3f}**  
- Slope: **{rel0["slope_mps_per_m"]:.3f} m/s per m**
"""
        )
    else:
        st.markdown("_Raw depth–velocity relationship: not enough data to compute._")

    if "depth_velocity_relationship_clean" in stats:
        rel = stats["depth_velocity_relationship_clean"]
        st.markdown(
            f"""
**Cleaned data**  
- Correlation coefficient: **{rel["corr"]:.3f}**  
- Slope: **{rel["slope_mps_per_m"]:.3f} m/s per m**
"""
        )
        if rel["corr"] < 0.3 or rel["slope_mps_per_m"] <= 0:
            st.warning(
                "Depth–velocity relationship (cleaned) looks weak or non-physical "
                "(low correlation or non-positive slope). Please review the dataset, "
                "sensor configuration, and QC thresholds."
            )
    else:
        st.markdown("_Cleaned depth–velocity relationship: not enough data to compute._")

    st.subheader("Manual Good/Bad Selection for Interpolation")
    st.markdown(
        "Mark the samples you trust as hydraulically sound. Only these rows will seed "
        "rating-curve and diurnal infill on the next rebuild, while unchecked rows are "
        "treated as bad and will be regenerated."
    )

    qc_depth_reference = (
        df_clean["depth_flag_auto"]
        if "depth_flag_auto" in df_clean.columns
        else df_clean["depth_flag"]
    )
    qc_velocity_reference = (
        df_clean["velocity_flag_auto"]
        if "velocity_flag_auto" in df_clean.columns
        else df_clean["velocity_flag"]
    )

    base_depth_defaults = df_clean["depth_flag"].astype(bool)
    base_velocity_defaults = df_clean["velocity_flag"].astype(bool)

    def current_manual_mask(name: str, base: pd.Series) -> pd.Series:
        mask = st.session_state.get(name)
        if isinstance(mask, pd.Series) and len(mask) == len(base):
            return mask.reindex(base.index).fillna(base).astype(bool)
        return base.astype(bool)

    manual_depth_defaults = current_manual_mask("manual_depth_mask", base_depth_defaults)
    manual_velocity_defaults = current_manual_mask("manual_velocity_mask", base_velocity_defaults)

    # Graphical selection helper
    st.markdown("### Graphical selection assistant")
    selection_metric = st.radio(
        "Metric to flag from chart",
        options=["Depth", "Velocity"],
        index=0,
        horizontal=True,
        key="graph_selection_metric",
    )

    selection_cols = ["timestamp", "depth_clean", "velocity_clean"]
    if "depth_quality" in df_clean.columns:
        selection_cols.append("depth_quality")
    if "velocity_quality" in df_clean.columns:
        selection_cols.append("velocity_quality")
    plot_df = df_clean[selection_cols].copy()
    plot_df["row_id"] = df_clean.index
    y_col = "depth_clean" if selection_metric == "Depth" else "velocity_clean"
    color_col = "depth_quality" if selection_metric == "Depth" else "velocity_quality"
    fig_select = px.scatter(
        plot_df,
        x="timestamp",
        y=y_col,
        color=color_col if color_col in plot_df.columns else None,
        title=f"Select {selection_metric.lower()} points (box-select / lasso)",
        labels={y_col: f"{selection_metric} (clean)", "timestamp": "Time"},
    )
    fig_select.update_layout(dragmode="select")
    fig_select.update_traces(marker=dict(size=6))

    selected_points = plotly_events(
        fig_select,
        select_event=True,
        override_height=420,
        key=f"{selection_metric.lower()}_plotly_select",
    )

    selected_row_ids: List[int] = []
    if selected_points:
        selected_indices = [
            pt.get("pointIndex", pt.get("pointNumber"))
            for pt in selected_points
            if pt.get("pointIndex", pt.get("pointNumber")) is not None
        ]
        if selected_indices:
            selected_row_ids = plot_df.loc[selected_indices, "row_id"].tolist()
            st.info(
                f"Selected {len(selected_row_ids)} samples for {selection_metric.lower()} classification."
            )

    def update_manual_mask(name: str, base: pd.Series, row_ids: List[int], value: bool):
        mask = current_manual_mask(name, base).copy()
        if row_ids:
            mask.loc[row_ids] = value
        st.session_state[name] = mask

    select_cols = st.columns(2)
    if selection_metric == "Depth":
        if select_cols[0].button(
            "Mark selection as good depth",
            disabled=not selected_row_ids,
            key="btn_depth_good",
        ):
            update_manual_mask("manual_depth_mask", base_depth_defaults, selected_row_ids, True)
            st.success("Marked selection as trusted depth data.")
            st.rerun()
        if select_cols[1].button(
            "Mark selection as bad depth",
            disabled=not selected_row_ids,
            key="btn_depth_bad",
        ):
            update_manual_mask("manual_depth_mask", base_depth_defaults, selected_row_ids, False)
            st.success("Marked selection as bad depth data.")
            st.rerun()
    else:
        if select_cols[0].button(
            "Mark selection as good velocity",
            disabled=not selected_row_ids,
            key="btn_vel_good",
        ):
            update_manual_mask("manual_velocity_mask", base_velocity_defaults, selected_row_ids, True)
            st.success("Marked selection as trusted velocity data.")
            st.rerun()
        if select_cols[1].button(
            "Mark selection as bad velocity",
            disabled=not selected_row_ids,
            key="btn_vel_bad",
        ):
            update_manual_mask("manual_velocity_mask", base_velocity_defaults, selected_row_ids, False)
            st.success("Marked selection as bad velocity data.")
            st.rerun()

    manual_selector_source = pd.DataFrame(
        {
            "timestamp": df_clean["timestamp"],
            "depth": df_clean["depth"],
            "velocity": df_clean["velocity"],
            "qc_depth_flag": qc_depth_reference.astype(bool),
            "qc_velocity_flag": qc_velocity_reference.astype(bool),
            "manual_depth_good": manual_depth_defaults,
            "manual_velocity_good": manual_velocity_defaults,
        }
    )

    manual_selector_editor = st.data_editor(
        manual_selector_source,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", disabled=True),
            "depth": st.column_config.NumberColumn("Depth (m)", disabled=True, format="%.3f"),
            "velocity": st.column_config.NumberColumn("Velocity (m/s)", disabled=True, format="%.3f"),
            "qc_depth_flag": st.column_config.CheckboxColumn("QC depth good", disabled=True),
            "qc_velocity_flag": st.column_config.CheckboxColumn("QC velocity good", disabled=True),
            "manual_depth_good": st.column_config.CheckboxColumn("Manual depth good"),
            "manual_velocity_good": st.column_config.CheckboxColumn("Manual velocity good"),
        },
        key="manual_selector_editor",
    )

    if st.button("Rebuild using manual selections", type="primary"):
        if st.session_state.raw_df is None or st.session_state.last_run_params is None:
            st.warning("Load a dataset and run analysis before applying manual selections.")
        else:
            try:
                manual_depth_mask = pd.Series(
                    manual_selector_editor["manual_depth_good"].astype(bool).values,
                    index=st.session_state.raw_df.index,
                    name="manual_depth_good",
                )
                manual_velocity_mask = pd.Series(
                    manual_selector_editor["manual_velocity_good"].astype(bool).values,
                    index=st.session_state.raw_df.index,
                    name="manual_velocity_good",
                )

                new_df_clean, _ = process_dataset(
                    st.session_state.raw_df.copy(),
                    manual_depth_mask=manual_depth_mask,
                    manual_velocity_mask=manual_velocity_mask,
                    **st.session_state.last_run_params,
                )

                st.session_state.df_clean = new_df_clean
                st.session_state.manual_depth_mask = manual_depth_mask
                st.session_state.manual_velocity_mask = manual_velocity_mask
                st.success("Manual selections applied. Interpolation rebuilt using trusted samples.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to rebuild with manual selections: {e}")
                st.exception(e)

    with st.expander("Depth sources & quality breakdown"):
        st.markdown("**Depth source counts**")
        st.json(stats["depth_source_counts"])
        st.markdown("**Depth quality classes**")
        st.json(stats["depth_quality_counts"])

    with st.expander("Velocity sources & quality breakdown"):
        st.markdown("**Velocity source counts**")
        st.json(stats["velocity_source_counts"])
        st.markdown("**Velocity quality classes**")
        st.json(stats["velocity_quality_counts"])
        if "froude_stats" in stats:
            st.markdown("**Froude number distribution (cleaned)**")
            st.json(stats["froude_stats"])
        if "flow_stats" in stats:
            st.markdown("**Flow (L/s) statistics**")
            st.json(stats["flow_stats"])

    # Interactive time series
    st.subheader("Interactive time series (zoom, pan, toggle traces)")

    df_plot = df_clean.copy()

    show_depth = st.checkbox("Show depth (raw vs cleaned)", value=True)
    if show_depth:
        fig_depth = px.line(
            df_plot,
            x="timestamp",
            y=["depth", "depth_clean"],
            labels={"value": "Depth (m)", "timestamp": "Time", "variable": "Series"},
            title="Depth vs Time (raw vs cleaned)",
        )
        st.plotly_chart(fig_depth, use_container_width=True)

    show_depth_quality = st.checkbox("Highlight depth quality classes", value=True)
    if show_depth_quality:
        fig_dq = px.scatter(
            df_plot,
            x="timestamp",
            y="depth_clean",
            color="depth_quality",
            title="Depth (cleaned) coloured by quality class",
            labels={"depth_clean": "Depth (m)", "timestamp": "Time", "depth_quality": "Quality"},
        )
        st.plotly_chart(fig_dq, use_container_width=True)

    show_vel = st.checkbox("Show velocity (raw vs cleaned)", value=True)
    if show_vel:
        fig_vel = px.line(
            df_plot,
            x="timestamp",
            y=["velocity", "velocity_clean"],
            labels={"value": "Velocity (m/s)", "timestamp": "Time", "variable": "Series"},
            title="Velocity vs Time (raw vs cleaned)",
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    show_vel_quality = st.checkbox("Highlight velocity quality classes", value=True)
    if show_vel_quality:
        fig_vq = px.scatter(
            df_plot,
            x="timestamp",
            y="velocity_clean",
            color="velocity_quality",
            title="Velocity (cleaned) coloured by quality class",
            labels={"velocity_clean": "Velocity (m/s)", "timestamp": "Time", "velocity_quality": "Quality"},
        )
        st.plotly_chart(fig_vq, use_container_width=True)

    # NEW: Depth–Velocity scatter plots (before / after)
    st.subheader("Depth–Velocity scatter (velocity on X, depth on Y)")

    show_dv_raw_scatter = st.checkbox(
        "Depth vs velocity (raw)", value=True
    )
    if show_dv_raw_scatter:
        fig_dv_raw = px.scatter(
            df_plot,
            x="velocity",
            y="depth",
            title="Depth vs Velocity (raw)",
            labels={"velocity": "Velocity (m/s)", "depth": "Depth (m)"},
        )
        st.plotly_chart(fig_dv_raw, use_container_width=True)

    show_dv_clean_scatter = st.checkbox(
        "Depth vs velocity (cleaned, coloured by velocity quality)", value=True
    )
    if show_dv_clean_scatter:
        fig_dv_clean = px.scatter(
            df_plot,
            x="velocity_clean",
            y="depth_clean",
            color="velocity_quality",
            title="Depth vs Velocity (cleaned) – expecting a tight upward trend",
            labels={
                "velocity_clean": "Velocity (m/s)",
                "depth_clean": "Depth (m)",
                "velocity_quality": "Quality",
            },
        )
        st.plotly_chart(fig_dv_clean, use_container_width=True)

    show_flow = st.checkbox(
        "Show flow (L/s)", value=(pipe_diam_m_saved is not None and "flow_lps" in df_plot.columns)
    )
    if show_flow and pipe_diam_m_saved is not None and "flow_lps" in df_plot.columns:
        fig_flow = px.line(
            df_plot,
            x="timestamp",
            y="flow_lps",
            labels={"flow_lps": "Flow (L/s)", "timestamp": "Time"},
            title="Flow vs Time (cleaned)",
        )
        st.plotly_chart(fig_flow, use_container_width=True)

    # Manual adjustments
    st.subheader("Manual review & adjustments")

    st.markdown(
        """
Use this table to make **fine adjustments** to the cleaned data.
Quality, source, and flow columns are visible so you can see exactly
what was raw vs interpolated.
"""
    )

    edited_df = st.data_editor(
        df_clean,
        num_rows="fixed",
        use_container_width=True,
        key="editor_table",
    )

    if st.button("Apply manual edits & recompute flow"):
        df_manual = edited_df.copy()
        df_manual = add_flow_columns(df_manual, pipe_diam_m_saved)
        df_manual = assign_quality_labels(df_manual)
        st.session_state.df_clean = df_manual
        st.success("Manual edits applied. Charts and stats updated above.")

    st.subheader("Download final cleaned CSV")
    csv_bytes = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
    download_name = (uploaded_file.name.replace(".csv", "") if uploaded_file else "output") + "_cleaned.csv"

    st.download_button(
        label="Download cleaned CSV",
        data=csv_bytes,
        file_name=download_name,
        mime="text/csv",
    )
