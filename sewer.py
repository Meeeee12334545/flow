#!/usr/bin/env python3
import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


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

    # Enforce non-negative, non-decreasing velocity with depth
    vel_knots = np.maximum(vel_knots, 0.0)
    vel_knots = np.maximum.accumulate(vel_knots)

    def v_expected(d):
        d = np.asarray(d, dtype=float)
        return np.interp(
            d,
            depth_knots,
            vel_knots,
            left=0.0,
            right=vel_knots[-1],
        )

    def d_expected(v):
        v = np.asarray(v, dtype=float)
        v_knots = vel_knots.copy()
        if np.allclose(v_knots, 0.0):
            return np.interp(v, [0.0, 1.0], [depth_knots[0], depth_knots[-1]])
        return np.interp(
            v,
            v_knots,
            depth_knots,
            left=depth_knots[0],
            right=depth_knots[-1],
        )

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

    depth_flag = pd.Series(True, index=df.index)
    vel_flag = pd.Series(True, index=df.index)

    # NaNs
    depth_flag &= depth.notna()
    vel_flag &= vel.notna()

    # Depth plausibility
    d_pos = depth[depth > 0]
    if len(d_pos) > 0:
        d_p99 = d_pos.quantile(0.99)
        depth_max_limit = d_p99 * 1.5
        depth_max_limit = max(depth_max_limit, d_p99 + 0.5)
    else:
        depth_max_limit = np.inf

    depth_flag &= depth >= 0
    if pipe_diam_m is not None and pipe_diam_m > 0:
        depth_flag &= depth <= pipe_diam_m * 1.05
        depth_max_limit = min(depth_max_limit, pipe_diam_m * 1.05)
    else:
        depth_flag &= depth <= depth_max_limit

    # Measured depth limits
    if depth_min_meas is not None:
        depth_flag &= depth >= depth_min_meas
    if depth_max_meas is not None:
        depth_flag &= depth <= depth_max_meas

    # Velocity plausibility
    v_abs = vel.abs()[vel.notna()]
    if len(v_abs) > 0:
        v_p99 = v_abs.quantile(0.99)
        vel_max_limit = v_p99 * 1.5
        vel_max_limit = max(vel_max_limit, v_p99 + 0.5)
    else:
        vel_max_limit = np.inf

    # No negative velocities allowed in QC
    vel_flag &= vel >= 0.0
    vel_flag &= vel.abs() <= vel_max_limit

    # Measured velocity limits
    if vel_min_meas is not None:
        vel_flag &= vel >= vel_min_meas
    if vel_max_meas is not None:
        vel_flag &= vel <= vel_max_meas

    # Flatlines
    depth_flat = detect_flatlines(depth, min_run=flatline_run)
    vel_flat = detect_flatlines(vel, min_run=flatline_run)
    depth_flag &= ~depth_flat
    vel_flag &= ~vel_flat

    # Spikes
    depth_spike = robust_spike_mask(depth, window=spike_window, k=spike_k)
    vel_spike = robust_spike_mask(vel, window=spike_window, k=spike_k)
    depth_flag &= ~depth_spike
    vel_flag &= ~vel_spike

    # Rating curve consistency: depth is generally trusted, so we only
    # use the curve to mark velocity outliers here.
    combined_good = depth_flag & vel_flag
    d_knots, v_knots, v_expected, d_expected = build_rating_curve(
        depth, vel, combined_good
    )

    if v_expected is not None:
        # Velocity vs expected from depth
        v_pred = pd.Series(v_expected(depth), index=df.index)
        v_resid = vel - v_pred
        mad = v_resid.abs().median()
        scale = mad / 0.6745 if mad > 0 else v_resid.std()
        if scale and scale > 0:
            z_v = (v_resid.abs() / scale)
            curve_outliers_v = z_v > 4.0
            vel_flag &= ~curve_outliers_v

        # We *don't* mark depth bad from curve consistency; depth is the anchor.

    # Diurnal baseline anomalies
    depth_flag, vel_flag = detect_diurnal_baseline_anomalies(
        df,
        depth_flag,
        vel_flag,
        diurnal_freq=diurnal_freq_baseline,
        diurnal_min_samples=diurnal_min_samples_baseline,
        k=5.0,
        min_run_bins=6,
    )

    df["depth_flag"] = depth_flag
    df["velocity_flag"] = vel_flag
    return d_knots, v_knots, v_expected, d_expected


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

    if vel_profile is not None:
        missing_vel = df["velocity_clean"].isna()
        if missing_vel.any():
            vel_median_map = vel_profile["median"]
            diurnal_vel_vals = df["tod_bin"].map(vel_median_map)
            use_diurnal_vel = missing_vel & diurnal_vel_vals.notna()
            df.loc[use_diurnal_vel, "velocity_clean"] = diurnal_vel_vals[use_diurnal_vel]
            df.loc[use_diurnal_vel, "velocity_source"] = "diurnal_profile"

    df = df.drop(columns=["tod_bin"])
    df = df.reset_index()
    return df


def interpolate_signals(
    df: pd.DataFrame,
    v_expected,
    d_expected,
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
      3) time interpolation as a final step

    Measured min/max depth & velocity (if provided) are enforced at the end,
    and a global minimum of 0.01 m/s is applied to velocity_clean.
    """
    df = df.copy()

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

        # Secondary: good velocity, bad depth (less common, but allowed).
        mask_rating_depth = depth_bad & (~vel_bad) & df["velocity_clean"].notna()
        if mask_rating_depth.any():
            df.loc[mask_rating_depth, "depth_clean"] = d_expected(
                df.loc[mask_rating_depth, "velocity_clean"]
            )
            df.loc[mask_rating_depth, "depth_source"] = "rating_from_velocity"

    df = df.reset_index()

    # 2) Diurnal profile infill
    df = apply_diurnal_infill(
        df,
        depth_profile=depth_profile,
        vel_profile=vel_profile,
        diurnal_freq=diurnal_freq,
    )

    # 3) Time interpolation
    df = df.set_index("timestamp")

    depth_before = df["depth_clean"].copy()
    df["depth_clean"] = df["depth_clean"].interpolate(method="time", limit_direction="both")
    filled_depth = depth_before.isna() & df["depth_clean"].notna()
    df.loc[filled_depth & (df["depth_source"] == "bad"), "depth_source"] = "time_interp"

    vel_before = df["velocity_clean"].copy()
    df["velocity_clean"] = df["velocity_clean"].interpolate(method="time", limit_direction="both")
    filled_vel = vel_before.isna() & df["velocity_clean"].notna()
    df.loc[filled_vel & (df["velocity_source"] == "bad"), "velocity_source"] = "time_interp"

    df = df.reset_index()

    # 4) Enforce measured & global bounds on final cleaned series
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
):
    d_knots, v_knots, v_expected, d_expected = apply_quality_checks(
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
    )

    if v_expected is None or d_expected is None:
        v_expected = None
        d_expected = None

    df_clean = interpolate_signals(
        df,
        v_expected,
        d_expected,
        diurnal_freq=diurnal_freq,
        diurnal_min_samples=diurnal_min_samples,
        depth_min_meas=depth_min_meas,
        depth_max_meas=depth_max_meas,
        vel_min_meas=vel_min_meas,
        vel_max_meas=vel_max_meas,
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
            )

            st.session_state.df_clean = df_clean
            st.session_state.pipe_diam_m = pipe_diam_m

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
