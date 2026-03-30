import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from rdmpy.outputs.load_data import load_processed_data


DAY_SUFFIX = {0: "MO", 1: "TU", 2: "WE", 3: "TH", 4: "FR", 5: "SA", 6: "SU"}


def _parse_hhmm_to_time(value: object):
    if pd.isna(value):
        return None
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if len(text) < 4:
        return None
    text = text[:4]
    hour = int(text[:2]) % 24
    minute = int(text[2:4])
    if minute >= 60:
        return None
    return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()


def _build_effective_event_datetime(df: pd.DataFrame, analysis_start: datetime) -> pd.DataFrame:
    target_day = DAY_SUFFIX[analysis_start.weekday()]
    out = df.copy()

    if "EVENT_DATETIME" in out.columns:
        out["event_dt"] = pd.to_datetime(
            out["EVENT_DATETIME"], format="%d-%b-%Y %H:%M", errors="coerce"
        )
    else:
        out["event_dt"] = pd.NaT

    if "DAY" not in out.columns:
        out["DAY"] = None

    missing_mask = out["event_dt"].isna() & (out["DAY"] == target_day)
    if missing_mask.any():
        fallback_times = out.loc[missing_mask, "ACTUAL_CALLS"] if "ACTUAL_CALLS" in out.columns else pd.Series(index=out.index[missing_mask], dtype=object)
        if "PLANNED_CALLS" in out.columns:
            planned = out.loc[missing_mask, "PLANNED_CALLS"]
            fallback_times = fallback_times.where(fallback_times.notna(), planned)

        parsed_times = fallback_times.apply(_parse_hhmm_to_time)
        reconstructed = parsed_times.apply(
            lambda t: datetime.combine(analysis_start.date(), t) if t is not None else pd.NaT
        )
        out.loc[missing_mask, "event_dt"] = pd.to_datetime(reconstructed, errors="coerce")

    return out


def compute_window_metrics(
    analysis_date: str,
    analysis_hhmm: str,
    window_minutes: int = 5,
    processed_dir: str = "processed_data",
):
    analysis_start = datetime.strptime(
        f"{analysis_date} {analysis_hhmm[:2]}:{analysis_hhmm[2:]}", "%d-%b-%Y %H:%M"
    )
    analysis_end = analysis_start + timedelta(minutes=window_minutes)

    all_data = load_processed_data(base_dir=processed_dir)
    if all_data.empty:
        return pd.DataFrame(), pd.DataFrame(), analysis_start, analysis_end

    all_data = _build_effective_event_datetime(all_data, analysis_start)
    window_df = all_data[
        (all_data["event_dt"] >= analysis_start) & (all_data["event_dt"] < analysis_end)
    ].copy()

    if window_df.empty:
        return pd.DataFrame(), pd.DataFrame(), analysis_start, analysis_end

    if "STANOX" not in window_df.columns:
        window_df["STANOX"] = "UNKNOWN"
    if "TRAIN_SERVICE_CODE" not in window_df.columns:
        window_df["TRAIN_SERVICE_CODE"] = "UNKNOWN"
    if "PFPI_MINUTES" not in window_df.columns:
        window_df["PFPI_MINUTES"] = 0
    if "EVENT_TYPE" not in window_df.columns:
        window_df["EVENT_TYPE"] = None
    if "INCIDENT_NUMBER" not in window_df.columns:
        window_df["INCIDENT_NUMBER"] = None

    window_df["delay_minutes"] = pd.to_numeric(window_df["PFPI_MINUTES"], errors="coerce").fillna(0)
    window_df["delay_occurred"] = window_df["delay_minutes"] > 0

    train_station_points = window_df[
        [
            "STANOX",
            "TRAIN_SERVICE_CODE",
            "event_dt",
            "delay_occurred",
            "delay_minutes",
            "EVENT_TYPE",
            "INCIDENT_NUMBER",
        ]
    ].sort_values(["STANOX", "event_dt", "TRAIN_SERVICE_CODE"])

    station_summary = (
        train_station_points.groupby("STANOX", as_index=False)
        .agg(
            trains_received=("TRAIN_SERVICE_CODE", "nunique"),
            train_station_points=("TRAIN_SERVICE_CODE", "count"),
            delay_occurrences=("delay_occurred", "sum"),
            total_delay_minutes=("delay_minutes", "sum"),
        )
        .sort_values(["trains_received", "delay_occurrences"], ascending=[False, False])
    )

    return station_summary, train_station_points, analysis_start, analysis_end


def main():
    parser = argparse.ArgumentParser(
        description="Compute train counts and delay occurrences per station in a short time window."
    )
    parser.add_argument("--date", required=True, help="Analysis date in DD-MMM-YYYY format.")
    parser.add_argument("--time", required=True, help="Analysis start time in HHMM format.")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=5,
        help="Window length in minutes (default: 5).",
    )
    parser.add_argument(
        "--processed-dir",
        default="processed_data",
        help="Path to processed data directory (default: processed_data).",
    )
    parser.add_argument(
        "--outdir",
        default="experiments/five_min_window/output",
        help="Output directory for CSV results.",
    )
    args = parser.parse_args()

    station_summary, train_station_points, analysis_start, analysis_end = compute_window_metrics(
        analysis_date=args.date,
        analysis_hhmm=args.time,
        window_minutes=args.window_minutes,
        processed_dir=args.processed_dir,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stamp = analysis_start.strftime("%Y%m%d_%H%M")
    station_file = outdir / f"station_summary_{stamp}_w{args.window_minutes}.csv"
    points_file = outdir / f"train_station_points_{stamp}_w{args.window_minutes}.csv"

    station_summary.to_csv(station_file, index=False)
    train_station_points.to_csv(points_file, index=False)

    print(f"Window: {analysis_start} to {analysis_end} ({args.window_minutes} minutes)")
    print(f"Stations in window: {len(station_summary)}")
    print(f"Train-station points in window: {len(train_station_points)}")
    print(f"Saved station summary: {station_file}")
    print(f"Saved train-station points: {points_file}")


if __name__ == "__main__":
    main()
