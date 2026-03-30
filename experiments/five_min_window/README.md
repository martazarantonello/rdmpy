# Five-Minute Station Window Experiment

This experiment answers two questions for a given datetime window:

1. How many trains each station received during a five-minute window.
2. Delay occurrences for each train-station point in that same window.

## Inputs

- Analysis date in `DD-MMM-YYYY` format (example: `07-DEC-2024`)
- Analysis time in `HHMM` format (example: `1830`)
- Optional window size (default 5 minutes)

## What the script does

1. Loads all processed station parquet files using `rdmpy.outputs.load_data.load_processed_data`.
2. Builds an event timestamp for each row:
   - Uses `EVENT_DATETIME` when available.
   - Falls back to `ACTUAL_CALLS` or `PLANNED_CALLS` combined with the analysis date when `DAY` matches the target weekday.
3. Filters rows to the requested window.
4. Produces:
   - Station-level counts.
   - Train-station point records with delay flags and delay minutes.
5. Saves outputs to CSV.

## Run

From repository root:

    python experiments/five_min_window/window_metrics.py --date 07-DEC-2024 --time 1830

Optional arguments:

- `--window-minutes` (default `5`)
- `--processed-dir` (default `processed_data`)
- `--outdir` (default `experiments/five_min_window/output`)

Example:

    python experiments/five_min_window/window_metrics.py --date 07-DEC-2024 --time 1830 --window-minutes 5 --outdir experiments/five_min_window/output

## Output files

- `station_summary_YYYYMMDD_HHMM_w5.csv`
- `train_station_points_YYYYMMDD_HHMM_w5.csv`
