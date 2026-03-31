"""
Experiment 15a: Citi Bike NYC — Data Download and Preprocessing

Downloads monthly Citi Bike trip records, filters to Manhattan core
(~150 stations), computes hourly activity distributions, and builds
a trip-weighted station graph.

Outputs:
  data/ex15_bikeshare_data.npz
  experiments/ex15a_station_map.png
  experiments/ex15a_distributions.png

Run: uv run experiments/ex15a_bikeshare_setup.py [--year 2023] [--months 6 7 8]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import zipfile
import urllib.request
import shutil
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')

# Jersey City Citi Bike network (also includes cross-river Manhattan edge stations)
# To use full Manhattan, set: LAT=(40.695,40.775), LON=(-74.025,-73.930)
LAT_MIN, LAT_MAX = 40.700, 40.755
LON_MIN, LON_MAX = -74.095, -74.015


# ── Download helpers ──────────────────────────────────────────────────────────

def _citibike_urls(year, month):
    """Return candidate URLs for a Citi Bike monthly data file."""
    tag = f'{year}{month:02d}'
    return [
        f'https://s3.amazonaws.com/tripdata/{tag}-citibike-tripdata.csv.zip',
        f'https://s3.amazonaws.com/tripdata/{tag}-citibike-tripdata.zip',
        f'https://s3.amazonaws.com/tripdata/JC-{tag}-citibike-tripdata.csv.zip',
    ]


def download_month(data_dir, year, month):
    """
    Download one month of Citi Bike trip data.
    Returns the path to the extracted CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)
    tag      = f'{year}{month:02d}'
    csv_path = os.path.join(data_dir, f'{tag}-citibike-tripdata.csv')

    if os.path.exists(csv_path):
        print(f"  {os.path.basename(csv_path)} already exists, skipping download")
        return csv_path

    for url in _citibike_urls(year, month):
        zip_path = os.path.join(data_dir, f'{tag}-citibike-tripdata.zip')
        try:
            print(f"  Downloading from {url} ...")
            with urllib.request.urlopen(url, timeout=60) as resp:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r    {pct:.0f}% ({downloaded/1e6:.0f} MB / {total/1e6:.0f} MB)",
                                  end='', flush=True)
            print()

            # Extract CSV from zip
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                if not csv_names:
                    raise ValueError(f"No CSV in zip file")
                # Extract the first (largest) CSV
                csv_name = sorted(csv_names, key=lambda n: zf.getinfo(n).file_size)[-1]
                print(f"  Extracting {csv_name}...")
                with zf.open(csv_name) as src, open(csv_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

            os.remove(zip_path)
            return csv_path

        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

    raise RuntimeError(
        f"Could not download {year}/{month:02d} data.\n"
        f"Please manually download to: {csv_path}\n"
        f"from: https://citibikenyc.com/system-data")


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _detect_format(header_cols):
    """Detect old vs new Citi Bike CSV column format."""
    header = set(c.strip().lower() for c in header_cols)
    if 'started_at' in header:
        return 'new'
    elif 'starttime' in header:
        return 'old'
    else:
        raise ValueError(f"Unknown Citi Bike CSV format. Columns: {header_cols[:6]}")


def load_trips_csv(csv_path, lat_min, lat_max, lon_min, lon_max):
    """
    Load trip data from a Citi Bike CSV, returning a filtered DataFrame-like
    structure as a dict of numpy arrays.

    Handles both old (pre-2021) and new (2021+) column formats.
    Returns only trips where both endpoints are inside the bounding box.
    """
    import csv
    from datetime import datetime

    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fmt = _detect_format(reader.fieldnames)

        if fmt == 'new':
            s_time_col = 'started_at'
            e_time_col = 'ended_at'
            s_id_col   = 'start_station_id'
            e_id_col   = 'end_station_id'
            s_lat_col  = 'start_lat'
            s_lng_col  = 'start_lng'
            e_lat_col  = 'end_lat'
            e_lng_col  = 'end_lng'
        else:
            s_time_col = 'starttime'
            e_time_col = 'stoptime'
            s_id_col   = 'start station id'
            e_id_col   = 'end station id'
            s_lat_col  = 'start station latitude'
            s_lng_col  = 'start station longitude'
            e_lat_col  = 'end station latitude'
            e_lng_col  = 'end station longitude'

        for row in reader:
            try:
                slat = float(row[s_lat_col])
                slng = float(row[s_lng_col])
                elat = float(row[e_lat_col])
                elng = float(row[e_lng_col])
            except (ValueError, KeyError):
                continue

            # Keep only trips whose start station is in bounding box
            if not (lat_min <= slat <= lat_max and lon_min <= slng <= lon_max):
                continue

            try:
                s_id = str(row[s_id_col]).strip()
                e_id = str(row[e_id_col]).strip()
            except KeyError:
                continue
            if not s_id or not e_id or s_id == 'nan' or e_id == 'nan':
                continue

            # Parse start time
            t_str = row[s_time_col].strip()
            try:
                for fmt_str in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
                                '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M'):
                    try:
                        t = datetime.strptime(t_str, fmt_str)
                        break
                    except ValueError:
                        continue
                else:
                    continue
            except Exception:
                continue

            rows.append({
                'hour':     t.hour,
                'dow':      t.weekday(),   # 0=Monday, 6=Sunday
                'date':     t.date(),
                'start_id': s_id,
                'end_id':   e_id,
                'start_lat': slat,
                'start_lng': slng,
                'end_lat':   elat,
                'end_lng':   elng,
            })

    print(f"  Parsed {len(rows):,} in-bounds trips from {os.path.basename(csv_path)}")
    return rows


# ── Station selection ─────────────────────────────────────────────────────────

def build_station_table(all_rows, n_stations=150):
    """
    Build a table of the most-active stations.
    Returns: stations dict {id: {lat, lng, activity}}
    """
    # Count activity per start station
    counts = {}
    coords = {}
    for row in all_rows:
        sid = row['start_id']
        counts[sid] = counts.get(sid, 0) + 1
        if sid not in coords:
            coords[sid] = (row['start_lat'], row['start_lng'])

    # Sort by activity, take top N
    sorted_ids = sorted(counts, key=lambda k: counts[k], reverse=True)
    selected   = sorted_ids[:n_stations]

    stations = {sid: {'lat': coords[sid][0], 'lng': coords[sid][1],
                       'activity': counts[sid]}
                for sid in selected}
    return stations


# ── Graph construction ────────────────────────────────────────────────────────

def build_trip_graph(all_rows, station_to_idx, N, min_trips=5):
    """
    Build trip-weighted adjacency matrix.
    W[i,j] = log(1 + #trips between i and j), symmetrized.
    """
    counts = {}
    for row in all_rows:
        i = station_to_idx.get(row['start_id'])
        j = station_to_idx.get(row['end_id'])
        if i is None or j is None or i == j:
            continue
        key = (min(i, j), max(i, j))
        counts[key] = counts.get(key, 0) + 1

    adj = np.zeros((N, N))
    for (i, j), cnt in counts.items():
        if cnt >= min_trips:
            w = np.log1p(cnt)
            adj[i, j] = w
            adj[j, i] = w

    # Fallback: add k-NN edges for any isolated stations
    connected = (adj.sum(axis=1) > 0)
    if not connected.all():
        n_isolated = int((~connected).sum())
        print(f"  Warning: {n_isolated} stations have no trip edges — adding geographic k-NN fallback")
        station_ids = sorted(station_to_idx, key=lambda k: station_to_idx[k])
        # positions are indexed by station_to_idx
        # We need lat/lng — stored separately; use GeodesicCache workaround
        # (positions must be passed in separately)

    return adj


def knn_fallback_adj(adj, positions, k=4):
    """Add k-NN geographic edges for nodes with no trip connections."""
    N = adj.shape[0]
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    for i in range(N):
        if adj[i].sum() == 0:
            nn = np.argsort(dists[i])[:k]
            for j in nn:
                if adj[i, j] == 0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    return adj


# ── Hourly distributions ──────────────────────────────────────────────────────

def compute_hourly_distributions(all_rows, station_to_idx, N, smoothing=0.5):
    """
    Compute hourly activity distributions: for each (date, hour), the
    fraction of total station activity at each station.

    Activity = trips starting + trips ending at each station per hour.

    Returns:
        snapshots: list of dicts with keys:
            dist   : (N,) float, normalized distribution
            hour   : int, hour of day (0-23)
            dow    : int, day of week (0=Mon)
            date   : date object
        date_hours: sorted list of (date, hour) keys present
    """
    from collections import defaultdict

    # Accumulate activity: {(date, hour): {station_idx: count}}
    activity = defaultdict(lambda: np.zeros(N))

    for row in all_rows:
        i = station_to_idx.get(row['start_id'])
        j = station_to_idx.get(row['end_id'])
        key = (row['date'], row['hour'])
        if i is not None:
            activity[key][i] += 1   # departure
        if j is not None:
            activity[key][j] += 1   # arrival

    snapshots = []
    for (date, hour), acts in sorted(activity.items()):
        total = acts.sum() + smoothing * N
        dist  = (acts + smoothing) / total
        snapshots.append({
            'dist': dist.astype(np.float32),
            'hour': hour,
            'dow':  date.weekday(),
            'date': str(date),
        })

    return snapshots


def compute_historical_mean(snapshots):
    """
    Compute mean distribution per (hour, dow) pair.
    Returns: (24, 7, N) array.
    """
    N = snapshots[0]['dist'].shape[0]
    acc   = np.zeros((24, 7, N))
    count = np.zeros((24, 7), dtype=int)
    for s in snapshots:
        h, d = s['hour'], s['dow']
        acc[h, d]   += s['dist']
        count[h, d] += 1
    count = np.maximum(count, 1)
    return acc / count[:, :, None]


# ── Positions (meters from centroid) ─────────────────────────────────────────

def latlon_to_meters(lats, lons):
    lat_c = lats.mean()
    x = (lons - lons.mean()) * 111320 * np.cos(np.radians(lat_c))
    y = (lats - lats.mean()) * 110540
    return np.stack([x, y], axis=1)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_station_map(stations_list, positions, adj, out_path):
    """Draw stations on a scatter map with trip edges."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Ex15a: Citi Bike NYC Station Graph', fontsize=12)

    N = len(stations_list)
    x, y = positions[:, 0], positions[:, 1]

    # Panel A: just stations
    ax = axes[0]
    acts = np.array([s['activity'] for s in stations_list])
    sc   = ax.scatter(x, y, c=acts, cmap='viridis', s=30, alpha=0.8,
                      norm=matplotlib.colors.LogNorm())
    plt.colorbar(sc, ax=ax, label='Trip activity (log scale)')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_title(f'A: {N} stations (coloured by activity)')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    # Panel B: trip graph edges (top 300 by weight)
    ax = axes[1]
    rows_e, cols_e = np.where(np.triu(adj > 0))
    weights = adj[rows_e, cols_e]
    order   = np.argsort(weights)[-300:]
    w_max   = weights[order].max()
    for k in order:
        i, j = rows_e[k], cols_e[k]
        alpha = float(weights[k] / w_max) * 0.6
        ax.plot([x[i], x[j]], [y[i], y[j]],
                '-', color='steelblue', lw=0.5, alpha=alpha)
    ax.scatter(x, y, s=15, c='black', zorder=3)
    n_edges = int((adj > 0).sum() // 2)
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_title(f'B: Trip graph ({n_edges} edges, top-300 shown)')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_distribution_examples(snapshots, hist_mean, positions, out_path):
    """Show a few hourly distributions and historical means."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Ex15a: Hourly Activity Distributions', fontsize=12)

    x, y = positions[:, 0], positions[:, 1]

    # Pick 4 representative snapshots
    pick_idx = np.linspace(0, len(snapshots) - 1, 4, dtype=int)
    hours_dow = [(snapshots[i]['hour'], snapshots[i]['dow']) for i in pick_idx]
    hour_names = ['12am Mon', '6am Tue', '12pm Wed', '6pm Thu',
                  '8am Fri', '5pm Fri', '10pm Sat', '3pm Sun']

    for col, idx in enumerate(pick_idx):
        snap = snapshots[idx]
        h, d = snap['hour'], snap['dow']
        vm   = max(snap['dist'].max(), hist_mean[h, d].max())

        ax = axes[0, col]
        sc = ax.scatter(x, y, c=snap['dist'], cmap='YlOrRd', s=25, vmin=0, vmax=vm)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Observed h={h} dow={d}", fontsize=8)

        ax = axes[1, col]
        sc = ax.scatter(x, y, c=hist_mean[h, d], cmap='YlOrRd', s=25, vmin=0, vmax=vm)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Historical mean h={h} dow={d}", fontsize=8)
        if col == 0:
            axes[0, col].set_ylabel('Observed', fontsize=9)
            axes[1, col].set_ylabel('Hist. mean', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex15a: Citi Bike data download and preprocessing')
    parser.add_argument('--year',        type=int,   default=2023)
    parser.add_argument('--months',      type=int,   nargs='+', default=[6, 7, 8])
    parser.add_argument('--n-stations',  type=int,   default=150,
                        help='Number of most-active stations to include')
    parser.add_argument('--min-trips',   type=int,   default=5,
                        help='Min trips for a trip edge in the graph')
    parser.add_argument('--data-dir',    type=str,   default=None,
                        help='Directory for downloaded trip CSVs '
                             '(default: data/citibike_raw)')
    parser.add_argument('--recompute',   action='store_true')
    args = parser.parse_args()

    raw_dir  = args.data_dir or os.path.join(DATA_DIR, 'citibike_raw')
    npz_path = os.path.join(DATA_DIR, 'ex15_bikeshare_data.npz')

    os.makedirs(raw_dir,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(npz_path) and not args.recompute:
        print(f"Data already exists at {npz_path}")
        print("Run with --recompute to regenerate.")
        _verify_and_report(npz_path)
        return

    print(f"=== Experiment 15a: Citi Bike Data Setup ===\n")
    print(f"Year: {args.year}, Months: {args.months}, Stations: {args.n_stations}")

    # ── Step 1: Download and load trip data ────────────────────────────────────
    all_rows = []
    for month in args.months:
        print(f"\nMonth {args.year}/{month:02d}:")
        csv_path = download_month(raw_dir, args.year, month)
        rows     = load_trips_csv(csv_path, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        all_rows.extend(rows)

    print(f"\nTotal in-bounds trips loaded: {len(all_rows):,}")

    # ── Step 2: Select top-N stations ─────────────────────────────────────────
    print(f"\nSelecting top {args.n_stations} stations by departure activity...")
    stations = build_station_table(all_rows, n_stations=args.n_stations)
    station_ids  = sorted(stations.keys())
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    N = len(station_ids)

    # Filter rows to only those starting at selected stations
    all_rows_filt = [r for r in all_rows if r['start_id'] in station_to_idx]
    print(f"  {N} stations, {len(all_rows_filt):,} trips with known start station")

    # ── Step 3: Build positions ────────────────────────────────────────────────
    lats = np.array([stations[sid]['lat'] for sid in station_ids])
    lons = np.array([stations[sid]['lng'] for sid in station_ids])
    positions = latlon_to_meters(lats, lons)   # (N, 2) meters

    # ── Step 4: Build trip graph ───────────────────────────────────────────────
    print(f"\nBuilding trip graph (min_trips={args.min_trips})...")
    adj = build_trip_graph(all_rows_filt, station_to_idx, N, min_trips=args.min_trips)
    adj = knn_fallback_adj(adj, positions, k=4)

    n_edges = int((adj > 0).sum() // 2)
    degrees = (adj > 0).sum(axis=1)
    print(f"  Edges: {n_edges}, mean degree: {degrees.mean():.1f}, "
          f"min: {degrees.min()}, max: {degrees.max()}")

    R = adj.copy().astype(float)
    np.fill_diagonal(R, -R.sum(axis=1))

    # ── Step 5: Compute hourly distributions ──────────────────────────────────
    print(f"\nComputing hourly activity distributions...")
    snapshots = compute_hourly_distributions(all_rows_filt, station_to_idx, N)
    print(f"  {len(snapshots)} hourly snapshots across {len(args.months)} months")

    # Train / test split: last 20% = test (at least 24h, at most 3 weeks)
    n_test  = max(24, min(24 * 21, len(snapshots) // 5))
    n_train = len(snapshots) - n_test
    train_snaps = snapshots[:n_train]
    test_snaps  = snapshots[n_train:]
    print(f"  Train: {len(train_snaps)} snapshots, Test: {len(test_snaps)} snapshots")

    # ── Step 6: Historical mean baseline ──────────────────────────────────────
    if not train_snaps:
        raise RuntimeError(
            f"Only {len(snapshots)} hourly snapshots found — not enough data.\n"
            f"Check that the bounds capture your data, or use more months.")
    print(f"\nComputing historical mean (from training snapshots)...")
    hist_mean = compute_historical_mean(train_snaps)   # (24, 7, N)

    # Evaluate historical mean on test set
    tvs_hist = []
    from graph_ot_fm import total_variation
    for s in test_snaps:
        h, d     = s['hour'], s['dow']
        tv       = total_variation(hist_mean[h, d], s['dist'])
        tvs_hist.append(tv)
    print(f"  Historical mean TV on test: {np.mean(tvs_hist):.4f} ± {np.std(tvs_hist):.4f}")

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    print(f"\nSaving data to {npz_path}...")
    station_names = np.array(station_ids)
    train_dists   = np.array([s['dist'] for s in train_snaps])
    train_hours   = np.array([s['hour'] for s in train_snaps], dtype=np.int32)
    train_dows    = np.array([s['dow']  for s in train_snaps], dtype=np.int32)
    train_dates   = np.array([s['date'] for s in train_snaps])
    test_dists    = np.array([s['dist'] for s in test_snaps])
    test_hours    = np.array([s['hour'] for s in test_snaps], dtype=np.int32)
    test_dows     = np.array([s['dow']  for s in test_snaps], dtype=np.int32)
    test_dates    = np.array([s['date'] for s in test_snaps])

    np.savez(npz_path,
             # Graph
             R=R, adj=adj, positions=positions, lats=lats, lons=lons,
             station_names=station_names,
             # Distributions
             train_dists=train_dists, train_hours=train_hours,
             train_dows=train_dows,   train_dates=train_dates,
             test_dists=test_dists,   test_hours=test_hours,
             test_dows=test_dows,     test_dates=test_dates,
             # Historical mean baseline
             hist_mean=hist_mean,
             # Metadata
             n_stations=np.array(N),
             n_train=np.array(len(train_snaps)),
             n_test=np.array(len(test_snaps)))
    print(f"  Saved.")

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    stations_list = [stations[sid] for sid in station_ids]
    plot_station_map(
        stations_list, positions, adj,
        os.path.join(HERE, 'ex15a_station_map.png'))
    plot_distribution_examples(
        train_snaps, hist_mean, positions,
        os.path.join(HERE, 'ex15a_distributions.png'))

    print(f"\n=== Setup complete ===")
    print(f"  {N} stations, {n_edges} edges")
    print(f"  {len(train_snaps)} train + {len(test_snaps)} test hourly snapshots")
    print(f"  Historical mean TV: {np.mean(tvs_hist):.4f}")
    print(f"\nNext: uv run experiments/ex15b_bikeshare_train.py")


def _verify_and_report(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    N  = int(data['n_stations'])
    nt = int(data['n_train'])
    nv = int(data['n_test'])
    print(f"  Stations: {N}, Train snapshots: {nt}, Test: {nv}")
    R  = data['R']
    adj = data['adj']
    n_edges = int((adj > 0).sum() // 2)
    print(f"  Graph: {N} nodes, {n_edges} edges")


if __name__ == '__main__':
    main()
