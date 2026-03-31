# Experiment 15: Bike Sharing Density Estimation

## Why This Is a Perfect Fit

Bikes in a sharing system are literal mass on a graph. At any moment,
each station holds some fraction of the total bikes. This fraction
evolves over time as people ride bikes between stations. The dynamics
are mass transport on the station graph — exactly what our framework
models.

The tasks:
1. **Interpolation:** Given bike counts at a subset of stations,
   reconstruct the full distribution (sparse sensor problem)
2. **Forecasting:** Given the current distribution, predict the
   distribution in 1 hour (forward transport problem)
3. **Posterior sampling:** Multiple plausible distributions given
   sparse observations, with uncertainty

## Data

### Citi Bike NYC

- Public data: https://citibikenyc.com/system-data
- Trip records: start station, end station, start time, end time
- Available monthly since 2013
- ~20M+ trips per year
- ~800 active stations in NYC (Manhattan + Brooklyn + Queens)

From trip records, we can reconstruct the bike distribution at any
time by tracking all bikes:

```
For each time snapshot t:
    For each station s:
        bikes[s] = initial_bikes[s]
                   + arrivals_before_t[s]
                   - departures_before_t[s]
```

Alternatively, Citi Bike publishes real-time station status feeds
(JSON), which directly give bike counts. Historical snapshots are
available from third-party archives.

### Simpler alternative: Station status snapshots

The General Bikeshare Feed Specification (GBFS) provides real-time
station status. Historical GBFS snapshots are archived at:
- https://github.com/BuzzFeedNews/2015-06-bikeshare-data
- Various civic data portals

Each snapshot gives: station_id, num_bikes_available, num_docks_available.

### Data processing pipeline

```python
def load_citibike_data(data_dir, year=2023, months=[6, 7, 8]):
    """
    Load Citi Bike trip data and compute hourly station occupancy.
    
    Returns:
        stations: DataFrame with station_id, lat, lon, capacity
        snapshots: list of (timestamp, occupancy_array) where
                   occupancy_array[i] = bikes at station i / total bikes
    """
    # 1. Load trip records
    trips = pd.concat([
        pd.read_csv(f'{data_dir}/{year}{m:02d}-citibike-tripdata.csv')
        for m in months
    ])
    
    # 2. Get unique stations and their positions
    stations = trips.groupby('start_station_id').agg(
        lat=('start_lat', 'mean'),
        lon=('start_lng', 'mean'),
    ).reset_index()
    
    # 3. For each hour, compute net flow to get occupancy
    # (or use station status snapshots directly if available)
    
    # 4. Normalize to distribution: bikes[s] / total_bikes
    
    return stations, snapshots
```

## Graph Construction

### Option A: k-nearest-neighbor on geographic coordinates

```python
from scipy.spatial.distance import cdist

def build_station_graph(stations, k=6):
    """
    Connect each station to its k nearest geographic neighbors.
    Edge weight = exp(-distance / median_distance).
    """
    coords = stations[['lat', 'lon']].values
    # Convert to approximate meters using Haversine or simple projection
    lat_center = coords[:, 0].mean()
    x = (coords[:, 1] - coords[:, 1].mean()) * 111320 * np.cos(np.radians(lat_center))
    y = (coords[:, 0] - coords[:, 0].mean()) * 110540
    positions = np.stack([x, y], axis=1)
    
    dists = cdist(positions, positions)
    adj = np.zeros((len(stations), len(stations)))
    for i in range(len(stations)):
        neighbors = np.argsort(dists[i])[1:k+1]
        for j in neighbors:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    R = adj.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, positions
```

### Option B: Trip-based adjacency

Connect stations that have direct trips between them. Weight by
trip frequency. This captures actual travel patterns, not just
geography.

```python
def build_trip_graph(trips, stations, min_trips=10):
    """
    Connect stations with at least min_trips direct trips.
    Weight = log(1 + trip_count).
    """
    trip_counts = trips.groupby(
        ['start_station_id', 'end_station_id']).size().reset_index(name='count')
    trip_counts = trip_counts[trip_counts['count'] >= min_trips]
    
    # Build adjacency
    adj = np.zeros((len(stations), len(stations)))
    for _, row in trip_counts.iterrows():
        i = station_to_idx[row['start_station_id']]
        j = station_to_idx[row['end_station_id']]
        adj[i, j] = np.log1p(row['count'])
        adj[j, i] = np.log1p(row['count'])  # symmetrize
    
    R = adj.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R
```

Option B is more interesting — it captures the actual transport
network, not just spatial proximity. Two stations across a bridge
might be strongly connected by trips but geographically distant.

### Scale considerations

~800 stations is too large for our exact OT solver (800×800 LP).
Options:
1. **Subsample:** Use Manhattan only (~400 stations) or a smaller
   neighborhood (~100-200 stations). Still larger than previous
   experiments.
2. **Cluster:** Group nearby stations into ~100 super-stations.
   Each super-station's bike count is the sum of its members.
3. **Use the GNN directly without exact solver:** Train on smaller
   subgraphs (random subsets of ~100 stations), test on the full
   network. Topology generalization should help.

Recommendation: Start with a neighborhood (~100-150 stations in
Manhattan below 60th St) to match the scale of our cortical mesh
experiments. Scale up later if it works.

## Tasks

### Task 1: Spatial interpolation (sparse observation)

Given bike counts at a random subset of stations (e.g., 30% observed),
reconstruct the full distribution.

This is directly analogous to Ex13 (sparse sensors on cube):
- Observed stations = sensor nodes
- Unobserved stations = hidden nodes
- Context: observed counts at observed stations + station mask
- FiLM: global summary of observed counts
- Flow: Dirichlet start → target distribution, conditioned on observations

Training data: take complete hourly snapshots, randomly mask 70% of
stations, train to reconstruct.

Baselines:
- **k-NN interpolation:** Fill each unobserved station with weighted
  average of observed neighbors.
- **Kriging / Gaussian process:** Spatial interpolation using geographic
  coordinates.
- **Graph Laplacian smoothing:** Harmonic extension from observed to
  unobserved (same as Ex12).
- **Historical mean:** Average occupancy for that station at that hour
  of day (strong baseline — bike patterns are very regular).

### Task 2: Temporal forecasting (forward prediction)

Given the current bike distribution, predict the distribution in 1 hour.

This is a forward transport problem — the easiest case for our framework:
- Input: current distribution μ_t
- Output: future distribution μ_{t+1h}
- Context: time of day, day of week (as FiLM conditioning)
- Flow: μ_t → μ_{t+1h} via learned transport

Training data: consecutive hourly snapshots as (source, target) pairs.

Baselines:
- **Persistence:** Predict μ_{t+1h} = μ_t (no change).
- **Historical mean:** Average distribution at hour t+1.
- **Linear extrapolation:** μ_{t+1h} = μ_t + (μ_t - μ_{t-1h}).
- **GNN regression:** Direct prediction with same architecture.

### Task 3: Posterior sampling

For Task 1 (interpolation): generate K=20 posterior samples from
Dirichlet starts. Show uncertainty map on the NYC station map.

For Task 2 (forecasting): generate K=20 posterior samples showing
different plausible future distributions. "If commuters go to
Midtown, bikes accumulate there; if they go to FiDi, bikes go there."

## Visualization

THE figure for the paper:

**A map of Manhattan** with Citi Bike stations as dots. Color = bike
density (dark = many bikes, light = few). Show:
1. True distribution (full observation)
2. Observed stations (sparse, 30% visible)
3. Reconstructed distribution (our model)
4. Uncertainty map (posterior std)
5. One or two alternative posterior samples

This is immediately intuitive. No domain expertise needed.

For forecasting: show current distribution → predicted distribution
→ actual future distribution, on the map. The flow visualization
(intermediate time steps on the map) would show bikes moving from
residential areas to work areas during morning commute.

## Implementation Plan

### Phase 1: Data pipeline (1 day)

1. Download Citi Bike trip data (or station status snapshots)
2. Select Manhattan subset (~100-150 stations)
3. Compute hourly bike distributions
4. Build station graph (trip-based adjacency)
5. Save processed data to .npz

### Phase 2: Interpolation experiment (1-2 days)

1. Split snapshots into train/test (e.g., June-July train, August test)
2. Build dataset: mask random stations, train flow to reconstruct
3. Train FiLM model (same architecture as Ex13)
4. Evaluate against baselines
5. Posterior sampling with Dirichlet starts

### Phase 3: Forecasting experiment (1-2 days)

1. Build dataset: consecutive hourly pairs
2. Train flow model with time-of-day conditioning
3. Evaluate against baselines
4. Posterior sampling showing future scenarios

### Phase 4: Visualization (1 day)

1. Plot on real NYC map using folium or matplotlib+basemap
2. Generate the "aha" figure

## Expected Outcome

The interpolation task should work well — it's structurally identical
to Ex13 (sparse sensors) which gave strong results. The trip-based
graph captures actual transport patterns, which should give our model
an advantage over spatial-only baselines (kriging, k-NN).

The forecasting task is the more novel application. If the model learns
to transport bike distributions along the trip graph according to
time-of-day patterns, it demonstrates genuine forward transport on
a real network — the core capability of the framework.

The posterior sampling on a real map is the visual highlight. Showing
uncertainty in bike availability — "we're confident about Midtown
stations but uncertain about the Upper East Side" — is immediately
understandable and practically useful for bike rebalancing operations.

## Dependencies

```
pandas
folium (or basemap/cartopy for static maps)
requests (for downloading data)
```

## CLI

```python
parser.add_argument('--city', type=str, default='nyc',
                    choices=['nyc', 'chicago', 'london'])
parser.add_argument('--n-stations', type=int, default=150,
                    help='Number of stations to include')
parser.add_argument('--task', type=str, default='interpolation',
                    choices=['interpolation', 'forecasting', 'both'])
parser.add_argument('--obs-fraction', type=float, default=0.3,
                    help='Fraction of stations observed (interpolation)')
parser.add_argument('--forecast-horizon', type=int, default=1,
                    help='Hours ahead to predict (forecasting)')
parser.add_argument('--mode', type=str, default='posterior',
                    choices=['point_estimate', 'posterior'])
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--regenerate', action='store_true')
```
