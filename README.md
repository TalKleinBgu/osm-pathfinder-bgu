# OSM Pathfinding System for Ben Gurion University

This project implements a comprehensive pathfinding system for the Ben Gurion University area using OpenStreetMap (OSM) data. The system calculates optimal routes from the Ben Gurion University gate (שער העלייה - Aliya Gate) to various points on the map using four different routing algorithms.

## Features

### Four Routing Algorithms

1. **Shortest Distance** 
   - **Objective**: Minimize total walking distance
   - **Cost Function**: `cost(u,v) = length(u,v)` (in meters)
   - **Heuristic**: Great circle distance (straight-line distance)
   - **Use Case**: When you want the most direct route

2. **Few Traffic Lights/Crossings**
   - **Objective**: Minimize waiting time at intersections
   - **Cost Function**: `cost = length + α × crossings + β × traffic_signals`
   - **Parameters**: α = 15m, β = 25m (equivalent time penalties)
   - **Features**: 
     - Identifies traffic signals: `highway=traffic_signals`
     - Identifies crossings: `highway=crossing` or `crossing=marked/uncontrolled/traffic_signs`
   - **Use Case**: When you want to avoid stopping at lights

3. **Safest Route**
   - **Objective**: Maximize pedestrian safety
   - **Cost Function**: `cost = length + safety_penalties`
   - **Safety Factors**:
     - **Lighting**: `lit=yes` (+), `lit=no` (−)
     - **Road Type**: Main streets (safer) vs alleys/service roads (less safe)
     - **Sidewalks**: `sidewalk=both/left/right` (+) vs no sidewalk (−)
     - **Crossings**: `crossing=traffic_signals` (+) vs `crossing=uncontrolled` (−)
   - **Day/Night Profiles**: Higher penalties at night
   - **Use Case**: When safety is the primary concern

4. **Fastest Walking Time**
   - **Objective**: Minimize total walking time
   - **Base Speed**: 1.3 m/s (4.7 km/h)
   - **Time Corrections**:
     - **Surface**: Asphalt (fast) vs gravel/sand (slow)
     - **Steps**: `highway=steps` → 1.5-2.0× time penalty
     - **Traffic Signals**: +10-30 seconds wait time
   - **Heuristic**: `h_time(n,t) = straight_line_distance(n,t) / v_max` (v_max = 1.6 m/s)
   - **Use Case**: When time is critical

## Technical Implementation

### Algorithm: Bidirectional A*

- **Forward Search**: From Ben Gurion gate to target
- **Backward Search**: From target to Ben Gurion gate  
- **Heuristics**: Admissible and consistent for optimal results
- **Graph Structure**: Directed graph supporting different costs per direction

### Key Technical Features

- **Cost Functions**: All non-negative to maintain A* optimality
- **Symmetric Penalties**: Applied to edges (not just nodes) for bidirectional compatibility
- **Additive Heuristics**: Great circle distance remains lower bound even with penalties
- **Multi-objective Support**: Lexicographic ordering or weighted scalarization

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- OSM data file (`map.osm`) in the project directory

### Quick Start

1. **Install dependencies and run complete analysis**:
   ```bash
   python run_analysis.py
   ```

2. **Manual installation**:
   ```bash
   pip install -r requirements.txt
   python osm_pathfinder.py
   python visualize_results.py
   ```

### Files Generated

- `pathfinding_results.json` - Raw pathfinding results
- `pathfinding_map.html` - Interactive map visualization
- `algorithm_comparison.png` - Performance comparison charts
- `pathfinding_summary_stats.csv` - Statistical analysis

## Project Structure

```
├── map.osm                    # OpenStreetMap data file
├── osm_pathfinder.py         # Main pathfinding implementation
├── visualize_results.py      # Visualization and analysis
├── run_analysis.py           # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## OSM Data Processing

### Node Processing
- Extracts coordinates (lat, lon) for all nodes
- Parses tags for traffic signals, crossings, etc.
- **Ben Gurion Gate**: Node ID `1486123013` (שער העלייה)

### Way Processing
- Filters for pedestrian-accessible ways
- **Walkable Types**: footway, pedestrian, path, steps, residential, tertiary, etc.
- **Excluded**: motorways, trunk roads, `foot=no` ways
- Handles bidirectional vs one-way (`oneway=yes`) routing

### Graph Construction
- Creates adjacency list representation
- Calculates haversine distances between connected nodes
- Stores way metadata for cost calculations

## Algorithm Details

### Cost Calculations

#### Shortest Distance
```python
cost = edge_distance  # meters
```

#### Few Traffic Lights
```python
cost = edge_distance + 25 * traffic_signals + 15 * crossings
```

#### Safest Route
```python
cost = edge_distance + penalties
penalties = lighting_penalty + road_type_penalty + sidewalk_penalty + tunnel_bridge_penalty
# Day penalties: 20-30m, Night penalties: 80-120m
```

#### Fastest Time
```python
time = distance / base_speed * surface_factor * steps_factor + signal_wait_time
base_speed = 1.3 m/s
surface_factor = 1.0 (asphalt) to 1.3 (rough terrain)
steps_factor = 1.8 for stairs
signal_wait_time = 20 seconds
```

### Heuristics

All heuristics use great circle distance to ensure admissibility:

```python
h_distance(n, target) = haversine_distance(n, target)
h_time(n, target) = haversine_distance(n, target) / max_walking_speed
```

## Visualization Features

### Interactive Map (`pathfinding_map.html`)
- **Ben Gurion Gate**: Red university marker
- **Target Points**: Blue flag markers  
- **Path Colors**:
  - 🔵 Blue: Shortest distance
  - 🟣 Purple: Few traffic lights
  - 🟠 Orange: Safest (day)
  - 🔴 Red: Safest (night)
  - 🟢 Green: Fastest time

### Performance Analysis (`algorithm_comparison.png`)
- Box plots comparing all algorithms
- Metrics: distance, time, traffic signals, crossings, safety penalties, detour factor

### Statistical Summary (`pathfinding_summary_stats.csv`)
- Mean, standard deviation, min/max for all metrics
- Algorithm performance comparison

## Customization

### Adjusting Parameters

Edit `osm_pathfinder.py` to modify:

```python
# Traffic light penalties (meters equivalent)
traffic_signal_penalty = 25
crossing_penalty = 15

# Safety penalties
lighting_penalty_day = 30
lighting_penalty_night = 120
road_type_penalty_day = 20
road_type_penalty_night = 80

# Walking parameters
base_walking_speed = 1.3  # m/s
max_walking_speed = 1.6   # m/s for heuristic
traffic_signal_wait = 20  # seconds
```

### Adding New Algorithms

1. Create new cost function in `OSMPathfinder` class
2. Add corresponding heuristic function
3. Update `find_path_astar()` method
4. Add to visualization color scheme

## Performance Considerations

- **Graph Size**: ~50 target nodes for demonstration (configurable)
- **Search Space**: Limited to 500m radius from Ben Gurion gate
- **Time Complexity**: O(E log V) per path with A*
- **Memory Usage**: Stores full OSM graph in memory

## Ben Gurion University Integration

### Gate Location
- **Name**: שער העלייה (Aliya Gate)  
- **Coordinates**: 31.2612934°N, 34.8011614°E
- **OSM Node ID**: 1486123013
- **Tags**: `barrier=gate`, `entrance=main`

### Campus Area Coverage
- **Bounds**: 31.2517°N to 31.2680°N, 34.7877°E to 34.8056°E
- **Coverage**: University campus and surrounding Beer Sheva area
- **Features**: Academic buildings, dormitories, medical facilities

## Future Enhancements

1. **Real-time Data**: Integration with traffic APIs
2. **Accessibility**: Wheelchair-accessible route options
3. **Weather Adaptation**: Different costs for weather conditions
4. **Public Transport**: Integration with bus/train connections
5. **User Preferences**: Personalized weight configurations
6. **Mobile App**: React Native or Flutter implementation

## Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## License

This project is open source. OSM data is licensed under the Open Database License (ODbL).

## References

- [OpenStreetMap Wiki - Routing](https://wiki.openstreetmap.org/wiki/Routing)
- [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [OSM Highway Tags](https://wiki.openstreetmap.org/wiki/Key:highway)
- [Pedestrian Routing](https://wiki.openstreetmap.org/wiki/Routing/Pedestrian)
