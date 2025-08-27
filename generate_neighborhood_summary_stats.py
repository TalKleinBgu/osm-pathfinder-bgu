#!/usr/bin/env python3
"""
Generate comprehensive pathfinding summary statistics CSV with neighborhood column.
Reads all JSON results from different neighborhoods and creates aggregate statistics.
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path
import re


def extract_algorithm_canonical_name(algorithm: str) -> str:
    """Normalize algorithm names to match existing CSV format."""
    if not algorithm:
        return "unknown"
    
    alg = algorithm.lower().strip()
    
    # Map to canonical names from existing CSV
    if "shortest" in alg:
        return "shortest path"
    elif "few" in alg and ("traffic" in alg or "light" in alg):
        return "few_traffic_lights path"
    elif "safest" in alg:
        if "night" in alg:
            return "safest path (night profile)"
        else:
            return "safest path"
    elif "fastest" in alg:
        return "fastest path"
    else:
        return f"{alg} (day profile)"


def calculate_detour_factor(distance_meters: float, straight_line_dist: float) -> float:
    """Calculate detour factor as ratio of actual distance to straight-line distance."""
    if straight_line_dist <= 0:
        return 1.0
    return distance_meters / straight_line_dist


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def extract_neighborhood_from_path(file_path: str) -> str:
    """Extract neighborhood name from file path."""
    path_parts = Path(file_path).parts
    
    # Look for neighborhood folder in path
    for part in path_parts:
        if 'neighborhood' in part.lower():
            return part
    
    # Fallback: use parent directory name
    return Path(file_path).parent.name


def process_json_file(file_path: str, target_coords: Tuple[float, float]) -> List[Dict]:
    """Process a single JSON file and extract path statistics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    neighborhood = extract_neighborhood_from_path(file_path)
    results = []
    
    # Extract target coordinates if available
    if '_target' in data:
        target_info = data['_target']
        if 'coordinates' in target_info:
            target_coords = (
                target_info['coordinates']['lat'],
                target_info['coordinates']['lon']
            )
    
    # Process each building
    for building_id, building_data in data.items():
        if building_id.startswith('_'):
            continue
            
        if 'paths' not in building_data:
            continue
            
        # Get building start coordinates from first path
        building_coords = None
        paths = building_data.get('paths', [])
        if paths and 'path_coordinates' in paths[0]:
            first_coord = paths[0]['path_coordinates'][0]
            if isinstance(first_coord, dict):
                building_coords = (first_coord['lat'], first_coord['lon'])
            else:
                building_coords = (first_coord[0], first_coord[1])
        
        # Calculate straight-line distance for detour factor
        straight_line_dist = 0.0
        if building_coords and target_coords:
            straight_line_dist = haversine_distance(
                building_coords[0], building_coords[1],
                target_coords[0], target_coords[1]
            )
        
        # Process each path/algorithm
        for path_data in paths:
            algorithm = extract_algorithm_canonical_name(path_data.get('algorithm', ''))
            
            # Extract all available metrics
            distance_m = float(path_data.get('distance_meters', 0.0))
            time_s = float(path_data.get('time_seconds', 0.0))
            signals = int(path_data.get('num_traffic_signals', 0))
            crossings = int(path_data.get('num_crossings', 0))
            safety_penalties = float(path_data.get('safety_penalties', 0.0))
            total_cost = float(path_data.get('total_cost', 0.0))
            
            # Calculate additional derived metrics
            detour_factor = calculate_detour_factor(distance_m, straight_line_dist)
            path_coordinates = path_data.get('path_coordinates', [])
            num_path_nodes = len(path_coordinates)
            
            # Calculate average walking speed (m/s)
            avg_walking_speed = distance_m / time_s if time_s > 0 else 0.0
            
            # Calculate time per distance ratio (seconds per meter)
            time_per_meter = time_s / distance_m if distance_m > 0 else 0.0
            
            # Calculate signal density (signals per 100m)
            signal_density_per_100m = (signals / distance_m * 100) if distance_m > 0 else 0.0
            
            # Calculate crossing density (crossings per 100m)
            crossing_density_per_100m = (crossings / distance_m * 100) if distance_m > 0 else 0.0
            
            # Calculate safety penalty ratio (penalties / distance)
            safety_penalty_ratio = safety_penalties / distance_m if distance_m > 0 else 0.0
            
            # Calculate cost efficiency (total_cost / distance)
            cost_efficiency = total_cost / distance_m if distance_m > 0 else 0.0
            
            results.append({
                'neighborhood': neighborhood,
                'building_id': building_id,
                'algorithm': algorithm,
                'distance_meters': distance_m,
                'time_seconds': time_s,
                'time_minutes': time_s / 60.0,
                'num_traffic_signals': signals,
                'num_crossings': crossings,
                'safety_penalties': safety_penalties,
                'total_cost': total_cost,
                'detour_factor': detour_factor,
                'straight_line_distance': straight_line_dist,
                'num_path_nodes': num_path_nodes,
                'avg_walking_speed_ms': avg_walking_speed,
                'avg_walking_speed_kmh': avg_walking_speed * 3.6,
                'time_per_meter': time_per_meter,
                'signal_density_per_100m': signal_density_per_100m,
                'crossing_density_per_100m': crossing_density_per_100m,
                'safety_penalty_ratio': safety_penalty_ratio,
                'cost_efficiency': cost_efficiency
            })
    
    return results


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics grouped by neighborhood and algorithm."""
    
    # Group by neighborhood and algorithm
    grouped = df.groupby(['neighborhood', 'algorithm'])
    
    summary_stats = []
    
    for (neighborhood, algorithm), group in grouped:
        if len(group) == 0:
            continue
            
        stats = {
            'neighborhood': neighborhood,
            'algorithm': algorithm,
            'count': len(group),
            
            # Distance metrics
            'distance_meters_mean': group['distance_meters'].mean(),
            'distance_meters_std': group['distance_meters'].std() if len(group) > 1 else np.nan,
            'distance_meters_min': group['distance_meters'].min(),
            'distance_meters_max': group['distance_meters'].max(),
            
            # Time metrics
            'time_seconds_mean': group['time_seconds'].mean(),
            'time_seconds_std': group['time_seconds'].std() if len(group) > 1 else np.nan,
            'time_seconds_min': group['time_seconds'].min(),
            'time_seconds_max': group['time_seconds'].max(),
            'time_minutes_mean': group['time_minutes'].mean(),
            
            # Infrastructure metrics
            'traffic_signals_mean': group['num_traffic_signals'].mean(),
            'traffic_signals_sum': group['num_traffic_signals'].sum(),
            'traffic_signals_max': group['num_traffic_signals'].max(),
            'crossings_mean': group['num_crossings'].mean(),
            'crossings_sum': group['num_crossings'].sum(),
            'crossings_max': group['num_crossings'].max(),
            
            # Safety and cost metrics
            'safety_penalties_mean': group['safety_penalties'].mean(),
            'safety_penalties_std': group['safety_penalties'].std() if len(group) > 1 else np.nan,
            'safety_penalties_max': group['safety_penalties'].max(),
            'total_cost_mean': group['total_cost'].mean(),
            'total_cost_std': group['total_cost'].std() if len(group) > 1 else np.nan,
            
            # Route quality metrics
            'detour_factor_mean': group['detour_factor'].mean(),
            'detour_factor_std': group['detour_factor'].std() if len(group) > 1 else np.nan,
            'detour_factor_max': group['detour_factor'].max(),
            'straight_line_distance_mean': group['straight_line_distance'].mean(),
            
            # Path complexity metrics
            'num_path_nodes_mean': group['num_path_nodes'].mean(),
            'num_path_nodes_max': group['num_path_nodes'].max(),
            
            # Speed and efficiency metrics
            'avg_walking_speed_ms_mean': group['avg_walking_speed_ms'].mean(),
            'avg_walking_speed_kmh_mean': group['avg_walking_speed_kmh'].mean(),
            'time_per_meter_mean': group['time_per_meter'].mean(),
            'cost_efficiency_mean': group['cost_efficiency'].mean(),
            
            # Density metrics (per 100m)
            'signal_density_per_100m_mean': group['signal_density_per_100m'].mean(),
            'signal_density_per_100m_max': group['signal_density_per_100m'].max(),
            'crossing_density_per_100m_mean': group['crossing_density_per_100m'].mean(),
            'crossing_density_per_100m_max': group['crossing_density_per_100m'].max(),
            'safety_penalty_ratio_mean': group['safety_penalty_ratio'].mean()
        }
        
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)


def main():
    """Main function to process all results and generate summary statistics."""
    
    results_base_dir = "results"
    output_file = "neighborhood_pathfinding_summary_stats.csv"
    
    if not os.path.exists(results_base_dir):
        print(f"❌ Results directory '{results_base_dir}' not found.")
        return
    
    print("🔍 Scanning for JSON result files...")
    
    # Find all JSON result files
    json_files = []
    target_coords = (31.2612934, 34.8011614)  # Default BGU gate coordinates
    
    for root, dirs, files in os.walk(results_base_dir):
        #skip for to_135310103 dir
        if 'to_135310103' in root:
            continue
        for file in files:
            if file.endswith('.json') and ('building_' in file or 'all_buildings' in file):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"❌ No JSON result files found in '{results_base_dir}'")
        return
    
    print(f"📁 Found {len(json_files)} JSON files")
    
    # Process all files
    all_data = []
    processed_neighborhoods = set()
    
    for json_file in json_files:
        print(f"   Processing: {os.path.relpath(json_file)}")
        
        # Skip individual building files if we have the master file for the same neighborhood
        if 'building_' in os.path.basename(json_file) and not 'all_buildings' in os.path.basename(json_file):
            continue
            
        file_data = process_json_file(json_file, target_coords)
        all_data.extend(file_data)
        
        if file_data:
            neighborhood = file_data[0]['neighborhood']
            processed_neighborhoods.add(neighborhood)
    
    if not all_data:
        print("❌ No valid data found in JSON files")
        return
    
    print(f"✅ Processed {len(all_data)} path records from {len(processed_neighborhoods)} neighborhoods")
    print(f"   Neighborhoods: {', '.join(sorted(processed_neighborhoods))}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Generate summary statistics
    print("📊 Generating summary statistics...")
    summary_df = generate_summary_statistics(df)
    
    # Sort by neighborhood and algorithm for better readability
    summary_df = summary_df.sort_values(['neighborhood', 'algorithm'])
    
    # Round numeric columns for cleaner output
    numeric_columns = [col for col in summary_df.columns if 'mean' in col or 'std' in col or 'min' in col or 'max' in col]
    for col in numeric_columns:
        summary_df[col] = summary_df[col].round(2)
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"📄 Summary statistics saved → {output_file}")
    
    # Display overview
    print("\n📈 Overview:")
    print(f"   Total neighborhoods: {summary_df['neighborhood'].nunique()}")
    print(f"   Total algorithms: {summary_df['algorithm'].nunique()}")
    print(f"   Total records: {len(summary_df)}")
    
    print("\n🔍 Algorithms found:")
    for alg in sorted(summary_df['algorithm'].unique()):
        count = summary_df[summary_df['algorithm'] == alg]['count'].sum()
        print(f"   • {alg}: {count} paths")
    
    print("\n🏘️ Neighborhoods found:")
    for neighborhood in sorted(summary_df['neighborhood'].unique()):
        count = summary_df[summary_df['neighborhood'] == neighborhood]['count'].sum()
        print(f"   • {neighborhood}: {count} paths")
    
    # Save detailed data as well for further analysis
    detailed_output = "neighborhood_pathfinding_detailed_data.csv"
    df.to_csv(detailed_output, index=False)
    print(f"📄 Detailed data saved → {detailed_output}")


if __name__ == "__main__":
    main()
