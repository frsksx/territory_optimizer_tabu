import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
import random
from joblib import Parallel, delayed
import json
import os
import time
import sys
from sklearn.cluster import KMeans  # fallback
import spopt.region  # AZP

# --- CONFIGURATION ---
INPUT_CSV = "subterritory_data.csv"
OUTPUT_MAP_BASE = 'territory_map_solution_'
OUTPUT_CSV_BASE = 'territory_assignments_solution_'

TARGET_K_TERRITORIES = 5
MAX_BOX_FACTOR = 2.5
MIN_AVG_INTERNAL_NEIGHBORS = 1.5
TABU_TENURE = 10
MAX_ITERATIONS = 200

N_START_ATTEMPTS = 10
NUM_TOP_MAPS = 5
NUM_TOP_CSVS = 5

COL_SUBTERRITORY_ID = 'subterritory_name'
COL_SALES = 'value'
COL_MANUAL_ASSIGNMENT = 'territory'


# --- HELPER FUNCTIONS ---
def parse_coordinates(coord_string):
    coord_list = json.loads(coord_string)
    if isinstance(coord_list[0][0][0], list):
        polygons = [Polygon([tuple(p) for p in poly[0]]) for poly in coord_list]
        return MultiPolygon(polygons)
    else:
        return Polygon([tuple(p) for p in coord_list[0]])


def build_contiguity_graph(gdf):
    w = nx.Graph()
    gdf = gdf.reset_index(drop=True)
    for idx in gdf.index:
        w.add_node(idx)

    sindex = gdf.sindex
    for idx, geom in gdf.geometry.items():
        possible_neighbors = list(sindex.intersection(geom.bounds))
        for nb_idx in possible_neighbors:
            if idx >= nb_idx:
                continue
            if geom.intersects(gdf.geometry[nb_idx]) and not geom.equals(gdf.geometry[nb_idx]):
                w.add_edge(idx, nb_idx)
    return w


def compute_territory_metrics(gdf, assignments):
    df = gdf.copy()
    df['territory'] = assignments
    metrics = {}
    for terr in set(assignments):
        sub_idxs = df.index[df['territory'] == terr].tolist()
        sub_geoms = df.loc[sub_idxs].geometry
        merged_geom = sub_geoms.union_all()  # updated to avoid deprecation
        bounds = merged_geom.bounds
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        total_sales = df.loc[sub_idxs, COL_SALES].sum()
        metrics[terr] = {
            "width": width,
            "height": height,
            "total_sales": total_sales,
            "sub_idxs": sub_idxs
        }
    return metrics


def avg_internal_neighbors(assignments, graph):
    territories = set(assignments)
    avg_dict = {}
    for t in territories:
        nodes = [i for i, a in enumerate(assignments) if a == t]
        internal_counts = []
        for n in nodes:
            neighbors = list(graph.neighbors(n))
            internal = sum(1 for nb in neighbors if assignments[nb] == t)
            internal_counts.append(internal)
        avg_dict[t] = np.mean(internal_counts) if internal_counts else 0
    return avg_dict


def is_feasible(assignments, gdf, graph, max_box_side, min_avg_neighbors):
    metrics = compute_territory_metrics(gdf, assignments)
    avg_neighbors = avg_internal_neighbors(assignments, graph)
    for t, m in metrics.items():
        if m['width'] > max_box_side or m['height'] > max_box_side:
            return False
        if avg_neighbors[t] < min_avg_neighbors:
            return False
    return True


def objective(assignments, gdf):
    metrics = compute_territory_metrics(gdf, assignments)
    sales = [m['total_sales'] for m in metrics.values()]
    return np.std(sales)


def manual_initial_solution(gdf, k, seed):
    if COL_MANUAL_ASSIGNMENT in gdf.columns:
        raw_labels = gdf[COL_MANUAL_ASSIGNMENT].factorize()[0]
        if len(set(raw_labels)) == k:
            print(f"Seed {seed}: Starting from manual assignment.")
            return raw_labels.tolist()
        print(f"Manual column mismatch, falling back to KMeans.")
    X = gdf[COL_SALES].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto', max_iter=300)
    kmeans.fit(X)
    labels = kmeans.labels_
    mapping = {old: new for new, old in enumerate(sorted(set(labels)))}
    return [mapping[l] for l in labels]


# --- HIGH-PERFORMANCE TABU SEARCH ---
def tabu_search_fast(gdf_raw, graph, k, max_iter, tabu_tenure,
                     max_box_side, min_avg_neighbors, seed,
                     sample_size=100):
    random.seed(seed)
    n = len(gdf_raw)
    current_solution = manual_initial_solution(gdf_raw, k, seed)
    best_solution = list(current_solution)

    def compute_all_metrics(solution):
        metrics = {}
        for t in range(k):
            idxs = [i for i, a in enumerate(solution) if a == t]
            if idxs:
                sub_geoms = gdf_raw.loc[idxs].geometry
                merged_geom = sub_geoms.union_all()
                bounds = merged_geom.bounds
                width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                total_sales = gdf_raw.loc[idxs, COL_SALES].sum()
                metrics[t] = {"width": width, "height": height,
                              "total_sales": total_sales, "sub_idxs": idxs}
            else:
                metrics[t] = {"width": 0, "height": 0, "total_sales": 0, "sub_idxs": []}
        return metrics

    metrics = compute_all_metrics(current_solution)
    avg_neighbors = avg_internal_neighbors(current_solution, graph)
    best_score = objective(current_solution, gdf_raw)
    valid_solution_found = is_feasible(current_solution, gdf_raw, graph,
                                       max_box_side, min_avg_neighbors)
    tabu_list = {}

    for iteration in range(max_iter):
        candidate_moves = []
        for _ in range(sample_size):
            i = random.randint(0, n-1)
            current_terr = current_solution[i]
            new_terr = random.choice([t for t in range(k) if t != current_terr])
            candidate_moves.append((i, new_terr))

        neighbors = []
        for i, new_terr in candidate_moves:
            move = (i, new_terr)
            if move in tabu_list and tabu_list[move] > iteration:
                continue
            old_terr = current_solution[i]
            temp_solution = list(current_solution)
            temp_solution[i] = new_terr
            affected = [old_terr, new_terr]
            temp_metrics = metrics.copy()
            for t in affected:
                idxs = [idx for idx, a in enumerate(temp_solution) if a == t]
                if idxs:
                    sub_geoms = gdf_raw.loc[idxs].geometry
                    merged_geom = sub_geoms.union_all()
                    bounds = merged_geom.bounds
                    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                    total_sales = gdf_raw.loc[idxs, COL_SALES].sum()
                    temp_metrics[t] = {"width": width, "height": height,
                                       "total_sales": total_sales, "sub_idxs": idxs}
                else:
                    temp_metrics[t] = {"width": 0, "height": 0, "total_sales": 0, "sub_idxs": []}

            feasible = True
            for t in affected:
                m = temp_metrics[t]
                if m['width'] > max_box_side or m['height'] > max_box_side:
                    feasible = False
                    break
            temp_avg_neighbors = avg_neighbors.copy()
            for t in affected:
                nodes = temp_metrics[t]['sub_idxs']
                internal_counts = []
                for n_idx in nodes:
                    neighbors_n = list(graph.neighbors(n_idx))
                    internal = sum(1 for nb in neighbors_n if temp_solution[nb] == t)
                    internal_counts.append(internal)
                temp_avg_neighbors[t] = np.mean(internal_counts) if internal_counts else 0
                if temp_avg_neighbors[t] < min_avg_neighbors:
                    feasible = False
                    break

            if feasible:
                score = objective(temp_solution, gdf_raw)
                neighbors.append((score, temp_solution, move, temp_metrics, temp_avg_neighbors))

        if not neighbors:
            continue

        neighbors.sort(key=lambda x: x[0])
        score, best_neighbor, best_move, new_metrics, new_avg_neighbors = neighbors[0]
        current_solution = best_neighbor
        metrics = new_metrics
        avg_neighbors = new_avg_neighbors
        if score < best_score and is_feasible(current_solution, gdf_raw, graph,
                                             max_box_side, min_avg_neighbors):
            best_score = score
            best_solution = list(current_solution)
            valid_solution_found = True
        tabu_list[best_move] = iteration + tabu_tenure

    if not valid_solution_found:
        return {'status': 'compactness_fail', 'score': np.inf, 'labels': None, 'seed': seed}

    final_metrics = compute_all_metrics(best_solution)
    final_avg_neighbors = avg_internal_neighbors(best_solution, graph)
    passed_box_count = sum(1 for m in final_metrics.values() if m['width'] <= max_box_side and m['height'] <= max_box_side)
    compact_score = passed_box_count / k if k > 0 else 0

    return {
        'status': 'success',
        'score': best_score,
        'labels': best_solution,
        'compact_score': compact_score,
        'seed': seed,
        'avg_neighbors': final_avg_neighbors
    }


# --- OUTPUT CREATION ---
def create_solution_outputs(gdf_base, gdf_proj_base, result, graph, max_box_side_km,
                            min_avg_internal_neighbors, avg_sales_per_territory,
                            avg_sales_per_territory_str, w):
    labels = result['labels']
    gdf = gdf_base.copy()
    gdf['new_territory'] = labels
    gdf_proj = gdf_proj_base.copy()
    gdf_proj['new_territory'] = labels

    territory_sales = gdf.groupby('new_territory')[COL_SALES].sum()
    gdf['territory_total_sales'] = gdf['new_territory'].map(territory_sales)
    gdf['Total Territory Sales'] = gdf['territory_total_sales'].apply(
        lambda v: f"{v/1000:,.1f}k units" if v >= 1000 else f"{v:,.0f} units"
    )
    gdf['National Avg Territory Sales'] = avg_sales_per_territory_str
    territory_neighbor_data = {t: f"{v:.2f}" if not np.isnan(v) else "N/A (1 Sub-T)"
                               for t, v in result['avg_neighbors'].items()}
    gdf['Avg Internal Neighbors'] = gdf['new_territory'].map(territory_neighbor_data)

    dissolved = gdf_proj.dissolve(by='new_territory')
    territory_box_data = {}
    for t, row in dissolved.iterrows():
        bounds = row.geometry.bounds
        width_km = (bounds[2] - bounds[0]) / 1000
        height_km = (bounds[3] - bounds[1]) / 1000
        territory_box_data[t] = f"{width_km:.1f}km x {height_km:.1f}km"
    gdf['Bounding Box (W x H)'] = gdf['new_territory'].map(territory_box_data)

    output_filename_csv = f"{OUTPUT_CSV_BASE}{result['rank']}.csv"
    output_cols = [COL_SUBTERRITORY_ID, 'new_territory', COL_SALES]
    if COL_MANUAL_ASSIGNMENT in gdf.columns:
        output_cols.insert(1, COL_MANUAL_ASSIGNMENT)
    gdf[output_cols].to_csv(output_filename_csv, index=False)

    output_filename_map = f"{OUTPUT_MAP_BASE}{result['rank']}.html"
    gdf.to_crs(epsg=4326).explore(
        column='new_territory',
        categorical=True,
        cmap='tab20',
        tooltip=[COL_SUBTERRITORY_ID, 'new_territory', COL_SALES],
        popup=True,
        style_kwds={'fillOpacity':0.7, 'weight':0.5, 'color':'black'}
    ).save(output_filename_map)

    return output_filename_csv, output_filename_map, result['score'], result['compact_score']


# --- MAIN OPTIMIZATION ---
def run_optimization():
    try:
        df_segments = pd.read_csv(INPUT_CSV)
        df_segments = df_segments[df_segments['territory'].str.contains("DEDR5")]
        print(df_segments.head)
    except FileNotFoundError:
        print(f"File {INPUT_CSV} not found.")
        return False
    df_segments['geometry'] = df_segments['coordinates'].apply(parse_coordinates)
    gdf_segments = gpd.GeoDataFrame(df_segments, geometry='geometry')
    gdf_segments.set_crs("EPSG:4269", inplace=True)

    agg_functions = {COL_SALES: 'first'}
    if COL_MANUAL_ASSIGNMENT in df_segments.columns:
        agg_functions[COL_MANUAL_ASSIGNMENT] = 'first'
    gdf_agg = gdf_segments.dissolve(by=COL_SUBTERRITORY_ID, aggfunc=agg_functions).reset_index()
    gdf = gdf_agg.rename(columns={'value': COL_SALES})
    print(f"Aggregated {len(df_segments)} segments into {len(gdf)} subterritories.")

    gdf_projected = gdf.to_crs(epsg=3857)
    graph = build_contiguity_graph(gdf_projected)

    total_sales = gdf[COL_SALES].sum()
    avg_sales_per_territory = total_sales / TARGET_K_TERRITORIES
    total_area = gdf_projected.geometry.union_all().area
    ideal_side = np.sqrt(total_area / TARGET_K_TERRITORIES)
    MAX_BOX_SIDE_METERS = ideal_side * MAX_BOX_FACTOR
    MAX_BOX_SIDE_KM = MAX_BOX_SIDE_METERS / 1000

    rng = np.random.default_rng()
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=N_START_ATTEMPTS)]
    start_time = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(tabu_search_fast)(gdf_projected.copy(), graph, TARGET_K_TERRITORIES,
                                  MAX_ITERATIONS, TABU_TENURE, MAX_BOX_SIDE_METERS,
                                  MIN_AVG_INTERNAL_NEIGHBORS, seed, sample_size=200)
        for seed in seeds
    )
    print(f"Optimization finished in {time.time()-start_time:.2f} seconds.")

    valid_results = [res for res in results if res['status'] == 'success']
    if not valid_results:
        print("No solution passed compactness constraints.")
        return False

    valid_results.sort(key=lambda item: item['score'])
    top_solutions = valid_results[:max(NUM_TOP_MAPS, NUM_TOP_CSVS)]
    best_result = top_solutions[0]
    avg_sales_per_territory_str = f"{avg_sales_per_territory:,.0f} units (Nat'l Avg)"

    for i, solution in enumerate(top_solutions):
        rank = i+1
        solution['rank'] = rank
        csv_path, map_path, std_dev, compact_score = create_solution_outputs(
            gdf.to_crs(epsg=4326),
            gdf_projected.copy(),
            solution,
            graph,
            MAX_BOX_SIDE_KM,
            MIN_AVG_INTERNAL_NEIGHBORS,
            avg_sales_per_territory,
            avg_sales_per_territory_str,
            None
        )
        print(f"#{rank}: Std Dev {std_dev:,.0f} | Compactness {compact_score:.1%} -> CSV: {csv_path}, MAP: {map_path}")

    print(f"Best Solution Sales Std Dev: {best_result['score']:,.0f} units")
    print(f"Compactness: {best_result['compact_score']:.1%}")
    print(f"Seed: {best_result['seed']}")
    return True


if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"Unexpected exception: {e}")
        sys.exit()
