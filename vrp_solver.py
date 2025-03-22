import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from streamlit_folium import st_folium, folium_static
from geopy import distance
import math
import requests
import polyline
from sklearn.cluster import KMeans
import random
import utils
from utils import compute_route_distance, two_opt, three_opt, simulated_annealing

###############################################
# Gelişmiş VRP Çözücü – MILP ve Heuristic (Çoklu Depo, Locker Ataması, Forbidden Node Kısıtı, OSRM)
###############################################
class AdvancedVRPSolver:
    def assign_cost(node_type, deliver_type):
        """
        Assign a cost based on node_type and deliver_type.
        Adjust the base_cost and add-ons as needed.
        """
        base_cost = 0
        
        # Base cost by node_type
        if node_type == 'depot':
            base_cost = 0
        elif node_type == 'locker':
            base_cost = 2
        elif node_type == 'customer':
            base_cost = 5
        
        # Additional cost by deliver_type
        if deliver_type == 'last_feet':
            base_cost += 3
        elif deliver_type == 'last_mile':
            base_cost += 4
        # If deliver_type is 'none', we add nothing
        
        return base_cost

    def __init__(self, data):
        """
        data: VRP’ye sokulacak (işlenmiş) DataFrame.
              – DataFrame’de node_type sütunu: depot, locker, customer şeklinde tanımlı.
        """
        self.data = data.reset_index(drop=True)
        if 'ID' not in self.data.columns:
            self.data['ID'] = self.data.index
        # Depo indekslerini belirle:
        self.depot_indices = self.data.index[self.data['node_type'] == 'depot'].tolist()
        if not self.depot_indices:
            raise ValueError("En az bir depo bulunmalıdır!")
        self.coordinates = self.data[['Latitude','Longitude']].values
        self.demands = self.data.apply(lambda row: row['demand'] if row['node_type'] != 'locker' else 0, axis=1).values
        self.dist_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        n = len(self.coordinates)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = distance.distance(self.coordinates[i], self.coordinates[j]).kilometers
        return matrix

    def get_osrm_route(self, start_coord, end_coord, cache):
        key = (start_coord, end_coord)
        if key in cache:
            return cache[key]
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coord[1]},{start_coord[0]};{end_coord[1]},{end_coord[0]}?overview=full&geometries=polyline"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and data.get('routes'):
                    polyline_str = data['routes'][0]['geometry']
                    coords = polyline.decode(polyline_str)
                    cache[key] = coords
                    return coords
            cache[key] = [start_coord, end_coord]
            return [start_coord, end_coord]
        except Exception as e:
            cache[key] = [start_coord, end_coord]
            return [start_coord, end_coord]

    def violates_forbidden(self, route, candidate, forbidden_groups):
        candidate_id = self.data.loc[candidate, 'ID']
        route_ids = [self.data.loc[node, 'ID'] for node in route]
        for group in forbidden_groups:
            if candidate_id in group:
                if any(rid in group for rid in route_ids):
                    return True
        return False

    def fix_route_forbidden(self, route, forbidden_groups):
        new_route = [route[0]]
        for node in route[1:-1]:
            candidate_id = self.data.loc[node, 'ID']
            conflict = False
            for group in forbidden_groups:
                if candidate_id in group and any(self.data.loc[n, 'ID'] in group for n in new_route):
                    conflict = True
                    break
            new_route.append(node)
        new_route.append(route[-1])
        return new_route

    ##########################################################
    # MILP Tabanlı Çözüm (Önceki versiyon gibi)
    ##########################################################
    def solve_vrp_milp(self, vehicles_df, time_limit=None):
        # (MILP çözüm kısmı; kod daha önceki versiyonlarda olduğu gibi)
        n = len(self.data)
        vehicle_ids = list(vehicles_df['vehicle_id'])
        vehicles = {}
        for _, row in vehicles_df.iterrows():
            vehicles[row['vehicle_id']] = {
                'capacity': row['capacity'],
                'max_duration': row['max_duration'] / 60.0,
                'cost_per_km': row['cost_per_km'],
                'fixed_cost': row['fixed_cost']
            }
        N = list(range(n))
        D = self.depot_indices
        C = [i for i in N if i not in D]
        
        prob = pulp.LpProblem("MultiDepot_VRP", pulp.LpMinimize)
        
        x = {}
        for i in N:
            for j in N:
                if i == j: continue
                for v in vehicle_ids:
                    x[i, j, v] = pulp.LpVariable(f'x_{i}_{j}_{v}', cat='Binary')
        
        y = {}
        for v in vehicle_ids:
            y[v] = pulp.LpVariable(f'y_{v}', cat='Binary')
        
        u = {}
        for i in C:
            for v in vehicle_ids:
                u[i, v] = pulp.LpVariable(f'u_{i}_{v}', lowBound=0)
        
        prob += pulp.lpSum(
            vehicles[v]['fixed_cost'] * y[v] +
            pulp.lpSum(self.dist_matrix[i][j] * vehicles[v]['cost_per_km'] * x[i, j, v]
                       for i in N for j in N if i != j)
            for v in vehicle_ids
        )
        
        for j in C:
            prob += pulp.lpSum(x[i, j, v] for i in N if i != j for v in vehicle_ids) == 1
        
        for v in vehicle_ids:
            prob += pulp.lpSum(x[d, j, v] for d in D for j in N if d != j) == y[v]
            prob += pulp.lpSum(x[i, d, v] for d in D for i in N if d != i) == y[v]
        
        for v in vehicle_ids:
            for k in C:
                prob += pulp.lpSum(x[i, k, v] for i in N if i != k) == pulp.lpSum(x[k, j, v] for j in N if j != k)
        
        for v in vehicle_ids:
            cap = vehicles[v]['capacity']
            for i in C:
                for j in C:
                    if i == j: continue
                    prob += u[i, v] + self.demands[j] <= u[j, v] + cap * (1 - x[i, j, v])
            for i in C:
                prob += u[i, v] >= self.demands[i]
        
        speed = 50.0
        for v in vehicle_ids:
            prob += pulp.lpSum(self.dist_matrix[i][j] / speed * x[i, j, v]
                               for i in N for j in N if i != j) <= vehicles[v]['max_duration']
        
        if time_limit:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
            prob.solve(solver)
        else:
            prob.solve()
        
        routes = {v: [] for v in vehicle_ids}
        if pulp.LpStatus[prob.status] in ["Optimal", "Feasible"]:
            for v in vehicle_ids:
                start = None
                for d in D:
                    for j in N:
                        if d != j and pulp.value(x.get((d, j, v), 0)) is not None and pulp.value(x[d, j, v]) > 0.5:
                            start = d
                            break
                    if start is not None:
                        break
                if start is None:
                    continue
                current = start
                route = [current]
                while True:
                    next_node = None
                    for j in N:
                        if j == current:
                            continue
                        if pulp.value(x.get((current, j, v), 0)) is not None and pulp.value(x[current, j, v]) > 0.5:
                            next_node = j
                            break
                    if next_node is None or next_node in D:
                        route.append(next_node if next_node is not None else start)
                        break
                    else:
                        route.append(next_node)
                        current = next_node
                routes[v] = route
        else:
            st.warning("MILP çözümü bulunamadı veya zaman limiti aşıldı.")
            routes = None
        total_cost = pulp.value(prob.objective) if routes is not None else None
        return routes, total_cost

    def find_closest_locker(self, current_node):
        """
        Returns the index of the closest locker to 'current_node'
        or None if no lockers exist.
        """
        locker_indices = [
            i for i in range(len(self.data)) 
            if self.data.loc[i, 'node_type'] == 'locker'
        ]
        if not locker_indices:
            return None
        
        # Pick the locker that minimizes distance from current_node.
        closest = min(
            locker_indices, 
            key=lambda locker: self.dist_matrix[current_node][locker]
        )
        return closest

# Heuristic Yöntem – Feasible Sweep Algoritması (Last Mile Versiyonu)
##########################################################
    def solve_vrp_heuristic(self, vehicles_df, forbidden_groups=None):
        """
        Hybrid heuristic for VRP with mixed home vs. locker deliveries.
        
        This function does the following:
        
        1. Data Preparation: Copies merged data and stores original customer coordinates.
        
        2. Clustering & Aggregation:
            - Uses DBSCAN to cluster customers with deliver_type 'last_mile'.
            - For each cluster, computes:
                        total_direct_cost = sum(distance(nearest_depot, customer_home) * cost_per_km)
                        total_locker_cost = sum((distance(nearest_depot, candidate_locker) * cost_per_km)
                                                + (distance(candidate_locker, customer_home) * customer_cost))
            - If the best locker option is cheaper than direct cost, the cluster is aggregated:
                    * For each customer in the cluster, update its coordinates to the chosen locker’s
                    and set deliver_type to 'locker_pickup'.
                    * Create an aggregated node (of type 'locker_cluster') representing the cluster
                    (with demand = sum of customer demands, a field 'served_customers', and its ID set to the actual locker’s ID).
            - Otherwise, customers remain as individual nodes.
            - Post-processing: Merge aggregated nodes that are at nearly identical coordinates.
        
        3. Initial Route Construction using Clarke–Wright Savings:
            - Build a new distance matrix (with distances converted to km).
            - For each non-depot node, find its nearest depot and initialize its route as:
                        [nearest_depot, node, nearest_depot]
            - Merge routes only if they share the same starting depot.
        
        4. Local Improvement (Simulated Annealing, SA):
            - SA explores neighborhood moves (e.g., toggling a customer’s mode).
            - The objective function is the sum of travel cost (distance × cost_per_km) plus a penalty
                for locker-served customers (penalty = distance(customer’s original home, locker) × customer_cost).
        
        5. Vehicle Assignment:
            - Greedily assign the final routes to vehicles.
            - Ensure each route ends with the depot it started from.
            - **Updated:** When computing each route’s cost, add the penalty from aggregated nodes.
        
        Returns:
        route_assignments: dict mapping vehicle_id to route info (routes are lists of integer indices into the routing data).
        total_cost: Total cost (sum of fixed cost and route cost) for all vehicles.
        """
        import math, numpy as np, pandas as pd, random, copy
        from sklearn.cluster import DBSCAN

        if forbidden_groups is None:
            forbidden_groups = []

        # PARAMETERS
        cost_per_km = getattr(self, 'cost_per_km', vehicles_df['cost_per_km'].iloc[0])
        vehicle_capacity = vehicles_df['capacity'].max()
        # SA parameters:
        T = 100.0
        T_min = 1e-3
        alpha = 0.95
        max_iter = 2000
        # Conversion factor: ~111 km per degree.
        km_per_degree = 111.0

        # ------------------
        # Phase 0: Data Preparation.
        original_df = self.data.copy()  # Keep original data for markers and penalty lookup.
        routing_df = self.data.copy()   # Work on a copy for routing.
        cust_mask = routing_df['node_type'].str.lower() == 'customer'
        if 'orig_Latitude' not in routing_df.columns:
            routing_df.loc[cust_mask, 'orig_Latitude'] = routing_df.loc[cust_mask, 'Latitude']
        if 'orig_Longitude' not in routing_df.columns:
            routing_df.loc[cust_mask, 'orig_Longitude'] = routing_df.loc[cust_mask, 'Longitude']

        # ------------------
        # Helper: distance in km between two points.
        def deg_distance(lat1, lon1, lat2, lon2):
            return math.hypot((lat1 - lat2) * km_per_degree, (lon1 - lon2) * km_per_degree)

        # ------------------
        # Phase 1: Clustering & Aggregation.
        mask_last_mile = cust_mask & (routing_df['deliver_type'].str.lower() == 'last_mile')
        last_mile_indices = routing_df[mask_last_mile].index.tolist()
        aggregated_nodes = []
        aggregated_customer_indices = set()
        if last_mile_indices:
            coords = routing_df.loc[last_mile_indices, ['Latitude', 'Longitude']].values
            # eps=0.01 ~ 1.11 km; adjust if needed.
            clustering = DBSCAN(eps=0.01, min_samples=1).fit(coords)
            labels = clustering.labels_
            cluster_map = {}
            for idx, label in zip(last_mile_indices, labels):
                cluster_map.setdefault(label, []).append(idx)
            locker_indices = routing_df[routing_df['node_type'].str.lower() == 'locker'].index.tolist()
            for label, indices in cluster_map.items():
                total_direct_cost = 0.0
                for i in indices:
                    nearest_depot = min(self.depot_indices, key=lambda d: deg_distance(
                        routing_df.loc[d, 'Latitude'], routing_df.loc[d, 'Longitude'],
                        routing_df.loc[i, 'Latitude'], routing_df.loc[i, 'Longitude']
                    ))
                    total_direct_cost += deg_distance(
                        routing_df.loc[nearest_depot, 'Latitude'], routing_df.loc[nearest_depot, 'Longitude'],
                        routing_df.loc[i, 'Latitude'], routing_df.loc[i, 'Longitude']
                    ) * cost_per_km
                best_total_locker_cost = float('inf')
                best_locker_for_cluster = None
                for l in locker_indices:
                    cluster_cost = 0.0
                    for i in indices:
                        nearest_depot = min(self.depot_indices, key=lambda d: deg_distance(
                            routing_df.loc[d, 'Latitude'], routing_df.loc[d, 'Longitude'],
                            routing_df.loc[i, 'Latitude'], routing_df.loc[i, 'Longitude']
                        ))
                        cost_i = (deg_distance(routing_df.loc[nearest_depot, 'Latitude'],
                                                routing_df.loc[nearest_depot, 'Longitude'],
                                                routing_df.loc[l, 'Latitude'],
                                                routing_df.loc[l, 'Longitude']) * cost_per_km) + \
                                (deg_distance(routing_df.loc[l, 'Latitude'],
                                                routing_df.loc[l, 'Longitude'],
                                                routing_df.loc[i, 'orig_Latitude'],
                                                routing_df.loc[i, 'orig_Longitude']) * float(routing_df.loc[i, 'customer_cost']))
                        cluster_cost += cost_i
                    if cluster_cost < best_total_locker_cost:
                        best_total_locker_cost = cluster_cost
                        best_locker_for_cluster = l
                if best_total_locker_cost < total_direct_cost and best_locker_for_cluster is not None:
                    for i in indices:
                        routing_df.loc[i, 'Latitude'] = routing_df.loc[best_locker_for_cluster, 'Latitude']
                        routing_df.loc[i, 'Longitude'] = routing_df.loc[best_locker_for_cluster, 'Longitude']
                        routing_df.loc[i, 'deliver_type'] = 'locker_pickup'
                        self.locker_assignments = getattr(self, 'locker_assignments', {})
                        self.locker_assignments[i] = {'assigned_locker': best_locker_for_cluster,
                                                    'cluster': label,
                                                    'locker_cost': best_total_locker_cost / len(indices)}
                    agg_demand = sum(routing_df.loc[i, 'demand'] for i in indices)
                    agg_node = {
                        'new_index': None,
                        'original_index': indices,
                        'node_type': 'locker_cluster',
                        'deliver_type': 'locker_pickup',
                        'Latitude': routing_df.loc[best_locker_for_cluster, 'Latitude'],
                        'Longitude': routing_df.loc[best_locker_for_cluster, 'Longitude'],
                        'demand': agg_demand,
                        'served_customers': indices,
                        # Set the ID to be the actual locker ID.
                        'ID': routing_df.loc[best_locker_for_cluster, 'ID'] if 'ID' in routing_df.columns 
                            else f"Locker({best_locker_for_cluster})",
                        'assigned_locker': best_locker_for_cluster
                    }
                    aggregated_nodes.append(agg_node)
                    aggregated_customer_indices.update(indices)
        # End Phase 1.
        
        # --- Post-Processing: Merge Aggregated Nodes That Are Nearly the Same ---
        tolerance = 0.001  # degrees
        unique_aggregated = []
        for node in aggregated_nodes:
            merged = False
            for u_node in unique_aggregated:
                if (abs(u_node['Latitude'] - node['Latitude']) < tolerance and
                    abs(u_node['Longitude'] - node['Longitude']) < tolerance):
                    u_node['served_customers'].extend(node['served_customers'])
                    u_node['demand'] += node['demand']
                    # Update the ID to be the actual locker ID if available.
                    if 'assigned_locker' in node:
                        u_node['ID'] = routing_df.loc[node['assigned_locker'], 'ID'] if 'ID' in routing_df.columns else f"Locker({node['assigned_locker']})"
                        u_node['assigned_locker'] = node['assigned_locker']
                    else:
                        u_node['ID'] = f"LockerCluster({','.join(str(x) for x in u_node['served_customers'])})"
                    merged = True
                    break
            if not merged:
                new_node = node.copy()
                if not isinstance(new_node.get('served_customers', None), list):
                    new_node['served_customers'] = [new_node['original_index']]
                unique_aggregated.append(new_node)
        aggregated_nodes = unique_aggregated

        # ------------------
        # Build new routing nodes:
        new_nodes = []
        # 1. Add depot nodes.
        depot_mask = routing_df['node_type'].str.lower() == 'depot'
        for idx, row in routing_df[depot_mask].iterrows():
            new_nodes.append({
                'node_type': row['node_type'],
                'deliver_type': row['deliver_type'],
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'demand': 0.0,
                'orig_Latitude': row.get('orig_Latitude', row['Latitude']),
                'orig_Longitude': row.get('orig_Longitude', row['Longitude']),
                'ID': row.get('ID', idx)
            })
        # 2. Add individual customer nodes not aggregated.
        non_agg_mask = cust_mask & (~routing_df.index.isin(aggregated_customer_indices))
        for idx, row in routing_df[non_agg_mask].iterrows():
            new_nodes.append({
                'node_type': row['node_type'],
                'deliver_type': row['deliver_type'],
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'demand': row['demand'],
                'orig_Latitude': row.get('orig_Latitude', row['Latitude']),
                'orig_Longitude': row.get('orig_Longitude', row['Longitude']),
                'ID': row.get('ID', idx)
            })
        # 3. Add aggregated locker nodes.
        for node in aggregated_nodes:
            new_nodes.append({
                'node_type': node['node_type'],
                'deliver_type': node['deliver_type'],
                'Latitude': node['Latitude'],
                'Longitude': node['Longitude'],
                'demand': node['demand'],
                'served_customers': node.get('served_customers', []),
                'ID': node['ID']
            })
        new_routing_df = pd.DataFrame(new_nodes).reset_index(drop=True)
        new_routing_df.index.name = 'new_index'
        new_routing_df.reset_index(inplace=True)
        routing_df = new_routing_df.copy()
        # (Original data remains unchanged for markers.)

        # ------------------
        # Phase 2: Build new distance matrix for routing_df.
        num_nodes = len(routing_df)
        new_dist_mat = [[0.0] * num_nodes for _ in range(num_nodes)]
        for a in range(num_nodes):
            for b in range(num_nodes):
                if a == b:
                    continue
                latA, lonA = routing_df.loc[a, 'Latitude'], routing_df.loc[a, 'Longitude']
                latB, lonB = routing_df.loc[b, 'Latitude'], routing_df.loc[b, 'Longitude']
                new_dist_mat[a][b] = deg_distance(latA, lonA, latB, lonB)
        # Get all depot indices from routing_df.
        depot_rows = routing_df[routing_df['node_type'].str.lower() == 'depot']
        if depot_rows.empty:
            raise ValueError("No depot found in routing data.")
        depot_indices_final = depot_rows.index.tolist()

        # ------------------
        # Phase 3: Initial solution using Clarke–Wright Savings.
        # For each non-depot node, determine its nearest depot from routing_df.
        initial_routes = {}
        for idx in routing_df.index:
            if routing_df.loc[idx, 'node_type'].strip().lower() != 'depot':
                nearest_depot = min(depot_indices_final, key=lambda d: new_dist_mat[d][idx])
                initial_routes[idx] = [nearest_depot, idx, nearest_depot]
        # Merge routes only if they share the same starting depot.
        savings = []
        for i in initial_routes:
            for j in initial_routes:
                if i >= j:
                    continue
                if initial_routes[i][0] == initial_routes[j][0]:
                    s = new_dist_mat[initial_routes[i][0]][i] + new_dist_mat[initial_routes[i][0]][j] - new_dist_mat[i][j]
                    savings.append((s, i, j))
        savings.sort(reverse=True, key=lambda x: x[0])
        route_of = {i: i for i in initial_routes}
        for s, i, j in savings:
            ri, rj = route_of[i], route_of[j]
            if ri != rj and initial_routes[ri][0] == initial_routes[rj][0]:
                merged = initial_routes[ri][:-1] + initial_routes[rj][1:]
                initial_routes[ri] = merged
                for node in initial_routes[rj][1:-1]:
                    route_of[node] = ri
                del initial_routes[rj]
        final_routes = list(initial_routes.values())
        # Ensure each route returns to its starting depot.
        for r in final_routes:
            depot_start = r[0]
            if r[-1] != depot_start:
                r.append(depot_start)

        # ------------------
        # Phase 4: Simulated Annealing (SA) for local improvement.
        def compute_solution_cost(routes_list, dist_mat, df):
            total = 0.0
            for r in routes_list:
                for k in range(len(r)-1):
                    total += dist_mat[r[k]][r[k+1]] * cost_per_km
                for n in r:
                    # For individual customer nodes served via locker.
                    if df.loc[n, 'node_type'].strip().lower() == 'customer' and \
                    df.loc[n, 'deliver_type'].strip().lower() == 'locker_pickup':
                        lat_orig = df.loc[n, 'orig_Latitude']
                        lon_orig = df.loc[n, 'orig_Longitude']
                        lat_curr = df.loc[n, 'Latitude']
                        lon_curr = df.loc[n, 'Longitude']
                        total += deg_distance(lat_orig, lon_orig, lat_curr, lon_curr) * float(df.loc[n, 'customer_cost'])
                    # For aggregated locker nodes, sum penalty over served customers.
                    elif df.loc[n, 'node_type'].strip().lower() == 'locker_cluster':
                        served = df.loc[n, 'served_customers']
                        # For each served customer, look up original coordinates in original_df.
                        for cust_idx in served:
                            cust_row = original_df.loc[cust_idx]
                            total += deg_distance(cust_row['Latitude'], cust_row['Longitude'],
                                                    df.loc[n, 'Latitude'], df.loc[n, 'Longitude']) * float(cust_row['customer_cost'])
            return total

        current_routes = copy.deepcopy(final_routes)
        current_cost = compute_solution_cost(current_routes, new_dist_mat, routing_df)
        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost

        def random_move(df):
            new_df = df.copy()
            cust_idx = new_df[new_df['node_type'].str.lower() == 'customer'].index.tolist()
            if not cust_idx:
                return new_df
            i = random.choice(cust_idx)
            current_mode = new_df.loc[i, 'deliver_type'].strip().lower()
            nearest_depot = min(depot_indices_final, key=lambda d: deg_distance(
                new_df.loc[d, 'Latitude'], new_df.loc[d, 'Longitude'],
                new_df.loc[i, 'Latitude'], new_df.loc[i, 'Longitude']
            ))
            if current_mode == 'locker_pickup':
                new_df.loc[i, 'Latitude'] = new_df.loc[i, 'orig_Latitude']
                new_df.loc[i, 'Longitude'] = new_df.loc[i, 'orig_Longitude']
                new_df.loc[i, 'deliver_type'] = 'last_mile'
            else:
                best_locker = None
                best_cost_local = float('inf')
                locker_idx = new_df[new_df['node_type'].str.lower() == 'locker'].index.tolist()
                for l in locker_idx:
                    candidate_cost = (deg_distance(new_df.loc[nearest_depot, 'Latitude'],
                                                new_df.loc[nearest_depot, 'Longitude'],
                                                new_df.loc[l, 'Latitude'],
                                                new_df.loc[l, 'Longitude']) * cost_per_km) + \
                                    (deg_distance(new_df.loc[l, 'Latitude'],
                                                new_df.loc[l, 'Longitude'],
                                                new_df.loc[i, 'Latitude'],
                                                new_df.loc[i, 'Longitude']) * float(new_df.loc[i, 'customer_cost']))
                    if candidate_cost < best_cost_local:
                        best_cost_local = candidate_cost
                        best_locker = l
                if best_locker is not None:
                    new_df.loc[i, 'Latitude'] = new_df.loc[best_locker, 'Latitude']
                    new_df.loc[i, 'Longitude'] = new_df.loc[best_locker, 'Longitude']
                    new_df.loc[i, 'deliver_type'] = 'locker_pickup'
            return new_df

        iter_count = 0
        while T > T_min and iter_count < max_iter:
            candidate_df = random_move(routing_df)
            num_nodes_candidate = len(candidate_df)
            cand_dist_mat = [[0.0] * num_nodes_candidate for _ in range(num_nodes_candidate)]
            for a in range(num_nodes_candidate):
                for b in range(num_nodes_candidate):
                    if a == b:
                        continue
                    latA = candidate_df.loc[a, 'Latitude']
                    lonA = candidate_df.loc[a, 'Longitude']
                    latB = candidate_df.loc[b, 'Latitude']
                    lonB = candidate_df.loc[b, 'Longitude']
                    cand_dist_mat[a][b] = deg_distance(latA, lonA, latB, lonB)
            candidate_cost = compute_solution_cost(current_routes, cand_dist_mat, candidate_df)
            delta = candidate_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                routing_df = candidate_df
                new_dist_mat = cand_dist_mat
                current_cost = candidate_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_routes = copy.deepcopy(current_routes)
            T *= alpha
            iter_count += 1

        final_routes = best_routes

        # ------------------
        # Phase 5: Greedy Assignment of Routes to Vehicles.
        vehicles_sorted = vehicles_df.sort_values(by='capacity', ascending=False)
        route_assignments = {}
        fr_idx = 0
        for idx, veh in vehicles_sorted.iterrows():
            if fr_idx < len(final_routes):
                r = final_routes[fr_idx]
                display_route = []
                for n in r:
                    row = routing_df.loc[n]
                    if row['node_type'].strip().lower() == 'customer' and row['deliver_type'].strip().lower() == 'locker_pickup':
                        display_route.append(f"Locker({row.get('ID', n)})")
                    else:
                        display_route.append(row.get('ID', n))
                route_distance = sum(new_dist_mat[r[k]][r[k+1]] for k in range(len(r)-1))
                route_demand = sum(routing_df.loc[n, 'demand'] for n in r if routing_df.loc[n, 'node_type'].strip().lower() == 'customer')
                route_cost = route_distance * cost_per_km
                route_penalty = 0.0
                for n in r:
                    if routing_df.loc[n, 'node_type'].strip().lower() == 'customer' and \
                    routing_df.loc[n, 'deliver_type'].strip().lower() == 'locker_pickup':
                        lat_orig = routing_df.loc[n, 'orig_Latitude']
                        lon_orig = routing_df.loc[n, 'orig_Longitude']
                        lat_curr = routing_df.loc[n, 'Latitude']
                        lon_curr = routing_df.loc[n, 'Longitude']
                        route_penalty += deg_distance(lat_orig, lon_orig, lat_curr, lon_curr) * float(routing_df.loc[n, 'customer_cost'])
                    elif routing_df.loc[n, 'node_type'].strip().lower() == 'locker_cluster':
                        # For an aggregated node, compute penalty for each served customer using original_df.
                        served = routing_df.loc[n, 'served_customers']
                        for cust_idx in served:
                            cust_row = original_df.loc[cust_idx]
                            route_penalty += deg_distance(cust_row['Latitude'], cust_row['Longitude'],
                                                        routing_df.loc[n, 'Latitude'], routing_df.loc[n, 'Longitude']) * float(cust_row['customer_cost'])
                route_cost += route_penalty
                route_assignments[veh['vehicle_id']] = {
                    'route': r,
                    'display_route': display_route,
                    'distance': route_distance,
                    'penalty': route_penalty,
                    'cost': route_cost,
                    'demand': route_demand,
                    'vehicle': veh.to_dict()
                }
                fr_idx += 1

        total_cost = 0.0
        for vid, rdata in route_assignments.items():
            fixed = rdata['vehicle'].get('fixed_cost', 0.0)
            total_cost += fixed + rdata['cost']

        self.data_routing = routing_df.copy()
        return route_assignments, total_cost








    def solve_vrp_heuristic_with_sa(self, vehicles_df, forbidden_groups=None):
        # Step 1: Generate initial solution using heuristic
        route_assignments, _ = self.solve_vrp_heuristic(vehicles_df, forbidden_groups)

        # Step 2: Apply Simulated Annealing to each route
        sa_optimized_routes = {}
        for vid, rdata in route_assignments.items():
            initial_route = rdata['route']
            best_route, best_distance = simulated_annealing(initial_route, self.dist_matrix)
            sa_optimized_routes[vid] = {
                'route': best_route,
                'distance': best_distance,
                'demand': rdata['demand'],
                'vehicle': rdata['vehicle']
            }

        # Step 3: Reassign routes to vehicles
        vehicles_sorted = vehicles_df.sort_values(by='capacity', ascending=False)
        final_route_assignments = {}
        i = 0
        for idx, row in vehicles_sorted.iterrows():
            if i < len(sa_optimized_routes):
                vid = list(sa_optimized_routes.keys())[i]
                final_route_assignments[row['vehicle_id']] = sa_optimized_routes[vid]
                i += 1
        if i < len(sa_optimized_routes):
            highest_vehicle = vehicles_sorted.iloc[0]
            existing = final_route_assignments.get(highest_vehicle['vehicle_id'], {})
            if existing and 'routes' in existing:
                routes_list = existing['routes']
            else:
                routes_list = [existing['route']] if existing else []
            for vid in list(sa_optimized_routes.keys())[i:]:
                routes_list.append(sa_optimized_routes[vid]['route'])
            total_distance = sum(compute_route_distance(r, self.dist_matrix) for r in routes_list)
            total_demand = sum(sum(self.demands[node] for node in r if node not in self.depot_indices) for r in routes_list)
            final_route_assignments[highest_vehicle['vehicle_id']] = {
                'route': routes_list,
                'distance': total_distance,
                'demand': total_demand,
                'vehicle': highest_vehicle.to_dict()
            }
        return final_route_assignments, []


    ##########################################################
    # OSRM tabanlı interaktif rota görselleştirme
    ##########################################################
    def create_advanced_route_map(self, route_costs, original_data, routing_data):
        import folium
        import pandas as pd
        from folium.plugins import PolyLineTextPath  # added for arrows

        # Center the map using the original data (so all nodes are visible).
        center_lat = original_data['Latitude'].mean()
        center_lon = original_data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        # --- Draw Markers for All Nodes from Original Data ---
        for idx, row in original_data.iterrows():
            node_type = row['node_type'].lower()
            deliver_type = str(row.get('deliver_type', 'none')).lower()
            lat, lon = row['Latitude'], row['Longitude']
            if node_type == 'depot':
                marker_color = 'black'
                icon_ = 'home'
                popup_text = f"Depo (ID: {row.get('ID', idx)})"
            elif node_type == 'locker':
                marker_color = 'gray'
                icon_ = 'lock'
                popup_text = f"Locker (ID: {row.get('ID', idx)})"
            else:
                marker_color = 'blue' if deliver_type == 'last_feet' else 'orange'
                icon_ = 'info-sign'
                popup_text = f"Müşteri (ID: {row.get('ID', idx)}) - Talep: {row['demand']} - {deliver_type}"
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon=icon_)
            ).add_to(m)

        # --- Draw Dashed Lines for Last Mile Assignments (from Original Data) ---
        assigned_lockers = original_data[original_data['deliver_type'] == 'last_mile']
        for _, row in assigned_lockers.iterrows():
            if pd.notnull(row.get('assigned_locker')):
                cust_coord = (row['Latitude'], row['Longitude'])
                locker_row = original_data[original_data['ID'] == row['assigned_locker']]
                if not locker_row.empty:
                    locker_coord = (locker_row.iloc[0]['Latitude'], locker_row.iloc[0]['Longitude'])
                    folium.PolyLine(
                        [cust_coord, locker_coord],
                        color='black',
                        weight=2,
                        opacity=0.7,
                        dash_array='5, 5',
                        tooltip=f"Last Mile: Müşteri {row.get('ID', '')} -> Locker {row.get('assigned_locker', '')}"
                    ).add_to(m)

        # --- Draw Routes using OSRM (based on routing_data) ---
        osrm_cache = {}
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                'darkblue', 'cadetblue', 'darkgreen', 'darkpurple', 'pink',
                'lightred', 'beige', 'lightblue', 'lightgreen', 'gray']
        for idx, (vid, rdata) in enumerate(route_costs.items()):
            route = rdata.get('route', None)
            if route is None or len(route) < 2:
                continue

            # Build coordinates for the route using routing_data.
            coords = []
            for node in route:
                try:
                    lat = routing_data.iloc[node]['Latitude']
                    lon = routing_data.iloc[node]['Longitude']
                    coords.append((lat, lon))
                except Exception as e:
                    continue

            # Use OSRM to fetch route segments.
            full_route_coords = []
            for i in range(len(coords) - 1):
                start_coord = coords[i]
                end_coord = coords[i+1]
                segment_coords = self.get_osrm_route(start_coord, end_coord, osrm_cache)
                if not segment_coords:
                    segment_coords = [start_coord, end_coord]
                if full_route_coords:
                    full_route_coords.extend(segment_coords[1:])
                else:
                    full_route_coords.extend(segment_coords)
            if full_route_coords:
                poly = folium.PolyLine(
                    full_route_coords,
                    color=colors[idx % len(colors)],
                    weight=4,
                    opacity=0.7,
                    popup=f"Araç {vid} Rotası (Maliyet: {rdata.get('cost', 0):.2f})",
                    tooltip=f"Araç ID: {vid}"
                ).add_to(m)
                # --- Add arrow symbols along the route for direction ---
                arrow = PolyLineTextPath(
                    poly,
                    ' ► ',
                    repeat=True,
                    offset=5,
                    attributes={'fill': 'black', 'font-size': '16', 'font-weight': 'bold'}
                )
                arrow.add_to(m)

        folium.LayerControl().add_to(m)
        return m


    def create_dashed_lines_map(self, m, routing_data, original_data):
        """
        Adds dashed lines on the Folium map (m) for any customer who was originally 'last_mile'
        but is now served via a locker. For an individual customer node with deliver_type 'locker_pickup',
        draws a dashed line from the customer's original home coordinates (orig_Latitude, orig_Longitude)
        to the current (locker) coordinates.
        For an aggregated node (node_type 'locker_cluster'), draws a dashed line from each served customer's 
        original coordinates (found in original_data using the served customer indices) to the locker cluster's 
        coordinates.
        
        Parameters:
        m: A Folium map object.
        routing_data: The updated routing DataFrame (with locker assignments).
        original_data: The original merged data (which contains each customer's original location).
        
        Returns:
        The Folium map object with dashed lines added.
        """
        import folium

        # For individual customer nodes
        for idx, row in routing_data.iterrows():
            node_type = row['node_type'].strip().lower()
            deliver_type = row.get('deliver_type', '').strip().lower()
            # For a single customer served via locker, draw a line.
            if node_type == 'customer' and deliver_type == 'locker_pickup':
                home_coord = (row['orig_Latitude'], row['orig_Longitude'])
                locker_coord = (row['Latitude'], row['Longitude'])
                folium.PolyLine(
                    locations=[home_coord, locker_coord],
                    color='black',
                    weight=2,
                    opacity=0.7,
                    dash_array='5, 5',
                    tooltip=f"Customer {row.get('ID', idx)}: Home -> Locker"
                ).add_to(m)
            # For aggregated nodes (locker_cluster), draw a dashed line for each served customer.
            elif node_type == 'locker_cluster':
                locker_coord = (row['Latitude'], row['Longitude'])
                served = row.get('served_customers', [])
                # served should be a list of customer indices.
                for cust_idx in served:
                    # Retrieve the customer's original coordinates from original_data.
                    if cust_idx in original_data.index:
                        cust_row = original_data.loc[cust_idx]
                        home_coord = (cust_row.get('orig_Latitude', cust_row['Latitude']),
                                    cust_row.get('orig_Longitude', cust_row['Longitude']))
                        folium.PolyLine(
                            locations=[home_coord, locker_coord],
                            color='black',
                            weight=2,
                            opacity=0.7,
                            dash_array='5, 5',
                            tooltip=f"Customer {cust_row.get('ID', cust_idx)}: Home -> Locker"
                        ).add_to(m)
        return m

