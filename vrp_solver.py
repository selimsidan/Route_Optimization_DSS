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
        import math, numpy as np, pandas as pd, random, copy
        from sklearn.cluster import DBSCAN

        if forbidden_groups is None:
            forbidden_groups = []

        # PARAMETERS
        cost_per_km      = getattr(self, 'cost_per_km', vehicles_df['cost_per_km'].iloc[0])
        vehicle_capacity = vehicles_df['capacity'].max()
        # SA parameters:
        T        = 500.0
        T_min    = 1e-10
        alpha    = 0.99
        max_iter = 10000
        km_per_degree = 111.0

        # ------------------
        # Phase 0: Data Preparation.
        original_df = self.data.copy()
        routing_df  = self.data.copy()
        cust_mask   = routing_df['node_type'].str.lower() == 'customer'
        if 'orig_Latitude' not in routing_df.columns:
            routing_df.loc[cust_mask, 'orig_Latitude']  = routing_df.loc[cust_mask, 'Latitude']
        if 'orig_Longitude' not in routing_df.columns:
            routing_df.loc[cust_mask, 'orig_Longitude'] = routing_df.loc[cust_mask, 'Longitude']

        # ------------------
        # Helper: distance in km between two points.
        def deg_distance(lat1, lon1, lat2, lon2):
            return math.hypot((lat1 - lat2) * km_per_degree,
                            (lon1 - lon2) * km_per_degree)

        # ------------------
        # Phase 1: Clustering & Aggregation.
        mask_last_mile = cust_mask & (routing_df['deliver_type'].str.lower() == 'last_mile')
        last_mile_indices = routing_df[mask_last_mile].index.tolist()
        aggregated_nodes = []
        aggregated_customer_indices = set()
        if last_mile_indices:
            coords = routing_df.loc[last_mile_indices, ['Latitude','Longitude']].values
            clustering = DBSCAN(eps=0.01, min_samples=1).fit(coords)
            labels = clustering.labels_
            cluster_map = {}
            for idx, label in zip(last_mile_indices, labels):
                cluster_map.setdefault(label, []).append(idx)
            locker_indices = routing_df[routing_df['node_type'].str.lower()=='locker'].index.tolist()
            for label, indices in cluster_map.items():
                total_direct_cost = 0.0
                for i in indices:
                    nearest_depot = min(self.depot_indices, key=lambda d: deg_distance(
                        routing_df.loc[d,'Latitude'], routing_df.loc[d,'Longitude'],
                        routing_df.loc[i,'Latitude'], routing_df.loc[i,'Longitude']
                    ))
                    total_direct_cost += deg_distance(
                        routing_df.loc[nearest_depot,'Latitude'], routing_df.loc[nearest_depot,'Longitude'],
                        routing_df.loc[i,'Latitude'], routing_df.loc[i,'Longitude']
                    ) * cost_per_km

                best_total_locker_cost = float('inf')
                best_locker_for_cluster = None
                for l in locker_indices:
                    cluster_cost = 0.0
                    for i in indices:
                        nearest_depot = min(self.depot_indices, key=lambda d: deg_distance(
                            routing_df.loc[d,'Latitude'], routing_df.loc[d,'Longitude'],
                            routing_df.loc[i,'Latitude'], routing_df.loc[i,'Longitude']
                        ))
                        cost_i = (
                            deg_distance(routing_df.loc[nearest_depot,'Latitude'],
                                        routing_df.loc[nearest_depot,'Longitude'],
                                        routing_df.loc[l,'Latitude'],
                                        routing_df.loc[l,'Longitude']) * cost_per_km
                            + deg_distance(routing_df.loc[l,'Latitude'],
                                        routing_df.loc[l,'Longitude'],
                                        routing_df.loc[i,'orig_Latitude'],
                                        routing_df.loc[i,'orig_Longitude']) * float(routing_df.loc[i,'customer_cost'])
                        )
                        cluster_cost += cost_i
                    if cluster_cost < best_total_locker_cost:
                        best_total_locker_cost = cluster_cost
                        best_locker_for_cluster = l

                if best_total_locker_cost < total_direct_cost and best_locker_for_cluster is not None:
                    for i in indices:
                        routing_df.loc[i,'Latitude']     = routing_df.loc[best_locker_for_cluster,'Latitude']
                        routing_df.loc[i,'Longitude']    = routing_df.loc[best_locker_for_cluster,'Longitude']
                        routing_df.loc[i,'deliver_type'] = 'locker_pickup'
                        self.locker_assignments = getattr(self,'locker_assignments',{})
                        self.locker_assignments[i] = {
                            'assigned_locker': best_locker_for_cluster,
                            'cluster': label,
                            'locker_cost': best_total_locker_cost/len(indices)
                        }
                    agg_demand = sum(routing_df.loc[i,'demand'] for i in indices)
                    aggregated_nodes.append({
                        'new_index': None,
                        'original_index': indices,
                        'node_type': 'locker_cluster',
                        'deliver_type': 'locker_pickup',
                        'Latitude': routing_df.loc[best_locker_for_cluster,'Latitude'],
                        'Longitude': routing_df.loc[best_locker_for_cluster,'Longitude'],
                        'demand': agg_demand,
                        'served_customers': indices,
                        'ID': routing_df.loc[best_locker_for_cluster,'ID'] if 'ID' in routing_df.columns else f"Locker({best_locker_for_cluster})",
                        'assigned_locker': best_locker_for_cluster
                    })
                    aggregated_customer_indices.update(indices)

        # Merge nearly‑identical aggregated nodes
        tolerance = 0.001
        unique_agg = []
        for node in aggregated_nodes:
            merged = False
            for u in unique_agg:
                if (abs(u['Latitude'] - node['Latitude']) < tolerance and
                    abs(u['Longitude'] - node['Longitude']) < tolerance):
                    u['served_customers'].extend(node['served_customers'])
                    u['demand'] += node['demand']
                    if 'assigned_locker' in node:
                        u['ID'] = routing_df.loc[node['assigned_locker'],'ID']
                        u['assigned_locker'] = node['assigned_locker']
                    else:
                        u['ID'] = f"LockerCluster({','.join(str(x) for x in u['served_customers'])})"
                    merged = True
                    break
            if not merged:
                cpy = node.copy()
                if not isinstance(cpy.get('served_customers'), list):
                    cpy['served_customers'] = [cpy['original_index']]
                unique_agg.append(cpy)
        aggregated_nodes = unique_agg

        # ------------------
        # Build new routing_df
        new_nodes = []
        # 1) depots
        for idx, row in routing_df[routing_df['node_type'].str.lower()=='depot'].iterrows():
            new_nodes.append({
                'node_type': row['node_type'],
                'deliver_type': row['deliver_type'],
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'demand': 0.0,
                'orig_Latitude': row.get('orig_Latitude',row['Latitude']),
                'orig_Longitude':row.get('orig_Longitude',row['Longitude']),
                'ID': row.get('ID',idx)
            })
        # 2) non‑agg customers
        non_agg_mask = cust_mask & (~routing_df.index.isin(aggregated_customer_indices))
        for idx, row in routing_df[non_agg_mask].iterrows():
            new_nodes.append({
                'node_type': row['node_type'],
                'deliver_type': row['deliver_type'],
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'demand': row['demand'],
                'orig_Latitude': row.get('orig_Latitude',row['Latitude']),
                'orig_Longitude':row.get('orig_Longitude',row['Longitude']),
                'ID': row.get('ID',idx)
            })
        # 3) aggregated clusters
        for node in aggregated_nodes:
            new_nodes.append({
                'node_type': node['node_type'],
                'deliver_type': node['deliver_type'],
                'Latitude': node['Latitude'],
                'Longitude': node['Longitude'],
                'demand': node['demand'],
                'served_customers': node['served_customers'],
                'ID': node['ID']
            })

        new_routing_df = pd.DataFrame(new_nodes).reset_index(drop=True)
        new_routing_df.index.name = 'new_index'
        new_routing_df.reset_index(inplace=True)
        routing_df = new_routing_df.copy()

        # ------------------
        # Phase 2: Distance matrix & depot list
        def build_dist_mat(df):
            n = len(df)
            mat = [[0.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    mat[i][j] = deg_distance(df.loc[i,'Latitude'], df.loc[i,'Longitude'],
                                            df.loc[j,'Latitude'], df.loc[j,'Longitude'])
            return mat

        new_dist_mat = build_dist_mat(routing_df)
        depot_indices_final = routing_df[routing_df['node_type'].str.lower()=='depot'].index.tolist()

        # ------------------
        # Phase 3: Clarke–Wright Savings
        def run_clarke_wright(dist_mat, df, depots):
            init_routes = {}
            for idx in df.index:
                if df.loc[idx,'node_type'].strip().lower()!='depot':
                    d0 = min(depots, key=lambda d: dist_mat[d][idx])
                    init_routes[idx] = [d0, idx, d0]
            savings = []
            for i in init_routes:
                for j in init_routes:
                    if i>=j: continue
                    if init_routes[i][0]==init_routes[j][0]:
                        s = dist_mat[init_routes[i][0]][i] + dist_mat[init_routes[i][0]][j] - dist_mat[i][j]
                        savings.append((s,i,j))
            savings.sort(reverse=True, key=lambda x: x[0])
            route_of = {i:i for i in init_routes}
            for s,i,j in savings:
                ri, rj = route_of[i], route_of[j]
                if ri!=rj and init_routes[ri][0]==init_routes[rj][0]:
                    merged = init_routes[ri][:-1] + init_routes[rj][1:]
                    init_routes[ri] = merged
                    for node in init_routes[rj][1:-1]:
                        route_of[node] = ri
                    del init_routes[rj]
            final = list(init_routes.values())
            for r in final:
                if r[-1]!=r[0]:
                    r.append(r[0])
            return final

        final_routes = run_clarke_wright(new_dist_mat, routing_df, depot_indices_final)

        # ------------------
        # NEW: Move and Swap operators
        # ------------------
        
        def move_operator(routes, dist_mat):
            """Move a customer from one route to another"""
            if len(routes) < 2:
                return routes
                
            # Pick random route and customer
            from_idx = random.randrange(len(routes))
            from_route = routes[from_idx]
            
            # Find customers (non-depot nodes) in the route
            customers = [i for i, node in enumerate(from_route) 
                        if from_route[0] != node != from_route[-1]]
            
            if not customers:
                return routes
                
            # Select random customer to move
            cust_idx = random.choice(customers)
            customer = from_route[cust_idx]
            
            # Remove from original route
            new_from_route = from_route[:cust_idx] + from_route[cust_idx+1:]
            
            # Select destination route
            to_idx = random.choice([i for i in range(len(routes)) if i != from_idx])
            to_route = routes[to_idx]
            
            # Find best insertion position
            best_cost = float('inf')
            best_pos = 1
            for pos in range(1, len(to_route)):
                cost = (dist_mat[to_route[pos-1]][customer] + 
                    dist_mat[customer][to_route[pos]] -
                    dist_mat[to_route[pos-1]][to_route[pos]])
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            # Insert customer at best position
            new_to_route = to_route[:best_pos] + [customer] + to_route[best_pos:]
            
            # Update routes
            new_routes = routes.copy()
            new_routes[from_idx] = new_from_route
            new_routes[to_idx] = new_to_route
            
            return new_routes
        
        def swap_operator(routes, dist_mat):
            """Swap customers between two routes"""
            if len(routes) < 2:
                return routes
                
            # Pick two different routes
            route_indices = random.sample(range(len(routes)), 2)
            route1_idx, route2_idx = route_indices
            route1, route2 = routes[route1_idx], routes[route2_idx]
            
            # Find customers in each route
            customers1 = [i for i, node in enumerate(route1) 
                        if route1[0] != node != route1[-1]]
            customers2 = [i for i, node in enumerate(route2) 
                        if route2[0] != node != route2[-1]]
            
            if not customers1 or not customers2:
                return routes
                
            # Select random customers to swap
            cust1_idx = random.choice(customers1)
            cust2_idx = random.choice(customers2)
            cust1, cust2 = route1[cust1_idx], route2[cust2_idx]
            
            # Swap customers
            new_route1 = route1.copy()
            new_route2 = route2.copy()
            new_route1[cust1_idx] = cust2
            new_route2[cust2_idx] = cust1
            
            # Update routes
            new_routes = routes.copy()
            new_routes[route1_idx] = new_route1
            new_routes[route2_idx] = new_route2
            
            return new_routes
        
        # ------------------
        # Phase 4: Simulated Annealing with multiple operators
        # ------------------
        
        def compute_solution_cost(routes, dist_mat, df):
            total = 0.0
            for r in routes:
                for a,b in zip(r,r[1:]):
                    total += dist_mat[a][b] * cost_per_km
                for n in r:
                    row = df.loc[n]
                    nt = row['node_type'].strip().lower()
                    dt = row['deliver_type'].strip().lower()
                    if nt=='customer' and dt=='locker_pickup':
                        total += deg_distance(
                            row['orig_Latitude'], row['orig_Longitude'],
                            row['Latitude'],      row['Longitude']
                        ) * float(row['customer_cost'])
                    elif nt=='locker_cluster':
                        for ci in row['served_customers']:
                            cro = original_df.loc[ci]
                            total += deg_distance(
                                cro['Latitude'], cro['Longitude'],
                                row['Latitude'],  row['Longitude']
                            ) * float(cro['customer_cost'])
            return total

        def two_opt_simple(route, dist_mat):
            """Simple 2-opt for a single route"""
            best_route = route.copy()
            best_distance = sum(dist_mat[best_route[i]][best_route[i+1]] 
                            for i in range(len(best_route)-1))
            
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+2, len(route)):
                        if j - i == 1: continue
                        new_route = route[:i] + route[i:j][::-1] + route[j:]
                        new_distance = sum(dist_mat[new_route[k]][new_route[k+1]] 
                                        for k in range(len(new_route)-1))
                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance
                            route = new_route
                            improved = True
            return best_route

        current_routes = copy.deepcopy(final_routes)
        current_cost   = compute_solution_cost(current_routes, new_dist_mat, routing_df)
        best_routes    = copy.deepcopy(current_routes)
        best_cost      = current_cost

        iter_count = 0
        while T > T_min and iter_count < max_iter:
            # Choose operator randomly
            operator_choice = random.random()
            
            if operator_choice < 0.2:  # 20% probability for mode toggle
                # Original mode‑toggle + re‑Clarke–Wright
                cand_df      = routing_df  # No mode toggle needed for now
                cand_dist_mat= new_dist_mat
                cand_routes  = run_clarke_wright(cand_dist_mat, cand_df, depot_indices_final)
            
            elif operator_choice < 0.4:  # 20% probability for move
                cand_df      = routing_df
                cand_dist_mat= new_dist_mat
                cand_routes  = move_operator(copy.deepcopy(current_routes), cand_dist_mat)
            
            elif operator_choice < 0.6:  # 20% probability for swap
                cand_df      = routing_df
                cand_dist_mat= new_dist_mat
                cand_routes  = swap_operator(copy.deepcopy(current_routes), cand_dist_mat)
            
            else:  # 40% probability for 2-opt
                cand_df      = routing_df
                cand_dist_mat= new_dist_mat
                cand_routes  = copy.deepcopy(current_routes)
                if cand_routes:
                    ridx = random.randrange(len(cand_routes))
                    cand_routes[ridx] = two_opt_simple(cand_routes[ridx], cand_dist_mat)
            
            cand_cost = compute_solution_cost(cand_routes, cand_dist_mat, cand_df)
            delta     = cand_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta/T):
                routing_df     = cand_df
                new_dist_mat   = cand_dist_mat
                current_routes = cand_routes
                current_cost   = cand_cost
                
                if current_cost < best_cost:
                    best_cost   = current_cost
                    best_routes = copy.deepcopy(cand_routes)
            
            T *= alpha
            iter_count += 1

        final_routes = best_routes

        # ------------------
        # Phase 5: Greedy Vehicle Assignment.
        vehicles_sorted = vehicles_df.sort_values(by='cost_per_km', ascending=True)
        route_assignments = {}
        fr_idx = 0
        for _, veh in vehicles_sorted.iterrows():
            if fr_idx >= len(final_routes):
                break
            r = final_routes[fr_idx]
            display = []
            for n in r:
                row = routing_df.loc[n]
                nt, dt = row['node_type'].lower(), row['deliver_type'].lower()
                if nt=='customer' and dt=='locker_pickup':
                    display.append(f"Locker({row['ID']})")
                else:
                    display.append(row['ID'])
            dist = sum(new_dist_mat[a][b] for a,b in zip(r,r[1:]))
            demand = sum(routing_df.loc[n,'demand']
                        for n in r if routing_df.loc[n,'node_type'].lower()=='customer')
            cost = dist * veh['cost_per_km']
            penalty = 0.0
            for n in r:
                row = routing_df.loc[n]
                nt, dt = row['node_type'].lower(), row['deliver_type'].lower()
                if nt=='customer' and dt=='locker_pickup':
                    penalty += deg_distance(row['orig_Latitude'], row['orig_Longitude'],
                                            row['Latitude'], row['Longitude']) * float(row['customer_cost'])
                elif nt=='locker_cluster':
                    for ci in row['served_customers']:
                        cro = original_df.loc[ci]
                        penalty += deg_distance(cro['Latitude'], cro['Longitude'],
                                                row['Latitude'], row['Longitude']) * float(cro['customer_cost'])
            total = cost + penalty
            route_assignments[veh['vehicle_id']] = {
                'route': r,
                'display_route': display,
                'distance': dist,
                'penalty': penalty,
                'cost': total,
                'demand': demand,
                'vehicle': veh.to_dict()
            }
            fr_idx += 1

        # FIX: pull fixed_cost from nested dict
        total_cost = sum(
            rdata['vehicle'].get('fixed_cost', 0.0) + rdata['cost']
            for rdata in route_assignments.values()
        )

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

