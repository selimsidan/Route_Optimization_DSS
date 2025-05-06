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

    def get_osrm_distance(self, start_coord, end_coord, cache):
        """
        Query OSRM for the road-network distance (in km) between two (lat,lon) points.
        Falls back to straight‐line if the request fails.
        """
        key = (start_coord, end_coord)
        if key in cache:
            return cache[key]
        
        lon1, lat1 = start_coord[1], start_coord[0]
        lon2, lat2 = end_coord[1],   end_coord[0]
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}"
            "?overview=false&annotations=distance"
        )
        try:
            r = requests.get(url, timeout=5).json()
            if r.get("code") == "Ok" and r.get("routes"):
                meters = r["routes"][0]["distance"]
                km = meters / 1000.0
                cache[key] = km
                return km
        except Exception:
            pass
        
        # fallback: planar distance
        km_per_degree = 111.0
        dlat = (lat1 - lat2) * km_per_degree
        dlon = (lon1 - lon2) * km_per_degree
        km = math.hypot(dlat, dlon)
        cache[key] = km
        return km

    def solve_vrp_heuristic(self, vehicles_df, forbidden_groups=None, locker_km_limit=100):
        import math, numpy as np, pandas as pd, random, copy

        if forbidden_groups is None:
            forbidden_groups = []

        # ─ Fleet inference ─
        fleet = vehicles_df.to_dict('records')
        fleet.sort(key=lambda v: v['cost_per_km'])
        available_vehicles = fleet.copy()
        max_vehicle_capacity = max(v['capacity'] for v in available_vehicles)

        # ─ Parameters ─
        cost_per_km = getattr(self, 'cost_per_km', vehicles_df['cost_per_km'].iloc[0])
        T, T_min, α, max_iter = 500.0, 1e-10, 0.99, 10000
        km_per_degree = 111.0

        # ─ Phase 0: Data Prep ─
        original_df = self.data.copy()
        routing_df  = self.data.copy()
        cust_mask   = routing_df['node_type'].str.lower() == 'customer'
        if 'orig_Latitude' not in routing_df.columns:
            routing_df.loc[cust_mask, 'orig_Latitude']  = routing_df.loc[cust_mask, 'Latitude']
            routing_df.loc[cust_mask, 'orig_Longitude'] = routing_df.loc[cust_mask, 'Longitude']

        def deg_distance(lat1, lon1, lat2, lon2):
            return math.hypot((lat1 - lat2) * km_per_degree,
                            (lon1 - lon2) * km_per_degree)

        # ─ Phase 1: Per-customer locker assignment with knapsack ─
        lm_mask        = cust_mask
        lm_indices     = routing_df[lm_mask].index.tolist()
        locker_indices = routing_df[routing_df['node_type'].str.lower() == 'locker'].index.tolist()

        best_choice = {}
        for i in lm_indices:
            # find nearest depot & direct cost
            d0 = min(self.depot_indices,
                    key=lambda d: deg_distance(
                        routing_df.at[d,'Latitude'], routing_df.at[d,'Longitude'],
                        routing_df.at[i,'Latitude'], routing_df.at[i,'Longitude']))
            direct_cost = deg_distance(
                routing_df.at[d0,'Latitude'], routing_df.at[d0,'Longitude'],
                routing_df.at[i,'Latitude'], routing_df.at[i,'Longitude']
            ) * cost_per_km

            best_sav, best_l = 0.0, None
            for l in locker_indices:
                # **NEW: skip lockers too far from this customer**
                dist_cl = deg_distance(
                    routing_df.at[l,'Latitude'], routing_df.at[l,'Longitude'],
                    routing_df.at[i,'orig_Latitude'], routing_df.at[i,'orig_Longitude']
                )
                if dist_cl > locker_km_limit:
                    continue

                # compute saving via locker l
                via_cost = (
                    deg_distance(
                        routing_df.at[d0,'Latitude'], routing_df.at[d0,'Longitude'],
                        routing_df.at[l,'Latitude'], routing_df.at[l,'Longitude']
                    ) * cost_per_km
                    + dist_cl * float(routing_df.at[i,'cost'])
                )
                sav = direct_cost - via_cost
                if sav > best_sav:
                    best_sav, best_l = sav, l

            if best_l is not None and best_sav > 0:
                best_choice[i] = (best_l, best_sav)

        # group by locker and run 0-1 knapsack
        aggregated_nodes = []
        agg_customers   = set()
        by_locker = {}
        for i,(l,sav) in best_choice.items():
            by_locker.setdefault(l, []).append((sav,
                                                float(routing_df.at[i,'demand']),
                                                i))

        for l, items in by_locker.items():
            locker_capacity = float(routing_df.at[l,'capacity'])
            pack_capacity   = int(min(locker_capacity, max_vehicle_capacity))

            # build knapsack tables
            n = len(items)
            weights = [int(round(dem)) for sav,dem,idx in items]
            values  = [sav for sav,dem,idx in items]
            idxs    = [idx for sav,dem,idx in items]

            dp_vals  = [0.0] * (pack_capacity + 1)
            dp_items = [[]    for _ in range(pack_capacity + 1)]

            for k in range(n):
                w, v, idx = weights[k], values[k], idxs[k]
                for cap in range(pack_capacity, w-1, -1):
                    if dp_vals[cap-w] + v > dp_vals[cap]:
                        dp_vals[cap] = dp_vals[cap-w] + v
                        dp_items[cap] = dp_items[cap-w] + [idx]

            best_cap = max(range(pack_capacity+1), key=lambda c: dp_vals[c])
            subset  = dp_items[best_cap]
            cum_d   = sum(routing_df.at[j,'demand'] for j in subset)
            if not subset:
                continue

            for i in subset:
                routing_df.at[i,'deliver_type'] = 'locker_pickup'
                routing_df.at[i,'Latitude']     = routing_df.at[l,'Latitude']
                routing_df.at[i,'Longitude']    = routing_df.at[l,'Longitude']

            aggregated_nodes.append({
                'node_type':        'locker_cluster',
                'deliver_type':     'locker_pickup',
                'Latitude':         routing_df.at[l,'Latitude'],
                'Longitude':        routing_df.at[l,'Longitude'],
                'load':             cum_d,
                'capacity':         locker_capacity,
                'served_customers': subset,
                'ID':               routing_df.at[l,'ID']
            })
            agg_customers.update(subset)

        # ─────────────── rebuild routing_df ───────────────
        new_nodes = []
        # depots
        for _, row in routing_df[routing_df['node_type'].str.lower()=='depot'].iterrows():
            new_nodes.append({
                'node_type':      row['node_type'],
                'deliver_type':   row['deliver_type'],
                'Latitude':       row['Latitude'],
                'Longitude':      row['Longitude'],
                'demand':         0.0,
                'orig_Latitude':  row.get('orig_Latitude', row['Latitude']),
                'orig_Longitude': row.get('orig_Longitude', row['Longitude']),
                'ID':             row.get('ID'),
                'cost':  0.0
            })
        # remaining customers
        non_agg = cust_mask & (~routing_df.index.isin(agg_customers))
        for _, row in routing_df[non_agg].iterrows():
            new_nodes.append({
                'node_type':      row['node_type'],
                'deliver_type':   row['deliver_type'],
                'Latitude':       row['Latitude'],
                'Longitude':      row['Longitude'],
                'demand':         row['demand'],
                'orig_Latitude':  row.get('orig_Latitude', row['Latitude']),
                'orig_Longitude': row.get('orig_Longitude', row['Longitude']),
                'ID':             row.get('ID'),
                'cost':  float(row['cost'])
            })
        # locker clusters
        for node in aggregated_nodes:
            node['cost'] = 0.0
            new_nodes.append(node.copy())

        routing_df = pd.DataFrame(new_nodes).reset_index(drop=True)
        routing_df.index.name = 'new_index'; routing_df.reset_index(inplace=True)

        # ─────────────── Phase 2: Distance matrix ───────────────
        def build_dist_mat(df):
            n = len(df)
            M = [[0.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        M[i][j] = deg_distance(
                            df.loc[i,'Latitude'], df.loc[i,'Longitude'],
                            df.loc[j,'Latitude'], df.loc[j,'Longitude']
                        )
            return M

        dist_mat = build_dist_mat(routing_df)
        depot_indices_final = routing_df[routing_df['node_type'].str.lower()=='depot'].index.tolist()

        # ─────────────── Phase 3: Clarke–Wright with safe merges ───────────────
        def run_clarke_wright(dm, df, depots, capacity):
            init_routes, route_of, route_demand = {}, {}, {}
            # initial single‐customer routes
            for i in df.index:
                if df.loc[i,'node_type'].strip().lower() != 'depot':
                    d0 = min(depots, key=lambda d: dm[d][i])
                    init_routes[i]  = [d0, i, d0]
                    route_of[i]     = i
                    route_demand[i] = float(df.loc[i,'load'] if df.loc[i,'node_type']=='locker_cluster'
                                            else df.loc[i,'demand'])
            # savings list
            sav = []
            for i in init_routes:
                for j in init_routes:
                    if i < j and init_routes[i][0] == init_routes[j][0]:
                        s = dm[init_routes[i][0]][i] + dm[init_routes[j][0]][j] - dm[i][j]
                        sav.append((s,i,j))
            sav.sort(reverse=True, key=lambda x: x[0])

            # safe merging
            for _, i, j in sav:
                # current route IDs
                ri = route_of.get(i)
                rj = route_of.get(j)
                # skip if already merged or invalid
                if ri is None or rj is None or ri == rj:
                    continue
                if ri not in init_routes or rj not in init_routes:
                    continue
                # capacity check
                if route_demand[ri] + route_demand[rj] > capacity:
                    continue

                # merge j into i
                init_routes[ri] = init_routes[ri][:-1] + init_routes[rj][1:]
                route_demand[ri] += route_demand[rj]
                # reassign all nodes in rj to route ri
                for node in init_routes[rj][1:-1]:
                    route_of[node] = ri
                # remove old route
                del init_routes[rj]
                del route_demand[rj]

            # finalize routes
            final = []
            for r in init_routes.values():
                if r[-1] != r[0]:
                    r.append(r[0])
                final.append(r)
            return final

        final_routes = run_clarke_wright(dist_mat,
                                        routing_df,
                                        depot_indices_final,
                                        max_vehicle_capacity)

        # ─────────────── Phase 4: SA + operators ───────────────
        def compute_cost(routes):
            tot = 0.0
            for r in routes:
                load = 0.0
                for n in r:
                    nd = routing_df.loc[n]
                    load += float(nd['load'] if nd['node_type']=='locker_cluster'
                                else nd['demand'])
                if load > max_vehicle_capacity:
                    return float('inf')
                for a,b in zip(r,r[1:]):
                    tot += dist_mat[a][b] * cost_per_km
                for n in r:
                    row = routing_df.loc[n]
                    nt, dt = row['node_type'].lower(), row['deliver_type'].lower()
                    if nt=='customer' and dt=='locker_pickup':
                        tot += deg_distance(
                            row['orig_Latitude'], row['orig_Longitude'],
                            row['Latitude'], row['Longitude']
                        ) * float(row['cost'])
                    elif nt=='locker_cluster':
                        for ci in row['served_customers']:
                            cro = original_df.loc[ci]
                            tot += deg_distance(
                                cro['Latitude'], cro['Longitude'],
                                row['Latitude'], row['Longitude']
                            ) * float(cro['cost'])
            return tot

        def move_op(routes):
            cand = copy.deepcopy(routes)
            if len(cand) < 2: return routes
            fr = random.randrange(len(cand)); r0 = cand[fr]
            custs = [k for k,n in enumerate(r0) if r0[0]!=n!=r0[-1]]
            if not custs: return routes
            ci = random.choice(custs); c = r0[ci]
            cand[fr] = r0[:ci] + r0[ci+1:]
            to = random.choice([k for k in range(len(cand)) if k!=fr])
            r1 = cand[to]
            best_c, best_p = float('inf'), 1
            for p in range(1,len(r1)):
                cc = dist_mat[r1[p-1]][c] + dist_mat[c][r1[p]] - dist_mat[r1[p-1]][r1[p]]
                if cc < best_c:
                    best_c, best_p = cc, p
            cand[to] = r1[:best_p] + [c] + r1[best_p:]
            return cand

        def swap_op(routes):
            cand = copy.deepcopy(routes)
            if len(cand) < 2: return routes
            a,b = random.sample(range(len(cand)),2)
            r1, r2 = cand[a], cand[b]
            c1 = [k for k,n in enumerate(r1) if r1[0]!=n!=r1[-1]]
            c2 = [k for k,n in enumerate(r2) if r2[0]!=n!=r2[-1]]
            if not c1 or not c2: return routes
            i1,i2 = random.choice(c1), random.choice(c2)
            r1[i1], r2[i2] = r2[i2], r1[i1]
            return cand

        def two_opt(r):
            best, improved = r.copy(), True
            while improved:
                improved = False
                for i in range(1,len(r)-2):
                    for j in range(i+2,len(r)):
                        if j-i == 1: continue
                        cand = r[:i] + r[i:j][::-1] + r[j:]
                        old = sum(dist_mat[r[k]][r[k+1]] for k in range(len(r)-1))
                        new = sum(dist_mat[cand[k]][cand[k+1]] for k in range(len(cand)-1))
                        if new < old:
                            best, improved, r = cand, True, cand
            return best

        current, best_ = copy.deepcopy(final_routes), copy.deepcopy(final_routes)
        c_cost, b_cost = compute_cost(current), None
        b_cost = c_cost
        it = 0
        while T > T_min and it < max_iter:
            u = random.random()
            if   u < 0.2:
                cand = run_clarke_wright(dist_mat, routing_df, depot_indices_final, max_vehicle_capacity)
            elif u < 0.4:
                cand = move_op(current)
            elif u < 0.6:
                cand = swap_op(current)
            else:
                cand = copy.deepcopy(current)
                if cand:
                    idx = random.randrange(len(cand))
                    cand[idx] = two_opt(cand[idx])

            cand_cost = compute_cost(cand)
            delta     = cand_cost - c_cost
            if delta < 0 or random.random() < math.exp(-delta/T):
                current, c_cost = cand, cand_cost
                if c_cost < b_cost:
                    best_, b_cost = copy.deepcopy(cand), c_cost

            T *= α; it += 1

        final_routes = best_

        # ─────────────── Phase 5: Assign non-empty routes ───────────────
        filtered = []
        for r in final_routes:
            load = sum(float(routing_df.loc[n,'load'] if routing_df.loc[n,'node_type']=='locker_cluster'
                            else routing_df.loc[n,'demand'])
                    for n in r)
            if load > 0:
                filtered.append((r, load))

        if not filtered:
            return {}, 0.0

        routes, demands = zip(*filtered)
        order = sorted(range(len(routes)), key=lambda i: demands[i], reverse=True)
        routes  = [routes[i]  for i in order]
        demands = [demands[i] for i in order]

        route_assignments = {}
        for r, rd in zip(routes, demands):
            fit = [v for v in available_vehicles if v['capacity'] >= rd]
            if not fit:
                raise RuntimeError(f"No single vehicle can carry demand={rd}")
            v = min(fit, key=lambda v: v['capacity'])
            available_vehicles.remove(v)

            display, road_km, penalty = [], 0.0, 0.0
            osrm_cache = {}
            for a,b in zip(r, r[1:]):
                sc = (routing_df.loc[a,'Latitude'], routing_df.loc[a,'Longitude'])
                ec = (routing_df.loc[b,'Latitude'], routing_df.loc[b,'Longitude'])
                road_km += self.get_osrm_distance(sc, ec, osrm_cache)

            for n in r:
                row = routing_df.loc[n]
                nt, dt = row['node_type'].lower(), row['deliver_type'].lower()
                if nt=='customer' and dt=='locker_pickup':
                    penalty += deg_distance(
                        row['orig_Latitude'], row['orig_Longitude'],
                        row['Latitude'], row['Longitude']
                    ) * float(row['cost'])
                elif nt=='locker_cluster':
                    for ci in row['served_customers']:
                        cro = original_df.loc[ci]
                        penalty += deg_distance(
                            cro['Latitude'], cro['Longitude'],
                            row['Latitude'], row['Longitude']
                        ) * float(cro['cost'])
                    print(f"Locker {row['ID']} carrying {row['load']} / {row['capacity']}")
                display.append(
                    f"Locker({row['ID']})" 
                    if nt=='customer' and dt=='locker_pickup' 
                    else row['ID']
                )

            cost  = road_km * v['cost_per_km']
            total = cost + penalty + v.get('fixed_cost', 0.0)
            vid   = v['vehicle_id']
            route_assignments[vid] = {
                'route':         r,
                'display_route': display,
                'distance':      road_km,
                'demand':        rd,
                'penalty':       penalty,
                'cost':          total,
                'vehicle':       v
            }

        total_cost = sum(r['cost'] for r in route_assignments.values())
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
                        opacity=0.1,
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
                    weight=1,
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