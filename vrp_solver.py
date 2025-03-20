import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from folium import Icon
import pandas as pd
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

    ##########################################################
    # Heuristic Yöntem – Feasible Sweep Algoritması
    ##########################################################
    def solve_vrp_heuristic(self, vehicles_df, forbidden_groups=None):
        if forbidden_groups is None:
            forbidden_groups = []
        speed = 50.0
        max_capacity = vehicles_df['capacity'].max()
        max_duration_minutes = vehicles_df['max_duration'].max()
        max_distance = (max_duration_minutes / 60.0) * speed
        N = list(range(len(self.data)))
        D = self.depot_indices

        # (A) Her non‑depot node, en yakın depoya göre gruplandırılsın.
        depot_groups = {d: [] for d in D}
        for i in N:
            if i in D:
                continue
            d_nearest = min(D, key=lambda d: self.dist_matrix[d][i])
            depot_groups[d_nearest].append(i)

        # (B) Her depo grubu için sweep algoritması.
        routes_all = []
        for depot, nodes in depot_groups.items():
            if not nodes:
                continue
            depot_coord = self.coordinates[depot]
            sorted_nodes = sorted(nodes, key=lambda i: math.atan2(
                self.data.loc[i, 'Latitude'] - depot_coord[0],
                self.data.loc[i, 'Longitude'] - depot_coord[1]
            ))
            current_route = [depot]
            current_demand = 0.0
            current_distance = 0.0
            for node in sorted_nodes:
                node_demand = self.demands[node]
                candidate_id = self.data.loc[node, 'ID']
                route_ids = [self.data.loc[n, 'ID'] for n in current_route if n != depot]
                conflict = any(candidate_id in group and any(rid in group for rid in route_ids)
                               for group in forbidden_groups)
                if conflict:
                    current_route.append(depot)
                    routes_all.append(current_route)
                    current_route = [depot]
                    current_demand = 0.0
                    current_distance = 0.0
                last_node = current_route[-1]
                extra = self.dist_matrix[last_node][node] + self.dist_matrix[node][depot] - self.dist_matrix[last_node][depot]
                if (current_demand + node_demand <= max_capacity) and (current_distance + extra <= max_distance):
                    current_route.append(node)
                    current_demand += node_demand
                    current_distance += extra
                else:
                    current_route.append(depot)
                    routes_all.append(current_route)
                    current_route = [depot, node]
                    current_demand = node_demand
                    current_distance = self.dist_matrix[depot][node] + self.dist_matrix[node][depot]
            if current_route[-1] != depot:
                current_route.append(depot)
            routes_all.append(current_route)

        # (C) Optimizasyon ve forbidden kontrolü.
        optimized_routes = []
        for route in routes_all:
            if len(route) > 3:
                route_opt = two_opt(route, self.dist_matrix)
                route_opt = three_opt(route_opt, self.dist_matrix)
            else:
                route_opt = route
            route_opt = self.fix_route_forbidden(route_opt, forbidden_groups)
            optimized_routes.append(route_opt)

        # (D) Rotaların araçlara ataması.
        vehicles_sorted = vehicles_df.sort_values(by='capacity', ascending=False)
        route_assignments = {}
        i = 0
        for idx, row in vehicles_sorted.iterrows():
            if i < len(optimized_routes):
                route = optimized_routes[i]
                route_demand = sum(self.demands[node] for node in route if node not in D)
                route_distance = compute_route_distance(route, self.dist_matrix)
                route_assignments[row['vehicle_id']] = {
                    'route': route,
                    'distance': route_distance,
                    'demand': route_demand,
                    'vehicle': row.to_dict()
                }
                i += 1
        if i < len(optimized_routes):
            highest_vehicle = vehicles_sorted.iloc[0]
            existing = route_assignments.get(highest_vehicle['vehicle_id'], {})
            if existing and 'routes' in existing:
                routes_list = existing['routes']
            else:
                routes_list = [existing['route']] if existing else []
            for r in optimized_routes[i:]:
                routes_list.append(r)
            total_distance = sum(compute_route_distance(r, self.dist_matrix) for r in routes_list)
            total_demand = sum(sum(self.demands[node] for node in r if node not in D) for r in routes_list)
            route_assignments[highest_vehicle['vehicle_id']] = {
                'route': routes_list,
                'distance': total_distance,
                'demand': total_demand,
                'vehicle': highest_vehicle.to_dict()
            }
        return route_assignments, []

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
    def create_advanced_route_map(self, route_costs, data_source):
        # Harita merkezi
        center_lat = self.data['Latitude'].mean()
        center_lon = self.data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Daha belirgin ve ayırt edilebilir renkler
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                'darkblue', 'cadetblue', 'darkgreen', 'darkpurple', 'pink',
                'lightred', 'beige', 'lightblue', 'lightgreen', 'gray']
        
        # Araçları renklerle eşleştiren bir sözlük oluşturalım 
        vehicle_colors = {}
        for idx, vid in enumerate(route_costs.keys()):
            vehicle_colors[vid] = colors[idx % len(colors)]
        
        # Araç renk lejantını ekleyelim
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; 
        background-color: white; border: 2px solid grey; z-index: 9999; font-size: 14px;
        padding: 10px; border-radius: 5px;">
        <p style="margin:0; font-weight: bold;">Araç Renkleri</p>
        '''

        for vid, color in vehicle_colors.items():
            legend_html += f'<p style="margin:0;"><span style="background-color:{color}; ' + \
                        f'display: inline-block; width: 15px; height: 15px; margin-right: 5px;"></span>Araç {vid}</p>'

        legend_html += '</div>'
        
        # Tüm nodelara simge ekleyelim
        for idx, row in data_source.iterrows():
            node_type = row['node_type']
            deliver_type = row.get('deliver_type', 'none')
            lat, lon = row['Latitude'], row['Longitude']
            if node_type == 'depot':
                marker_color = 'black'
                icon_ = 'factory'  # Depo için fabrika simgesi
                popup_text = f"Depo (ID: {row['ID']})"
            elif node_type == 'locker':
                marker_color = 'gray'
                icon_ = 'lock'
                popup_text = f"Locker (ID: {row['ID']}) - Kapasite: {row.get('remaining_capacity', row['demand'])}"
            else:
                marker_color = 'blue' if deliver_type == 'last_feet' else 'orange'
                icon_ = 'home'  # Müşteri için ev simgesi
                popup_text = f"Müşteri (ID: {row['ID']}) - Talep: {row['demand']} - {deliver_type}"

            # Marker'ı ekleyelim
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon=icon_)
            ).add_to(m)

        # Last Mile atamalarını kesikli çizgi ile gösterelim
        assigned_lockers = data_source[data_source['deliver_type'] == 'last_mile']
        for _, row in assigned_lockers.iterrows():
            if pd.notnull(row.get('assigned_locker')):
                cust_coord = (row['Latitude'], row['Longitude'])
                locker_row = data_source[data_source['ID'] == row['assigned_locker']]
                if not locker_row.empty:
                    locker_coord = (locker_row.iloc[0]['Latitude'], locker_row.iloc[0]['Longitude'])
                    folium.PolyLine(
                        [cust_coord, locker_coord],
                        color='black',
                        weight=2,
                        opacity=0.7,
                        dash_array='5, 5',
                        tooltip=f"Last Mile: Müşteri {row['ID']} -> Locker {row['assigned_locker']}"
                    ).add_to(m)

        # Rota segmentlerini OSRM ile çizelim ve oklar ekleyelim
        osrm_cache = {}
        for vid, rdata in route_costs.items():
            route_val = rdata.get('route', None)
            if route_val is None or isinstance(route_val, int):
                continue
            if isinstance(route_val, list):
                if len(route_val) > 0 and isinstance(route_val[0], int):
                    routes = [route_val]
                else:
                    routes = route_val
            else:
                continue
                
            color = vehicle_colors[vid]
            for route_idx, route in enumerate(routes):
                if not isinstance(route, list):
                    continue
                if not route or len(route) < 2:
                    continue
                    
                # Rotanın noktalarını ve detaylarını tutacak liste
                route_points = []
                node_ids = []
                
                for i in range(len(route) - 1):
                    start_node = route[i]
                    end_node = route[i + 1]
                    start_coord = (self.data.iloc[start_node]['Latitude'], self.data.iloc[start_node]['Longitude'])
                    end_coord = (self.data.iloc[end_node]['Latitude'], self.data.iloc[end_node]['Longitude'])
                    
                    # Düğüm ID'lerini saklayalım
                    start_id = self.data.iloc[start_node]['ID']
                    end_id = self.data.iloc[end_node]['ID']
                    node_ids.append((start_id, end_id))
                    
                    # OSRM ile gerçek yol koordinatlarını alalım
                    segment_coords = self.get_osrm_route(start_coord, end_coord, osrm_cache)
                    if not segment_coords:
                        segment_coords = [start_coord, end_coord]
                    
                    # Segment bilgilerini saklayalım
                    route_points.append({
                        'start': start_coord,
                        'end': end_coord,
                        'coords': segment_coords,
                        'start_id': start_id,
                        'end_id': end_id
                    })
                    
                    # Ana rotayı çizelim
                    line = folium.PolyLine(
                        segment_coords,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f"Araç {vid} - Segment {i+1}: {start_id} → {end_id}",
                        tooltip=f"Araç {vid}: {start_id} → {end_id}"
                    ).add_to(m)
                    
                    # Her segment için ok simgesi ekleyelim
                    if len(segment_coords) >= 2:
                        mid_idx = len(segment_coords) // 2
                        midpoint = segment_coords[mid_idx]
                        
                        # Add an arrow at the midpoint of the segment
                        folium.RegularPolygonMarker(
                            location=midpoint,
                            color=color,
                            number_of_sides=3,
                            radius=6,
                            rotation=0,  # No need for rotation calculation
                            fill_color=color,
                            fill_opacity=0.8,
                            popup=f"Yön: {start_id} → {end_id}"
                        ).add_to(m)
                # Tüm rotayı birleştirip genel bilgi ekleyelim
                all_coords = []
                for point_data in route_points:
                    if not all_coords:
                        all_coords.extend(point_data['coords'])
                    else:
                        all_coords.extend(point_data['coords'][1:])
                
                # Rota bilgisini ekleyelim
                route_id_str = " → ".join([str(self.data.iloc[n]['ID']) for n in route])
                folium.Popup(
                    f"""
                    <div style='width:200px'>
                    <b>Araç {vid} - Rota {route_idx+1}</b><br>
                    Toplam Mesafe: {compute_route_distance(route, self.dist_matrix):.2f} km<br>
                    Sıralama: {route_id_str}
                    </div>
                    """
                ).add_to(m)
        
        # Haritaya araç renk lejantını ekleyelim
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        
        return m