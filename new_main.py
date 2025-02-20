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

###############################################
# Yardımcı Fonksiyonlar: 2‑opt, 3‑opt ve rota mesafesi hesaplama
###############################################
def compute_route_distance(route, dist_matrix):
    """Verilen rota (indeks listesi) için toplam mesafeyi hesaplar.
       Rota; başlangıç ve bitiş depot’u içerir."""
    total = 0.0
    for i in range(len(route)-1):
        total += dist_matrix[route[i]][route[i+1]]
    return total

def two_opt(route, dist_matrix):
    """İki‑opt yerelleştirme algoritması."""
    best_route = route.copy()
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_distance = compute_route_distance(new_route, dist_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return best_route

def three_opt(route, dist_matrix):
    """Çok basit 3‑opt yerelleştirme; kısıtlı kombinasyon deneniyor."""
    best_route = route.copy()
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True
    n = len(route)
    while improved:
        improved = False
        for i in range(1, n-4):
            for j in range(i+1, n-2):
                for k in range(j+1, n):
                    segments = [best_route[:i], best_route[i:j], best_route[j:k], best_route[k:]]
                    candidate = segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3]
                    candidate_distance = compute_route_distance(candidate, dist_matrix)
                    if candidate_distance < best_distance:
                        best_route = candidate
                        best_distance = candidate_distance
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_route

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

    ##########################################################
    # OSRM tabanlı interaktif rota görselleştirme
    ##########################################################
    def create_advanced_route_map(self, route_costs, data_source):
        # data_source: görselleştirmede kullanılacak veri (orijinal ya da güncel VRP verisi)
        center_lat = self.data['Latitude'].mean()
        center_lon = self.data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                  'darkblue', 'cadetblue', 'darkgreen', 'darkpurple', 'pink',
                  'lightred', 'beige', 'lightblue', 'lightgreen', 'gray']

        # Tüm noktaları ekleyelim:
        for idx, row in data_source.iterrows():
            node_type = row['node_type']
            deliver_type = row.get('deliver_type', 'none')
            lat, lon = row['Latitude'], row['Longitude']
            if node_type == 'depot':
                marker_color = 'black'
                icon_ = 'home'
                popup_text = f"Depo (ID: {row['ID']})"
            elif node_type == 'locker':
                marker_color = 'gray'
                icon_ = 'lock'
                popup_text = f"Locker (ID: {row['ID']}) - Kapasite: {row.get('remaining_capacity', row['demand'])}"
            else:
                marker_color = 'blue' if deliver_type == 'last_feet' else 'orange'
                icon_ = 'info-sign'
                popup_text = f"Müşteri (ID: {row['ID']}) - Talep: {row['demand']} - {deliver_type}"
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon=icon_)
            ).add_to(m)

        # Last Mile atamalarını kesikli çizgi ile gösterelim.
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

        # Rota segmentlerini OSRM kullanarak çizelim.
        osrm_cache = {}
        for idx, (vid, rdata) in enumerate(route_costs.items()):
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
            color = colors[idx % len(colors)]
            for route in routes:
                if not isinstance(route, list):
                    continue
                if not route or len(route) < 2:
                    continue
                full_route_coords = []
                for i in range(len(route)-1):
                    start_node = route[i]
                    end_node = route[i+1]
                    start_coord = (self.data.iloc[start_node]['Latitude'], self.data.iloc[start_node]['Longitude'])
                    end_coord = (self.data.iloc[end_node]['Latitude'], self.data.iloc[end_node]['Longitude'])
                    segment_coords = self.get_osrm_route(start_coord, end_coord, osrm_cache)
                    if not segment_coords:
                        segment_coords = [start_coord, end_coord]
                    if full_route_coords:
                        full_route_coords.extend(segment_coords[1:])
                    else:
                        full_route_coords.extend(segment_coords)
                folium.PolyLine(
                    full_route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Araç {vid} Rotası (Maliyet: {rdata.get('cost', 0):.2f})",
                    tooltip=f"Araç ID: {vid}"
                ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

###############################################
# Streamlit Uygulaması – Dosya Yükleme ve Node Yönetimi
###############################################
st.set_page_config(page_title="Gelişmiş VRP Çözümleme", layout="wide")
st.title("🚚 Gelişmiş Araç Rotalama Problemi Çözücü (İstanbul)")

# Sidebar’da forbidden node grupları (önceki kısım)…
st.sidebar.subheader("Aynı Rotada Bulunamayacak Nodelar")
if 'forbidden_groups' not in st.session_state:
    st.session_state.forbidden_groups = []
all_node_ids = []
if st.session_state.get('original_data') is not None:
    all_node_ids = sorted(st.session_state.original_data['ID'].unique())
else:
    all_node_ids = list(range(100))
selected_nodes = st.sidebar.multiselect("Rota içinde yan yana bulunmasın istenen node ID’leri:", options=all_node_ids, key='forbidden_nodes_multiselect')
if st.sidebar.button("Ekle"):
    if selected_nodes:
        st.session_state.forbidden_groups.append(selected_nodes)
        st.sidebar.success(f"Eklendi: {selected_nodes}")
    else:
        st.sidebar.warning("Lütfen en az bir node seçiniz.")
if st.session_state.forbidden_groups:
    st.sidebar.write("Eklenen forbidden gruplar:")
    for grp in st.session_state.forbidden_groups:
        st.sidebar.write(grp)

# Dosya yükleme
uploaded_nodes_file = st.file_uploader("Lokasyon Dosyası Yükle", type="xlsx", key="nodes")
uploaded_vehicles_file = st.file_uploader("Araç Dosyası Yükle", type="xlsx", key="vehicles")

if uploaded_nodes_file is not None and uploaded_vehicles_file is not None:
    original_data = pd.read_excel(uploaded_nodes_file)
    vehicles_df = pd.read_excel(uploaded_vehicles_file)
    st.session_state.original_data = original_data.copy()
    needed_node_cols = ['Latitude', 'Longitude', 'demand', 'node_type', 'deliver_type']
    if not all(col in original_data.columns for col in needed_node_cols):
        st.error(f"Lokasyon dosyasında şu kolonlar eksik veya hatalı: {needed_node_cols}")
        st.stop()
    needed_vehicle_cols = ['vehicle_id', 'capacity', 'max_duration', 'cost_per_km', 'fixed_cost']
    if not all(col in vehicles_df.columns for col in needed_vehicle_cols):
        st.error(f"Araç dosyasında şu kolonlar eksik veya hatalı: {needed_vehicle_cols}")
        st.stop()
    depot_df = original_data[original_data['node_type'] == 'depot'].copy()
    locker_df = original_data[original_data['node_type'] == 'locker'].copy()
    customer_df = original_data[original_data['node_type'] == 'customer'].copy()
    if depot_df.empty:
        st.error("En az bir depo bulunmalıdır!")
        st.stop()
    if not locker_df.empty:
        locker_df['remaining_capacity'] = locker_df['demand']
    max_lock_distance = st.sidebar.number_input("Maksimum Locker Atama Mesafesi (km):", min_value=0.1, value=2.0, step=0.1)
    last_mile_mask = (customer_df['deliver_type'] == 'last_mile')
    for idx, row in customer_df[last_mile_mask].iterrows():
        cust_coord = (row['Latitude'], row['Longitude'])
        candidate_lockers = []
        for l_idx, l_row in locker_df.iterrows():
            locker_coord = (l_row['Latitude'], l_row['Longitude'])
            d_km = distance.distance(cust_coord, locker_coord).kilometers
            if d_km <= max_lock_distance and l_row['remaining_capacity'] >= row['demand']:
                candidate_lockers.append((l_idx, d_km))
        if candidate_lockers:
            candidate_lockers.sort(key=lambda x: x[1])
            chosen_locker_idx = candidate_lockers[0][0]
            locker_df.at[chosen_locker_idx, 'remaining_capacity'] -= row['demand']
            assigned_locker_id = locker_df.at[chosen_locker_idx, 'ID'] if 'ID' in locker_df.columns else chosen_locker_idx
            customer_df.at[idx, 'assigned_locker'] = assigned_locker_id
            customer_df.drop(idx, inplace=True)
        else:
            customer_df.at[idx, 'deliver_type'] = 'last_feet'
    data_for_vrp = pd.concat([depot_df, locker_df, customer_df], ignore_index=True)
    # Veriyi session_state’de saklayalım.
    st.session_state.data_for_vrp = data_for_vrp.copy()
    if "vehicles_df" not in st.session_state:
        st.session_state.vehicles_df = vehicles_df.copy()

    # ---------------------------
    # Node Yönetimi (Sil / Ekle) Kontrolleri
    # ---------------------------
    node_management_mode = st.sidebar.radio("Node Yönetimi", ["Yok", "Sil", "Ekle"])
    if node_management_mode == "Sil":
        st.sidebar.info("Haritada tıklayarak silmek istediğiniz customer veya locker’ı seçin.")
        # Solver'ı güncel veriden oluşturuyoruz.
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic(st.session_state.vehicles_df, forbidden_groups=st.session_state.forbidden_groups)
        # Haritayı oluşturuyoruz (veri olarak güncel data_for_vrp kullanıyoruz)
        current_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        map_data = st_folium(current_map, width=700, height=500)
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            df = st.session_state.data_for_vrp
            # Sadece customer ve locker'ları hedefleyelim.
            df_non_depot = df[df["node_type"] != "depot"]
            if not df_non_depot.empty:
                distances = df_non_depot.apply(lambda row: distance.distance((row["Latitude"], row["Longitude"]), (click_lat, click_lon)).kilometers, axis=1)
                closest_idx = distances.idxmin()
                closest_distance = distances.min()
                if closest_distance < 0.5:  # 500 m içinde ise
                    st.write("Silinecek Node Bilgisi:", df_non_depot.loc[closest_idx])
                    if st.button("Bu node'u sil"):
                        st.session_state.data_for_vrp = st.session_state.data_for_vrp.drop(closest_idx).reset_index(drop=True)
                        st.success("Node silindi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")
    elif node_management_mode == "Ekle":
        st.sidebar.info("Yeni node eklemek için bilgileri girin.")
        with st.form(key="add_node_form"):
            lat = st.number_input("Enlem", value=st.session_state.data_for_vrp["Latitude"].mean())
            lon = st.number_input("Boylam", value=st.session_state.data_for_vrp["Longitude"].mean())
            demand = st.number_input("Demand", value=1.0)
            node_type = st.selectbox("Node Tipi", ["customer", "locker"])
            deliver_type = "last_feet" if node_type == "customer" else "none"
            submit = st.form_submit_button("Ekle")
            if submit:
                if st.session_state.data_for_vrp.empty:
                    new_id = 0
                else:
                    new_id = st.session_state.data_for_vrp["ID"].max() + 1
                new_row = {"ID": new_id, "Latitude": lat, "Longitude": lon, "demand": demand,
                           "node_type": node_type, "deliver_type": deliver_type}
                if node_type == "locker":
                    new_row["remaining_capacity"] = demand
                st.session_state.data_for_vrp = st.session_state.data_for_vrp.append(new_row, ignore_index=True)
                st.success("Yeni node eklendi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    # ---------------------------
    # Rota Çözümleme: Kullanıcının seçtiği yönteme göre
    # ---------------------------
    method = st.sidebar.selectbox("Çözüm Yöntemi Seçiniz:", ["Heuristic (Hızlı)", "MILP (Optimal Ama Yavaş)"])
    if method == "MILP (Optimal Ama Yavaş)":
        routes, total_cost = AdvancedVRPSolver(st.session_state.data_for_vrp).solve_vrp_milp(st.session_state.vehicles_df, 
                                                                                             time_limit=st.sidebar.number_input("MILP Zaman Limiti (saniye):", min_value=1, value=600, step=1))
        if routes is not None:
            st.header("Rotalama Sonuçları (MILP)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, route in routes.items():
                    if route:
                        route_ids = [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in route]
                        st.write(f"Araç {vid}: Ziyaret Sırası (ID'ler) -> {route_ids}")
            route_map = AdvancedVRPSolver(st.session_state.data_for_vrp).create_advanced_route_map(
                {vid: {'route': route,
                       'distance': compute_route_distance(route, AdvancedVRPSolver(st.session_state.data_for_vrp).dist_matrix),
                       'cost': None,
                       'vehicle': {}}
                 for vid, route in routes.items()},
                st.session_state.data_for_vrp
            )
            folium_static(route_map, width=1000, height=600)
        else:
            st.error("MILP çözümü bulunamadı.")
    else:
        route_costs, _ = AdvancedVRPSolver(st.session_state.data_for_vrp).solve_vrp_heuristic(st.session_state.vehicles_df, 
                                                                                              forbidden_groups=st.session_state.forbidden_groups)
        total_cost = sum(rdata['vehicle']['fixed_cost'] + rdata['vehicle']['cost_per_km'] * rdata['distance']
                         for rdata in route_costs.values())
        st.header("Rotalama Sonuçları (Heuristic)")
        st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
        with st.expander("Rota Detayları"):
            for vid, rdata in route_costs.items():
                if isinstance(rdata['route'], list) and len(rdata['route']) > 0 and isinstance(rdata['route'][0], list):
                    route_ids = [ [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in r] for r in rdata['route'] ]
                elif isinstance(rdata['route'], list):
                    route_ids = [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in rdata['route']]
                else:
                    route_ids = []
                st.write(f"Araç {vid}: Rota (ID'ler) -> {route_ids} | Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}")
        route_map = AdvancedVRPSolver(st.session_state.data_for_vrp).create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        folium_static(route_map, width=1000, height=600)

else:
    st.warning("Lütfen hem lokasyon/tip bilgilerini hem de araç bilgilerini içeren Excel dosyalarını yükleyiniz.")

# ---------------------------
# Rota Yenileme Butonu: Node yönetimi sonrası güncel rotayı oluşturur.
if st.button("Rota Yenile"):
    solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
    # Heuristic yöntemi kullanılarak yeniden rota oluşturuluyor.
    route_costs, _ = solver.solve_vrp_heuristic(st.session_state.vehicles_df, forbidden_groups=st.session_state.forbidden_groups)
    st.session_state.last_route_costs = route_costs
    updated_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
    folium_static(updated_map, width=1000, height=600)
    st.success("Rota güncellendi!")
