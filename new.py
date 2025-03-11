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
import pulp
import math
from sklearn.cluster import KMeans
import random
import time  # Progress bar için gerekli
import base64  # Örnek dosya indirmek için gerekli
import io

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

# Simulated Annealing function
def simulated_annealing(initial_route, dist_matrix, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
    def get_route_distance(route):
        return compute_route_distance(route, dist_matrix)
    def get_neighbor(route):
        new_route = route.copy()
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route

    current_route = initial_route
    current_distance = get_route_distance(current_route)
    best_route = current_route
    best_distance = current_distance
    temperature = initial_temp

    while temperature > stopping_temp:
        neighbor_route = get_neighbor(current_route)
        neighbor_distance = get_route_distance(neighbor_route)
        delta_distance = neighbor_distance - current_distance

        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_route = neighbor_route
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        temperature *= cooling_rate

    return best_route, best_distance

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
        # Customer'ların demand değeri, locker'lar için 0 kabul ediliyor.
        self.demands = self.data.apply(lambda row: row['demand'] if row['node_type'] != 'locker' else 0, axis=1).values
        self.dist_matrix = self._calculate_distance_matrix()
        # Customer indekslerini belirleyelim:
        self.customer_indices = self.data.index[self.data['node_type'] == 'customer'].tolist()
        # Locker indekslerini belirleyelim:
        self.locker_indices = self.data.index[self.data['node_type'] == 'locker'].tolist()

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
    # MILP Tabanlı Çözüm
    ##########################################################
    def solve_vrp_milp(self, vehicles_df, forbidden_groups=None, time_limit=None, locker_km_limit=10.0):
        """
        Revize edilmiş MILP tabanlı VRP çözümü:
        - Tek depo, çok locker
        - Müşteriler:
            * last_feet: Doğrudan ziyaret edilmeli.
            * last_mile: Doğrudan ziyaret edilmemeli; Excel’deki "cost" bilgisine göre ve 
                km limiti içinde bulunan locker'lara atanarak, toplam maliyete yansıtılsın.
        - M_last_feet: last_feet müşteri düğümleri.
        - Locker düğümleri: Tüm locker'lar, ancak rotada yalnızca atama (w) sonucu aktif olanlar ziyaret edilir.
        - z[lm,lk]: last_mile müşteri lm'nin, km limiti içinde yer alan locker lk'ya atanmasını sağlayan binary değişken.
        - d[lk]: Locker lk'ya atanan toplam last_mile müşteri talebi (adjusted demand).
        - w[lk]: Locker lk'nın etkin (ziyaret edilecek) olup olmadığını belirten binary değişken.
        - M_all = M_last_feet ∪ L: Rota ve kapasite (MTZ) kısıtları bu küme üzerinden uygulanır.
        - En az bir aracın kullanılma zorunluluğu vardır.
        
        Model, hem rota optimizasyonunu hem de last_mile müşterilerin locker atamasını aynı anda yapar.
        """
        import pulp

        if forbidden_groups is None:
            forbidden_groups = []
        
        # ---------------------
        # 1) Araç Bilgilerinin Tanımlanması
        # ---------------------
        vehicle_ids = list(vehicles_df['vehicle_id'])
        vehicles = {}
        for _, row in vehicles_df.iterrows():
            vehicles[row['vehicle_id']] = {
                'capacity': row['capacity'],
                'max_duration': row['max_duration'] / 60.0,  # dakika -> saat dönüşümü
                'cost_per_km': row['cost_per_km'],
                'fixed_cost': row['fixed_cost']
            }
        
        # ---------------------
        # 2) Düğümlerin Tanımlanması ve Temizlenmesi
        # ---------------------
        N = list(range(self.data.shape[0]))  # Tüm düğümler
        # Depot, Locker ve Customer düğümleri
        D = [i for i in N if str(self.data.loc[i, 'node_type']).strip().lower() == 'depot']
        L = [i for i in N if str(self.data.loc[i, 'node_type']).strip().lower() == 'locker']
        
        # Müşteri düğümleri: last_mile ve last_feet
        M_last_mile = [
            i for i in N 
            if str(self.data.loc[i, 'node_type']).strip().lower() == 'customer'
            and str(self.data.loc[i, 'deliver_type']).strip().lower() == 'last_mile'
        ]
        M_last_feet = [
            i for i in N 
            if str(self.data.loc[i, 'node_type']).strip().lower() == 'customer'
            and str(self.data.loc[i, 'deliver_type']).strip().lower() == 'last_feet'
        ]
        
        print("M_last_mile:", M_last_mile)
        print("M_last_feet:", M_last_feet)
        
        if not L and len(M_last_mile) > 0:
            raise ValueError("Last mile müşteriler var fakat hiç locker yok!")
        
        # ---------------------
        # 3) MILP Modeli ve Rota Karar Değişkenleri
        # ---------------------
        prob = pulp.LpProblem("SingleDepot_VRP_LastMileFeet", pulp.LpMinimize)
        
        # Rota karar değişkenleri: x[i,j,v] ∈ {0,1}
        x = {}
        for i in N:
            for j in N:
                if i == j:
                    continue
                for v in vehicle_ids:
                    x[i, j, v] = pulp.LpVariable(f"x_{i}_{j}_{v}", cat="Binary")
        # Araç kullanım değişkeni: y[v]
        y = {v: pulp.LpVariable(f"y_{v}", cat="Binary") for v in vehicle_ids}
        
        # ---------------------
        # 4) Last_mile Müşteri Atama Karar Değişkenleri (z) ve Atama Kısıtları
        # ---------------------
        # Her last_mile müşteri lm için, km limiti (locker_km_limit) içinde yer alan locker'lara atama yapılır.
        z = {}  # z[lm,lk] ∈ {0,1}
        for lm in M_last_mile:
            candidate_lockers = [lk for lk in L if self.dist_matrix[lm][lk] <= locker_km_limit]
            if not candidate_lockers:
                raise ValueError(f"Last mile müşteri {lm} için km limiti içerisinde uygun locker bulunamadı!")
            # Her lm için aday locker'ların toplamı 1 olmalı:
            prob += pulp.lpSum(z.setdefault((lm, lk), pulp.LpVariable(f"z_{lm}_{lk}", cat="Binary")) 
                                for lk in candidate_lockers) == 1, f"Assign_{lm}"
        
        # ---------------------
        # 5) Locker Kullanım Karar Değişkeni (w) ve Atanmış Talep (d) Değişkenleri
        # ---------------------
        # w[lk]: locker'ın etkin olarak kullanılıp kullanılmadığını belirler.
        w = {lk: pulp.LpVariable(f"w_{lk}", cat="Binary") for lk in L}
        # d[lk]: locker lk'ya atanan toplam last_mile müşteri talebi
        d = {}
        for lk in L:
            d[lk] = pulp.LpVariable(f"d_{lk}", lowBound=0)
            # lm'lerden, yalnızca (lm,lk) tanımlı olanları toplayarak d[lk] tanımlanır.
            candidate_lm = [lm for lm in M_last_mile if (lm, lk) in z]
            if candidate_lm:
                prob += d[lk] == pulp.lpSum(self.demands[lm] * z[lm, lk] for lm in candidate_lm), f"DemandAssign_{lk}"
            else:
                prob += d[lk] == 0, f"DemandAssign_{lk}"
            # Locker kapasite kısıtı: d[lk] <= capacity[lk]
            locker_capacity = self.data.loc[lk, 'capacity']
            prob += d[lk] <= locker_capacity, f"LockerCap_{lk}"
            # Locker kullanım gösterimi: d[lk] > 0 olduğunda w[lk] = 1 (aksi halde d[lk] = 0)
            prob += d[lk] <= locker_capacity * w[lk], f"LockerUsage_{lk}"
            prob += pulp.lpSum(z[lm, lk] for lm in candidate_lm) >= w[lk], f"LockerActive_{lk}"
        
        # ---------------------
        # 6) Adjusted Demands ve M_all
        # ---------------------
        # M_all: Ziyaret edilecek düğümler = M_last_feet ∪ L (rotada tüm locker'lar var, ancak ziyaret kısıtı w ile sağlanır)
        M_all = M_last_feet + L
        # Last_feet için demand sabit, locker için adjusted demand d[lk]'dır.
        def adjusted_demand(i):
            if i in M_last_feet:
                return self.demands[i]
            elif i in L:
                return d[i]
            else:
                return 0
        
        # ---------------------
        # 7) MTZ ve Rota Yük Değişkenleri (u)
        # ---------------------
        # u[i,v]: araç v'nin, M_all üzerindeki düğüm i'ye geldiğinde taşıdığı yük.
        u = {}
        for i in M_all:
            for v in vehicle_ids:
                lb = self.demands[i] if i in M_last_feet else 0  # Locker için alt sınır daha sonra u[i,v] >= d[i] ile sağlanır.
                u[i, v] = pulp.LpVariable(f"u_{i}_{v}", lowBound=lb)
                prob += u[i, v] <= vehicles[v]['capacity'], f"LoadMax_{i}_{v}"
                if i in L:
                    prob += u[i, v] >= d[i], f"LockerLoadMin_{i}_{v}"
        
        # ---------------------
        # 8) Objective Function
        # ---------------------
        # Rota maliyeti: her araç için sabit maliyet + kat edilen mesafe * km başına maliyet
        routing_cost = pulp.lpSum(
            vehicles[v]['fixed_cost'] * y[v] +
            pulp.lpSum(self.dist_matrix[i][j] * vehicles[v]['cost_per_km'] * x[i, j, v]
                    for i in N for j in N if i != j)
            for v in vehicle_ids
        )
        # Atama avantajı (cost saving): her last_mile müşteri için, atanılan locker ile arasındaki mesafe * Excel'deki "cost" değeri
        assignment_saving = pulp.lpSum(
            self.dist_matrix[lm][lk] * self.data.loc[lm, 'cost'] * z[lm, lk]
            for (lm, lk) in z
        )
        # Nihai amaç: toplam rota maliyeti - atama avantajı
        prob += routing_cost - assignment_saving, "Total_Cost"
        
        # ---------------------
        # 9) Ziyaret Kısıtları
        # ---------------------
        # (a) last_feet müşterileri: tam 1 kez ziyaret
        for j in M_last_feet:
            prob += pulp.lpSum(x[i, j, v] for i in N if i != j for v in vehicle_ids) == 1, f"Visit_LastFeet_{j}"
        # (b) Locker düğümleri: w[j] aktifse tam 1 ziyaret, değilse ziyaret edilmemeli
        for j in L:
            prob += pulp.lpSum(x[i, j, v] for i in N if i != j for v in vehicle_ids) == w[j], f"Visit_Locker_{j}"
        # (c) last_mile müşteri düğümleri: rota dışı, atama yoluyla ele alınıyor
        for j in M_last_mile:
            prob += pulp.lpSum(x[i, j, v] for i in N if i != j for v in vehicle_ids) == 0, f"NoVisit_LastMile_{j}"
        
        # ---------------------
        # 10) Depodan Çıkış ve Depoya Dönüş Kısıtları
        # ---------------------
        for v in vehicle_ids:
            prob += pulp.lpSum(x[d, j, v] for d in D for j in N if d != j) == y[v], f"Depart_{v}"
            prob += pulp.lpSum(x[i, d, v] for d in D for i in N if i != d) == y[v], f"Return_{v}"
        
        # ---------------------
        # 11) Akış (Flow) Korunumu
        # ---------------------
        for v in vehicle_ids:
            for k in M_all:
                prob += (pulp.lpSum(x[i, k, v] for i in N if i != k) ==
                        pulp.lpSum(x[k, j, v] for j in N if j != k)), f"Flow_{k}_{v}"
        
        # ---------------------
        # 12) MTZ (Kapasite/Load) Kısıtları
        # ---------------------
        for v in vehicle_ids:
            for i in M_all:
                for j in M_all:
                    if i == j:
                        continue
                    prob += (u[i, v] + adjusted_demand(j) <= 
                            u[j, v] + vehicles[v]['capacity'] * (1 - x[i, j, v])), f"MTZ_{i}_{j}_{v}"
        
        # ---------------------
        # 13) Süre (Time) Kısıtı
        # ---------------------
        speed = 60.0  # km/saat
        for v in vehicle_ids:
            prob += pulp.lpSum(
                (self.dist_matrix[i][j] / speed) * x[i, j, v]
                for i in N for j in N if i != j
            ) <= vehicles[v]['max_duration'], f"TimeLimit_{v}"
        
        # ---------------------
        # 14) Forbidden Group Kısıtları (Varsa)
        # ---------------------
        for group in forbidden_groups:
            for v in vehicle_ids:
                prob += pulp.lpSum(x[i, j, v] for i in group for j in group if i != j) <= 1, f"Forbidden_{v}_{str(group)}"
        
        # ---------------------
        # 15) Modelin Çözümü
        # ---------------------
        if time_limit:
            solver_instance = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
            prob.solve(solver_instance)
        else:
            prob.solve()
        
        print("Solver Status:", pulp.LpStatus[prob.status])
        
        # ---------------------
        # 16) Rota Çıkarma
        # ---------------------
        routes = {}
        for v in vehicle_ids:
            if pulp.value(y[v]) < 0.5:
                continue
            start_depot = None
            for dpt in D:
                for j in N:
                    if dpt != j and pulp.value(x[dpt, j, v]) > 0.5:
                        start_depot = dpt
                        break
                if start_depot is not None:
                    break
            if start_depot is None:
                routes[v] = [D[0], D[0]]
                continue
            route = [start_depot]
            visited = {start_depot}
            current = start_depot
            while True:
                next_node = None
                for j in N:
                    if j != current:
                        val = pulp.value(x[current, j, v])
                        if val is not None and val > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node in visited:
                    if route[-1] not in D:
                        route.append(D[0])
                    break
                else:
                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node
            routes[v] = route
        
        total_cost = pulp.value(prob.objective) if routes else None
        return routes, total_cost

    ##########################################################
    # Heuristic Yöntem – Feasible Sweep Algoritması
    ##########################################################
    def solve_vrp_heuristic(self, vehicles_df, forbidden_groups=None):
        if forbidden_groups is None:
            forbidden_groups = []
        speed = 60.0
        max_capacity = vehicles_df['capacity'].max()
        max_duration_minutes = vehicles_df['max_duration'].max()
        max_distance = (max_duration_minutes / 60.0) * speed
        N = list(range(self.data.shape[0]))
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

        # (C) Yerel iyileştirme ve forbidden kontrolü.
        optimized_routes = []
        for route in routes_all:
            if len(route) > 3:
                route_opt = two_opt(route, self.dist_matrix)
                route_opt = three_opt(route_opt, self.dist_matrix)
            else:
                route_opt = route
            route_opt = self.fix_route_forbidden(route_opt, forbidden_groups)
            optimized_routes.append(route_opt)

        # (D) Araçlara atama
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
        """
        Heuristic çözüm üzerinde Simulated Annealing (SA) uygulayarak rota iyileştirmesi yapar.
         - İlk çözüm, solve_vrp_heuristic ile oluşturulur.
         - Her rota üzerinde SA uygulanarak, rota sıralaması (ve dolayısıyla mesafe ve maliyet) iyileştirilir.
         - Son olarak, her rota için araç ataması maliyetler üzerinden tekrar değerlendirilir.
         - Bu adım, araçların km başına maliyetleri ve sabit maliyetleri göz önünde bulundurarak en uygun atanmayı sağlar.
        """
        # Adım 1: Temel heuristic çözüm
        route_assignments, _ = self.solve_vrp_heuristic(vehicles_df, forbidden_groups)
    
        # Adım 2: Her rota üzerinde SA uygulayarak iyileştirme
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
    
        # Adım 3: Araç Maliyetlerini Göz Önüne Alarak Rota Ataması
        final_route_assignments = {}
        for vid, rdata in sa_optimized_routes.items():
            route = rdata['route']
            route_distance = compute_route_distance(route, self.dist_matrix)
            best_vehicle = None
            best_vehicle_cost = float('inf')
            for v in vehicles_df['vehicle_id']:
                # Hesaplama: Sabit maliyet + (mesafe * cost_per_km)
                v_row = vehicles_df[vehicles_df['vehicle_id'] == v].iloc[0]
                cost = v_row['fixed_cost'] + route_distance * v_row['cost_per_km']
                if cost < best_vehicle_cost:
                    best_vehicle_cost = cost
                    best_vehicle = v
            final_route_assignments[best_vehicle] = {
                'route': route,
                'distance': route_distance,
                'demand': rdata['demand'],
                'vehicle': vehicles_df[vehicles_df['vehicle_id'] == best_vehicle].iloc[0].to_dict()
            }
        return final_route_assignments, []

    ##########################################################
    # OSRM tabanlı interaktif rota görselleştirme
    ##########################################################
    def create_advanced_route_map(self, route_costs, data_source):
        """
        route_costs: { vehicle_id: { 'route': [...], 'distance': float, 'cost': float, ... }, ... }
        data_source: Gösterimde kullanılacak DataFrame (örneğin, st.session_state.data_for_vrp)
        """
        center_lat = self.data['Latitude'].mean()
        center_lon = self.data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'darkblue', 'cadetblue', 'darkgreen', 'darkpurple', 'pink',
                  'lightred', 'beige', 'lightblue', 'lightgreen', 'gray']
        
        # 1) Nokta Marker'ları
        for idx, row in data_source.iterrows():
            node_type = row['node_type']
            deliver_type = row.get('deliver_type', 'none')
            is_last_mile = row.get('is_last_mile', False)
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
                if is_last_mile:
                    marker_color = 'orange'
                    icon_ = 'motorcycle'
                else:
                    marker_color = 'blue' if deliver_type == 'last_feet' else 'orange'
                    icon_ = 'info-sign'
                popup_text = f"Müşteri (ID: {row['ID']}) - Talep: {row['demand']} - {deliver_type}"
            
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon=icon_)
            ).add_to(m)
        
        # 2) Last Mile Atamalarını Kesikli Çizgi ile Göster
        # 2) Last Mile Atamalarını Kesikli Çizgi ile Göster
        for _, row in data_source.iterrows():
            if row.get('is_last_mile', False) and pd.notnull(row.get('assigned_locker')):
                cust_coord = (row['Latitude'], row['Longitude'])
                locker_row = data_source[data_source['ID'] == row['assigned_locker']]
                if not locker_row.empty:
                    locker_coord = (locker_row.iloc[0]['Latitude'], locker_row.iloc[0]['Longitude'])
                    folium.PolyLine(
                        [cust_coord, locker_coord],
                        color='black',
                        weight=2,
                        opacity=0.7,
                        dash_array='5, 5',  # Kesikli çizgi
                        tooltip=f"Last Mile: Müşteri {row['ID']} -> Locker {row['assigned_locker']}"
                    ).add_to(m)

        
        # 3) Rota Segmentlerinin Çizimi (OSRM)
        osrm_cache = {}
        for idx, (vid, rdata) in enumerate(route_costs.items()):
            route_val = rdata.get('route', None)
            if route_val is None or not isinstance(route_val, list) or len(route_val) < 2:
                continue
            
            # Eğer rota tek liste ise; liste içinde listeye dönüştürelim:
            if not isinstance(route_val[0], list):
                routes = [route_val]
            else:
                routes = route_val
            
            color = colors[idx % len(colors)]
            total_distance = rdata.get('distance', 0)
            total_cost = rdata.get('cost', 0)
            
            for sub_route in routes:
                if not isinstance(sub_route, list) or len(sub_route) < 2:
                    continue
                
                full_route_coords = []
                for i in range(len(sub_route)-1):
                    start_node = sub_route[i]
                    end_node = sub_route[i+1]
                    start_coord = (self.data.iloc[start_node]['Latitude'], self.data.iloc[start_node]['Longitude'])
                    end_coord = (self.data.iloc[end_node]['Latitude'], self.data.iloc[end_node]['Longitude'])
                    segment_coords = self.get_osrm_route(start_coord, end_coord, osrm_cache)
                    if not segment_coords:
                        segment_coords = [start_coord, end_coord]
                    if full_route_coords:
                        full_route_coords.extend(segment_coords[1:])
                    else:
                        full_route_coords.extend(segment_coords)
                
                popup_text = f"Araç {vid} - Toplam Mesafe: {total_distance:.2f} km"
                if total_cost:
                    popup_text += f", Maliyet: {total_cost:.2f}"
                
                folium.PolyLine(
                    full_route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=popup_text,
                    tooltip=f"Araç ID: {vid}"
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m

###############################################
# Streamlit Uygulaması – Dosya Yükleme ve Node Yönetimi
###############################################
st.set_page_config(page_title="Gelişmiş VRP Çözümleme", layout="wide")

# Tema stilini uygula (Koyu arka plan + açık renk yazı)
dark_theme = """
<style>
    /* Global ayarlar */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #121212;
        color: #F5F5F5;
        font-family: 'Roboto', sans-serif; /* Dilerseniz istediğiniz fontu kullanabilirsiniz */
    }
    .stApp {
        background-color: #121212;
        color: #F5F5F5;
    }

    /* Sidebar ayarları */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    /* Sidebar içindeki text, label, vs. */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stCaption {
        color: #FFFFFF !important;
    }

    /* Butonlar */
    .stButton > button {
        background-color: #333;
        color: #F5F5F5;
        border: none;
    }
    .stButton > button:hover {
        background-color: #444;
        color: #FFFFFF;
    }

    /* Metin kutuları, file uploader vb. */
    .stTextInput > div > div > input {
        background-color: #333;
        color: #F5F5F5;
    }
    .stFileUploader > div, .stFileUploader label, .stFileUploader span {
        background-color: #333;
        color: #F5F5F5 !important;
    }
    .stDownloadButton > button {
        background-color: #333;
        color: #F5F5F5;
    }
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

st.title("🚚 :blue[Gelişmiş Araç Rotalama Problemi Çözücü (İstanbul)]")

# Sidebar’da forbidden node grupları…
st.sidebar.subheader(":blue[Aynı Rotada Bulunamayacak Nodelar]")
if 'forbidden_groups' not in st.session_state:
    st.session_state.forbidden_groups = []
all_node_ids = []
if st.session_state.get('original_data') is not None:
    all_node_ids = sorted(st.session_state.original_data['ID'].unique())
else:
    all_node_ids = list(range(100))
selected_nodes = st.sidebar.multiselect(":blue[Rota içinde yan yana bulunmasın istenen node ID’leri:]", 
                                        options=all_node_ids, 
                                        key='forbidden_nodes_multiselect')
if st.sidebar.button(":blue[Ekle]"):
    if selected_nodes:
        st.session_state.forbidden_groups.append(selected_nodes)
        st.sidebar.success(f"Eklendi: {selected_nodes}")
    else:
        st.sidebar.warning(":blue[Lütfen en az bir node seçiniz.]")
if st.session_state.forbidden_groups:
    st.sidebar.write(":blue[Eklenen forbidden gruplar:]")
    for grp in st.session_state.forbidden_groups:
        st.sidebar.write(grp)

st.title(":blue[Veri Yükleme Paneli] 📂")

st.markdown("""
**Lütfen aşağıdaki dosyaları yükleyin:**
- **Lokasyon Dosyası**: `Latitude`, `Longitude`, `demand`, `node_type`, `deliver_type` kolonlarını içermelidir.
- **Araç Dosyası**: `vehicle_id`, `capacity`, `max_duration`, `cost_per_km`, `fixed_cost` kolonlarını içermelidir.
""")

# Örnek Excel oluşturan fonksiyon
def create_example_file():
    sample_data = {
        "id": ["Adana1", "Locker2"],
        "Latitude": [40.123, 41.456],
        "Longitude": [29.789, 30.123],
        "demand": [5, 10],
        "node_type": ["depot", "customer"],
        "deliver_type": ["normal", "last_mile"]
    }
    df = pd.DataFrame(sample_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Örnek dosya indirme bağlantısı
st.download_button(
    label="📥 :blue[Örnek Lokasyon Dosyasını İndir]",
    data=create_example_file(),
    file_name="ornek_lokasyon.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Dosya yükleme alanı
st.markdown("### 📍 :blue[Lokasyon Dosyası Yükle]")
uploaded_nodes_file = st.file_uploader(":blue[Lokasyon Dosyası (Excel formatında)]", type="xlsx", key="nodes")

st.markdown("### 🚗 :blue[Araç Dosyası Yükle]")
uploaded_vehicles_file = st.file_uploader(":blue[Araç Dosyası (Excel formatında)]", type="xlsx", key="vehicles")

def preview_uploaded_file(file, file_type):
    if file is not None:
        try:
            df = pd.read_excel(file)
            st.write(f"📋 **{file_type} Dosyası Önizleme:**")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"❌ Hata: Dosya okunamadı! Geçerli bir Excel dosyası yüklediğinizden emin olun. ({str(e)})")
            return None
    return None

# Yüklenen dosyaların önizlemesini göster
original_data = preview_uploaded_file(uploaded_nodes_file, "Lokasyon")
vehicles_df = preview_uploaded_file(uploaded_vehicles_file, "Araç")

# Eğer iki dosya da yüklenmişse
if uploaded_nodes_file is not None and uploaded_vehicles_file is not None:
    with st.spinner(":blue[Dosyalar yükleniyor...] ⏳"):
        time.sleep(1)

    st.success("✔️ :blue[Dosyalar başarıyla yüklendi!]")

    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    # Okunan excel’leri dataframe’e al
    original_data = pd.read_excel(uploaded_nodes_file)
    vehicles_df = pd.read_excel(uploaded_vehicles_file)

    st.session_state.original_data = original_data.copy()

    needed_node_cols = ['Latitude', 'Longitude', 'demand', 'node_type', 'deliver_type']
    needed_vehicle_cols = ['vehicle_id', 'capacity', 'max_duration', 'cost_per_km', 'fixed_cost']

    if not all(col in original_data.columns for col in needed_node_cols):
        st.error(f"❌ Lokasyon dosyasında şu kolonlar eksik veya hatalı: {needed_node_cols}")
        st.stop()

    if not all(col in vehicles_df.columns for col in needed_vehicle_cols):
        st.error(f"❌ Araç dosyasında şu kolonlar eksik veya hatalı: {needed_vehicle_cols}")
        st.stop()

    # Depo, locker, müşteri ayrıştırma
    depot_df = original_data[original_data['node_type'] == 'depot'].copy()
    locker_df = original_data[original_data['node_type'] == 'locker'].copy()
    customer_df = original_data[original_data['node_type'] == 'customer'].copy()

    if depot_df.empty:
        st.error("❌ :blue[En az bir depo bulunmalıdır!]")
        st.stop()

    if not locker_df.empty:
        locker_df['remaining_capacity'] = locker_df['demand']

    max_lock_distance = st.sidebar.number_input(":blue[Maksimum Locker Atama Mesafesi (km):]", min_value=0.1, value=2.0, step=0.1)

    # Last mile teslimatlar için locker ataması
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
            customer_df.at[idx, 'is_last_mile'] = True  # Last mile olduğunu işaretle
            # customer_df.drop(idx, inplace=True)  -> Bu satırı kaldırıyoruz, böylece last_mile nodeları veri setinde kalır.
        else:
            customer_df.at[idx, 'deliver_type'] = 'last_feet'

    data_for_vrp = pd.concat([depot_df, locker_df, customer_df], ignore_index=True)

    st.session_state.data_for_vrp = data_for_vrp.copy()
    if "vehicles_df" not in st.session_state:
        st.session_state.vehicles_df = vehicles_df.copy()

    st.success("✅ :blue[Veriler başarıyla işlendi ve kaydedildi!]")

    # Node Yönetimi (Sil / Ekle) – Örnek yaklaşım
    node_management_mode = st.sidebar.radio("Node Yönetimi", ["Yok", "Sil", "Ekle"])

    if node_management_mode == "Sil":
        st.sidebar.info(":blue[Haritada tıklayarak silmek istediğiniz customer veya locker’ı seçin.]")
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic(st.session_state.vehicles_df, 
                                                    forbidden_groups=st.session_state.forbidden_groups)
        current_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        map_data = st_folium(current_map, width=700, height=500)
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            df = st.session_state.data_for_vrp
            df_non_depot = df[df["node_type"] != "depot"]
            if not df_non_depot.empty:
                distances = df_non_depot.apply(
                    lambda row: distance.distance((row["Latitude"], row["Longitude"]), (click_lat, click_lon)).kilometers,
                    axis=1
                )
                closest_idx = distances.idxmin()
                closest_distance = distances.min()
                if closest_distance < 0.5:
                    st.write("Silinecek Node Bilgisi:", df_non_depot.loc[closest_idx])
                    if st.button("Bu node'u sil"):
                        st.session_state.data_for_vrp = st.session_state.data_for_vrp.drop(closest_idx).reset_index(drop=True)
                        st.success("Node silindi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    elif node_management_mode == "Ekle":
        st.sidebar.info(":blue[Yeni node eklemek için bilgileri girin.]")
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
                    # ID kolonunuzun ismi “ID” değil de “id” ise ona göre değiştiriniz
                    new_id = st.session_state.data_for_vrp["ID"].max() + 1 if "ID" in st.session_state.data_for_vrp.columns else 0
                new_row = {
                    "ID": new_id,
                    "Latitude": lat,
                    "Longitude": lon,
                    "demand": demand,
                    "node_type": node_type,
                    "deliver_type": deliver_type
                }
                if node_type == "locker":
                    new_row["remaining_capacity"] = demand
                st.session_state.data_for_vrp = st.session_state.data_for_vrp.append(new_row, ignore_index=True)
                st.success("Yeni node eklendi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    # Rota çözüm yöntemi seçimi
    method = st.sidebar.selectbox("Çözüm Yöntemi Seçiniz:", ["Heuristic (Hızlı)",
                                                             "MILP (Optimal Ama Yavaş)",
                                                             "Heuristic + Simulated Annealing (SA)"])

    if method == "Heuristic + Simulated Annealing (SA)":
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic_with_sa(
            st.session_state.vehicles_df, 
            forbidden_groups=st.session_state.forbidden_groups
        )
        total_cost = sum(rdata['vehicle']['fixed_cost'] 
                         + rdata['vehicle']['cost_per_km'] * rdata['distance']
                         for rdata in route_costs.values())
        
        st.header("Rotalama Sonuçları (Heuristic + Simulated Annealing)")
        st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
        with st.expander("Rota Detayları"):
            for vid, rdata in route_costs.items():
                route_ids = [str(solver.data.iloc[node]['ID']) for node in rdata['route']]
                st.write(f"Araç {vid}: Rota (ID'ler) -> {route_ids} | Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}")
        route_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        folium_static(route_map, width=1000, height=600)

    elif method == "MILP (Optimal Ama Yavaş)":
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        routes, total_cost = solver.solve_vrp_milp(
            st.session_state.vehicles_df, 
            time_limit=st.sidebar.number_input("MILP Zaman Limiti (saniye):", min_value=1, value=600, step=1),
            locker_km_limit = max_lock_distance
        )

        
        if routes is not None:
            st.header("Rotalama Sonuçları (MILP)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, route in routes.items():
                    if route:
                        route_ids = [str(solver.data.iloc[node]['ID']) for node in route]
                        st.write(f"Araç {vid}: Ziyaret Sırası (ID'ler) -> {route_ids}")
            route_map = solver.create_advanced_route_map(
                {
                    vid: {
                        'route': route,
                        'distance': compute_route_distance(route, solver.dist_matrix),
                        'cost': None,
                        'vehicle': {}
                    }
                    for vid, route in routes.items()
                },
                st.session_state.data_for_vrp
            )
            folium_static(route_map, width=1000, height=600)
        else:
            st.error("MILP çözümü bulunamadı.")

    else:
        # Heuristic (hızlı) yöntem
        with st.spinner('Rota hesaplanıyor...'):
            solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
            route_costs, _ = solver.solve_vrp_heuristic(
                st.session_state.vehicles_df, 
                forbidden_groups=st.session_state.forbidden_groups
            )
            total_cost = sum(rdata['vehicle']['fixed_cost'] 
                             + rdata['vehicle']['cost_per_km'] * rdata['distance']
                             for rdata in route_costs.values())

            st.header("Rotalama Sonuçları (Heuristic)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, rdata in route_costs.items():
                    if isinstance(rdata['route'], list) and len(rdata['route']) > 0 and isinstance(rdata['route'][0], list):
                        route_ids = [
                            [str(solver.data.iloc[node]['ID']) for node in r] for r in rdata['route']
                        ]
                    elif isinstance(rdata['route'], list):
                        route_ids = [str(solver.data.iloc[node]['ID']) for node in rdata['route']]
                    else:
                        route_ids = []
                    st.write(
                        f"Araç {vid}: Rota (ID'ler) -> {route_ids} | "
                        f"Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}"
                    )
            route_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
            folium_static(route_map, width=1000, height=600)

else:
    st.warning("Lütfen hem lokasyon/tip bilgilerini hem de araç bilgilerini içeren Excel dosyalarını yükleyiniz.")

# Rota Yenile Butonu
if st.button("Rota Yenile"):
    solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
    route_costs, _ = solver.solve_vrp_heuristic(
        st.session_state.vehicles_df, 
        forbidden_groups=st.session_state.forbidden_groups
    )
    st.session_state.last_route_costs = route_costs
    updated_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
    folium_static(updated_map, width=1000, height=600)
    st.success("Rota güncellendi!")
